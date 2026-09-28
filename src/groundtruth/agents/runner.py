from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from typing import Any

from openai import OpenAI

from groundtruth.agents.protocol import system_prompt, tool_definitions
from groundtruth.cache import get_or_compute
from groundtruth.costs import UsageLedger
from groundtruth.retrieval.base import Retriever
from groundtruth.schemas import AgentRun, Chunk, ToolCall


class AgentRunner:
    def __init__(
        self,
        client: OpenAI,
        ledger: UsageLedger,
        retriever: Retriever,
        chunks: Sequence[Chunk],
        backend: str,
        mode: str = "controlled",
        model: str = "gpt-4.1-mini",
        reasoning_effort: str | None = None,
        prompt_version: str = "shared_v1",
        max_tool_calls: int = 5,
        max_steps: int = 8,
        cache_dir: str = "cache/agent_runs",
        graph_max_hops: int = 2,
    ):
        if max_tool_calls < 0 or max_steps < 1:
            raise ValueError("max_tool_calls must be >= 0 and max_steps >= 1")
        self.client = client
        self.ledger = ledger
        self.retriever = retriever
        self.chunks = list(chunks)
        self.backend = backend
        self.mode = mode
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.prompt_version = prompt_version
        self.max_tool_calls = max_tool_calls
        self.max_steps = max_steps
        self.cache_dir = cache_dir
        self.graph_max_hops = graph_max_hops
        self.chunks_by_id = {chunk.id: chunk for chunk in chunks}
        self.document_ids = sorted({chunk.document_id for chunk in chunks})
        self.document_titles = {
            chunk.document_id: chunk.document_title for chunk in chunks
        }
        self.tools = tool_definitions(mode, backend, self.document_titles)
        serialized_tools = json.dumps(self.tools, sort_keys=True, ensure_ascii=False)
        self.tool_schema_version = hashlib.sha256(serialized_tools.encode("utf-8")).hexdigest()

    def run(self, query_id: str, query: str, index_fingerprint: str) -> AgentRun:
        key = {
            "query_id": query_id,
            "query": query,
            "backend": self.backend,
            "index_fingerprint": index_fingerprint,
            "mode": self.mode,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "prompt_version": self.prompt_version,
            "tool_schema_version": self.tool_schema_version,
            "max_tool_calls": self.max_tool_calls,
            "max_steps": self.max_steps,
        }
        payload, hit = get_or_compute(
            self.cache_dir,
            key,
            lambda: self._run_uncached(query_id, query),
        )
        cached_run = AgentRun.from_record(payload)
        if hit:
            return AgentRun(
                query_id=cached_run.query_id,
                answer=cached_run.answer,
                contexts=cached_run.contexts,
                tool_calls=cached_run.tool_calls,
                input_tokens=cached_run.input_tokens,
                output_tokens=cached_run.output_tokens,
                cost_usd=cached_run.cost_usd,
                latency_ms=cached_run.latency_ms,
                error=cached_run.error,
                cache_hit=True,
            )
        return cached_run

    def _run_uncached(self, query_id: str, query: str) -> dict[str, Any]:
        start = time.perf_counter()
        starting_cost = self.ledger.cost_usd
        tools_called: list[ToolCall] = []
        contexts: list[str] = []
        input_tokens = output_tokens = 0
        error: str | None = None
        answer = ""
        messages: list[Any] = [{"role": "user", "content": query}]
        prompt = system_prompt(self.mode, self.backend, self.prompt_version)
        try:
            for _ in range(self.max_steps):
                request = {
                    "model": self.model,
                    "instructions": prompt,
                    "input": messages,
                    "tools": self.tools,
                    "tool_choice": "auto",
                }
                if self.reasoning_effort:
                    request["reasoning"] = {"effort": self.reasoning_effort}
                response = self.client.responses.create(**request)
                usage = response.usage
                current_input = int(getattr(usage, "input_tokens", 0) or 0)
                current_output = int(getattr(usage, "output_tokens", 0) or 0)
                input_details = getattr(usage, "input_tokens_details", None)
                cached_input = int(getattr(input_details, "cached_tokens", 0) or 0)
                input_tokens += current_input
                output_tokens += current_output
                self.ledger.record(
                    self.model,
                    input_tokens=current_input,
                    output_tokens=current_output,
                    cached_input_tokens=cached_input,
                )
                function_calls = [item for item in response.output if item.type == "function_call"]
                if not function_calls:
                    answer = response.output_text or ""
                    break
                messages.extend(item.model_dump(exclude_none=True) for item in response.output)
                for call in function_calls:
                    if len(tools_called) >= self.max_tool_calls:
                        error = "maximum_tool_calls_reached"
                        break
                    call_start = time.perf_counter()
                    arguments = json.loads(call.arguments)
                    try:
                        result, retrieved = self._execute_tool(call.name, arguments)
                        contexts.extend(text for text in retrieved if text not in contexts)
                        trace = ToolCall(call.name, arguments, True,
                                         (time.perf_counter() - call_start) * 1000)
                    except Exception as exc:
                        result = {"error": str(exc)}
                        trace = ToolCall(call.name, arguments, False,
                                         (time.perf_counter() - call_start) * 1000,
                                         error=str(exc))
                    tools_called.append(trace)
                    messages.append({
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    })
                else:
                    continue
                break
            else:
                error = "maximum_steps_reached"
            if not answer and not error:
                error = "empty_model_answer"
        except Exception as exc:
            error = f"agent_request_failed: {type(exc).__name__}: {exc}"
        result = AgentRun(
            query_id=query_id,
            answer=answer,
            contexts=tuple(contexts),
            tool_calls=tuple(tools_called),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=max(0.0, self.ledger.cost_usd - starting_cost),
            latency_ms=(time.perf_counter() - start) * 1000,
            error=error,
        )
        return result.as_record()

    def _execute_tool(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        requested_k = max(1, min(int(args.get("k", 10)), 50))
        if name == "retrieve":
            scope = args.get("scope")
            results = self.retriever.retrieve(args["query"], self.chunks, requested_k, scope)
        elif name == "search_all":
            results = self.retriever.retrieve(args["query"], self.chunks, requested_k)
        elif name == "search_by_document":
            document_id = args["document_id"]
            if document_id not in self.document_ids:
                raise ValueError(f"Unknown document_id: {document_id}")
            results = self.retriever.retrieve(
                args["query"], self.chunks, requested_k, [document_id]
            )
        elif name == "graph_search":
            if hasattr(self.retriever, "max_hops"):
                self.retriever.max_hops = max(0, min(int(args["depth"]), 4))
            results = self.retriever.retrieve(args["query"], self.chunks, requested_k)
        elif name == "fetch_source_chunks":
            found = [self.chunks_by_id[item] for item in args["chunk_ids"] if item in self.chunks_by_id]
            texts = [f"[{chunk.id} | {chunk.document_title}]\n{chunk.text}" for chunk in found]
            return {"chunks": texts, "missing_ids": sorted(set(args["chunk_ids"]) - set(self.chunks_by_id))}, texts
        else:
            raise ValueError(f"Unknown tool: {name}")
        texts = [
            f"[{item.chunk.id} | {item.chunk.document_title} | score={item.score:.4f}]\n{item.chunk.text}"
            for item in results
        ]
        return {"results": texts}, texts
