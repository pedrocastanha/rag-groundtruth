from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from groundtruth.agents.trace import summarize_tool_calls
from groundtruth.cache import load_cache, save_cache
from groundtruth.schemas import AgentRun, QueryCase


def summarize_agent_runs(runs: Sequence[AgentRun]) -> dict[str, Any]:
    if not runs:
        return {"count": 0, "metrics": {}}
    trace_summaries = [summarize_tool_calls(run.tool_calls) for run in runs]
    denominator = len(runs)
    return {
        "count": denominator,
        "successful": sum(not run.error for run in runs),
        "failed": sum(bool(run.error) for run in runs),
        "cache_hits": sum(run.cache_hit for run in runs),
        "metrics": {
            "tool_calls_per_answer": sum(item["tool_calls"] for item in trace_summaries) / denominator,
            "tool_calls_by_name": _tool_counts(runs),
            "unique_tools_per_answer": sum(item["unique_tools"] for item in trace_summaries) / denominator,
            "failed_tool_calls": sum(item["failed_tool_calls"] for item in trace_summaries),
            "repeated_tool_calls": sum(item["repeated_tool_calls"] for item in trace_summaries),
            "cost_usd_per_answer": sum(run.cost_usd for run in runs) / denominator,
            "latency_ms_per_answer": sum(run.latency_ms for run in runs) / denominator,
        },
    }


def _tool_counts(runs: Sequence[AgentRun]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for run in runs:
        for call in run.tool_calls:
            counts[call.name] += 1
    return dict(sorted(counts.items()))


async def _score_one_metric(
    metric_name: str,
    cache_key: dict[str, object],
    score_call: Callable[[], Awaitable[Any]],
    semaphore: asyncio.Semaphore,
    timeout_seconds: float = 180,
) -> dict[str, Any]:
    cache_dir = Path("cache/generation_eval")
    cached = load_cache(cache_dir, cache_key)
    if cached is not None:
        return cached
    try:
        async with semaphore:
            metric_result = await asyncio.wait_for(
                score_call(), timeout=timeout_seconds
            )
        value = metric_result.value
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"{metric_name} returned a non-finite score: {value!r}")
        result = {"metric": metric_name, "score": float(value), "error": None}
        save_cache(cache_dir, cache_key, result)
        return result
    except Exception as exc:
        return {
            "metric": metric_name,
            "score": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def evaluate_generation_with_ragas(
    cases: Sequence[QueryCase], runs: Sequence[AgentRun], model: str = "gpt-4.1-mini"
) -> list[dict[str, Any]]:
    from openai import AsyncOpenAI, OpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness
    from dotenv import load_dotenv

    load_dotenv()
    client = AsyncOpenAI(timeout=45, max_retries=0)
    embedding_client = OpenAI(timeout=45, max_retries=0)
    evaluator_llm = llm_factory(model, client=client, max_tokens=4096)
    evaluator_embeddings = embedding_factory(
        "openai", model="text-embedding-3-small", client=embedding_client
    )
    faithfulness = Faithfulness(llm=evaluator_llm)
    relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    by_id = {run.query_id: run for run in runs}
    semaphore = asyncio.Semaphore(3)
    cache_dir = Path("cache/generation_eval")

    async def score_case(case: QueryCase) -> dict[str, Any]:
        run = by_id[case.id]
        if run.error:
            return {"query_id": case.id, "faithfulness": None,
                    "faithfulness_error": run.error,
                    "answer_relevancy": None,
                    "answer_relevancy_error": run.error,
                    "error": run.error}
        key = {
            "metric_version": "ragas-0.3.9-generation-v1",
            "query_id": case.id,
            "query": case.query,
            "answer_sha256": hashlib.sha256(run.answer.encode()).hexdigest(),
            "contexts_sha256": hashlib.sha256(
                json.dumps(list(run.contexts), ensure_ascii=False).encode()
            ).hexdigest(),
            "model": model,
        }

        legacy = load_cache(cache_dir, key)

        async def one_metric(name, call):
            metric_key = {
                **key,
                "metric_version": f"ragas-0.3.9-{name.replace('_', '-')}-v2",
                "metric": name,
            }
            cached_metric = load_cache(cache_dir, metric_key)
            if cached_metric is not None:
                return cached_metric
            if (
                legacy is not None
                and legacy.get("error") is None
                and legacy.get(name) is not None
            ):
                migrated = {"metric": name, "score": legacy[name], "error": None}
                save_cache(cache_dir, metric_key, migrated)
                return migrated
            return await _score_one_metric(
                name,
                metric_key,
                call,
                semaphore,
            )

        faith_result, relevancy_result = await asyncio.gather(
            one_metric(
                "faithfulness",
                lambda: faithfulness.ascore(
                    user_input=case.query,
                    response=run.answer,
                    retrieved_contexts=list(run.contexts),
                ),
            ),
            one_metric(
                "answer_relevancy",
                lambda: relevancy.ascore(
                    user_input=case.query,
                    response=run.answer,
                ),
            ),
        )
        errors = [
            f"{result['metric']}: {result['error']}"
            for result in (faith_result, relevancy_result)
            if result["error"]
        ]
        return {
            "query_id": case.id,
            "faithfulness": faith_result["score"],
            "faithfulness_error": faith_result["error"],
            "answer_relevancy": relevancy_result["score"],
            "answer_relevancy_error": relevancy_result["error"],
            "error": "; ".join(errors) if errors else None,
        }

    try:
        return await asyncio.gather(*(score_case(case) for case in cases))
    finally:
        await client.close()
        embedding_client.close()
