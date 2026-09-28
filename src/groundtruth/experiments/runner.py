from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from groundtruth.agents.runner import AgentRunner
from groundtruth.cache import get_or_compute
from groundtruth.costs import UsageLedger
from groundtruth.corpus.chunking import chunk_documents
from groundtruth.corpus.fingerprint import documents_fingerprint
from groundtruth.corpus.io import load_documents
from groundtruth.evaluation.agent import evaluate_generation_with_ragas, summarize_agent_runs
from groundtruth.evaluation.retrieval import evaluate_retrieval
from groundtruth.experiments.config import load_query_cases, require_approved_dataset
from groundtruth.knowledge_graph.builder import KnowledgeGraphBuilder
from groundtruth.knowledge_graph.index import KnowledgeGraphIndex
from groundtruth.retrieval.dense import DenseRetriever, EmbeddingService
from groundtruth.retrieval.filtered import FilteredRetriever
from groundtruth.retrieval.graph import GraphRetriever
from groundtruth.schemas import Chunk, Document


def _client() -> OpenAI:
    load_dotenv(dotenv_path=Path(".env"))
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for API-backed experiments")
    return OpenAI()


def _load_ledger(config: dict[str, Any]) -> UsageLedger:
    return UsageLedger.from_yaml(config.get("pricing", "configs/model_pricing.yaml"))


def load_corpus(config: dict[str, Any]) -> tuple[list[Document], list[Chunk], str]:
    documents = load_documents(config["corpus"]["sources"])
    chunking = config.get("chunking", {})
    chunk_size = int(chunking.get("size_words", 200))
    overlap = int(chunking.get("overlap_words", 40))
    fingerprint = documents_fingerprint(documents)
    chunk_key = {"corpus_fingerprint": fingerprint, "chunk_size_words": chunk_size,
                 "overlap_words": overlap, "chunker_version": "word-offset-v1"}
    rows, _ = get_or_compute(
        "cache/corpus",
        chunk_key,
        lambda: [chunk.as_record() for chunk in chunk_documents(documents, chunk_size, overlap)],
    )
    return documents, [Chunk.from_record(row) for row in rows], fingerprint


def _prepare_retrievers(
    config: dict[str, Any], documents: list[Document], chunks: list[Chunk], corpus_fingerprint: str,
    ledger: UsageLedger, client: OpenAI, include_graph: bool,
) -> tuple[EmbeddingService, list[Chunk], DenseRetriever, GraphRetriever | None, str]:
    embedding_model = config.get("embedding_model", "text-embedding-3-small")
    embedding_service = EmbeddingService(client, ledger, embedding_model)
    embedded_chunks = embedding_service.embed_chunks(chunks)
    chunking = config.get("chunking", {})
    chunk_size = int(chunking.get("size_words", 200))
    overlap = int(chunking.get("overlap_words", 40))
    chunking_version = f"word{chunk_size}-overlap{overlap}"
    index_fingerprint = f"{corpus_fingerprint}:{chunking_version}:{embedding_model}"
    dense = DenseRetriever(embedding_service)
    graph_retriever = None
    if include_graph:
        graph_cfg = config.get("graph", {})
        graph = KnowledgeGraphBuilder(
            client, ledger,
            cache_dir=graph_cfg.get("cache_dir", "cache/knowledge_graph"),
            model=graph_cfg.get("model", "gpt-4.1-mini"),
            batch_size=int(graph_cfg.get("batch_size", 5)),
        ).build(documents, embedded_chunks, chunking_version)
        graph_retriever = GraphRetriever(
            dense, KnowledgeGraphIndex(graph),
            seed_k=int(graph_cfg.get("seed_k", 12)),
            max_hops=int(graph_cfg.get("max_hops", 2)),
            graph_weight=float(graph_cfg.get("graph_weight", 0.2)),
        )
    return embedding_service, embedded_chunks, dense, graph_retriever, index_fingerprint


def run_retrieval_experiment(config: dict[str, Any]) -> dict[str, Any]:
    cases = load_query_cases(config)
    require_approved_dataset(cases)
    documents, chunks, corpus_fingerprint = load_corpus(config)
    client, ledger = _client(), _load_ledger(config)
    backends = config.get("retrieval_backends", ["flat", "filtered", "graph"])
    embedding_service, embedded_chunks, dense, graph, index_fingerprint = _prepare_retrievers(
        config, documents, chunks, corpus_fingerprint, ledger, client, "graph" in backends
    )
    output: dict[str, Any] = {
        "experiment": config["name"], "dataset": config["dataset"],
        "dataset_size": len(cases), "index_fingerprint": index_fingerprint,
        "shared_setup_cost_usd": ledger.cost_usd,
        "cache": {"embedding_hits": embedding_service.cache_hits,
                  "embedding_misses": embedding_service.cache_misses},
        "configurations": {},
    }
    top_k = int(config.get("top_k", 10))
    for backend in backends:
        cost_before_backend = ledger.cost_usd
        dense.query_cache_scope = config.get("query_cache_scope", f"{config['name']}:{backend}")
        if backend == "flat":
            retriever = dense
        elif backend == "filtered":
            retriever = FilteredRetriever(dense)
        elif backend == "graph" and graph is not None:
            retriever = graph
        else:
            raise ValueError(f"Unknown retrieval backend or graph not initialized: {backend}")
        retrieved = {}
        for case in cases:
            scope = list(case.expected_document_ids) if backend == "filtered" else None
            retrieved[case.id] = retriever.retrieve(case.query, embedded_chunks, top_k, scope)
        report = evaluate_retrieval(cases, embedded_chunks, retrieved, top_k)
        report["configuration_cost_usd"] = ledger.cost_usd - cost_before_backend
        output["configurations"][backend] = report
    _write_result(config["name"], output)
    return output


def run_agent_experiment(
    config: dict[str, Any], include_generation_eval: bool = True
) -> dict[str, Any]:
    cases = load_query_cases(config)
    require_approved_dataset(cases)
    documents, chunks, corpus_fingerprint = load_corpus(config)
    client = _client()
    backends = config.get("retrieval_backends", ["flat", "filtered", "graph"])
    ledger = _load_ledger(config)
    embedding_service, embedded_chunks, dense, graph, index_fingerprint = _prepare_retrievers(
        config, documents, chunks, corpus_fingerprint, ledger, client, "graph" in backends
    )
    agent_cfg = config["agent"]
    output: dict[str, Any] = {
        "experiment": config["name"], "dataset": config["dataset"],
        "dataset_size": len(cases), "index_fingerprint": index_fingerprint,
        "agent": {
            "mode": agent_cfg["mode"],
            "model": agent_cfg.get("model", "gpt-4.1-mini"),
            "reasoning_effort": agent_cfg.get("reasoning_effort"),
        },
        "shared_setup_cost_usd": ledger.cost_usd,
        "configurations": {},
    }
    for backend in backends:
        dense.query_cache_scope = config.get("query_cache_scope", f"{config['name']}:{backend}")
        retriever = {"flat": dense, "filtered": FilteredRetriever(dense), "graph": graph}.get(backend)
        if retriever is None:
            raise ValueError(f"Unknown retrieval backend: {backend}")
        strategy = config.get("strategies", {}).get(backend, {})
        runner = AgentRunner(
            client=client, ledger=ledger, retriever=retriever, chunks=embedded_chunks,
            backend=backend, mode=agent_cfg["mode"], model=agent_cfg.get("model", "gpt-4.1-mini"),
            reasoning_effort=agent_cfg.get("reasoning_effort"),
            prompt_version=strategy.get("prompt_version", agent_cfg.get("prompt_version", "shared_v1")),
            max_tool_calls=int(agent_cfg.get("max_tool_calls", 5)),
            max_steps=int(agent_cfg.get("max_steps", 8)),
            cache_dir=(
                f"{config.get('cache', {}).get('root', 'cache')}/agent_runs/"
                f"{config.get('cache', {}).get('agent_run_namespace', config['name'])}"
            ),
        )
        graph_identity = config.get("graph", {}) if backend == "graph" else {}
        backend_fingerprint = f"{index_fingerprint}:{backend}:{json.dumps(graph_identity, sort_keys=True)}"
        runs = [runner.run(case.id, case.query, backend_fingerprint) for case in cases]
        run_records = [run.as_record() for run in runs]
        report: dict[str, Any] = {"operations": summarize_agent_runs(runs), "runs": run_records}
        if include_generation_eval:
            generation_cfg = config.get("generation_eval", {})
            sample_ids = generation_cfg.get("query_ids")
            if sample_ids is not None:
                cases_by_id = {case.id: case for case in cases}
                runs_by_id = {run.query_id: run for run in runs}
                unknown_ids = set(sample_ids) - cases_by_id.keys()
                if unknown_ids:
                    raise ValueError(f"Unknown generation evaluation query IDs: {sorted(unknown_ids)}")
                generation_cases = [cases_by_id[query_id] for query_id in sample_ids]
                generation_runs = [runs_by_id[query_id] for query_id in sample_ids]
            else:
                generation_cases, generation_runs = cases, runs
            evaluation_model = agent_cfg.get(
                "evaluation_model", generation_cfg.get("model", "gpt-4.1-mini")
            )
            quality = await_generation_eval(generation_cases, generation_runs, evaluation_model)
            report["generation"] = _generation_summary(quality)
            report["generation_details"] = quality
            report["generation_evaluation"] = {
                "model": evaluation_model,
                "sample_size": len(generation_cases),
                "query_ids": [case.id for case in generation_cases],
            }
        output["configurations"][backend] = report
    output["usage"] = ledger.snapshot()
    output["cache"] = {"embedding_hits": embedding_service.cache_hits,
                        "embedding_misses": embedding_service.cache_misses}
    _write_result(config["name"], output)
    return output


def await_generation_eval(cases, runs, model):
    import asyncio
    return asyncio.run(evaluate_generation_with_ragas(cases, runs, model))


def _generation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    faithfulness_scores = [
        float(row["faithfulness"])
        for row in rows
        if row.get("faithfulness") is not None
    ]
    relevancy_scores = [
        float(row["answer_relevancy"])
        for row in rows
        if row.get("answer_relevancy") is not None
    ]
    complete = [
        row for row in rows
        if row.get("faithfulness") is not None
        and row.get("answer_relevancy") is not None
    ]
    return {
        "total": len(rows),
        "successful": len(complete),
        "failed": len(rows) - len(complete),
        "faithfulness_valid": len(faithfulness_scores),
        "faithfulness_failed": len(rows) - len(faithfulness_scores),
        "answer_relevancy_valid": len(relevancy_scores),
        "answer_relevancy_failed": len(rows) - len(relevancy_scores),
        "mean_faithfulness": _mean(faithfulness_scores),
        "mean_answer_relevancy": _mean(relevancy_scores),
    }


def _mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _write_result(experiment_name: str, output: dict[str, Any]) -> Path:
    path = Path("results/multidoc") / f"{experiment_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
