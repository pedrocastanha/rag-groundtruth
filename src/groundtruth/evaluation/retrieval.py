from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from groundtruth.corpus.chunking import attach_relevant_chunk_ids
from groundtruth.schemas import QueryCase, ScoredChunk


def recall_at_k(relevant: set[str], ranked_ids: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked_ids[:k])) / len(relevant)


def reciprocal_rank(relevant: set[str], ranked_ids: Sequence[str]) -> float:
    return next((1.0 / rank for rank, item in enumerate(ranked_ids, 1) if item in relevant), 0.0)


def ndcg_at_k(relevant: set[str], ranked_ids: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, item in enumerate(ranked_ids[:k], 1) if item in relevant)
    ideal_count = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def evaluate_retrieval(
    cases: Sequence[QueryCase],
    chunks: Sequence,
    retrieved: dict[str, Sequence[ScoredChunk]],
    k: int = 10,
) -> dict[str, object]:
    relevant_by_query = attach_relevant_chunk_ids(cases, chunks)
    details = []
    for case in cases:
        ranked = list(retrieved.get(case.id, ()))
        ranked_ids = [item.chunk.id for item in ranked]
        relevant = set(relevant_by_query[case.id])
        details.append({
            "query_id": case.id,
            "query": case.query,
            "category": case.category,
            "difficulty": case.difficulty,
            "question_type": case.question_type,
            "expected_document_ids": list(case.expected_document_ids),
            "relevant_chunk_ids": sorted(relevant),
            "retrieved_chunk_ids": ranked_ids,
            f"recall_at_{k}": recall_at_k(relevant, ranked_ids, k),
            "mrr": reciprocal_rank(relevant, ranked_ids),
            f"ndcg_at_{k}": ndcg_at_k(relevant, ranked_ids, k),
        })
    return {"metrics": _average(details, k), "slices": _slices(details, k), "details": details}


def _average(rows: Sequence[dict], k: int) -> dict[str, float]:
    denominator = max(len(rows), 1)
    return {
        f"recall_at_{k}": sum(row[f"recall_at_{k}"] for row in rows) / denominator,
        "mrr": sum(row["mrr"] for row in rows) / denominator,
        f"ndcg_at_{k}": sum(row[f"ndcg_at_{k}"] for row in rows) / denominator,
    }


def _slices(rows: Sequence[dict], k: int) -> dict[str, dict[str, dict[str, float]]]:
    output = {}
    for dimension in ("category", "difficulty", "question_type"):
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row[dimension]].append(row)
        output[dimension] = {key: _average(group, k) for key, group in sorted(grouped.items())}
    return output
