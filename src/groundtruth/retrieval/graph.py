from __future__ import annotations

from collections.abc import Sequence

from groundtruth.knowledge_graph.index import KnowledgeGraphIndex
from groundtruth.retrieval.dense import DenseRetriever
from groundtruth.schemas import Chunk, ScoredChunk


class GraphRetriever:
    name = "graph"

    def __init__(
        self,
        dense: DenseRetriever,
        graph_index: KnowledgeGraphIndex,
        seed_k: int = 12,
        max_hops: int = 2,
        graph_weight: float = 0.2,
    ):
        if seed_k <= 0 or max_hops < 0 or not 0 <= graph_weight <= 1:
            raise ValueError("Invalid graph retrieval settings")
        self.dense = dense
        self.graph_index = graph_index
        self.seed_k = seed_k
        self.max_hops = max_hops
        self.graph_weight = graph_weight

    def retrieve(
        self,
        query: str,
        chunks: Sequence[Chunk],
        k: int,
        scope_document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        if k <= 0:
            return []
        allowed_docs = set(scope_document_ids or ())
        seeds = self.dense.retrieve(query, chunks, max(k, self.seed_k), scope_document_ids)
        dense_by_id = {item.chunk.id: item for item in seeds}
        seed_nodes = [f"chunk:{item.chunk.id}" for item in seeds]
        expanded = self.graph_index.expand(seed_nodes, max_hops=self.max_hops)
        distances = {
            node_id.removeprefix("chunk:"): distance
            for node_id, distance in expanded
            if node_id.startswith("chunk:")
        }
        by_id = {chunk.id: chunk for chunk in chunks}
        candidates = set(dense_by_id) | set(distances)
        scored: list[ScoredChunk] = []
        for chunk_id in candidates:
            chunk = by_id.get(chunk_id)
            if chunk is None or (allowed_docs and chunk.document_id not in allowed_docs):
                continue
            dense_item = dense_by_id.get(chunk_id)
            distance = distances.get(chunk_id)

            dense_score = (
                (1 - self.graph_weight) / (dense_item.rank + 1)
                if dense_item is not None
                else 0.0
            )
            graph_score = (
                self.graph_weight / max(distance, 1)
                if distance is not None and dense_item is None
                else 0.0
            )
            score = dense_score + graph_score
            scored.append(ScoredChunk(chunk=chunk, score=score, rank=0))
        scored.sort(key=lambda item: (-item.score, item.chunk.id))
        from dataclasses import replace

        return [replace(item, rank=rank) for rank, item in enumerate(scored[:k], 1)]
