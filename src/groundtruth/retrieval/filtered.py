from __future__ import annotations

from collections.abc import Sequence

from groundtruth.retrieval.dense import DenseRetriever
from groundtruth.schemas import Chunk, ScoredChunk


class FilteredRetriever:
    name = "filtered"

    def __init__(self, dense: DenseRetriever):
        self.dense = dense

    def retrieve(
        self,
        query: str,
        chunks: Sequence[Chunk],
        k: int,
        scope_document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        if not scope_document_ids:
            # No scope means the agent did not request a filter; preserve recall.
            return self.dense.retrieve(query, chunks, k)
        allowed = set(scope_document_ids)
        scoped = [chunk for chunk in chunks if chunk.document_id in allowed]
        return self.dense.retrieve(query, scoped, k)
