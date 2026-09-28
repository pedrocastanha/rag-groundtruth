from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace

from groundtruth.retrieval.dense import EmbeddingService, cosine_similarity
from groundtruth.schemas import Chunk, ScoredChunk

_TOKEN = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> set[str]:
    return set(_TOKEN.findall(text.casefold()))


class HybridRetriever:
    name = "hybrid"

    def __init__(self, embedding_service: EmbeddingService, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.embedding_service = embedding_service
        self.alpha = alpha

    def retrieve(
        self,
        query: str,
        chunks: Sequence[Chunk],
        k: int,
        scope_document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        query_vector, _ = self.embedding_service.embed(query)
        query_terms = tokenize(query)
        candidates = [
            chunk for chunk in chunks
            if not scope_document_ids or chunk.document_id in scope_document_ids
        ]
        scored = []
        for chunk in candidates:
            dense_score = (cosine_similarity(query_vector, chunk.embedding) + 1) / 2
            terms = tokenize(chunk.text)
            lexical_score = len(query_terms & terms) / len(query_terms) if query_terms else 0.0
            score = self.alpha * dense_score + (1 - self.alpha) * lexical_score
            scored.append(ScoredChunk(chunk=chunk, score=score, rank=0))
        scored.sort(key=lambda item: (-item.score, item.chunk.id))
        return [replace(item, rank=rank) for rank, item in enumerate(scored[:k], 1)]
