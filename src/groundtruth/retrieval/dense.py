from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from openai import OpenAI

from groundtruth.cache import get_or_compute
from groundtruth.costs import UsageLedger
from groundtruth.schemas import Chunk, ScoredChunk


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(left * right for left, right in zip(a, b))
    magnitude_a = math.sqrt(sum(value * value for value in a))
    magnitude_b = math.sqrt(sum(value * value for value in b))
    if not magnitude_a or not magnitude_b:
        return 0.0
    return dot / (magnitude_a * magnitude_b)


class EmbeddingService:
    def __init__(
        self,
        client: OpenAI,
        ledger: UsageLedger,
        model: str = "text-embedding-3-small",
        cache_dir: str = "cache/text_embeddings",
    ):
        self.client = client
        self.ledger = ledger
        self.model = model
        self.cache_dir = cache_dir
        self.cache_hits = 0
        self.cache_misses = 0

    def embed(
        self, text: str, cache_scope: str | None = None
    ) -> tuple[list[float], bool]:
        key = {"text": text, "model": self.model}
        if cache_scope is not None:
            key["cache_scope"] = cache_scope
        cached = get_or_compute(
            self.cache_dir,
            key,
            lambda: self._request_embedding(text),
        )
        value, hit = cached
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return value["embedding"], hit

    def _request_embedding(self, text: str) -> dict[str, Any]:
        response = self.client.embeddings.create(model=self.model, input=text)
        tokens = int(getattr(response.usage, "prompt_tokens", 0) or 0)
        self.ledger.record(self.model, input_tokens=tokens)
        return {"embedding": response.data[0].embedding, "input_tokens": tokens}

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        embedded = []
        for chunk in chunks:
            vector, _ = self.embed(chunk.text)
            embedded.append(replace(chunk, embedding=tuple(vector)))
        return embedded


class DenseRetriever:
    name = "flat"

    def __init__(
        self, embedding_service: EmbeddingService, query_cache_scope: str | None = None
    ):
        self.embedding_service = embedding_service
        self.query_cache_scope = query_cache_scope

    def retrieve(
        self,
        query: str,
        chunks: Sequence[Chunk],
        k: int,
        scope_document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0:
            return []
        query_vector, _ = self.embedding_service.embed(query, self.query_cache_scope)
        candidates = [
            chunk for chunk in chunks
            if not scope_document_ids or chunk.document_id in scope_document_ids
        ]
        scored = [
            ScoredChunk(
                chunk=chunk,
                score=cosine_similarity(query_vector, chunk.embedding),
                rank=0,
            )
            for chunk in candidates
        ]
        scored.sort(key=lambda item: (-item.score, item.chunk.id))
        return [replace(item, rank=rank) for rank, item in enumerate(scored[:k], 1)]
