from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from groundtruth.schemas import Chunk, ScoredChunk


class Retriever(Protocol):
    name: str

    def retrieve(
        self,
        query: str,
        chunks: Sequence[Chunk],
        k: int,
        scope_document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]: ...
