from __future__ import annotations

import re
from collections.abc import Iterable

from groundtruth.schemas import Chunk, Document, QueryCase

_WORD = re.compile(r"\S+")


def chunk_document(
    document: Document,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    words = list(_WORD.finditer(document.text))
    chunks = []
    step = chunk_size - overlap
    for index, start in enumerate(range(0, len(words), step)):
        end = min(start + chunk_size, len(words))
        start_char = words[start].start()
        end_char = words[end - 1].end()
        chunks.append(Chunk(
            id=f"{document.id}_{index + 1:04d}",
            document_id=document.id,
            document_title=document.title,
            text=document.text[start_char:end_char],
            start_char=start_char,
            end_char=end_char,
            chunk_index=index,
        ))
    return chunks


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[Chunk]:
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_size, overlap)
    ]


def find_reference_chunks(reference_text: str, chunks: Iterable[Chunk]) -> list[str]:
    normalized_reference = " ".join(reference_text.split()).casefold()
    if not normalized_reference:
        raise ValueError("Reference text cannot be empty")
    matches: list[Chunk] = []
    for chunk in chunks:
        normalized_chunk = " ".join(chunk.text.split()).casefold()
        if normalized_reference in normalized_chunk:
            matches.append(chunk)
    if not matches:
        raise ValueError(f"Reference text not found in chunks: {reference_text[:80]!r}")

    best_match = min(matches, key=lambda chunk: (len(chunk.text), chunk.chunk_index, chunk.id))
    return [best_match.id]


def attach_relevant_chunk_ids(
    cases: Iterable[QueryCase],
    chunks: Iterable[Chunk],
) -> dict[str, list[str]]:
    chunk_list = list(chunks)
    mapped = {}
    for case in cases:
        ids = set()
        for reference in case.reference_texts:
            ids.update(find_reference_chunks(reference, chunk_list))
        mapped[case.id] = sorted(ids)
    return mapped
