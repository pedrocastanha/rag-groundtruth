from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    path: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    document_title: str
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    embedding: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunk_index": self.chunk_index,
            "embedding": list(self.embedding),
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Chunk":
        return cls(
            id=record["id"],
            document_id=record["document_id"],
            document_title=record["document_title"],
            text=record["text"],
            start_char=record["start_char"],
            end_char=record["end_char"],
            chunk_index=record["chunk_index"],
            embedding=tuple(record.get("embedding", ())),
            metadata=record.get("metadata", {}),
        )


@dataclass(frozen=True)
class QueryCase:
    id: str
    query: str
    category: str
    difficulty: str
    question_type: str
    expected_document_ids: tuple[str, ...]
    reference_texts: tuple[str, ...]
    review_status: str = "pending_human_confirmation"
    reference_answer: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "QueryCase":
        return cls(
            id=record["id"],
            query=record["query"],
            category=record["category"],
            difficulty=record["difficulty"],
            question_type=record["question_type"],
            expected_document_ids=tuple(record["expected_document_ids"]),
            reference_texts=tuple(record["reference_texts"]),
            review_status=record.get("review_status", "pending_human_confirmation"),
            reference_answer=record.get("reference_answer", ""),
        )


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    succeeded: bool
    duration_ms: float
    error: str | None = None


@dataclass(frozen=True)
class AgentRun:
    query_id: str
    answer: str
    contexts: tuple[str, ...]
    tool_calls: tuple[ToolCall, ...]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    error: str | None = None
    cache_hit: bool = False

    def as_record(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "answer": self.answer,
            "contexts": list(self.contexts),
            "tool_calls": [call.__dict__ for call in self.tool_calls],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "cache_hit": self.cache_hit,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "AgentRun":
        return cls(
            query_id=record["query_id"],
            answer=record["answer"],
            contexts=tuple(record["contexts"]),
            tool_calls=tuple(ToolCall(**call) for call in record["tool_calls"]),
            input_tokens=record["input_tokens"],
            output_tokens=record["output_tokens"],
            cost_usd=record["cost_usd"],
            latency_ms=record["latency_ms"],
            error=record.get("error"),
            cache_hit=record.get("cache_hit", False),
        )
