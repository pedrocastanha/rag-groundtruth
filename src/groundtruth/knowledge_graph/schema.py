from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GraphNode:
    id: str
    kind: str
    label: str
    document_id: str | None = None
    chunk_ids: tuple[str, ...] = ()
    properties: dict[str, Any] | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "document_id": self.document_id,
            "chunk_ids": list(self.chunk_ids),
            "properties": self.properties or {},
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "GraphNode":
        return cls(
            id=record["id"],
            kind=record["kind"],
            label=record["label"],
            document_id=record.get("document_id"),
            chunk_ids=tuple(record.get("chunk_ids", ())),
            properties=record.get("properties", {}),
        )


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    relation: str
    evidence_chunk_ids: tuple[str, ...] = ()
    weight: float = 1.0

    def as_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "evidence_chunk_ids": list(self.evidence_chunk_ids),
            "weight": self.weight,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "GraphEdge":
        return cls(
            source_id=record["source_id"],
            target_id=record["target_id"],
            relation=record["relation"],
            evidence_chunk_ids=tuple(record.get("evidence_chunk_ids", ())),
            weight=float(record.get("weight", 1.0)),
        )


@dataclass(frozen=True)
class KnowledgeGraph:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "nodes": [node.as_record() for node in self.nodes],
            "edges": [edge.as_record() for edge in self.edges],
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "KnowledgeGraph":
        return cls(
            nodes=tuple(GraphNode.from_record(row) for row in record["nodes"]),
            edges=tuple(GraphEdge.from_record(row) for row in record["edges"]),
        )
