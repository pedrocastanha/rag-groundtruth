from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from openai import OpenAI

from groundtruth.cache import get_or_compute
from groundtruth.costs import UsageLedger
from groundtruth.corpus.fingerprint import documents_fingerprint
from groundtruth.knowledge_graph.schema import GraphEdge, GraphNode, KnowledgeGraph
from groundtruth.schemas import Chunk, Document

GRAPH_PROMPT_VERSION = "entity-relations-v1"
GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "kind": {"type": "string"},
                            },
                            "required": ["name", "kind"],
                            "additionalProperties": False,
                        },
                    },
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "relation": {"type": "string"},
                                "target": {"type": "string"},
                            },
                            "required": ["source", "relation", "target"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["chunk_id", "entities", "relations"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["chunks"],
    "additionalProperties": False,
}


def _stable_id(prefix: str, value: str) -> str:
    normalized = " ".join(value.casefold().split())
    return f"{prefix}:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]}"


def _index_chunk_extractions(
    payload: dict[str, Any], chunks: list[Chunk]
) -> tuple[dict[str, dict[str, Any]], list[Chunk]]:
    requested = {chunk.id for chunk in chunks}
    indexed: dict[str, dict[str, Any]] = {}
    for item in payload.get("chunks", []):
        chunk_id = item.get("chunk_id")
        if chunk_id in requested and chunk_id not in indexed:
            indexed[chunk_id] = item
    missing = [chunk for chunk in chunks if chunk.id not in indexed]
    return indexed, missing


class KnowledgeGraphBuilder:
    def __init__(
        self,
        client: OpenAI,
        ledger: UsageLedger,
        cache_dir: str = "cache/knowledge_graph",
        model: str = "gpt-4.1-mini",
        batch_size: int = 5,
    ):
        self.client = client
        self.ledger = ledger
        self.cache_dir = cache_dir
        self.model = model
        self.batch_size = batch_size
        self.cache_hits = 0
        self.cache_misses = 0

    def build(
        self,
        documents: list[Document],
        chunks: list[Chunk],
        chunking_version: str,
    ) -> KnowledgeGraph:
        corpus_hash = documents_fingerprint(documents)
        graph_key = {
            "corpus_fingerprint": corpus_hash,
            "chunking_version": chunking_version,
            "extractor_model": self.model,
            "prompt_version": GRAPH_PROMPT_VERSION,
        }
        payload, hit = get_or_compute(
            self.cache_dir,
            graph_key,
            lambda: self._build_payload(documents, chunks, corpus_hash, chunking_version),
        )
        self.cache_hits += int(hit)
        self.cache_misses += int(not hit)
        return KnowledgeGraph.from_record(payload)

    def _build_payload(
        self,
        documents: list[Document],
        chunks: list[Chunk],
        corpus_hash: str,
        chunking_version: str,
    ) -> dict[str, Any]:
        extracted = []
        for offset in range(0, len(chunks), self.batch_size):
            batch = chunks[offset:offset + self.batch_size]
            key = {
                "corpus_fingerprint": corpus_hash,
                "chunking_version": chunking_version,
                "extractor_model": self.model,
                "prompt_version": GRAPH_PROMPT_VERSION,
                "chunks": [{"id": chunk.id, "text": chunk.text} for chunk in batch],
            }
            payload, hit = get_or_compute(
                f"{self.cache_dir}/extractions",
                key,
                lambda batch=batch: self._extract_batch(batch),
            )
            self.cache_hits += int(hit)
            self.cache_misses += int(not hit)
            extracted.extend(payload["chunks"])
        graph = self._assemble(documents, chunks, extracted)
        return graph.as_record()

    def _extract_batch(self, chunks: list[Chunk]) -> dict[str, Any]:
        indexed = self._extract_with_recovery(chunks)
        return {"chunks": [indexed[chunk.id] for chunk in chunks]}

    def _extract_with_recovery(self, chunks: list[Chunk]) -> dict[str, dict[str, Any]]:
        result = self._request_extraction(chunks)
        indexed, missing = _index_chunk_extractions(result, chunks)
        if not missing:
            return indexed
        if len(missing) == len(chunks):
            if len(chunks) == 1:
                raise ValueError(f"Graph extractor omitted chunk {chunks[0].id}")

            midpoint = len(chunks) // 2
            recovered = self._extract_with_recovery(chunks[:midpoint])
            recovered.update(self._extract_with_recovery(chunks[midpoint:]))
            return recovered

        indexed.update(self._extract_with_recovery(missing))
        return indexed

    def _request_extraction(self, chunks: list[Chunk]) -> dict[str, Any]:
        prompt = (
            "Extract concrete entities and explicit relationships from each chunk. "
            "Use stable, concise entity names. Do not infer personal or technical facts "
            "not stated in the text. Return exactly one result for each chunk ID.\n\n"
            + json.dumps([{"chunk_id": c.id, "text": c.text} for c in chunks], ensure_ascii=False)
        )
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "chunk_graph_extraction",
                    "strict": True,
                    "schema": GRAPH_SCHEMA,
                }
            },
        )
        usage = response.usage
        self.ledger.record(
            self.model,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )
        result = json.loads(response.output_text)
        return result

    @staticmethod
    def _assemble(
        documents: list[Document],
        chunks: list[Chunk],
        extracted: list[dict[str, Any]],
    ) -> KnowledgeGraph:
        nodes: dict[str, GraphNode] = {}
        edges: dict[tuple[str, str, str], GraphEdge] = {}
        entity_chunks: dict[str, set[str]] = defaultdict(set)
        entity_kinds: dict[str, str] = {}
        normalized_name_to_id: dict[str, str] = {}

        for document in documents:
            doc_node = f"document:{document.id}"
            nodes[doc_node] = GraphNode(doc_node, "document", document.title, document.id)
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        for chunk in chunks:
            chunk_node = f"chunk:{chunk.id}"
            nodes[chunk_node] = GraphNode(
                chunk_node, "chunk", chunk.id, chunk.document_id, (chunk.id,)
            )
            doc_node = f"document:{chunk.document_id}"
            edges[(doc_node, "CONTAINS", chunk_node)] = GraphEdge(
                doc_node, chunk_node, "CONTAINS", (chunk.id,)
            )

        normalized_extractions = []
        for item in extracted:
            chunk_id = item["chunk_id"]
            chunk = chunks_by_id[chunk_id]
            chunk_node = f"chunk:{chunk_id}"
            entities = {}
            for entity in item["entities"]:
                label = " ".join(entity["name"].split())
                if not label:
                    continue
                normalized = label.casefold()
                entity_id = normalized_name_to_id.setdefault(
                    normalized, _stable_id("entity", normalized)
                )
                entity_kinds.setdefault(entity_id, entity["kind"])
                entity_chunks[entity_id].add(chunk_id)
                entities[normalized] = entity_id
                edges[(chunk_node, "MENTIONS", entity_id)] = GraphEdge(
                    chunk_node, entity_id, "MENTIONS", (chunk_id,)
                )
            normalized_extractions.append((chunk_id, chunk, entities, item["relations"]))

        for normalized, entity_id in normalized_name_to_id.items():
            nodes[entity_id] = GraphNode(
                id=entity_id,
                kind=entity_kinds[entity_id],
                label=normalized,
                chunk_ids=tuple(sorted(entity_chunks[entity_id])),
            )

        for chunk_id, chunk, entities, relations in normalized_extractions:
            for relation in relations:
                source_name = " ".join(relation["source"].casefold().split())
                target_name = " ".join(relation["target"].casefold().split())
                source_id = entities.get(source_name) or normalized_name_to_id.get(source_name)
                target_id = entities.get(target_name) or normalized_name_to_id.get(target_name)
                if not source_id or not target_id or source_id == target_id:
                    continue
                relation_name = "_".join(relation["relation"].upper().split())[:48]
                if not relation_name:
                    continue
                edge_key = (source_id, relation_name, target_id)
                previous = edges.get(edge_key)
                evidence = set(previous.evidence_chunk_ids if previous else ())
                evidence.add(chunk_id)
                edges[edge_key] = GraphEdge(
                    source_id, target_id, relation_name, tuple(sorted(evidence))
                )
        return KnowledgeGraph(
            nodes=tuple(nodes[key] for key in sorted(nodes)),
            edges=tuple(edges[key] for key in sorted(edges)),
        )
