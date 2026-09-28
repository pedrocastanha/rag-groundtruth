from __future__ import annotations

import pytest

from groundtruth.agents.trace import summarize_tool_calls
from groundtruth.agents.protocol import tool_definitions
from groundtruth.cache import get_or_compute
from groundtruth.corpus.chunking import chunk_document, find_reference_chunks
from groundtruth.experiments.config import require_approved_dataset
from groundtruth.knowledge_graph.index import KnowledgeGraphIndex
from groundtruth.knowledge_graph.builder import KnowledgeGraphBuilder, _index_chunk_extractions
from groundtruth.knowledge_graph.schema import GraphEdge, GraphNode, KnowledgeGraph
from groundtruth.retrieval.graph import GraphRetriever
from groundtruth.schemas import Chunk, Document, QueryCase, ScoredChunk, ToolCall
from groundtruth.evaluation.retrieval import ndcg_at_k, reciprocal_rank, recall_at_k
from groundtruth.evaluation.gate import passes_recall_gate


def test_content_cache_computes_once_and_reads_back(tmp_path):
    calls = []
    compute = lambda: calls.append(1) or {"vector": [0.1, 0.2]}
    first, first_hit = get_or_compute(tmp_path, {"text": "hello", "model": "fake"}, compute)
    second, second_hit = get_or_compute(tmp_path, {"text": "hello", "model": "fake"}, compute)
    assert first == second
    assert first_hit is False and second_hit is True
    assert len(calls) == 1


def test_reference_in_chunk_overlap_is_counted_once():
    document = Document("doc", "Doc", "one two alpha beta three four five six", "doc.md")
    chunks = chunk_document(document, chunk_size=4, overlap=2)
    assert find_reference_chunks("alpha beta", chunks) == ["doc_0001"]


def test_metrics_handle_relevance_and_ranking():
    relevant = {"a", "b"}
    ranking = ["x", "a", "b"]
    assert recall_at_k(relevant, ranking, 2) == 0.5
    assert reciprocal_rank(relevant, ranking) == 0.5
    assert ndcg_at_k(relevant, ranking, 3) < 1.0
    assert ndcg_at_k(relevant, ["a", "b"], 2) == 1.0


def test_recall_gate_allows_exactly_one_percentage_point_but_not_more():
    assert passes_recall_gate(1.0, 0.99)
    assert not passes_recall_gate(1.0, 0.9899)


def test_pending_queries_block_metric_collection():
    pending = QueryCase("q", "question", "cat", "easy", "single_document", ("doc",), ("evidence",))
    with pytest.raises(RuntimeError, match="Metric collection blocked"):
        require_approved_dataset([pending])
    approved = QueryCase("q", "question", "cat", "easy", "single_document", ("doc",),
                         ("evidence",), review_status="approved")
    require_approved_dataset([approved])


def test_graph_expansion_finds_neighboring_chunk():
    graph = KnowledgeGraph(
        nodes=(GraphNode("chunk:a", "chunk", "a"), GraphNode("entity:x", "entity", "x"),
               GraphNode("chunk:b", "chunk", "b")),
        edges=(GraphEdge("chunk:a", "entity:x", "MENTIONS"),
               GraphEdge("chunk:b", "entity:x", "MENTIONS")),
    )
    assert dict(KnowledgeGraphIndex(graph).expand(["chunk:a"], max_hops=2))["chunk:b"] == 2


def test_graph_expansion_candidate_can_compete_with_dense_seeds():
    chunks = [Chunk(f"doc_{index:04}", "doc", "Doc", "text", index, index + 4, index)
              for index in range(13)]

    class StubDense:
        def retrieve(self, query, candidates, k, scope_document_ids=None):
            return [ScoredChunk(chunk, 1 / rank, rank)
                    for rank, chunk in enumerate(chunks[1:13], 1)]

    graph = KnowledgeGraph(
        nodes=(GraphNode("chunk:doc_0000", "chunk", "expanded"),
               GraphNode("chunk:doc_0001", "chunk", "seed"),
               GraphNode("entity:shared", "entity", "shared")),
        edges=(GraphEdge("chunk:doc_0000", "entity:shared", "MENTIONS"),
               GraphEdge("chunk:doc_0001", "entity:shared", "MENTIONS")),
    )
    retriever = GraphRetriever(StubDense(), KnowledgeGraphIndex(graph), seed_k=12)
    results = retriever.retrieve("query", chunks, 10)
    result_ids = [item.chunk.id for item in results]
    assert "doc_0000" in result_ids
    assert "doc_0012" not in result_ids


def test_graph_builder_keeps_all_evidence_for_repeated_relation():
    document = Document("doc", "Doc", "source", "doc.md")
    chunks = [
        Chunk("doc_0001", "doc", "Doc", "A uses B", 0, 8, 0),
        Chunk("doc_0002", "doc", "Doc", "A uses B again", 9, 23, 1),
    ]
    extraction = [
        {"chunk_id": chunk.id,
         "entities": [{"name": "A", "kind": "person"}, {"name": "B", "kind": "tool"}],
         "relations": [{"source": "A", "relation": "uses", "target": "B"}]}
        for chunk in chunks
    ]
    graph = KnowledgeGraphBuilder._assemble([document], chunks, extraction)
    relation = next(edge for edge in graph.edges if edge.relation == "USES")
    assert relation.evidence_chunk_ids == ("doc_0001", "doc_0002")


def test_graph_extraction_deduplicates_rows_and_reports_missing_chunks():
    chunks = [
        Chunk("doc_0001", "doc", "Doc", "first", 0, 5, 0),
        Chunk("doc_0002", "doc", "Doc", "second", 6, 12, 1),
    ]
    first = {"chunk_id": "doc_0001", "entities": [], "relations": []}
    indexed, missing = _index_chunk_extractions(
        {"chunks": [first, first, {"chunk_id": "unknown", "entities": [], "relations": []}]},
        chunks,
    )
    assert list(indexed) == ["doc_0001"]
    assert [chunk.id for chunk in missing] == ["doc_0002"]


def test_controlled_tool_contract_stays_same_across_retrievers():
    names = [tool_definitions("controlled", backend)[0]["name"]
             for backend in ("flat", "filtered", "graph")]
    assert names == ["retrieve", "retrieve", "retrieve"]
    assert tool_definitions("tuned", "graph")[0]["name"] == "graph_search"


def test_filtered_agent_tool_exposes_valid_document_ids():
    tool = tool_definitions(
        "tuned", "filtered", {"ai_engineering": "AI Handbook", "pedro": "Pedro"}
    )[0]
    schema = tool["parameters"]["properties"]["document_id"]
    assert schema["enum"] == ["ai_engineering", "pedro"]
    assert "ai_engineering = AI Handbook" in tool["description"]


def test_trace_counts_repeated_calls_and_failures():
    call = ToolCall("search", {"query": "x"}, True, 1.0)
    failed = ToolCall("search", {"query": "y"}, False, 2.0, "oops")
    summary = summarize_tool_calls([call, call, failed])
    assert summary["tool_calls"] == 3
    assert summary["unique_tools"] == 1
    assert summary["failed_tool_calls"] == 1
    assert summary["repeated_tool_calls"] == 1
