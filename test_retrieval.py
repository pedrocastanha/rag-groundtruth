from main import passes_recall_gate, load_golden_set, load_chunks, embed_chunks, run_experiment, \
    retrieve_dense_reranked, retrieve_broken
import json

def load_baseline(file_path: str) -> dict:
    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)

def test_recall_gate_fail_when_drop_is_too_large():
    baseline_recall = 0.92
    candidate_recall = 0.90

    result = passes_recall_gate(baseline_recall, candidate_recall)

    assert result is False

def test_recall_gate_passes_when_drop_is_allowed():
    baseline_recall = 0.92
    candidate_recall = 0.91

    result = passes_recall_gate(
        baseline_recall,
        candidate_recall,
    )

    assert result is True

def test_recall_at_10_does_not_regress():
    golden_set = load_golden_set("golden_set.jsonl")
    chunks = load_chunks("chunks.jsonl")

    embedded_chunks = embed_chunks(chunks)

    candidate = run_experiment(
        golden_set,
        retrieve_dense_reranked,
        embedded_chunks,
        k=10,
    )

    baseline = load_baseline("baselines/retrieval_v1.json")
    baseline_recall = baseline["metrics"]["recall_at_10"]

    candidate_recall = candidate["overall"]["mean_recall_at_k"]

    assert passes_recall_gate(
        baseline_recall,
        candidate_recall,
        max_drop=0.01,
    )

def test_recall_gate_detects_bad_retriever():
    golden_set = load_golden_set("golden_set.jsonl")
    chunks = load_chunks("chunks.jsonl")

    candidate = run_experiment(
        golden_set,
        retrieve_broken,
        chunks,
        k=10,
    )

    baseline = load_baseline(
        "baselines/retrieval_v1.json"
    )

    baseline_recall = (
        baseline["metrics"]["recall_at_10"]
    )

    candidate_recall = (
        candidate["overall"]["mean_recall_at_k"]
    )

    assert not passes_recall_gate(
        baseline_recall,
        candidate_recall,
        max_drop=0.01,
    )
