from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from groundtruth.evaluation.agent import _score_one_metric
from groundtruth.evaluation.agent_matrix import (
    EXPERIMENTS,
    BACKENDS,
    build_agent_matrix,
)
from groundtruth.experiments.runner import _generation_summary


def test_generation_metrics_cache_and_fail_independently(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = {"faithfulness": 0, "answer_relevancy": 0}

    async def faithfulness():
        calls["faithfulness"] += 1
        return SimpleNamespace(value=0.75)

    async def answer_relevancy():
        calls["answer_relevancy"] += 1
        raise ConnectionError("embedding endpoint unavailable")

    async def score_both():
        semaphore = asyncio.Semaphore(2)
        return await asyncio.gather(
            _score_one_metric(
                "faithfulness", {"metric": "faithfulness"}, faithfulness, semaphore
            ),
            _score_one_metric(
                "answer_relevancy", {"metric": "answer_relevancy"},
                answer_relevancy, semaphore,
            ),
        )

    first = asyncio.run(score_both())
    second = asyncio.run(score_both())

    assert first[0]["score"] == 0.75
    assert first[0]["error"] is None
    assert first[1]["score"] is None
    assert "ConnectionError" in first[1]["error"]
    assert second[0] == first[0]
    assert calls == {"faithfulness": 1, "answer_relevancy": 2}


def test_generation_summary_averages_each_metric_independently():
    summary = _generation_summary([
        {"faithfulness": 1.0, "answer_relevancy": None, "error": "answer_relevancy failed"},
        {"faithfulness": 0.5, "answer_relevancy": 0.8, "error": None},
    ])

    assert summary["faithfulness_valid"] == 2
    assert summary["mean_faithfulness"] == 0.75
    assert summary["answer_relevancy_valid"] == 1
    assert summary["mean_answer_relevancy"] == 0.8
    assert summary["successful"] == 1


def _write_complete_fixture(root: Path, baseline_path: Path) -> None:
    fingerprint = "same-index"
    baseline_path.write_text(json.dumps({
        "index_fingerprint": fingerprint,
        "metrics": {"recall_at_10": 0.95},
        "recall_gate_max_drop": 0.01,
    }))
    (root / "retrieval_controlled.json").write_text(json.dumps({
        "index_fingerprint": fingerprint,
        "configurations": {"flat": {"metrics": {"recall_at_10": 0.95}}},
    }))
    for _, _, experiment in EXPERIMENTS:
        configurations = {}
        for backend in BACKENDS:
            configurations[backend] = {
                "operations": {
                    "successful": 18,
                    "failed": 0,
                    "metrics": {
                        "tool_calls_per_answer": 1.0,
                        "cost_usd_per_answer": 0.001,
                        "latency_ms_per_answer": 1000,
                        "tool_calls_by_name": {"retrieve": 18},
                    },
                },
                "generation_details": [
                    {
                        "query_id": f"q{index}",
                        "faithfulness": 1.0,
                        "answer_relevancy": 0.9,
                        "faithfulness_error": None,
                        "answer_relevancy_error": None,
                        "error": None,
                    }
                    for index in range(4)
                ],
            }
        (root / f"{experiment}.json").write_text(json.dumps({
            "configurations": configurations,
        }))


def test_agent_matrix_requires_all_18_cells_and_full_generation_sample(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    baseline = tmp_path / "baseline.json"
    _write_complete_fixture(results_dir, baseline)

    complete = build_agent_matrix(results_dir, baseline)
    assert complete["complete"] is True
    assert complete["cells_expected"] == complete["cells_present"] == 18
    assert complete["retrieval_gate"]["status"] == "pass"

    partial_path = results_dir / "agent_tuned.json"
    partial = json.loads(partial_path.read_text())
    partial["configurations"]["flat"]["generation_details"][0]["faithfulness"] = None
    partial_path.write_text(json.dumps(partial))
    incomplete = build_agent_matrix(results_dir, baseline)
    assert incomplete["complete"] is False
    assert any("Faithfulness scores" in error for error in incomplete["errors"])
