from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from groundtruth.evaluation.gate import passes_recall_gate


EXPERIMENTS = (
    ("controlled", "gpt-4.1-mini", "agent_controlled"),
    ("controlled", "gpt-4o-mini", "agent_model_gpt4o_mini_controlled"),
    ("controlled", "gpt-6-luna", "agent_model_gpt6_luna_controlled"),
    ("tuned", "gpt-4.1-mini", "agent_tuned"),
    ("tuned", "gpt-4o-mini", "agent_model_gpt4o_mini"),
    ("tuned", "gpt-6-luna", "agent_model_gpt6_luna"),
)
BACKENDS = ("flat", "filtered", "graph")


def build_agent_matrix(
    results_dir: str | Path,
    baseline_path: str | Path,
    expected_generation_samples: int = 4,
) -> dict[str, Any]:
    result_root = Path(results_dir)
    errors: list[str] = []
    rows: list[dict[str, Any]] = []

    for mode, model, experiment in EXPERIMENTS:
        result_path = result_root / f"{experiment}.json"
        if not result_path.exists():
            errors.append(f"Missing experiment result: {result_path}")
            continue
        try:
            report = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Cannot read {result_path}: {exc}")
            continue

        for backend in BACKENDS:
            cell = report.get("configurations", {}).get(backend)
            if cell is None:
                errors.append(f"{experiment} is missing retrieval strategy {backend}")
                continue
            operations = cell.get("operations", {})
            op_metrics = operations.get("metrics", {})
            generation = cell.get("generation", {})
            details = cell.get("generation_details", [])
            faithfulness = [
                float(item["faithfulness"])
                for item in details
                if item.get("faithfulness") is not None
            ]
            relevancy = [
                float(item["answer_relevancy"])
                for item in details
                if item.get("answer_relevancy") is not None
            ]
            row = {
                "mode": mode,
                "model": model,
                "retrieval": backend,
                "experiment": experiment,
                "successful_runs": operations.get("successful", 0),
                "failed_runs": operations.get("failed", 0),
                "tool_calls_per_answer": op_metrics.get("tool_calls_per_answer"),
                "tool_calls_by_name": op_metrics.get("tool_calls_by_name", {}),
                "cost_usd_per_answer": op_metrics.get("cost_usd_per_answer"),
                "latency_ms_per_answer": op_metrics.get("latency_ms_per_answer"),
                "faithfulness_valid": len(faithfulness),
                "faithfulness_mean": _mean(faithfulness),
                "answer_relevancy_valid": len(relevancy),
                "answer_relevancy_mean": _mean(relevancy),
                "generation_errors": [
                    {
                        "query_id": item.get("query_id"),
                        "faithfulness": item.get("faithfulness_error"),
                        "answer_relevancy": item.get("answer_relevancy_error"),
                        "legacy_error": item.get("error"),
                    }
                    for item in details
                    if item.get("error")
                    or item.get("faithfulness_error")
                    or item.get("answer_relevancy_error")
                ],
            }
            rows.append(row)
            if operations.get("failed", 0):
                errors.append(
                    f"{experiment}/{backend} has {operations['failed']} failed agent runs"
                )
            if len(faithfulness) < expected_generation_samples:
                errors.append(
                    f"{experiment}/{backend} has {len(faithfulness)}/"
                    f"{expected_generation_samples} Faithfulness scores"
                )
            if len(relevancy) < expected_generation_samples:
                errors.append(
                    f"{experiment}/{backend} has {len(relevancy)}/"
                    f"{expected_generation_samples} Answer Relevancy scores"
                )

    retrieval_gate = _retrieval_gate(result_root, Path(baseline_path), errors)
    complete = len(rows) == len(EXPERIMENTS) * len(BACKENDS) and not errors
    return {
        "matrix": "controlled_tuned_x_3_models_x_3_retrieval",
        "cells_expected": len(EXPERIMENTS) * len(BACKENDS),
        "cells_present": len(rows),
        "expected_generation_samples_per_metric": expected_generation_samples,
        "complete": complete,
        "retrieval_gate": retrieval_gate,
        "errors": errors,
        "rows": rows,
    }


def _retrieval_gate(
    results_dir: Path, baseline_path: Path, errors: list[str]
) -> dict[str, Any]:
    retrieval_path = results_dir / "retrieval_controlled.json"
    if not baseline_path.exists():
        errors.append(f"Missing retrieval baseline: {baseline_path}")
        return {"status": "missing_baseline"}
    if not retrieval_path.exists():
        errors.append(f"Missing retrieval experiment result: {retrieval_path}")
        return {"status": "missing_candidate"}
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        candidate = json.loads(retrieval_path.read_text(encoding="utf-8"))
        baseline_recall = float(baseline["metrics"]["recall_at_10"])
        candidate_recall = float(
            candidate["configurations"]["flat"]["metrics"]["recall_at_10"]
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"Cannot compare retrieval candidate to baseline: {exc}")
        return {"status": "invalid_comparison", "error": str(exc)}

    baseline_fingerprint = baseline.get("index_fingerprint")
    candidate_fingerprint = candidate.get("index_fingerprint")
    if baseline_fingerprint and candidate_fingerprint != baseline_fingerprint:
        errors.append("Retrieval baseline and candidate use different index fingerprints")
        return {
            "status": "incompatible_index",
            "baseline_fingerprint": baseline_fingerprint,
            "candidate_fingerprint": candidate_fingerprint,
        }

    max_drop = float(baseline.get("recall_gate_max_drop", 0.01))
    passed = passes_recall_gate(baseline_recall, candidate_recall, max_drop)
    if not passed:
        errors.append(
            f"Normal retrieval Recall@10 dropped more than {max_drop:.2%}: "
            f"{baseline_recall:.6f} -> {candidate_recall:.6f}"
        )
    return {
        "status": "pass" if passed else "fail",
        "baseline_recall_at_10": baseline_recall,
        "candidate_recall_at_10": candidate_recall,
        "max_drop": max_drop,
    }


def render_agent_matrix_markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# Comparação completa dos agentes",
        "",
        f"Estado: **{'PASS' if matrix['complete'] else 'INCOMPLETO/FAIL'}**",
        "",
        "| Modo | Modelo | Busca | Sucesso | Chamadas/resposta | Custo/resposta | Latência | Faithfulness (n; média) | Answer Relevancy (n; média) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in matrix["rows"]:
        lines.append(
            "| {mode} | {model} | {retrieval} | {successful}/{total} | "
            "{calls} | {cost} | {latency} | {faith} | {relevancy} |".format(
                mode=row["mode"],
                model=row["model"],
                retrieval=row["retrieval"],
                successful=row["successful_runs"],
                total=row["successful_runs"] + row["failed_runs"],
                calls=_fmt(row["tool_calls_per_answer"]),
                cost=_fmt(row["cost_usd_per_answer"]),
                latency=_fmt(row["latency_ms_per_answer"], suffix=" ms"),
                faith=_fmt_metric(row["faithfulness_valid"], row["faithfulness_mean"]),
                relevancy=_fmt_metric(
                    row["answer_relevancy_valid"], row["answer_relevancy_mean"]
                ),
            )
        )
    gate = matrix["retrieval_gate"]
    lines.extend(
        [
            "",
            "## Gate de Recall@10",
            "",
            f"{gate.get('status', 'unknown')}: baseline "
            f"{_fmt(gate.get('baseline_recall_at_10'))}, candidato "
            f"{_fmt(gate.get('candidate_recall_at_10'))}, queda máxima "
            f"{_fmt(gate.get('max_drop'))}.",
            "",
            "## Problemas",
            "",
        ]
    )
    lines.extend(f"- {error}" for error in matrix["errors"])
    if not matrix["errors"]:
        lines.append("- Nenhum.")
    lines.append("")
    return "\n".join(lines)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: Any, suffix: str = "") -> str:
    return "—" if value is None else f"{float(value):.4f}{suffix}"


def _fmt_metric(count: int, mean: float | None) -> str:
    return f"{count}; {_fmt(mean)}"
