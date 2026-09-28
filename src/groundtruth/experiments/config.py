from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from groundtruth.corpus.io import load_jsonl
from groundtruth.schemas import QueryCase


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in experiment config {config_path}")
    config["_config_path"] = str(config_path)
    return config


def load_query_cases(config: dict[str, Any]) -> list[QueryCase]:
    records = load_jsonl(config["dataset"])
    cases = [QueryCase.from_record(record) for record in records]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Golden set contains duplicate query IDs")
    return cases


def require_approved_dataset(cases: list[QueryCase]) -> None:
    if not cases:
        raise RuntimeError("Metric collection blocked: golden set is empty")
    pending = [case.id for case in cases if case.review_status != "approved"]
    missing_reference = [case.id for case in cases if not case.reference_texts]
    if pending or missing_reference:
        raise RuntimeError(
            "Metric collection blocked: every query must have review_status='approved' "
            "and at least one reference_text. "
            f"Pending={pending[:10]}, missing_reference={missing_reference[:10]}"
        )


def unresolved_reviews(cases: list[QueryCase]) -> list[str]:
    return [case.id for case in cases if case.review_status != "approved"]


def validate_dataset_references(cases: list[QueryCase], chunks: list) -> list[dict[str, str]]:
    from groundtruth.corpus.chunking import find_reference_chunks

    problems = []
    for case in cases:
        for reference in case.reference_texts:
            try:
                find_reference_chunks(reference, chunks)
            except ValueError as exc:
                problems.append({"query_id": case.id, "error": str(exc)})
    known_documents = {chunk.document_id for chunk in chunks}
    for case in cases:
        for document_id in case.expected_document_ids:
            if document_id not in known_documents:
                problems.append({"query_id": case.id,
                                 "error": f"Unknown expected document: {document_id}"})
    return problems
