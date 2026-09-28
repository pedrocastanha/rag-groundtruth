from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from groundtruth.schemas import Document


def load_documents(sources: Iterable[dict[str, str]]) -> list[Document]:
    documents = []
    for source in sources:
        path = Path(source["path"])
        documents.append(Document(
            id=source["id"],
            title=source["title"],
            text=path.read_text(encoding="utf-8"),
            path=str(path),
        ))
    return documents


def load_jsonl(file_path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(file_path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object on line {line_number}")
            records.append(record)
    return records


def save_jsonl(records: Iterable[dict[str, Any]], file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
