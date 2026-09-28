from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable


def cache_path_for(
    cache_dir: str | Path,
    key_data: dict[str, object],
) -> Path:
    serialized_key = json.dumps(
        key_data,
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(serialized_key.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.json"


def load_cache(
    cache_dir: str | Path,
    key_data: dict[str, object],
) -> Any | None:
    path = cache_path_for(cache_dir, key_data)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_cache(
    cache_dir: str | Path,
    key_data: dict[str, object],
    value: object,
) -> Path:
    path = cache_path_for(cache_dir, key_data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = file.name
            json.dump(value, file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return path


def get_or_compute(
    cache_dir: str | Path,
    key_data: dict[str, object],
    compute: Callable[[], Any],
) -> tuple[Any, bool]:
    cached = load_cache(cache_dir, key_data)
    if cached is not None:
        return cached, True
    value = compute()
    save_cache(cache_dir, key_data, value)
    return value, False
