from __future__ import annotations

from typing import Any, Callable

from groundtruth.cache import get_or_compute


def get_or_build_graph(
    cache_dir: str,
    key: dict[str, object],
    build: Callable[[], Any],
) -> tuple[Any, bool]:
    return get_or_compute(cache_dir, key, build)
