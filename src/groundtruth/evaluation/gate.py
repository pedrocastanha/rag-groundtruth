from __future__ import annotations

import math


def passes_recall_gate(
    baseline_recall: float,
    candidate_recall: float,
    max_drop: float = 0.01,
    tolerance: float = 1e-9,
) -> bool:
    drop = baseline_recall - candidate_recall
    return drop < max_drop or math.isclose(drop, max_drop, abs_tol=tolerance)
