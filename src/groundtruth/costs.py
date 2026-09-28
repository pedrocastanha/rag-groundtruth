from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class UsageLedger:
    prices: dict[str, dict[str, float]]
    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "UsageLedger":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(prices=raw["models"])

    def record(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
    ) -> float:
        if model not in self.prices:
            raise KeyError(f"Missing pricing for model {model!r}")
        rates = self.prices[model]
        cached = min(input_tokens, cached_input_tokens)
        billable_input = input_tokens - cached
        cost = (
            billable_input * rates["input_per_million"]
            + cached * rates.get("cached_input_per_million", rates["input_per_million"])
            + output_tokens * rates["output_per_million"]
        ) / 1_000_000
        self.calls += 1
        self.input_tokens += input_tokens
        self.cached_input_tokens += cached
        self.output_tokens += output_tokens
        self.cost_usd += cost
        model_stats = self.by_model.setdefault(model, {"calls": 0, "cost_usd": 0.0})
        model_stats["calls"] += 1
        model_stats["cost_usd"] += cost
        return cost

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "by_model": self.by_model,
        }
