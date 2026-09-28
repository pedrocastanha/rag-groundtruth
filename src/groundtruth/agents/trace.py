from __future__ import annotations

import json
from collections import Counter

from groundtruth.schemas import ToolCall


def call_signature(call: ToolCall) -> str:
    return f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"


def summarize_tool_calls(calls: tuple[ToolCall, ...] | list[ToolCall]) -> dict[str, object]:
    counts = Counter(call.name for call in calls)
    signatures = Counter(call_signature(call) for call in calls)
    return {
        "tool_calls": len(calls),
        "tool_calls_by_name": dict(sorted(counts.items())),
        "unique_tools": len(counts),
        "failed_tool_calls": sum(not call.succeeded for call in calls),
        "repeated_tool_calls": sum(count - 1 for count in signatures.values() if count > 1),
    }
