"""Structured audit tracing for agent workflow steps."""

from __future__ import annotations

import json
import time
from typing import Any

from guardrails.pii import redact_pii


def _safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, list):
        return [_safe_value(item) for item in value[:10]]
    if isinstance(value, dict):
        return {key: _safe_value(val) for key, val in list(value.items())[:20]}
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


class AuditCollector:
    """Collect per-node trace events during LangGraph execution."""

    def __init__(self) -> None:
        self._started_at = time.perf_counter()
        self._last_mark = self._started_at
        self.steps: list[dict[str, Any]] = []

    def record_step(self, node_name: str, node_output: dict[str, Any]) -> None:
        now = time.perf_counter()
        safe_output = {
            key: _safe_value(value)
            for key, value in node_output.items()
            if key not in {"messages"}
        }
        self.steps.append(
            {
                "node": node_name,
                "duration_ms": round((now - self._last_mark) * 1000, 2),
                "output_keys": list(node_output.keys()),
                "output": safe_output,
            }
        )
        self._last_mark = now

    def finalize(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        total_ms = round((time.perf_counter() - self._started_at) * 1000, 2)
        trace = {
            "total_duration_ms": total_ms,
            "step_count": len(self.steps),
            "steps": self.steps,
        }
        if extra:
            trace["summary"] = _safe_value(extra)
        return trace

    def to_json(self, extra: dict[str, Any] | None = None) -> str:
        return json.dumps(self.finalize(extra), ensure_ascii=True)
