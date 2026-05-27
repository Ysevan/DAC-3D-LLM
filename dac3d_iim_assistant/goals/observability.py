"""Read-only observability snapshot for DAC-Agent workspace."""

from __future__ import annotations

from typing import Any


def _count_by(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _tool_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("tool") or value.get("tool_name") or "unknown")
    return str(value or "unknown")


class ObservabilityReporter:
    """Aggregate trace and workspace signals without creating new side effects."""

    def __init__(
        self,
        *,
        trace_logger: Any = None,
        event_queue_store: Any = None,
        verification_store: Any = None,
        review_handoff_store: Any = None,
        checkpoint_store: Any = None,
    ) -> None:
        self.trace_logger = trace_logger
        self.event_queue_store = event_queue_store
        self.verification_store = verification_store
        self.review_handoff_store = review_handoff_store
        self.checkpoint_store = checkpoint_store

    def snapshot(self, *, recent_trace_limit: int = 20) -> dict[str, Any]:
        """Return a compact observability view for dashboards and agents."""
        safe_limit = max(1, min(200, int(recent_trace_limit or 20)))
        traces = self._recent_traces(limit=safe_limit)
        event_summary = self._describe(self.event_queue_store, "local_agent_event_queue")
        verification_summary = self._describe(self.verification_store, "local_verification_runner")
        review_summary = self._describe(self.review_handoff_store, "local_review_handoff_queue")
        checkpoint_summary = self._describe(self.checkpoint_store, "local_agent_checkpoint_store")
        by_event_type = _count_by([str(trace.get("event_type") or "unknown") for trace in traces])
        by_intent = _count_by([str(trace.get("intent") or "unknown") for trace in traces])
        tool_counts = _count_by(
            [
                _tool_name(tool_call)
                for trace in traces
                for tool_call in list(trace.get("tool_calls") or [])
            ]
        )
        return {
            "enabled": True,
            "backend": "local_agent_observability",
            "traces": {
                "recent_limit": safe_limit,
                "recent_count": len(traces),
                "latest_trace_id": traces[-1].get("trace_id") if traces else "",
                "by_event_type": by_event_type,
                "by_intent": by_intent,
                "tool_calls": tool_counts,
                "recent": traces,
            },
            "signals": {
                "events": event_summary,
                "verifications": verification_summary,
                "reviews": review_summary,
                "checkpoints": checkpoint_summary,
            },
            "attention": {
                "queued_events": int((event_summary.get("by_status") or {}).get("queued") or 0),
                "failed_verifications": int(
                    (verification_summary.get("by_status") or {}).get("failed") or 0
                ),
                "pending_reviews": int((review_summary.get("by_status") or {}).get("pending") or 0),
                "active_checkpoints": int(
                    (checkpoint_summary.get("by_status") or {}).get("active") or 0
                ),
            },
            "workflow": "traces + workspace stores -> observability snapshot -> agent dashboard",
        }

    def describe(self) -> dict[str, Any]:
        """Return a minimal runtime summary for Agent workspace."""
        snapshot = self.snapshot(recent_trace_limit=5)
        return {
            "enabled": True,
            "backend": "local_agent_observability",
            "recent_trace_count": snapshot["traces"]["recent_count"],
            "latest_trace_id": snapshot["traces"]["latest_trace_id"],
            "attention": snapshot["attention"],
            "workflow": "collect_signals -> summarize_attention -> inspect_recent_traces",
        }

    def _recent_traces(self, *, limit: int) -> list[dict[str, Any]]:
        if self.trace_logger is None or not hasattr(self.trace_logger, "iter_recent"):
            return []
        traces = self.trace_logger.iter_recent(limit=limit)
        return [trace for trace in traces if isinstance(trace, dict)]

    def _describe(self, store: Any, backend: str) -> dict[str, Any]:
        if store is None or not hasattr(store, "describe"):
            return {"enabled": False, "backend": backend}
        summary = store.describe()
        return summary if isinstance(summary, dict) else {"enabled": False, "backend": backend}
