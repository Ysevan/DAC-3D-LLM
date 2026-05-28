"""Local performance analytics store for DAC-Agent LLMOps workflows."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PERFORMANCE_STATUSES = ("active", "archived")


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(default)
    return payload if isinstance(payload, dict) else dict(default)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = _utc_now_iso()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _clip(value: Any, limit: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _strings(values: list[Any] | tuple[Any, ...] | None, *, limit: int = 80) -> list[str]:
    if not values:
        return []
    result: list[str] = []
    for value in values[:limit]:
        text = _clip(value, 220)
        if text and text not in result:
            result.append(text)
    return result


def _json_dict(value: dict[str, Any] | None, *, field_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(json.dumps(dict(value or {}), ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable.") from exc
    return payload if isinstance(payload, dict) else {}


def _number(value: Any, *, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc
    return round(number, 6)


def _normalize_status(value: str, *, default: str = "active") -> str:
    status = str(value or default).strip().lower()
    if status not in PERFORMANCE_STATUSES:
        raise ValueError(f"Performance status must be one of: {', '.join(PERFORMANCE_STATUSES)}.")
    return status


def _nested_number(payload: dict[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if current is None or current == "":
        return None
    return _number(current, field_name="trace metric")


def _search_text(metric: dict[str, Any]) -> str:
    parts = [
        str(metric.get("id") or ""),
        str(metric.get("metric_name") or ""),
        str(metric.get("category") or ""),
        str(metric.get("unit") or ""),
        str(metric.get("target") or ""),
        str(metric.get("source_type") or ""),
        str(metric.get("source_id") or ""),
        str(metric.get("session_id") or ""),
        " ".join(str(item) for item in metric.get("tags") or []),
        json.dumps(metric.get("metadata") or {}, ensure_ascii=False, sort_keys=True),
    ]
    return " ".join(parts).lower()


class AgentPerformanceStore:
    """JSON-backed performance metric store for local Agent operations."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_root(cls, root_dir: Path) -> "AgentPerformanceStore":
        return cls(root_dir / "agent_performance.json")

    def record_metric(
        self,
        metric_name: str,
        metric_value: Any,
        *,
        unit: str = "",
        category: str = "custom",
        target: str = "",
        source_type: str = "manual",
        source_id: str = "",
        session_id: str = "",
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        recorded_by: str = "agent",
    ) -> dict[str, Any]:
        clean_name = _clip(metric_name, 160)
        if not clean_name:
            raise ValueError("Performance metric_name is required.")
        value = _number(metric_value, field_name="Performance metric_value")
        now = _utc_now_iso()
        metric = {
            "id": f"perf-{uuid.uuid4().hex[:12]}",
            "metric_name": clean_name,
            "metric_value": value,
            "unit": _clip(unit, 80),
            "category": _clip(category or "custom", 120) or "custom",
            "target": _clip(target, 220),
            "source_type": _clip(source_type or "manual", 80) or "manual",
            "source_id": _clip(source_id, 220),
            "session_id": _clip(session_id, 220),
            "tags": _strings(tags),
            "metadata": _json_dict(metadata, field_name="Performance metadata"),
            "status": "active",
            "recorded_by": _clip(recorded_by, 120) or "agent",
            "recorded_at": now,
            "updated_at": now,
            "history": [
                {
                    "at": now,
                    "type": "recorded",
                    "actor": _clip(recorded_by, 120) or "agent",
                    "note": "",
                }
            ],
        }
        payload = self._load()
        metrics = self._metrics(payload)
        metrics.insert(0, metric)
        payload["metrics"] = metrics[:1000]
        _write_json(self.path, payload)
        return {"metric": metric, "created": True}

    def record_from_trace(
        self,
        trace: dict[str, Any],
        *,
        tags: list[Any] | None = None,
        recorded_by: str = "agent",
    ) -> dict[str, Any]:
        trace_id = _clip(trace.get("trace_id"), 220)
        if not trace_id:
            raise ValueError("Trace must include trace_id.")
        intent = _clip(trace.get("intent"), 120)
        base_tags = ["trace"]
        if intent:
            base_tags.append(intent)
        base_tags.extend(_strings(tags))
        payload = self._load()
        metrics = self._metrics(payload)
        observations = self._trace_observations(trace, tags=base_tags, recorded_by=recorded_by)
        metrics = observations + metrics
        payload["metrics"] = metrics[:1000]
        _write_json(self.path, payload)
        return {
            "enabled": True,
            "backend": "local_agent_performance_store",
            "trace_id": trace_id,
            "metrics": observations,
            "count": len(observations),
            "workflow": "trace -> performance_metrics -> llmops_summary",
        }

    def list_metrics(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        metric_name: str | None = None,
        source_type: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        metrics = self._metrics(payload)
        if status:
            normalized_status = _normalize_status(status)
            metrics = [metric for metric in metrics if str(metric.get("status") or "") == normalized_status]
        if category:
            clean_category = _clip(category, 120)
            metrics = [metric for metric in metrics if str(metric.get("category") or "") == clean_category]
        if metric_name:
            clean_metric_name = _clip(metric_name, 160)
            metrics = [
                metric for metric in metrics if str(metric.get("metric_name") or "") == clean_metric_name
            ]
        if source_type:
            clean_source_type = _clip(source_type, 80)
            metrics = [metric for metric in metrics if str(metric.get("source_type") or "") == clean_source_type]
        if tag:
            clean_tag = _clip(tag, 220)
            metrics = [
                metric for metric in metrics if clean_tag in [str(value) for value in metric.get("tags") or []]
            ]
        if query:
            needle = str(query or "").strip().lower()
            metrics = [metric for metric in metrics if needle and needle in _search_text(metric)]
        safe_limit = max(1, min(200, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_agent_performance_store",
            "path": str(self.path),
            "metrics": metrics[:safe_limit],
            "count": len(metrics[:safe_limit]),
            "total_count": len(self._metrics(payload)),
            "statuses": list(PERFORMANCE_STATUSES),
            "workflow": "record_metric -> filter_metrics -> performance_review",
        }

    def summarize(
        self,
        *,
        status: str = "active",
        category: str | None = None,
        metric_name: str | None = None,
        source_type: str | None = None,
    ) -> dict[str, Any]:
        metrics = self.list_metrics(
            status=status,
            category=category,
            metric_name=metric_name,
            source_type=source_type,
            limit=200,
        )["metrics"]
        return {
            "enabled": True,
            "backend": "local_agent_performance_store",
            "metric_count": len(metrics),
            "by_metric": self._aggregate(metrics, key="metric_name"),
            "by_category": self._aggregate(metrics, key="category"),
            "latest_metric": metrics[0] if metrics else {},
            "workflow": "performance_metrics -> aggregate -> bottleneck_review",
        }

    def update_status(
        self,
        metric_id: str,
        status: str,
        *,
        note: str = "",
        actor: str = "agent",
    ) -> dict[str, Any]:
        payload = self._load()
        metric = self._find_in_metrics(self._metrics(payload), metric_id)
        if metric is None:
            raise ValueError(f"Unknown performance metric: {metric_id}")
        old_status = str(metric.get("status") or "active")
        new_status = _normalize_status(status)
        metric["status"] = new_status
        metric["updated_at"] = _utc_now_iso()
        history = {
            "at": metric["updated_at"],
            "type": "status_changed",
            "actor": _clip(actor, 120) or "agent",
            "note": _clip(note or f"{old_status} -> {new_status}", 800),
            "from": old_status,
            "to": new_status,
        }
        events = list(metric.get("history") or [])
        events.append(history)
        metric["history"] = events[-50:]
        _write_json(self.path, payload)
        return {"metric": metric, "history": history}

    def describe(self) -> dict[str, Any]:
        metrics = self._metrics(self._load())
        active = [metric for metric in metrics if str(metric.get("status") or "") == "active"]
        by_status = {status: 0 for status in PERFORMANCE_STATUSES}
        for metric in metrics:
            status = str(metric.get("status") or "active")
            by_status[status] = by_status.get(status, 0) + 1
        return {
            "enabled": True,
            "backend": "local_agent_performance_store",
            "path": str(self.path),
            "metric_count": len(metrics),
            "active_metric_count": len(active),
            "by_status": by_status,
            "by_category": self._count_by(metrics, "category"),
            "latest_metric": metrics[0] if metrics else {},
            "workflow": "llmops_metrics -> performance_summary -> agent_improvement",
        }

    def _trace_observations(
        self,
        trace: dict[str, Any],
        *,
        tags: list[str],
        recorded_by: str,
    ) -> list[dict[str, Any]]:
        trace_id = _clip(trace.get("trace_id"), 220)
        session_id = _clip(trace.get("session_id"), 220)
        intent = _clip(trace.get("intent"), 120)
        now = _utc_now_iso()
        observations: list[tuple[str, float, str, str, dict[str, Any]]] = []
        duration_ms = _nested_number(trace, "duration_ms") or _nested_number(trace, "latency_ms")
        if duration_ms is not None:
            observations.append(("agent_latency_ms", duration_ms, "ms", "latency", {}))
        total_tokens = (
            _nested_number(trace, "token_usage", "total_tokens")
            or _nested_number(trace, "usage", "total_tokens")
            or _nested_number(trace, "usage", "total")
            or _nested_number(trace, "tokens", "total")
        )
        if total_tokens is not None:
            observations.append(("agent_total_tokens", total_tokens, "tokens", "cost", {}))
        cost_usd = _nested_number(trace, "cost_usd") or _nested_number(trace, "usage", "cost_usd")
        if cost_usd is not None:
            observations.append(("agent_cost_usd", cost_usd, "usd", "cost", {}))
        quality_score = _nested_number(trace, "quality_score")
        if quality_score is not None:
            observations.append(("agent_quality_score", quality_score, "score", "quality", {}))
        tool_calls = trace.get("tool_calls") if isinstance(trace.get("tool_calls"), list) else []
        observations.append(
            (
                "agent_tool_call_count",
                float(len(tool_calls)),
                "count",
                "tooling",
                {"tool_names": [self._tool_name(tool_call) for tool_call in tool_calls]},
            )
        )
        metrics: list[dict[str, Any]] = []
        for name, value, unit, category, metadata in observations:
            metrics.append(
                {
                    "id": f"perf-{uuid.uuid4().hex[:12]}",
                    "metric_name": name,
                    "metric_value": round(float(value), 6),
                    "unit": unit,
                    "category": category,
                    "target": intent,
                    "source_type": "trace",
                    "source_id": trace_id,
                    "session_id": session_id,
                    "tags": _strings(tags),
                    "metadata": _json_dict(metadata, field_name="Trace performance metadata"),
                    "status": "active",
                    "recorded_by": _clip(recorded_by, 120) or "agent",
                    "recorded_at": now,
                    "updated_at": now,
                    "history": [
                        {
                            "at": now,
                            "type": "recorded_from_trace",
                            "actor": _clip(recorded_by, 120) or "agent",
                            "note": trace_id,
                        }
                    ],
                }
            )
        return metrics

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.path, {"version": 1, "updated_at": "", "metrics": []})
        if not isinstance(payload.get("metrics"), list):
            payload["metrics"] = []
        return payload

    def _metrics(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [metric for metric in payload.get("metrics", []) if isinstance(metric, dict)]

    def _find_in_metrics(self, metrics: list[dict[str, Any]], metric_id: str) -> dict[str, Any] | None:
        for metric in metrics:
            if str(metric.get("id") or "") == str(metric_id or ""):
                return metric
        return None

    def _aggregate(self, metrics: list[dict[str, Any]], *, key: str) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[float]] = {}
        latest: dict[str, dict[str, Any]] = {}
        for metric in metrics:
            group = str(metric.get(key) or "unknown")
            value = float(metric.get("metric_value") or 0)
            grouped.setdefault(group, []).append(value)
            latest.setdefault(group, metric)
        return {
            group: {
                "count": len(values),
                "avg": round(sum(values) / len(values), 6) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "latest_metric_id": latest.get(group, {}).get("id", ""),
            }
            for group, values in grouped.items()
        }

    def _count_by(self, metrics: list[dict[str, Any]], key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for metric in metrics:
            group = str(metric.get(key) or "unknown")
            counts[group] = counts.get(group, 0) + 1
        return counts

    def _tool_name(self, value: Any) -> str:
        if isinstance(value, dict):
            return _clip(value.get("name") or value.get("tool") or value.get("tool_name"), 120) or "unknown"
        return _clip(value, 120) or "unknown"
