"""Local event queue for DAC-Agent runtime workflows.

The queue gives the Agent workspace a durable place to put work events that a
future worker, CLI, or UI can claim and complete. It is intentionally local
JSON so demos can inspect and replay event state without a background service.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


EVENT_STATUSES = ("queued", "claimed", "completed", "failed", "cancelled")
EVENT_PRIORITIES = ("low", "normal", "high")
_PRIORITY_RANK = {"high": 0, "normal": 1, "low": 2}


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


def _normalize_status(value: str, *, default: str = "queued") -> str:
    status = str(value or default).strip().lower()
    if status not in EVENT_STATUSES:
        raise ValueError(f"Event status must be one of: {', '.join(EVENT_STATUSES)}.")
    return status


def _normalize_priority(value: str) -> str:
    priority = str(value or "normal").strip().lower()
    if priority not in EVENT_PRIORITIES:
        raise ValueError(f"Event priority must be one of: {', '.join(EVENT_PRIORITIES)}.")
    return priority


def _payload_dict(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Event payload must be a JSON object.")
    return dict(value)


class EventQueueStore:
    """JSON-backed event queue for local Agent workflow coordination."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_root(cls, root_dir: Path) -> "EventQueueStore":
        return cls(root_dir / "agent_events.json")

    def enqueue_event(
        self,
        event_type: str,
        *,
        payload: dict[str, Any] | None = None,
        session_id: str = "web",
        priority: str = "normal",
        scheduled_for: str = "",
        task_id: str = "",
        workflow_id: str = "",
        automation_id: str = "",
        goal_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_type = _clip(event_type, 120)
        if not clean_type:
            raise ValueError("Event type is required.")
        normalized_priority = _normalize_priority(priority)
        now = _utc_now_iso()
        event = {
            "id": f"event-{uuid.uuid4().hex[:12]}",
            "event_type": clean_type,
            "session_id": str(session_id or "web"),
            "status": "queued",
            "priority": normalized_priority,
            "payload": _payload_dict(payload),
            "scheduled_for": _clip(scheduled_for, 80),
            "task_id": str(task_id or ""),
            "workflow_id": str(workflow_id or ""),
            "automation_id": str(automation_id or ""),
            "goal_id": str(goal_id or ""),
            "worker_id": "",
            "attempts": 0,
            "created_at": now,
            "updated_at": now,
            "claimed_at": "",
            "completed_at": "",
            "result": {},
            "history": [
                {
                    "id": f"history-{uuid.uuid4().hex[:10]}",
                    "created_at": now,
                    "type": "queued",
                    "status": "queued",
                    "note": "Event queued.",
                }
            ],
            "metadata": dict(metadata or {}),
        }
        payload_doc = self._load()
        events = self._events(payload_doc)
        events.append(event)
        payload_doc["events"] = events
        _write_json(self.path, payload_doc)
        return {"event": event, "queued": True}

    def list_events(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        event_type: str | None = None,
        priority: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        events = self._events(payload)
        if session_id:
            events = [event for event in events if str(event.get("session_id") or "") == session_id]
        if status:
            normalized_status = _normalize_status(status)
            events = [event for event in events if str(event.get("status") or "") == normalized_status]
        if event_type:
            normalized_type = _clip(event_type, 120)
            events = [event for event in events if str(event.get("event_type") or "") == normalized_type]
        if priority:
            normalized_priority = _normalize_priority(priority)
            events = [event for event in events if str(event.get("priority") or "") == normalized_priority]
        safe_limit = max(1, min(200, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_agent_event_queue",
            "path": str(self.path),
            "events": list(reversed(events))[:safe_limit],
            "count": len(events[:safe_limit]),
            "total_count": len(self._events(payload)),
            "statuses": list(EVENT_STATUSES),
            "priorities": list(EVENT_PRIORITIES),
            "workflow": "enqueue_event -> due_discovery -> claim -> status_update",
        }

    def due_events(self, *, now: str | None = None, limit: int = 20) -> dict[str, Any]:
        current = now or _utc_now_iso()
        due = [
            event
            for event in self._events(self._load())
            if str(event.get("status") or "") == "queued" and self._is_due(event, current)
        ]
        due = self._sort_for_claim(due)
        safe_limit = max(1, min(100, int(limit or 20)))
        return {"enabled": True, "events": due[:safe_limit], "count": len(due[:safe_limit])}

    def claim_next(
        self,
        *,
        worker_id: str = "agent-worker",
        session_id: str | None = None,
        event_type: str | None = None,
    ) -> dict[str, Any]:
        payload = self._load()
        candidates = [
            event
            for event in self._events(payload)
            if str(event.get("status") or "") == "queued" and self._is_due(event, _utc_now_iso())
        ]
        if session_id:
            candidates = [event for event in candidates if str(event.get("session_id") or "") == session_id]
        if event_type:
            normalized_type = _clip(event_type, 120)
            candidates = [event for event in candidates if str(event.get("event_type") or "") == normalized_type]
        candidates = self._sort_for_claim(candidates)
        if not candidates:
            return {
                "enabled": True,
                "backend": "local_agent_event_queue",
                "claimed": False,
                "event": None,
            }
        event = candidates[0]
        now = _utc_now_iso()
        event["status"] = "claimed"
        event["worker_id"] = _clip(worker_id, 120)
        event["claimed_at"] = now
        event["updated_at"] = now
        event["attempts"] = int(event.get("attempts") or 0) + 1
        self._append_history(event, "claimed", status="claimed", note=f"Claimed by {event['worker_id']}.")
        payload["events"] = self._events(payload)
        _write_json(self.path, payload)
        return {
            "enabled": True,
            "backend": "local_agent_event_queue",
            "claimed": True,
            "event": event,
        }

    def update_status(
        self,
        event_id: str,
        status: str,
        *,
        note: str = "",
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_status = _normalize_status(status)
        payload = self._load()
        events = self._events(payload)
        event = self._find_event(events, event_id)
        if event is None:
            raise ValueError(f"Unknown event: {event_id}")
        previous_status = str(event.get("status") or "")
        now = _utc_now_iso()
        event["status"] = normalized_status
        event["updated_at"] = now
        if normalized_status in {"completed", "failed", "cancelled"}:
            event["completed_at"] = now
        elif previous_status in {"completed", "failed", "cancelled"}:
            event["completed_at"] = ""
        if result is not None:
            event["result"] = _payload_dict(result)
        history = self._append_history(
            event,
            "status_changed",
            status=normalized_status,
            note=note or f"{previous_status} -> {normalized_status}",
            extra={"from": previous_status, "to": normalized_status},
        )
        payload["events"] = events
        _write_json(self.path, payload)
        return {"event": event, "history": history}

    def describe(self) -> dict[str, Any]:
        events = self._events(self._load())
        by_status: dict[str, int] = {status: 0 for status in EVENT_STATUSES}
        by_type: dict[str, int] = {}
        for event in events:
            status = str(event.get("status") or "queued")
            event_type = str(event.get("event_type") or "unknown")
            by_status[status] = by_status.get(status, 0) + 1
            by_type[event_type] = by_type.get(event_type, 0) + 1
        return {
            "enabled": True,
            "backend": "local_agent_event_queue",
            "path": str(self.path),
            "event_count": len(events),
            "by_status": by_status,
            "by_type": by_type,
            "statuses": list(EVENT_STATUSES),
            "priorities": list(EVENT_PRIORITIES),
            "workflow": "enqueue -> claim_next -> complete_or_fail -> workspace",
        }

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.path, {"version": 1, "updated_at": "", "events": []})
        if not isinstance(payload.get("events"), list):
            payload["events"] = []
        return payload

    def _events(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [event for event in payload.get("events", []) if isinstance(event, dict)]

    def _find_event(
        self,
        events: list[dict[str, Any]],
        event_id: str,
    ) -> dict[str, Any] | None:
        for event in events:
            if str(event.get("id") or "") == str(event_id or ""):
                return event
        return None

    def _is_due(self, event: dict[str, Any], current_iso: str) -> bool:
        scheduled_for = str(event.get("scheduled_for") or "")
        return not scheduled_for or scheduled_for <= current_iso

    def _sort_for_claim(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            events,
            key=lambda event: (
                _PRIORITY_RANK.get(str(event.get("priority") or "normal"), 1),
                str(event.get("scheduled_for") or ""),
                str(event.get("created_at") or ""),
            ),
        )

    def _append_history(
        self,
        event: dict[str, Any],
        history_type: str,
        *,
        status: str,
        note: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = {
            "id": f"history-{uuid.uuid4().hex[:10]}",
            "created_at": _utc_now_iso(),
            "type": history_type,
            "status": status,
            "note": _clip(note, 500),
        }
        entry.update(dict(extra or {}))
        event.setdefault("history", [])
        if not isinstance(event["history"], list):
            event["history"] = []
        event["history"].append(entry)
        return entry
