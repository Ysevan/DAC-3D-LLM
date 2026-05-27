"""Local automation planner for DAC-Agent Runtime.

The planner stores scheduled automation definitions and run summaries, but it
does not start a background daemon. A UI, CLI, or future worker can read due
items and execute them through the existing Agent runtime.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


AUTOMATION_STATUSES = ("active", "paused", "archived")
SCHEDULE_TYPES = ("manual", "interval", "daily", "weekly")
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _utc_now_iso() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def _to_iso(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _normalize_status(value: str, *, default: str = "active") -> str:
    status = str(value or default).strip().lower()
    if status not in AUTOMATION_STATUSES:
        raise ValueError(f"Automation status must be one of: {', '.join(AUTOMATION_STATUSES)}.")
    return status


def _parse_time(value: Any) -> tuple[int, int]:
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{2}:\d{2}", text):
        raise ValueError("Schedule time must use HH:MM.")
    hour, minute = [int(part) for part in text.split(":", 1)]
    if hour > 23 or minute > 59:
        raise ValueError("Schedule time must use a valid 24-hour HH:MM value.")
    return hour, minute


def _normalize_schedule(schedule: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(schedule or {})
    schedule_type = str(payload.get("type") or "manual").strip().lower()
    if schedule_type not in SCHEDULE_TYPES:
        raise ValueError(f"Schedule type must be one of: {', '.join(SCHEDULE_TYPES)}.")
    if schedule_type == "manual":
        return {"type": "manual"}
    if schedule_type == "interval":
        try:
            minutes = int(payload.get("interval_minutes") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("interval schedule requires integer interval_minutes.") from exc
        if minutes < 5 or minutes > 10080:
            raise ValueError("interval_minutes must be between 5 and 10080.")
        return {"type": "interval", "interval_minutes": minutes}
    if schedule_type == "daily":
        hour, minute = _parse_time(payload.get("time") or "09:00")
        return {"type": "daily", "time": f"{hour:02d}:{minute:02d}"}
    weekdays = payload.get("weekdays") or ["mon"]
    if not isinstance(weekdays, list):
        raise ValueError("weekly schedule requires weekdays as a list.")
    normalized_weekdays = []
    for weekday in weekdays:
        value = str(weekday or "").strip().lower()[:3]
        if value not in WEEKDAYS:
            raise ValueError(f"Unknown weekday: {weekday}")
        if value not in normalized_weekdays:
            normalized_weekdays.append(value)
    if not normalized_weekdays:
        raise ValueError("weekly schedule requires at least one weekday.")
    hour, minute = _parse_time(payload.get("time") or "09:00")
    return {
        "type": "weekly",
        "weekdays": normalized_weekdays,
        "time": f"{hour:02d}:{minute:02d}",
    }


def _next_run_at(schedule: dict[str, Any], *, now: datetime | None = None) -> str:
    current = now or _utc_now()
    schedule_type = str(schedule.get("type") or "manual")
    if schedule_type == "manual":
        return ""
    if schedule_type == "interval":
        return _to_iso(current + timedelta(minutes=int(schedule.get("interval_minutes") or 5)))
    if schedule_type == "daily":
        hour, minute = _parse_time(schedule.get("time") or "09:00")
        candidate = current.replace(hour=hour, minute=minute, second=0)
        if candidate <= current:
            candidate += timedelta(days=1)
        return _to_iso(candidate)
    hour, minute = _parse_time(schedule.get("time") or "09:00")
    weekdays = [WEEKDAYS.index(day) for day in schedule.get("weekdays", ["mon"])]
    for offset in range(8):
        candidate_date = current + timedelta(days=offset)
        if candidate_date.weekday() not in weekdays:
            continue
        candidate = candidate_date.replace(hour=hour, minute=minute, second=0)
        if candidate > current:
            return _to_iso(candidate)
    return ""


def _schedule_summary(schedule: dict[str, Any]) -> str:
    schedule_type = str(schedule.get("type") or "manual")
    if schedule_type == "manual":
        return "manual"
    if schedule_type == "interval":
        return f"every {schedule.get('interval_minutes')} minutes"
    if schedule_type == "daily":
        return f"daily at {schedule.get('time')}"
    return f"weekly {','.join(schedule.get('weekdays', []))} at {schedule.get('time')}"


class AutomationPlannerStore:
    """JSON-backed scheduled automation definitions."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_root(cls, root_dir: Path) -> "AutomationPlannerStore":
        return cls(root_dir / "agent_automations.json")

    def create_automation(
        self,
        name: str,
        prompt: str,
        *,
        session_id: str = "web",
        schedule: dict[str, Any] | None = None,
        status: str = "active",
        task_id: str = "",
        goal_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_name = _clip(name, 160)
        clean_prompt = _clip(prompt, 1200)
        if not clean_name:
            raise ValueError("Automation name is required.")
        if not clean_prompt:
            raise ValueError("Automation prompt is required.")
        normalized_schedule = _normalize_schedule(schedule)
        normalized_status = _normalize_status(status)
        now = _utc_now_iso()
        automation = {
            "id": f"auto-{uuid.uuid4().hex[:12]}",
            "session_id": str(session_id or "web"),
            "task_id": str(task_id or ""),
            "goal_id": str(goal_id or ""),
            "name": clean_name,
            "prompt": clean_prompt,
            "schedule": normalized_schedule,
            "schedule_summary": _schedule_summary(normalized_schedule),
            "status": normalized_status,
            "created_at": now,
            "updated_at": now,
            "last_run_at": "",
            "next_run_at": _next_run_at(normalized_schedule) if normalized_status == "active" else "",
            "run_count": 0,
            "runs": [],
            "metadata": dict(metadata or {}),
        }
        payload = self._load()
        automations = self._automations(payload)
        automations.insert(0, automation)
        payload["automations"] = automations
        _write_json(self.path, payload)
        return {"automation": automation, "created": True}

    def list_automations(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        automations = self._automations(payload)
        if session_id:
            automations = [
                automation
                for automation in automations
                if str(automation.get("session_id") or "") == session_id
            ]
        if status:
            normalized_status = _normalize_status(status)
            automations = [
                automation
                for automation in automations
                if str(automation.get("status") or "") == normalized_status
            ]
        safe_limit = max(1, min(200, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_automation_planner",
            "path": str(self.path),
            "automations": automations[:safe_limit],
            "count": len(automations[:safe_limit]),
            "total_count": len(self._automations(payload)),
            "schedule_types": list(SCHEDULE_TYPES),
            "workflow": "automation_definition -> due_check -> external_runner -> run_record",
        }

    def update_status(self, automation_id: str, status: str, *, note: str = "") -> dict[str, Any]:
        normalized_status = _normalize_status(status)
        payload = self._load()
        automations = self._automations(payload)
        automation = self._find_automation(automations, automation_id)
        if automation is None:
            raise ValueError(f"Unknown automation: {automation_id}")
        automation["status"] = normalized_status
        automation["updated_at"] = _utc_now_iso()
        automation["next_run_at"] = (
            _next_run_at(dict(automation.get("schedule") or {}))
            if normalized_status == "active"
            else ""
        )
        if note:
            automation.setdefault("metadata", {})
            if isinstance(automation["metadata"], dict):
                automation["metadata"]["status_note"] = _clip(note, 500)
        payload["automations"] = automations
        _write_json(self.path, payload)
        return {"automation": automation}

    def record_run(
        self,
        automation_id: str,
        *,
        result: str = "",
        status: str = "completed",
        trace_id: str = "",
    ) -> dict[str, Any]:
        payload = self._load()
        automations = self._automations(payload)
        automation = self._find_automation(automations, automation_id)
        if automation is None:
            raise ValueError(f"Unknown automation: {automation_id}")
        now = _utc_now_iso()
        run = {
            "id": f"run-{uuid.uuid4().hex[:10]}",
            "created_at": now,
            "status": _clip(status, 80),
            "result": _clip(result, 900),
            "trace_id": str(trace_id or ""),
        }
        automation.setdefault("runs", [])
        if not isinstance(automation["runs"], list):
            automation["runs"] = []
        automation["runs"].insert(0, run)
        automation["runs"] = automation["runs"][:20]
        automation["run_count"] = int(automation.get("run_count") or 0) + 1
        automation["last_run_at"] = now
        automation["updated_at"] = now
        if str(automation.get("status") or "") == "active":
            automation["next_run_at"] = _next_run_at(dict(automation.get("schedule") or {}))
        payload["automations"] = automations
        _write_json(self.path, payload)
        return {"automation": automation, "run": run}

    def due_automations(self, *, now: datetime | None = None, limit: int = 20) -> dict[str, Any]:
        current_iso = _to_iso(now or _utc_now())
        due = []
        for automation in self._automations(self._load()):
            if str(automation.get("status") or "") != "active":
                continue
            next_run_at = str(automation.get("next_run_at") or "")
            if next_run_at and next_run_at <= current_iso:
                due.append(automation)
        safe_limit = max(1, min(100, int(limit or 20)))
        return {"enabled": True, "automations": due[:safe_limit], "count": len(due[:safe_limit])}

    def describe(self) -> dict[str, Any]:
        payload = self._load()
        automations = self._automations(payload)
        by_status: dict[str, int] = {status: 0 for status in AUTOMATION_STATUSES}
        for automation in automations:
            status = str(automation.get("status") or "active")
            by_status[status] = by_status.get(status, 0) + 1
        return {
            "enabled": True,
            "backend": "local_automation_planner",
            "path": str(self.path),
            "automation_count": len(automations),
            "by_status": by_status,
            "schedule_types": list(SCHEDULE_TYPES),
            "workflow": "define -> schedule_metadata -> due_discovery -> external_execution_trace",
            "runner": "external_or_future_worker",
        }

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.path, {"version": 1, "updated_at": "", "automations": []})
        if not isinstance(payload.get("automations"), list):
            payload["automations"] = []
        return payload

    def _automations(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            automation
            for automation in payload.get("automations", [])
            if isinstance(automation, dict)
        ]

    def _find_automation(
        self,
        automations: list[dict[str, Any]],
        automation_id: str,
    ) -> dict[str, Any] | None:
        for automation in automations:
            if str(automation.get("id") or "") == str(automation_id or ""):
                return automation
        return None
