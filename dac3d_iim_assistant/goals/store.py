"""Local goal tracker for DAC-Agent Runtime.

Goals are lightweight task objectives inspired by coding-agent sessions: a user
can create a durable objective, append progress notes, and mark it complete.
The data stays local and readable as JSON so it can be searched, audited, and
shown in the Agent workspace.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


GOAL_MARKERS = ("目标", "goal", "objective", "任务目标")


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


def _clip(text: Any, limit: int = 1200) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return f"{value[: max(0, limit - 3)]}..."


def _extract_objective(text: str) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""
    patterns = (
        r"(?:目标|任务目标|objective|goal)\s*[:：]\s*(.+)",
        r"(?:我的目标是|目标是|goal is|objective is)\s*(.+)",
    )
    for pattern in patterns:
        match = re.search(pattern, clean, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return _clip(match.group(1).strip(), 500)
    return _clip(clean, 500)


class GoalStore:
    """JSON-backed goal store used by the web UI and Agent runtime."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_root(cls, root_dir: Path) -> "GoalStore":
        return cls(root_dir / "agent_goals.json")

    def create_goal(
        self,
        objective: str,
        *,
        session_id: str = "web",
        source: str = "user",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_objective = _extract_objective(objective)
        if not clean_objective:
            raise ValueError("Goal objective is required.")
        payload = self._load()
        goals = self._goals(payload)
        existing = self._find_duplicate(goals, clean_objective, session_id=session_id)
        if existing is not None:
            return {"goal": existing, "created": False, "duplicate": True}
        now = _utc_now_iso()
        goal = {
            "id": f"goal-{uuid.uuid4().hex[:12]}",
            "session_id": str(session_id or "web"),
            "objective": clean_objective,
            "status": "active",
            "source": str(source or "user"),
            "created_at": now,
            "updated_at": now,
            "completed_at": "",
            "progress": [],
            "metadata": dict(metadata or {}),
        }
        goals.insert(0, goal)
        payload["goals"] = goals
        _write_json(self.path, payload)
        return {"goal": goal, "created": True, "duplicate": False}

    def maybe_create_from_message(
        self,
        message: str,
        *,
        session_id: str,
        source: str = "agent_chat",
    ) -> dict[str, Any] | None:
        text = str(message or "").strip()
        lowered = text.lower()
        if not any(marker in text or marker in lowered for marker in GOAL_MARKERS):
            return None
        objective = _extract_objective(text)
        if len(objective) < 6:
            return None
        return self.create_goal(
            objective,
            session_id=session_id,
            source=source,
            metadata={"captured_from": "chat_message"},
        )

    def list_goals(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        payload = self._load()
        goals = self._goals(payload)
        if session_id:
            goals = [goal for goal in goals if str(goal.get("session_id") or "") == session_id]
        if status:
            normalized_status = str(status).strip()
            goals = [goal for goal in goals if str(goal.get("status") or "") == normalized_status]
        safe_limit = max(1, min(100, int(limit or 20)))
        return {
            "enabled": True,
            "backend": "local_goal_store",
            "path": str(self.path),
            "goals": goals[:safe_limit],
            "count": len(goals[:safe_limit]),
            "total_count": len(self._goals(payload)),
            "workflow": "goal -> progress_notes -> complete -> searchable_history",
        }

    def append_progress(
        self,
        goal_id: str,
        note: str,
        *,
        evidence: dict[str, Any] | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        clean_note = _clip(note, 700)
        if not clean_note:
            raise ValueError("Progress note is required.")
        payload = self._load()
        goals = self._goals(payload)
        goal = self._find_goal(goals, goal_id)
        if goal is None:
            raise ValueError(f"Unknown goal: {goal_id}")
        entry = {
            "id": f"progress-{uuid.uuid4().hex[:10]}",
            "created_at": _utc_now_iso(),
            "note": clean_note,
            "evidence": dict(evidence or {}),
        }
        goal.setdefault("progress", [])
        if not isinstance(goal["progress"], list):
            goal["progress"] = []
        goal["progress"].append(entry)
        if status:
            goal["status"] = str(status)
        goal["updated_at"] = _utc_now_iso()
        payload["goals"] = goals
        _write_json(self.path, payload)
        return {"goal": goal, "progress": entry}

    def complete_goal(self, goal_id: str, *, note: str = "") -> dict[str, Any]:
        payload = self._load()
        goals = self._goals(payload)
        goal = self._find_goal(goals, goal_id)
        if goal is None:
            raise ValueError(f"Unknown goal: {goal_id}")
        if note:
            goal.setdefault("progress", [])
            if isinstance(goal["progress"], list):
                goal["progress"].append(
                    {
                        "id": f"progress-{uuid.uuid4().hex[:10]}",
                        "created_at": _utc_now_iso(),
                        "note": _clip(note, 700),
                        "evidence": {"source": "completion"},
                    }
                )
        goal["status"] = "completed"
        goal["completed_at"] = _utc_now_iso()
        goal["updated_at"] = goal["completed_at"]
        payload["goals"] = goals
        _write_json(self.path, payload)
        return {"goal": goal, "completed": True}

    def describe(self) -> dict[str, Any]:
        payload = self._load()
        goals = self._goals(payload)
        by_status: dict[str, int] = {}
        for goal in goals:
            status = str(goal.get("status") or "unknown")
            by_status[status] = by_status.get(status, 0) + 1
        return {
            "enabled": True,
            "backend": "local_goal_store",
            "path": str(self.path),
            "goal_count": len(goals),
            "by_status": by_status,
            "workflow": "create_goal -> track_progress -> complete_goal -> agent_workspace",
        }

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.path, {"version": 1, "updated_at": "", "goals": []})
        if not isinstance(payload.get("goals"), list):
            payload["goals"] = []
        return payload

    def _goals(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [goal for goal in payload.get("goals", []) if isinstance(goal, dict)]

    def _find_goal(self, goals: list[dict[str, Any]], goal_id: str) -> dict[str, Any] | None:
        for goal in goals:
            if str(goal.get("id") or "") == str(goal_id or ""):
                return goal
        return None

    def _find_duplicate(
        self,
        goals: list[dict[str, Any]],
        objective: str,
        *,
        session_id: str,
    ) -> dict[str, Any] | None:
        normalized = objective.strip().lower()
        for goal in goals:
            if str(goal.get("status") or "") != "active":
                continue
            if str(goal.get("session_id") or "") != str(session_id or "web"):
                continue
            if str(goal.get("objective") or "").strip().lower() == normalized:
                return goal
        return None
