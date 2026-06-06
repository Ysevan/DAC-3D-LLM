"""Local task board for DAC-Agent multi-session work.

The task board is inspired by coding-agent task boards: each card captures a
unit of Agent work, its target specialist path, candidate tools, dependencies,
and current status. It is intentionally JSON-backed so it stays local,
auditable, and easy to inspect alongside goals and traces.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_STATUSES = ("backlog", "ready", "in_progress", "blocked", "done")
TASK_PRIORITIES = ("low", "normal", "high")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _normalize_status(value: str, *, default: str = "backlog") -> str:
    status = str(value or default).strip().lower()
    if status not in TASK_STATUSES:
        raise ValueError(f"Task status must be one of: {', '.join(TASK_STATUSES)}.")
    return status


def _normalize_priority(value: str) -> str:
    priority = str(value or "normal").strip().lower()
    if priority not in TASK_PRIORITIES:
        raise ValueError(f"Task priority must be one of: {', '.join(TASK_PRIORITIES)}.")
    return priority


def _strings(values: list[Any] | tuple[Any, ...] | None, *, limit: int = 20) -> list[str]:
    if not values:
        return []
    result: list[str] = []
    for value in values[:limit]:
        text = _clip(value, 200)
        if text:
            result.append(text)
    return result


class TaskBoardStore:
    """JSON-backed task board used by the Agent workspace and Web API."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_root(cls, root_dir: Path) -> "TaskBoardStore":
        return cls(root_dir / "agent_task_board.json")

    def create_task(
        self,
        title: str,
        *,
        session_id: str = "web",
        description: str = "",
        status: str = "backlog",
        priority: str = "normal",
        goal_id: str = "",
        agent_path: list[Any] | None = None,
        tool_candidates: list[Any] | None = None,
        dependencies: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_title = _clip(title, 180)
        if not clean_title:
            raise ValueError("Task title is required.")
        normalized_status = _normalize_status(status)
        normalized_priority = _normalize_priority(priority)
        payload = self._load()
        tasks = self._tasks(payload)
        now = _utc_now_iso()
        task = {
            "id": f"task-{uuid.uuid4().hex[:12]}",
            "session_id": str(session_id or "web"),
            "goal_id": str(goal_id or ""),
            "title": clean_title,
            "description": _clip(description, 900),
            "status": normalized_status,
            "priority": normalized_priority,
            "agent_path": _strings(agent_path),
            "tool_candidates": _strings(tool_candidates),
            "dependencies": _strings(dependencies),
            "created_at": now,
            "updated_at": now,
            "completed_at": "",
            "events": [
                {
                    "id": f"event-{uuid.uuid4().hex[:10]}",
                    "created_at": now,
                    "type": "created",
                    "status": normalized_status,
                    "note": "Task card created.",
                }
            ],
            "metadata": dict(metadata or {}),
        }
        tasks.insert(0, task)
        payload["tasks"] = tasks
        _write_json(self.path, payload)
        return {"task": task, "created": True}

    def create_from_workflow_preview(
        self,
        preview: dict[str, Any],
        *,
        session_id: str = "web",
        goal_id: str = "",
        status: str = "ready",
        priority: str = "normal",
    ) -> dict[str, Any]:
        task_text = _clip(preview.get("task"), 180)
        if not task_text:
            raise ValueError("Workflow preview must include a task.")
        context_sections = preview.get("context_sections") if isinstance(preview.get("context_sections"), list) else []
        return self.create_task(
            task_text,
            session_id=session_id or str(preview.get("session_id") or "web"),
            description="Generated from Agent workflow preview.",
            status=status,
            priority=priority,
            goal_id=goal_id,
            agent_path=list(preview.get("agent_path") or []),
            tool_candidates=list(preview.get("tool_candidates") or []),
            metadata={
                "source": "workflow_preview",
                "preview_backend": preview.get("backend"),
                "context_section_count": len(context_sections),
                "node_count": len(preview.get("nodes") or []),
            },
        )

    def list_tasks(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        goal_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        tasks = self._tasks(payload)
        if session_id:
            tasks = [task for task in tasks if str(task.get("session_id") or "") == session_id]
        if status:
            normalized_status = _normalize_status(status)
            tasks = [task for task in tasks if str(task.get("status") or "") == normalized_status]
        if goal_id:
            tasks = [task for task in tasks if str(task.get("goal_id") or "") == goal_id]
        safe_limit = max(1, min(200, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_agent_task_board",
            "path": str(self.path),
            "tasks": tasks[:safe_limit],
            "count": len(tasks[:safe_limit]),
            "total_count": len(self._tasks(payload)),
            "columns": self.columns(),
            "workflow": "goal_or_workflow_preview -> task_card -> status_transitions -> workspace",
        }

    def update_status(self, task_id: str, status: str, *, note: str = "") -> dict[str, Any]:
        normalized_status = _normalize_status(status)
        payload = self._load()
        tasks = self._tasks(payload)
        task = self._find_task(tasks, task_id)
        if task is None:
            raise ValueError(f"Unknown task: {task_id}")
        now = _utc_now_iso()
        previous_status = str(task.get("status") or "")
        task["status"] = normalized_status
        task["updated_at"] = now
        if normalized_status == "done":
            task["completed_at"] = now
        elif previous_status == "done":
            task["completed_at"] = ""
        task.setdefault("events", [])
        if not isinstance(task["events"], list):
            task["events"] = []
        event = {
            "id": f"event-{uuid.uuid4().hex[:10]}",
            "created_at": now,
            "type": "status_changed",
            "from": previous_status,
            "to": normalized_status,
            "note": _clip(note, 500),
        }
        task["events"].append(event)
        payload["tasks"] = tasks
        _write_json(self.path, payload)
        return {"task": task, "event": event}

    def describe(self) -> dict[str, Any]:
        payload = self._load()
        tasks = self._tasks(payload)
        by_status = {status: 0 for status in TASK_STATUSES}
        for task in tasks:
            status = str(task.get("status") or "backlog")
            by_status[status] = by_status.get(status, 0) + 1
        return {
            "enabled": True,
            "backend": "local_agent_task_board",
            "path": str(self.path),
            "task_count": len(tasks),
            "columns": self.columns(),
            "by_status": by_status,
            "workflow": "create_task -> route_agent -> track_status -> done",
        }

    def columns(self) -> list[dict[str, str]]:
        return [
            {"id": "backlog", "label": "Backlog"},
            {"id": "ready", "label": "Ready"},
            {"id": "in_progress", "label": "In Progress"},
            {"id": "blocked", "label": "Blocked"},
            {"id": "done", "label": "Done"},
        ]

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.path, {"version": 1, "updated_at": "", "tasks": []})
        if not isinstance(payload.get("tasks"), list):
            payload["tasks"] = []
        return payload

    def _tasks(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [task for task in payload.get("tasks", []) if isinstance(task, dict)]

    def _find_task(self, tasks: list[dict[str, Any]], task_id: str) -> dict[str, Any] | None:
        for task in tasks:
            if str(task.get("id") or "") == str(task_id or ""):
                return task
        return None
