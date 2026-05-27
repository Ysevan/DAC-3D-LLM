"""Local workflow checkpoints for DAC-Agent Runtime.

Checkpoints persist compact state snapshots so a long-running Agent workflow can
leave a clear resume point. Restoring a checkpoint records the selected resume
point; it does not mutate files, tasks, or DAC command state.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


CHECKPOINT_STATUSES = ("active", "restored", "archived")


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


def _normalize_status(value: str, *, default: str = "active") -> str:
    status = str(value or default).strip().lower()
    if status not in CHECKPOINT_STATUSES:
        raise ValueError(f"Checkpoint status must be one of: {', '.join(CHECKPOINT_STATUSES)}.")
    return status


def _strings(values: list[Any] | tuple[Any, ...] | None, *, limit: int = 30) -> list[str]:
    if not values:
        return []
    result: list[str] = []
    for value in values[:limit]:
        text = _clip(value, 180)
        if text:
            result.append(text)
    return result


def _json_object(value: dict[str, Any] | None, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable.") from exc


class CheckpointStore:
    """JSON-backed workflow checkpoint store for Agent workspace state."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_root(cls, root_dir: Path) -> "CheckpointStore":
        return cls(root_dir / "agent_checkpoints.json")

    def create_checkpoint(
        self,
        title: str,
        *,
        state: dict[str, Any] | None = None,
        session_id: str = "web",
        summary: str = "",
        status: str = "active",
        task_id: str = "",
        workflow_id: str = "",
        event_id: str = "",
        review_id: str = "",
        trace_id: str = "",
        parent_checkpoint_id: str = "",
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_title = _clip(title, 180)
        if not clean_title:
            raise ValueError("Checkpoint title is required.")
        normalized_status = _normalize_status(status)
        clean_state = _json_object(state, field_name="Checkpoint state")
        now = _utc_now_iso()
        checkpoint = {
            "id": f"checkpoint-{uuid.uuid4().hex[:12]}",
            "session_id": str(session_id or "web"),
            "task_id": str(task_id or ""),
            "workflow_id": str(workflow_id or ""),
            "event_id": str(event_id or ""),
            "review_id": str(review_id or ""),
            "trace_id": str(trace_id or ""),
            "parent_checkpoint_id": str(parent_checkpoint_id or ""),
            "title": clean_title,
            "summary": _clip(summary, 1200),
            "status": normalized_status,
            "state": clean_state,
            "tags": _strings(tags),
            "created_at": now,
            "updated_at": now,
            "restored_at": now if normalized_status == "restored" else "",
            "history": [
                {
                    "id": f"history-{uuid.uuid4().hex[:10]}",
                    "created_at": now,
                    "type": "created",
                    "status": normalized_status,
                    "note": "Checkpoint created.",
                }
            ],
            "metadata": _json_object(metadata, field_name="Checkpoint metadata"),
        }
        payload = self._load()
        checkpoints = self._checkpoints(payload)
        checkpoints.insert(0, checkpoint)
        payload["checkpoints"] = checkpoints[:200]
        _write_json(self.path, payload)
        return {"checkpoint": checkpoint, "created": True}

    def list_checkpoints(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        tag: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        checkpoints = self._checkpoints(payload)
        if session_id:
            checkpoints = [
                checkpoint
                for checkpoint in checkpoints
                if str(checkpoint.get("session_id") or "") == session_id
            ]
        if status:
            normalized_status = _normalize_status(status)
            checkpoints = [
                checkpoint
                for checkpoint in checkpoints
                if str(checkpoint.get("status") or "") == normalized_status
            ]
        if tag:
            normalized_tag = _clip(tag, 180)
            checkpoints = [
                checkpoint
                for checkpoint in checkpoints
                if normalized_tag in list(checkpoint.get("tags") or [])
            ]
        safe_limit = max(1, min(200, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_agent_checkpoint_store",
            "path": str(self.path),
            "checkpoints": checkpoints[:safe_limit],
            "count": len(checkpoints[:safe_limit]),
            "total_count": len(self._checkpoints(payload)),
            "statuses": list(CHECKPOINT_STATUSES),
            "workflow": "capture_state -> checkpoint -> restore_marker -> resume_context",
        }

    def read_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        checkpoint = self._find_checkpoint(self._checkpoints(self._load()), checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Unknown checkpoint: {checkpoint_id}")
        return {"enabled": True, "checkpoint": checkpoint}

    def restore_checkpoint(
        self,
        checkpoint_id: str,
        *,
        note: str = "",
        actor: str = "agent",
    ) -> dict[str, Any]:
        payload = self._load()
        checkpoints = self._checkpoints(payload)
        checkpoint = self._find_checkpoint(checkpoints, checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Unknown checkpoint: {checkpoint_id}")
        now = _utc_now_iso()
        previous_status = str(checkpoint.get("status") or "")
        checkpoint["status"] = "restored"
        checkpoint["updated_at"] = now
        checkpoint["restored_at"] = now
        history = self._append_history(
            checkpoint,
            "restored",
            status="restored",
            note=note or "Checkpoint selected as resume point.",
            extra={"from": previous_status, "actor": _clip(actor or "agent", 120)},
        )
        payload["checkpoints"] = checkpoints
        payload["last_restored_checkpoint_id"] = checkpoint_id
        _write_json(self.path, payload)
        return {"checkpoint": checkpoint, "history": history, "restored": True}

    def archive_checkpoint(self, checkpoint_id: str, *, note: str = "") -> dict[str, Any]:
        payload = self._load()
        checkpoints = self._checkpoints(payload)
        checkpoint = self._find_checkpoint(checkpoints, checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Unknown checkpoint: {checkpoint_id}")
        now = _utc_now_iso()
        previous_status = str(checkpoint.get("status") or "")
        checkpoint["status"] = "archived"
        checkpoint["updated_at"] = now
        history = self._append_history(
            checkpoint,
            "archived",
            status="archived",
            note=note or "Checkpoint archived.",
            extra={"from": previous_status},
        )
        payload["checkpoints"] = checkpoints
        _write_json(self.path, payload)
        return {"checkpoint": checkpoint, "history": history}

    def describe(self) -> dict[str, Any]:
        payload = self._load()
        checkpoints = self._checkpoints(payload)
        by_status = {status: 0 for status in CHECKPOINT_STATUSES}
        for checkpoint in checkpoints:
            status = str(checkpoint.get("status") or "active")
            by_status[status] = by_status.get(status, 0) + 1
        latest = checkpoints[0] if checkpoints else {}
        return {
            "enabled": True,
            "backend": "local_agent_checkpoint_store",
            "path": str(self.path),
            "checkpoint_count": len(checkpoints),
            "by_status": by_status,
            "latest_checkpoint": latest,
            "last_restored_checkpoint_id": str(payload.get("last_restored_checkpoint_id") or ""),
            "workflow": "checkpoint_state -> persist -> restore_marker -> continue",
        }

    def _load(self) -> dict[str, Any]:
        payload = _read_json(
            self.path,
            {
                "version": 1,
                "updated_at": "",
                "last_restored_checkpoint_id": "",
                "checkpoints": [],
            },
        )
        if not isinstance(payload.get("checkpoints"), list):
            payload["checkpoints"] = []
        return payload

    def _checkpoints(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            checkpoint
            for checkpoint in payload.get("checkpoints", [])
            if isinstance(checkpoint, dict)
        ]

    def _find_checkpoint(
        self,
        checkpoints: list[dict[str, Any]],
        checkpoint_id: str,
    ) -> dict[str, Any] | None:
        for checkpoint in checkpoints:
            if str(checkpoint.get("id") or "") == str(checkpoint_id or ""):
                return checkpoint
        return None

    def _append_history(
        self,
        checkpoint: dict[str, Any],
        event_type: str,
        *,
        status: str,
        note: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        history_items = checkpoint.setdefault("history", [])
        if not isinstance(history_items, list):
            history_items = []
            checkpoint["history"] = history_items
        history = {
            "id": f"history-{uuid.uuid4().hex[:10]}",
            "created_at": _utc_now_iso(),
            "type": event_type,
            "status": status,
            "note": _clip(note, 800),
        }
        if extra:
            history.update(extra)
        history_items.append(history)
        return history
