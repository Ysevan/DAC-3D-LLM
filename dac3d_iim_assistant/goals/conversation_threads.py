"""Local conversation threads for multi-agent collaboration."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


THREAD_STATUSES = ("active", "paused", "resolved", "archived")
MESSAGE_ROLES = ("user", "assistant", "agent", "tool", "system")


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


def _strings(values: list[Any] | tuple[Any, ...] | None, *, limit: int = 50) -> list[str]:
    if not values:
        return []
    result: list[str] = []
    for value in values[:limit]:
        text = _clip(value, 180)
        if text and text not in result:
            result.append(text)
    return result


def _json_dict(value: dict[str, Any] | None, *, field_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(json.dumps(dict(value or {}), ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable.") from exc
    return payload if isinstance(payload, dict) else {}


def _json_list(values: list[Any] | None, *, field_name: str, limit: int = 50) -> list[Any]:
    try:
        payload = json.loads(json.dumps(list(values or [])[:limit], ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable.") from exc
    return payload if isinstance(payload, list) else []


def _normalize_status(value: str, *, default: str = "active") -> str:
    status = str(value or default).strip().lower()
    if status not in THREAD_STATUSES:
        raise ValueError(f"Thread status must be one of: {', '.join(THREAD_STATUSES)}.")
    return status


def _normalize_role(value: str, *, default: str = "agent") -> str:
    role = str(value or default).strip().lower()
    if role not in MESSAGE_ROLES:
        raise ValueError(f"Message role must be one of: {', '.join(MESSAGE_ROLES)}.")
    return role


def _search_text(thread: dict[str, Any]) -> str:
    message_text = " ".join(str(message.get("content") or "") for message in thread.get("messages") or [])
    parts = [
        str(thread.get("id") or ""),
        str(thread.get("title") or ""),
        str(thread.get("summary") or ""),
        str(thread.get("session_id") or ""),
        " ".join(str(item) for item in thread.get("participants") or []),
        " ".join(str(item) for item in thread.get("tags") or []),
        message_text,
    ]
    return " ".join(parts).lower()


class ConversationThreadStore:
    """JSON-backed multi-agent conversation thread workspace."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_root(cls, root_dir: Path) -> "ConversationThreadStore":
        return cls(root_dir / "agent_threads.json")

    def create_thread(
        self,
        title: str,
        *,
        session_id: str = "web",
        summary: str = "",
        participants: list[Any] | None = None,
        status: str = "active",
        tags: list[Any] | None = None,
        shared_state_ids: list[Any] | None = None,
        artifact_ids: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        created_by: str = "agent",
    ) -> dict[str, Any]:
        clean_title = _clip(title, 240)
        if not clean_title:
            raise ValueError("Thread title is required.")
        now = _utc_now_iso()
        thread = {
            "id": f"thread-{uuid.uuid4().hex[:12]}",
            "title": clean_title,
            "summary": _clip(summary, 1000),
            "session_id": str(session_id or "web"),
            "status": _normalize_status(status),
            "participants": _strings(participants),
            "tags": _strings(tags),
            "shared_state_ids": _strings(shared_state_ids),
            "artifact_ids": _strings(artifact_ids),
            "metadata": _json_dict(metadata, field_name="Thread metadata"),
            "created_by": _clip(created_by, 120) or "agent",
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "messages": [],
            "history": [
                {
                    "at": now,
                    "type": "created",
                    "actor": _clip(created_by, 120) or "agent",
                    "note": "Conversation thread created.",
                }
            ],
        }
        payload = self._load()
        threads = self._threads(payload)
        threads.insert(0, thread)
        payload["threads"] = threads[:200]
        _write_json(self.path, payload)
        return {"thread": thread, "created": True}

    def list_threads(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        participant: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        threads = self._threads(payload)
        if session_id:
            threads = [thread for thread in threads if str(thread.get("session_id") or "") == session_id]
        if status:
            normalized_status = _normalize_status(status)
            threads = [thread for thread in threads if str(thread.get("status") or "") == normalized_status]
        if participant:
            clean_participant = _clip(participant, 180)
            threads = [
                thread
                for thread in threads
                if clean_participant in [str(item) for item in thread.get("participants") or []]
            ]
        if tag:
            clean_tag = _clip(tag, 180)
            threads = [thread for thread in threads if clean_tag in [str(item) for item in thread.get("tags") or []]]
        if query:
            needle = str(query or "").strip().lower()
            threads = [thread for thread in threads if needle and needle in _search_text(thread)]
        safe_limit = max(1, min(100, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_agent_threads",
            "path": str(self.path),
            "threads": [self._thread_summary(thread) for thread in threads[:safe_limit]],
            "count": len(threads[:safe_limit]),
            "total_count": len(self._threads(payload)),
            "statuses": list(THREAD_STATUSES),
            "workflow": "conversation_thread -> participants -> shared_context_refs -> resume",
        }

    def read_thread(self, thread_id: str) -> dict[str, Any]:
        thread = self._find_thread(thread_id)
        if thread is None:
            raise ValueError(f"Unknown conversation thread: {thread_id}")
        return {"enabled": True, "backend": "local_agent_threads", "thread": thread}

    def append_message(
        self,
        thread_id: str,
        *,
        role: str,
        content: str,
        agent_role: str = "",
        tool_calls: list[Any] | None = None,
        artifact_ids: list[Any] | None = None,
        shared_state_ids: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_content = _clip(content, 4000)
        if not clean_content:
            raise ValueError("Message content is required.")
        payload = self._load()
        threads = self._threads(payload)
        thread = self._find_in_threads(threads, thread_id)
        if thread is None:
            raise ValueError(f"Unknown conversation thread: {thread_id}")
        now = _utc_now_iso()
        message = {
            "id": f"msg-{uuid.uuid4().hex[:12]}",
            "role": _normalize_role(role),
            "agent_role": _clip(agent_role, 120),
            "content": clean_content,
            "tool_calls": _json_list(tool_calls, field_name="Message tool_calls", limit=25),
            "artifact_ids": _strings(artifact_ids),
            "shared_state_ids": _strings(shared_state_ids),
            "metadata": _json_dict(metadata, field_name="Message metadata"),
            "created_at": now,
        }
        messages = list(thread.get("messages") or [])
        messages.append(message)
        thread["messages"] = messages[-200:]
        thread["message_count"] = int(thread.get("message_count") or 0) + 1
        thread["updated_at"] = now
        for participant in [message["agent_role"]]:
            if participant and participant not in list(thread.get("participants") or []):
                thread.setdefault("participants", []).append(participant)
        for state_id in message["shared_state_ids"]:
            if state_id not in list(thread.get("shared_state_ids") or []):
                thread.setdefault("shared_state_ids", []).append(state_id)
        for artifact_id in message["artifact_ids"]:
            if artifact_id not in list(thread.get("artifact_ids") or []):
                thread.setdefault("artifact_ids", []).append(artifact_id)
        history = self._append_history(
            thread,
            "message_appended",
            actor=message["agent_role"] or message["role"],
            note=f"Appended {message['role']} message.",
        )
        _write_json(self.path, payload)
        return {"thread": thread, "message": message, "history": history}

    def update_status(self, thread_id: str, status: str, *, note: str = "", actor: str = "agent") -> dict[str, Any]:
        payload = self._load()
        thread = self._find_in_threads(self._threads(payload), thread_id)
        if thread is None:
            raise ValueError(f"Unknown conversation thread: {thread_id}")
        old_status = str(thread.get("status") or "active")
        new_status = _normalize_status(status)
        thread["status"] = new_status
        thread["updated_at"] = _utc_now_iso()
        history = self._append_history(
            thread,
            "status_changed",
            actor=actor,
            note=note or f"{old_status} -> {new_status}",
            extra={"from": old_status, "to": new_status},
        )
        _write_json(self.path, payload)
        return {"thread": thread, "history": history}

    def describe(self) -> dict[str, Any]:
        threads = self._threads(self._load())
        by_status = {status: 0 for status in THREAD_STATUSES}
        participants: set[str] = set()
        message_count = 0
        for thread in threads:
            status = str(thread.get("status") or "active")
            by_status[status] = by_status.get(status, 0) + 1
            message_count += int(thread.get("message_count") or len(thread.get("messages") or []))
            for participant in thread.get("participants") or []:
                participants.add(str(participant))
        return {
            "enabled": True,
            "backend": "local_agent_threads",
            "path": str(self.path),
            "thread_count": len(threads),
            "message_count": message_count,
            "participant_count": len(participants),
            "by_status": by_status,
            "latest_thread": self._thread_summary(threads[0]) if threads else {},
            "workflow": "threaded_multi_agent_workspace -> shared_context_refs -> resumable_collaboration",
        }

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.path, {"version": 1, "updated_at": "", "threads": []})
        if not isinstance(payload.get("threads"), list):
            payload["threads"] = []
        return payload

    def _threads(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [thread for thread in payload.get("threads", []) if isinstance(thread, dict)]

    def _find_thread(self, thread_id: str) -> dict[str, Any] | None:
        return self._find_in_threads(self._threads(self._load()), thread_id)

    def _find_in_threads(self, threads: list[dict[str, Any]], thread_id: str) -> dict[str, Any] | None:
        for thread in threads:
            if str(thread.get("id") or "") == str(thread_id or ""):
                return thread
        return None

    def _append_history(
        self,
        thread: dict[str, Any],
        event_type: str,
        *,
        actor: str,
        note: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        history = {
            "at": _utc_now_iso(),
            "type": event_type,
            "actor": _clip(actor, 120) or "agent",
            "note": _clip(note, 800),
            **(extra or {}),
        }
        events = list(thread.get("history") or [])
        events.append(history)
        thread["history"] = events[-50:]
        return history

    def _thread_summary(self, thread: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": thread.get("id"),
            "title": thread.get("title"),
            "summary": thread.get("summary"),
            "session_id": thread.get("session_id"),
            "status": thread.get("status"),
            "participants": list(thread.get("participants") or []),
            "tags": list(thread.get("tags") or []),
            "shared_state_ids": list(thread.get("shared_state_ids") or []),
            "artifact_ids": list(thread.get("artifact_ids") or []),
            "message_count": int(thread.get("message_count") or len(thread.get("messages") or [])),
            "updated_at": thread.get("updated_at"),
        }
