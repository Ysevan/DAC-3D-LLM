"""Local shared browser context records for Agent workspaces."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


BROWSER_CONTEXT_STATUSES = ("active", "captured", "stale", "archived")


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


def _normalize_status(value: str, *, default: str = "active") -> str:
    status = str(value or default).strip().lower()
    if status not in BROWSER_CONTEXT_STATUSES:
        raise ValueError(
            f"Browser context status must be one of: {', '.join(BROWSER_CONTEXT_STATUSES)}."
        )
    return status


def _search_text(context: dict[str, Any]) -> str:
    observations = " ".join(
        " ".join(
            [
                str(observation.get("title") or ""),
                str(observation.get("text") or ""),
                str(observation.get("url") or ""),
            ]
        )
        for observation in context.get("observations") or []
    )
    parts = [
        str(context.get("id") or ""),
        str(context.get("title") or ""),
        str(context.get("url") or ""),
        str(context.get("summary") or ""),
        str(context.get("session_id") or ""),
        str(context.get("thread_id") or ""),
        " ".join(str(item) for item in context.get("tags") or []),
        observations,
    ]
    return " ".join(parts).lower()


class BrowserContextStore:
    """JSON-backed shared browser context metadata for Agent collaboration."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_root(cls, root_dir: Path) -> "BrowserContextStore":
        return cls(root_dir / "agent_browser_contexts.json")

    def create_context(
        self,
        title: str,
        *,
        url: str = "",
        session_id: str = "web",
        thread_id: str = "",
        owner_agent: str = "agent",
        summary: str = "",
        status: str = "active",
        tags: list[Any] | None = None,
        artifact_ids: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_title = _clip(title, 240)
        if not clean_title:
            raise ValueError("Browser context title is required.")
        now = _utc_now_iso()
        context = {
            "id": f"browser-{uuid.uuid4().hex[:12]}",
            "title": clean_title,
            "url": _clip(url, 1000),
            "summary": _clip(summary, 1200),
            "session_id": str(session_id or "web"),
            "thread_id": str(thread_id or ""),
            "owner_agent": _clip(owner_agent, 120) or "agent",
            "status": _normalize_status(status),
            "tags": _strings(tags),
            "artifact_ids": _strings(artifact_ids),
            "metadata": _json_dict(metadata, field_name="Browser context metadata"),
            "observations": [],
            "observation_count": 0,
            "created_at": now,
            "updated_at": now,
            "history": [
                {
                    "at": now,
                    "type": "created",
                    "actor": _clip(owner_agent, 120) or "agent",
                    "note": "Browser context created.",
                }
            ],
        }
        payload = self._load()
        contexts = self._contexts(payload)
        contexts.insert(0, context)
        payload["contexts"] = contexts[:200]
        _write_json(self.path, payload)
        return {"context": context, "created": True}

    def list_contexts(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        thread_id: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        contexts = self._contexts(payload)
        if session_id:
            contexts = [item for item in contexts if str(item.get("session_id") or "") == session_id]
        if status:
            normalized_status = _normalize_status(status)
            contexts = [item for item in contexts if str(item.get("status") or "") == normalized_status]
        if thread_id:
            contexts = [item for item in contexts if str(item.get("thread_id") or "") == thread_id]
        if tag:
            clean_tag = _clip(tag, 220)
            contexts = [item for item in contexts if clean_tag in [str(value) for value in item.get("tags") or []]]
        if query:
            needle = str(query or "").strip().lower()
            contexts = [item for item in contexts if needle and needle in _search_text(item)]
        safe_limit = max(1, min(100, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_browser_context_store",
            "path": str(self.path),
            "contexts": [self._context_summary(item) for item in contexts[:safe_limit]],
            "count": len(contexts[:safe_limit]),
            "total_count": len(self._contexts(payload)),
            "statuses": list(BROWSER_CONTEXT_STATUSES),
            "workflow": "browser_context -> observations -> shared_thread_context",
        }

    def read_context(self, context_id: str) -> dict[str, Any]:
        context = self._find_context(context_id)
        if context is None:
            raise ValueError(f"Unknown browser context: {context_id}")
        return {"enabled": True, "backend": "local_browser_context_store", "context": context}

    def append_observation(
        self,
        context_id: str,
        *,
        url: str = "",
        title: str = "",
        text: str = "",
        agent_role: str = "agent",
        selector: str = "",
        screenshot_path: str = "",
        artifact_ids: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not any(str(value or "").strip() for value in [url, title, text, screenshot_path]):
            raise ValueError("Observation requires at least one of url, title, text, or screenshot_path.")
        payload = self._load()
        contexts = self._contexts(payload)
        context = self._find_in_contexts(contexts, context_id)
        if context is None:
            raise ValueError(f"Unknown browser context: {context_id}")
        now = _utc_now_iso()
        observation = {
            "id": f"observation-{uuid.uuid4().hex[:12]}",
            "url": _clip(url or context.get("url"), 1000),
            "title": _clip(title, 240),
            "text": _clip(text, 4000),
            "agent_role": _clip(agent_role, 120) or "agent",
            "selector": _clip(selector, 240),
            "screenshot_path": _clip(screenshot_path, 1000),
            "artifact_ids": _strings(artifact_ids),
            "metadata": _json_dict(metadata, field_name="Browser observation metadata"),
            "created_at": now,
        }
        observations = list(context.get("observations") or [])
        observations.append(observation)
        context["observations"] = observations[-200:]
        context["observation_count"] = int(context.get("observation_count") or 0) + 1
        context["updated_at"] = now
        if observation["url"]:
            context["url"] = observation["url"]
        for artifact_id in observation["artifact_ids"]:
            if artifact_id not in list(context.get("artifact_ids") or []):
                context.setdefault("artifact_ids", []).append(artifact_id)
        history = self._append_history(
            context,
            "observation_appended",
            actor=observation["agent_role"],
            note="Browser observation appended.",
        )
        _write_json(self.path, payload)
        return {"context": context, "observation": observation, "history": history}

    def update_status(self, context_id: str, status: str, *, note: str = "", actor: str = "agent") -> dict[str, Any]:
        payload = self._load()
        context = self._find_in_contexts(self._contexts(payload), context_id)
        if context is None:
            raise ValueError(f"Unknown browser context: {context_id}")
        old_status = str(context.get("status") or "active")
        new_status = _normalize_status(status)
        context["status"] = new_status
        context["updated_at"] = _utc_now_iso()
        history = self._append_history(
            context,
            "status_changed",
            actor=actor,
            note=note or f"{old_status} -> {new_status}",
            extra={"from": old_status, "to": new_status},
        )
        _write_json(self.path, payload)
        return {"context": context, "history": history}

    def describe(self) -> dict[str, Any]:
        contexts = self._contexts(self._load())
        by_status = {status: 0 for status in BROWSER_CONTEXT_STATUSES}
        observation_count = 0
        thread_ids: set[str] = set()
        for context in contexts:
            status = str(context.get("status") or "active")
            by_status[status] = by_status.get(status, 0) + 1
            observation_count += int(context.get("observation_count") or len(context.get("observations") or []))
            if context.get("thread_id"):
                thread_ids.add(str(context.get("thread_id")))
        return {
            "enabled": True,
            "backend": "local_browser_context_store",
            "path": str(self.path),
            "context_count": len(contexts),
            "observation_count": observation_count,
            "thread_count": len(thread_ids),
            "by_status": by_status,
            "latest_context": self._context_summary(contexts[0]) if contexts else {},
            "workflow": "shared_browser_context -> observations -> agent_thread_resume",
        }

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.path, {"version": 1, "updated_at": "", "contexts": []})
        if not isinstance(payload.get("contexts"), list):
            payload["contexts"] = []
        return payload

    def _contexts(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [item for item in payload.get("contexts", []) if isinstance(item, dict)]

    def _find_context(self, context_id: str) -> dict[str, Any] | None:
        return self._find_in_contexts(self._contexts(self._load()), context_id)

    def _find_in_contexts(self, contexts: list[dict[str, Any]], context_id: str) -> dict[str, Any] | None:
        for context in contexts:
            if str(context.get("id") or "") == str(context_id or ""):
                return context
        return None

    def _append_history(
        self,
        context: dict[str, Any],
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
        events = list(context.get("history") or [])
        events.append(history)
        context["history"] = events[-50:]
        return history

    def _context_summary(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": context.get("id"),
            "title": context.get("title"),
            "url": context.get("url"),
            "summary": context.get("summary"),
            "session_id": context.get("session_id"),
            "thread_id": context.get("thread_id"),
            "owner_agent": context.get("owner_agent"),
            "status": context.get("status"),
            "tags": list(context.get("tags") or []),
            "artifact_ids": list(context.get("artifact_ids") or []),
            "observation_count": int(
                context.get("observation_count") or len(context.get("observations") or [])
            ),
            "updated_at": context.get("updated_at"),
        }
