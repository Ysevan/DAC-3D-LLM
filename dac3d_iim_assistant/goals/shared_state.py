"""Local shared state store for DAC-Agent collaboration."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


STATE_SCOPES = ("session", "workflow", "task", "global")


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


def _normalize_scope(value: str, *, default: str = "session") -> str:
    scope = str(value or default).strip().lower()
    if scope not in STATE_SCOPES:
        raise ValueError(f"Shared state scope must be one of: {', '.join(STATE_SCOPES)}.")
    return scope


def _json_value(value: Any, *, field_name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable.") from exc


def _strings(values: list[Any] | tuple[Any, ...] | None, *, limit: int = 30) -> list[str]:
    if not values:
        return []
    result: list[str] = []
    for value in values[:limit]:
        text = _clip(value, 160)
        if text:
            result.append(text)
    return result


class SharedStateStore:
    """JSON-backed shared state for multi-agent workspace coordination."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_root(cls, root_dir: Path) -> "SharedStateStore":
        return cls(root_dir / "agent_shared_state.json")

    def set_state(
        self,
        key: str,
        value: Any,
        *,
        session_id: str = "web",
        scope: str = "session",
        namespace: str = "default",
        owner_agent: str = "",
        task_id: str = "",
        workflow_id: str = "",
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        clean_key = _clip(key, 180)
        if not clean_key:
            raise ValueError("Shared state key is required.")
        normalized_scope = _normalize_scope(scope)
        clean_namespace = _clip(namespace or "default", 120) or "default"
        payload = self._load()
        states = self._states(payload)
        state = self._find_by_identity(
            states,
            key=clean_key,
            session_id=session_id,
            scope=normalized_scope,
            namespace=clean_namespace,
            task_id=task_id,
            workflow_id=workflow_id,
        )
        now = _utc_now_iso()
        if state is None:
            state = {
                "id": f"state-{uuid.uuid4().hex[:12]}",
                "key": clean_key,
                "session_id": str(session_id or "web"),
                "scope": normalized_scope,
                "namespace": clean_namespace,
                "owner_agent": _clip(owner_agent, 120),
                "task_id": str(task_id or ""),
                "workflow_id": str(workflow_id or ""),
                "value": _json_value(value, field_name="Shared state value"),
                "version": 1,
                "tags": _strings(tags),
                "created_at": now,
                "updated_at": now,
                "history": [],
                "metadata": _json_value(dict(metadata or {}), field_name="Shared state metadata"),
            }
            states.insert(0, state)
            created = True
            event_type = "created"
        else:
            state["value"] = _json_value(value, field_name="Shared state value")
            state["version"] = int(state.get("version") or 1) + 1
            state["updated_at"] = now
            if owner_agent:
                state["owner_agent"] = _clip(owner_agent, 120)
            if tags is not None:
                state["tags"] = _strings(tags)
            if metadata is not None:
                state["metadata"] = _json_value(dict(metadata), field_name="Shared state metadata")
            created = False
            event_type = "updated"
        history = self._append_history(
            state,
            event_type,
            note=note or f"Shared state {event_type}.",
            actor=owner_agent or "agent",
        )
        payload["states"] = states[:300]
        _write_json(self.path, payload)
        return {"state": state, "history": history, "created": created}

    def list_states(
        self,
        *,
        session_id: str | None = None,
        scope: str | None = None,
        namespace: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        states = self._states(payload)
        if session_id:
            states = [state for state in states if str(state.get("session_id") or "") == session_id]
        if scope:
            normalized_scope = _normalize_scope(scope)
            states = [state for state in states if str(state.get("scope") or "") == normalized_scope]
        if namespace:
            clean_namespace = _clip(namespace, 120)
            states = [state for state in states if str(state.get("namespace") or "") == clean_namespace]
        if tag:
            clean_tag = _clip(tag, 160)
            states = [state for state in states if clean_tag in list(state.get("tags") or [])]
        if query:
            needle = str(query or "").strip().lower()
            states = [
                state
                for state in states
                if needle
                and needle
                in " ".join(
                    [
                        str(state.get("key") or ""),
                        str(state.get("namespace") or ""),
                        str(state.get("owner_agent") or ""),
                        json.dumps(state.get("value"), ensure_ascii=False),
                    ]
                ).lower()
            ]
        safe_limit = max(1, min(200, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_agent_shared_state",
            "path": str(self.path),
            "states": states[:safe_limit],
            "count": len(states[:safe_limit]),
            "total_count": len(self._states(payload)),
            "scopes": list(STATE_SCOPES),
            "workflow": "agent_write_state -> scoped_context_share -> agent_read_state",
        }

    def read_state(
        self,
        state_id_or_key: str,
        *,
        session_id: str = "web",
        scope: str = "session",
        namespace: str = "default",
    ) -> dict[str, Any]:
        states = self._states(self._load())
        state = self._find_by_id(states, state_id_or_key)
        if state is None:
            state = self._find_by_identity(
                states,
                key=state_id_or_key,
                session_id=session_id,
                scope=_normalize_scope(scope),
                namespace=_clip(namespace or "default", 120) or "default",
                task_id="",
                workflow_id="",
            )
        if state is None:
            raise ValueError(f"Unknown shared state: {state_id_or_key}")
        return {"enabled": True, "state": state}

    def describe(self) -> dict[str, Any]:
        states = self._states(self._load())
        by_scope = {scope: 0 for scope in STATE_SCOPES}
        namespaces: set[str] = set()
        for state in states:
            scope = str(state.get("scope") or "session")
            by_scope[scope] = by_scope.get(scope, 0) + 1
            namespaces.add(str(state.get("namespace") or "default"))
        return {
            "enabled": True,
            "backend": "local_agent_shared_state",
            "path": str(self.path),
            "state_count": len(states),
            "by_scope": by_scope,
            "namespace_count": len(namespaces),
            "latest_state": states[0] if states else {},
            "workflow": "scoped_json_state -> shared_context -> isolated_resume",
        }

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.path, {"version": 1, "updated_at": "", "states": []})
        if not isinstance(payload.get("states"), list):
            payload["states"] = []
        return payload

    def _states(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [state for state in payload.get("states", []) if isinstance(state, dict)]

    def _find_by_id(self, states: list[dict[str, Any]], state_id: str) -> dict[str, Any] | None:
        for state in states:
            if str(state.get("id") or "") == str(state_id or ""):
                return state
        return None

    def _find_by_identity(
        self,
        states: list[dict[str, Any]],
        *,
        key: str,
        session_id: str,
        scope: str,
        namespace: str,
        task_id: str,
        workflow_id: str,
    ) -> dict[str, Any] | None:
        for state in states:
            if (
                str(state.get("key") or "") == str(key or "")
                and str(state.get("session_id") or "") == str(session_id or "web")
                and str(state.get("scope") or "") == str(scope or "session")
                and str(state.get("namespace") or "") == str(namespace or "default")
                and str(state.get("task_id") or "") == str(task_id or "")
                and str(state.get("workflow_id") or "") == str(workflow_id or "")
            ):
                return state
        return None

    def _append_history(
        self,
        state: dict[str, Any],
        event_type: str,
        *,
        note: str,
        actor: str,
    ) -> dict[str, Any]:
        history_items = state.setdefault("history", [])
        if not isinstance(history_items, list):
            history_items = []
            state["history"] = history_items
        history = {
            "id": f"history-{uuid.uuid4().hex[:10]}",
            "created_at": _utc_now_iso(),
            "type": event_type,
            "version": int(state.get("version") or 1),
            "actor": _clip(actor, 120),
            "note": _clip(note, 800),
        }
        history_items.append(history)
        return history
