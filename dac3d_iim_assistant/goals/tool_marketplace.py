"""Local tool marketplace catalog for DAC-Agent workflows."""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


TOOL_ENTRY_STATUSES = ("available", "installed", "disabled", "archived")


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


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "").strip().lower()).strip("-")
    return normalized[:120] or f"tool-pack-{uuid.uuid4().hex[:8]}"


def _strings(values: list[Any] | tuple[Any, ...] | None, *, limit: int = 60) -> list[str]:
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


def _normalize_status(value: str, *, default: str = "available") -> str:
    status = str(value or default).strip().lower()
    if status not in TOOL_ENTRY_STATUSES:
        raise ValueError(f"Tool marketplace status must be one of: {', '.join(TOOL_ENTRY_STATUSES)}.")
    return status


def _search_text(entry: dict[str, Any]) -> str:
    parts = [
        str(entry.get("id") or ""),
        str(entry.get("slug") or ""),
        str(entry.get("name") or ""),
        str(entry.get("description") or ""),
        str(entry.get("category") or ""),
        str(entry.get("provider") or ""),
        " ".join(str(item) for item in entry.get("tools") or []),
        " ".join(str(item) for item in entry.get("required_context") or []),
        " ".join(str(item) for item in entry.get("prompt_examples") or []),
        " ".join(str(item) for item in entry.get("tags") or []),
    ]
    return " ".join(parts).lower()


class ToolMarketplaceStore:
    """JSON-backed catalog of reusable Agent tool packs."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_root(cls, root_dir: Path) -> "ToolMarketplaceStore":
        return cls(root_dir / "agent_tool_marketplace.json")

    def seed_defaults(self, entries: list[dict[str, Any]]) -> dict[str, Any]:
        created = 0
        updated = 0
        for entry in entries:
            result = self.register_entry(
                name=str(entry.get("name") or ""),
                slug=str(entry.get("slug") or ""),
                description=str(entry.get("description") or ""),
                category=str(entry.get("category") or "agent_tools"),
                provider=str(entry.get("provider") or "dac-agent"),
                status=str(entry.get("status") or "available"),
                tools=entry.get("tools") if isinstance(entry.get("tools"), list) else None,
                required_context=entry.get("required_context")
                if isinstance(entry.get("required_context"), list)
                else None,
                prompt_examples=entry.get("prompt_examples")
                if isinstance(entry.get("prompt_examples"), list)
                else None,
                tags=entry.get("tags") if isinstance(entry.get("tags"), list) else None,
                metadata=entry.get("metadata") if isinstance(entry.get("metadata"), dict) else None,
                owner_agent=str(entry.get("owner_agent") or "system"),
                seed=True,
            )
            if result["created"]:
                created += 1
            else:
                updated += 1
        return {"enabled": True, "backend": "local_tool_marketplace", "created": created, "updated": updated}

    def register_entry(
        self,
        name: str,
        *,
        slug: str = "",
        description: str = "",
        category: str = "agent_tools",
        provider: str = "dac-agent",
        status: str = "available",
        tools: list[Any] | None = None,
        required_context: list[Any] | None = None,
        prompt_examples: list[Any] | None = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        owner_agent: str = "agent",
        seed: bool = False,
    ) -> dict[str, Any]:
        clean_name = _clip(name, 180)
        if not clean_name:
            raise ValueError("Tool marketplace entry name is required.")
        clean_slug = _slugify(slug or clean_name)
        payload = self._load()
        entries = self._entries(payload)
        entry = self._find_by_slug(entries, clean_slug)
        now = _utc_now_iso()
        event_type = "seeded" if seed else "registered"
        if entry is None:
            entry = {
                "id": f"tool-pack-{uuid.uuid4().hex[:12]}",
                "slug": clean_slug,
                "name": clean_name,
                "description": _clip(description, 1000),
                "category": _clip(category, 120) or "agent_tools",
                "provider": _clip(provider, 120) or "dac-agent",
                "status": _normalize_status(status),
                "tools": _strings(tools),
                "required_context": _strings(required_context),
                "prompt_examples": _strings(prompt_examples, limit=20),
                "tags": _strings(tags),
                "metadata": _json_dict(metadata, field_name="Tool marketplace metadata"),
                "owner_agent": _clip(owner_agent, 120) or "agent",
                "created_at": now,
                "updated_at": now,
                "history": [],
            }
            entries.insert(0, entry)
            created = True
        else:
            entry.update(
                {
                    "name": clean_name,
                    "description": _clip(description, 1000),
                    "category": _clip(category, 120) or "agent_tools",
                    "provider": _clip(provider, 120) or "dac-agent",
                    "status": _normalize_status(status),
                    "tools": _strings(tools),
                    "required_context": _strings(required_context),
                    "prompt_examples": _strings(prompt_examples, limit=20),
                    "tags": _strings(tags),
                    "metadata": _json_dict(metadata, field_name="Tool marketplace metadata"),
                    "owner_agent": _clip(owner_agent, 120) or "agent",
                    "updated_at": now,
                }
            )
            created = False
            event_type = "seed_refreshed" if seed else "updated"
        history = self._append_history(entry, event_type, actor=owner_agent or "agent")
        payload["entries"] = entries[:200]
        _write_json(self.path, payload)
        return {"entry": entry, "history": history, "created": created}

    def list_entries(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        provider: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        entries = self._entries(payload)
        if status:
            normalized_status = _normalize_status(status)
            entries = [entry for entry in entries if str(entry.get("status") or "") == normalized_status]
        if category:
            clean_category = _clip(category, 120)
            entries = [entry for entry in entries if str(entry.get("category") or "") == clean_category]
        if provider:
            clean_provider = _clip(provider, 120)
            entries = [entry for entry in entries if str(entry.get("provider") or "") == clean_provider]
        if tag:
            clean_tag = _clip(tag, 220)
            entries = [entry for entry in entries if clean_tag in [str(value) for value in entry.get("tags") or []]]
        if query:
            needle = str(query or "").strip().lower()
            entries = [entry for entry in entries if needle and needle in _search_text(entry)]
        safe_limit = max(1, min(100, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_tool_marketplace",
            "path": str(self.path),
            "entries": entries[:safe_limit],
            "count": len(entries[:safe_limit]),
            "total_count": len(self._entries(payload)),
            "statuses": list(TOOL_ENTRY_STATUSES),
            "workflow": "tool_catalog -> select_tool_pack -> workflow_builder",
        }

    def read_entry(self, entry_id_or_slug: str) -> dict[str, Any]:
        entries = self._entries(self._load())
        entry = self._find_by_id(entries, entry_id_or_slug) or self._find_by_slug(entries, entry_id_or_slug)
        if entry is None:
            raise ValueError(f"Unknown tool marketplace entry: {entry_id_or_slug}")
        return {"enabled": True, "backend": "local_tool_marketplace", "entry": entry}

    def update_status(
        self,
        entry_id_or_slug: str,
        status: str,
        *,
        note: str = "",
        actor: str = "agent",
    ) -> dict[str, Any]:
        payload = self._load()
        entries = self._entries(payload)
        entry = self._find_by_id(entries, entry_id_or_slug) or self._find_by_slug(entries, entry_id_or_slug)
        if entry is None:
            raise ValueError(f"Unknown tool marketplace entry: {entry_id_or_slug}")
        old_status = str(entry.get("status") or "available")
        new_status = _normalize_status(status)
        entry["status"] = new_status
        entry["updated_at"] = _utc_now_iso()
        history = self._append_history(
            entry,
            "status_changed",
            actor=actor,
            note=note or f"{old_status} -> {new_status}",
            extra={"from": old_status, "to": new_status},
        )
        _write_json(self.path, payload)
        return {"entry": entry, "history": history}

    def describe(self) -> dict[str, Any]:
        entries = self._entries(self._load())
        by_status = {status: 0 for status in TOOL_ENTRY_STATUSES}
        by_category: dict[str, int] = {}
        tools: set[str] = set()
        for entry in entries:
            status = str(entry.get("status") or "available")
            by_status[status] = by_status.get(status, 0) + 1
            category = str(entry.get("category") or "agent_tools")
            by_category[category] = by_category.get(category, 0) + 1
            for tool in entry.get("tools") or []:
                tools.add(str(tool))
        return {
            "enabled": True,
            "backend": "local_tool_marketplace",
            "path": str(self.path),
            "entry_count": len(entries),
            "tool_count": len(tools),
            "by_status": by_status,
            "by_category": by_category,
            "latest_entry": entries[0] if entries else {},
            "workflow": "local_tool_marketplace -> low_code_workflow_builder -> agent_runtime",
        }

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.path, {"version": 1, "updated_at": "", "entries": []})
        if not isinstance(payload.get("entries"), list):
            payload["entries"] = []
        return payload

    def _entries(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [entry for entry in payload.get("entries", []) if isinstance(entry, dict)]

    def _find_by_id(self, entries: list[dict[str, Any]], entry_id: str) -> dict[str, Any] | None:
        for entry in entries:
            if str(entry.get("id") or "") == str(entry_id or ""):
                return entry
        return None

    def _find_by_slug(self, entries: list[dict[str, Any]], slug: str) -> dict[str, Any] | None:
        clean_slug = _slugify(slug)
        for entry in entries:
            if str(entry.get("slug") or "") == clean_slug:
                return entry
        return None

    def _append_history(
        self,
        entry: dict[str, Any],
        event_type: str,
        *,
        actor: str,
        note: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        history = {
            "at": _utc_now_iso(),
            "type": event_type,
            "actor": _clip(actor, 120) or "agent",
            "note": _clip(note, 800),
            **(extra or {}),
        }
        events = list(entry.get("history") or [])
        events.append(history)
        entry["history"] = events[-50:]
        return history
