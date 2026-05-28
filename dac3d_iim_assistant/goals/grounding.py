"""Local grounding evidence store for DAC-Agent research workflows."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


GROUNDING_STATUSES = ("active", "stale", "archived")
GROUNDING_SOURCE_TYPES = ("web", "search", "document", "manual", "tool")


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


def _normalize_status(value: str, *, default: str = "active") -> str:
    status = str(value or default).strip().lower()
    if status not in GROUNDING_STATUSES:
        raise ValueError(f"Grounding status must be one of: {', '.join(GROUNDING_STATUSES)}.")
    return status


def _normalize_source_type(value: str, *, default: str = "manual") -> str:
    source_type = str(value or default).strip().lower()
    if source_type not in GROUNDING_SOURCE_TYPES:
        raise ValueError(
            f"Grounding source_type must be one of: {', '.join(GROUNDING_SOURCE_TYPES)}."
        )
    return source_type


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Grounding confidence must be a number from 0 to 1.") from exc
    if number < 0 or number > 1:
        raise ValueError("Grounding confidence must be a number from 0 to 1.")
    return round(number, 4)


def _search_text(source: dict[str, Any]) -> str:
    parts = [
        str(source.get("id") or ""),
        str(source.get("query") or ""),
        str(source.get("title") or ""),
        str(source.get("url") or ""),
        str(source.get("source_type") or ""),
        str(source.get("snippet") or ""),
        str(source.get("summary") or ""),
        " ".join(str(item) for item in source.get("citations") or []),
        " ".join(str(item) for item in source.get("tags") or []),
    ]
    return " ".join(parts).lower()


class AgentGroundingStore:
    """JSON-backed store for search/web/document grounding evidence."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_root(cls, root_dir: Path) -> "AgentGroundingStore":
        return cls(root_dir / "agent_grounding.json")

    def record_source(
        self,
        title: str,
        *,
        query: str,
        source_type: str = "manual",
        url: str = "",
        snippet: str = "",
        summary: str = "",
        citations: list[Any] | None = None,
        confidence: Any = 0.5,
        trace_id: str = "",
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        recorded_by: str = "agent",
    ) -> dict[str, Any]:
        clean_title = _clip(title, 220)
        clean_query = _clip(query, 500)
        if not clean_title:
            raise ValueError("Grounding title is required.")
        if not clean_query:
            raise ValueError("Grounding query is required.")
        payload = self._load()
        sources = self._sources(payload)
        now = _utc_now_iso()
        source = {
            "id": f"grounding-{uuid.uuid4().hex[:12]}",
            "query": clean_query,
            "title": clean_title,
            "source_type": _normalize_source_type(source_type),
            "url": _clip(url, 1000),
            "snippet": _clip(snippet, 2000),
            "summary": _clip(summary, 2000),
            "citations": _strings(citations),
            "confidence": _confidence(confidence),
            "trace_id": _clip(trace_id, 220),
            "tags": _strings(tags),
            "metadata": _json_dict(metadata, field_name="Grounding metadata"),
            "status": "active",
            "recorded_by": _clip(recorded_by, 120) or "agent",
            "created_at": now,
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
        sources.insert(0, source)
        payload["sources"] = sources[:1000]
        _write_json(self.path, payload)
        return {"source": source, "created": True}

    def list_sources(
        self,
        *,
        status: str | None = None,
        source_type: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        min_confidence: Any | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        sources = self._sources(payload)
        if status:
            normalized_status = _normalize_status(status)
            sources = [source for source in sources if str(source.get("status") or "") == normalized_status]
        if source_type:
            normalized_source_type = _normalize_source_type(source_type)
            sources = [
                source
                for source in sources
                if str(source.get("source_type") or "") == normalized_source_type
            ]
        if tag:
            clean_tag = _clip(tag, 220)
            sources = [
                source for source in sources if clean_tag in [str(value) for value in source.get("tags") or []]
            ]
        if min_confidence is not None and min_confidence != "":
            threshold = _confidence(min_confidence)
            sources = [source for source in sources if float(source.get("confidence") or 0) >= threshold]
        if query:
            needle = str(query or "").strip().lower()
            sources = [source for source in sources if needle and needle in _search_text(source)]
        safe_limit = max(1, min(200, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_agent_grounding_store",
            "path": str(self.path),
            "sources": [self._source_summary(source) for source in sources[:safe_limit]],
            "count": len(sources[:safe_limit]),
            "total_count": len(self._sources(payload)),
            "statuses": list(GROUNDING_STATUSES),
            "source_types": list(GROUNDING_SOURCE_TYPES),
            "workflow": "search_or_fetch -> record_grounding -> context_bundle",
        }

    def read_source(self, source_id: str) -> dict[str, Any]:
        source = self._find_source(source_id)
        if source is None:
            raise ValueError(f"Unknown grounding source: {source_id}")
        return {"enabled": True, "backend": "local_agent_grounding_store", "source": source}

    def update_status(
        self,
        source_id: str,
        status: str,
        *,
        note: str = "",
        actor: str = "agent",
    ) -> dict[str, Any]:
        payload = self._load()
        source = self._find_in_sources(self._sources(payload), source_id)
        if source is None:
            raise ValueError(f"Unknown grounding source: {source_id}")
        old_status = str(source.get("status") or "active")
        new_status = _normalize_status(status)
        source["status"] = new_status
        source["updated_at"] = _utc_now_iso()
        history = {
            "at": source["updated_at"],
            "type": "status_changed",
            "actor": _clip(actor, 120) or "agent",
            "note": _clip(note or f"{old_status} -> {new_status}", 800),
            "from": old_status,
            "to": new_status,
        }
        events = list(source.get("history") or [])
        events.append(history)
        source["history"] = events[-50:]
        _write_json(self.path, payload)
        return {"source": source, "history": history}

    def context_bundle(
        self,
        *,
        query: str = "",
        tag: str | None = None,
        min_confidence: Any = 0,
        limit: int = 8,
    ) -> dict[str, Any]:
        listed = self.list_sources(
            status="active",
            tag=tag,
            query=query,
            min_confidence=min_confidence,
            limit=limit,
        )
        sources = [self._find_source(str(item.get("id") or "")) for item in listed["sources"]]
        full_sources = [source for source in sources if isinstance(source, dict)]
        citations = []
        for index, source in enumerate(full_sources, start=1):
            citations.append(
                {
                    "index": index,
                    "title": source.get("title"),
                    "url": source.get("url"),
                    "source_type": source.get("source_type"),
                    "confidence": source.get("confidence"),
                    "summary": source.get("summary") or source.get("snippet"),
                    "citations": list(source.get("citations") or []),
                }
            )
        return {
            "enabled": True,
            "backend": "local_agent_grounding_store",
            "query": query,
            "count": len(citations),
            "citations": citations,
            "context": "\n".join(
                f"[{item['index']}] {item['title']}: {item['summary']}" for item in citations
            ),
            "workflow": "grounding_sources -> cited_context_bundle -> context_builder",
        }

    def describe(self) -> dict[str, Any]:
        sources = self._sources(self._load())
        by_status = {status: 0 for status in GROUNDING_STATUSES}
        by_source_type: dict[str, int] = {}
        confidence_total = 0.0
        for source in sources:
            status = str(source.get("status") or "active")
            by_status[status] = by_status.get(status, 0) + 1
            source_type = str(source.get("source_type") or "manual")
            by_source_type[source_type] = by_source_type.get(source_type, 0) + 1
            confidence_total += float(source.get("confidence") or 0)
        return {
            "enabled": True,
            "backend": "local_agent_grounding_store",
            "path": str(self.path),
            "source_count": len(sources),
            "active_source_count": by_status.get("active", 0),
            "by_status": by_status,
            "by_source_type": by_source_type,
            "average_confidence": round(confidence_total / len(sources), 4) if sources else 0,
            "latest_source": self._source_summary(sources[0]) if sources else {},
            "workflow": "web_fetch_or_search -> grounding_store -> cited_context",
        }

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.path, {"version": 1, "updated_at": "", "sources": []})
        if not isinstance(payload.get("sources"), list):
            payload["sources"] = []
        return payload

    def _sources(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [source for source in payload.get("sources", []) if isinstance(source, dict)]

    def _find_source(self, source_id: str) -> dict[str, Any] | None:
        return self._find_in_sources(self._sources(self._load()), source_id)

    def _find_in_sources(self, sources: list[dict[str, Any]], source_id: str) -> dict[str, Any] | None:
        for source in sources:
            if str(source.get("id") or "") == str(source_id or ""):
                return source
        return None

    def _source_summary(self, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": source.get("id"),
            "query": source.get("query"),
            "title": source.get("title"),
            "source_type": source.get("source_type"),
            "url": source.get("url"),
            "snippet": source.get("snippet"),
            "summary": source.get("summary"),
            "confidence": source.get("confidence"),
            "trace_id": source.get("trace_id"),
            "tags": list(source.get("tags") or []),
            "status": source.get("status"),
            "updated_at": source.get("updated_at"),
        }
