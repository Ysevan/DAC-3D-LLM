"""Local data-labeling queue for DAC-Agent LLMOps workflows."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


LABELING_STATUSES = ("pending", "labeled", "skipped", "exported", "archived")
DEFAULT_LABEL_SCHEMA = {
    "answer_quality": ["good", "partial", "bad"],
    "intent_correct": [True, False],
    "tool_call_correct": [True, False, "not_applicable"],
    "memory_action_correct": [True, False, "not_applicable"],
}


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


def _json_list(value: list[Any] | None, *, field_name: str) -> list[Any]:
    try:
        payload = json.loads(json.dumps(list(value or []), ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable.") from exc
    return payload if isinstance(payload, list) else []


def _normalize_status(value: str, *, default: str = "pending") -> str:
    status = str(value or default).strip().lower()
    if status not in LABELING_STATUSES:
        raise ValueError(f"Labeling status must be one of: {', '.join(LABELING_STATUSES)}.")
    return status


def _bounded_score(value: Any | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Label score must be an integer from 0 to 5.") from exc
    if number < 0 or number > 5:
        raise ValueError("Label score must be an integer from 0 to 5.")
    return number


def _tool_names(tool_calls: list[Any]) -> list[str]:
    names: list[str] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            name = _clip(tool_call.get("name") or tool_call.get("tool") or tool_call.get("tool_name"), 120)
        else:
            name = _clip(tool_call, 120)
        if name and name not in names:
            names.append(name)
    return names


def _search_text(item: dict[str, Any]) -> str:
    annotations = " ".join(
        " ".join(
            [
                str(annotation.get("labeler") or ""),
                str(annotation.get("outcome") or ""),
                str(annotation.get("comment") or ""),
                json.dumps(annotation.get("labels") or {}, ensure_ascii=False, sort_keys=True),
            ]
        )
        for annotation in item.get("annotations") or []
    )
    parts = [
        str(item.get("id") or ""),
        str(item.get("title") or ""),
        str(item.get("source_type") or ""),
        str(item.get("source_id") or ""),
        str(item.get("session_id") or ""),
        str(item.get("intent") or ""),
        str(item.get("input_text") or ""),
        str(item.get("agent_output") or ""),
        " ".join(str(value) for value in item.get("tool_names") or []),
        " ".join(str(value) for value in item.get("tags") or []),
        annotations,
    ]
    return " ".join(parts).lower()


class AgentLabelingStore:
    """JSON-backed queue for human-labeled Agent trace samples."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_root(cls, root_dir: Path) -> "AgentLabelingStore":
        return cls(root_dir / "agent_labeling.json")

    def create_item(
        self,
        title: str,
        *,
        source_type: str = "manual",
        source_id: str = "",
        session_id: str = "",
        input_text: str = "",
        agent_output: str = "",
        intent: str = "",
        tool_calls: list[Any] | None = None,
        expected: dict[str, Any] | None = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        created_by: str = "agent",
    ) -> dict[str, Any]:
        clean_title = _clip(title, 220)
        if not clean_title:
            raise ValueError("Labeling item title is required.")
        payload = self._load()
        items = self._items(payload)
        existing = self._find_by_source(items, source_type=source_type, source_id=source_id)
        now = _utc_now_iso()
        clean_tool_calls = _json_list(tool_calls, field_name="Labeling tool_calls")
        if existing is None:
            item = {
                "id": f"label-{uuid.uuid4().hex[:12]}",
                "title": clean_title,
                "source_type": _clip(source_type or "manual", 80) or "manual",
                "source_id": _clip(source_id, 220),
                "session_id": _clip(session_id, 220),
                "input_text": _clip(input_text, 3000),
                "agent_output": _clip(agent_output, 3000),
                "intent": _clip(intent, 120),
                "tool_calls": clean_tool_calls,
                "tool_names": _tool_names(clean_tool_calls),
                "expected": _json_dict(expected, field_name="Labeling expected"),
                "tags": _strings(tags),
                "metadata": _json_dict(metadata, field_name="Labeling metadata"),
                "status": "pending",
                "created_by": _clip(created_by, 120) or "agent",
                "created_at": now,
                "updated_at": now,
                "annotations": [],
                "history": [],
            }
            items.insert(0, item)
            created = True
            event_type = "created"
        else:
            item = existing
            item.update(
                {
                    "title": clean_title,
                    "session_id": _clip(session_id, 220),
                    "input_text": _clip(input_text, 3000),
                    "agent_output": _clip(agent_output, 3000),
                    "intent": _clip(intent, 120),
                    "tool_calls": clean_tool_calls,
                    "tool_names": _tool_names(clean_tool_calls),
                    "expected": _json_dict(expected, field_name="Labeling expected"),
                    "tags": _strings(tags),
                    "metadata": _json_dict(metadata, field_name="Labeling metadata"),
                    "updated_at": now,
                }
            )
            created = False
            event_type = "updated"
        history = self._append_history(item, event_type, actor=created_by or "agent")
        payload["items"] = items[:500]
        _write_json(self.path, payload)
        return {"item": item, "history": history, "created": created}

    def create_from_trace(
        self,
        trace: dict[str, Any],
        *,
        title: str = "",
        tags: list[Any] | None = None,
        created_by: str = "agent",
    ) -> dict[str, Any]:
        trace_id = _clip(trace.get("trace_id"), 220)
        if not trace_id:
            raise ValueError("Trace must include trace_id.")
        trace_tags = ["trace", str(trace.get("intent") or "unknown")]
        trace_tags.extend(_strings(tags))
        return self.create_item(
            title or f"Trace {trace_id}",
            source_type="trace",
            source_id=trace_id,
            session_id=str(trace.get("session_id") or ""),
            input_text=str(trace.get("user_message") or ""),
            agent_output=str(trace.get("final_response") or trace.get("assistant_answer") or ""),
            intent=str(trace.get("intent") or ""),
            tool_calls=trace.get("tool_calls") if isinstance(trace.get("tool_calls"), list) else None,
            expected={
                "intent": str(trace.get("intent") or ""),
                "final_response": str(trace.get("final_response") or ""),
            },
            tags=trace_tags,
            metadata={"source_timestamp": trace.get("timestamp"), "runtime": trace.get("runtime")},
            created_by=created_by,
        )

    def list_items(
        self,
        *,
        status: str | None = None,
        source_type: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        items = self._items(payload)
        if status:
            normalized_status = _normalize_status(status)
            items = [item for item in items if str(item.get("status") or "") == normalized_status]
        if source_type:
            clean_source_type = _clip(source_type, 80)
            items = [item for item in items if str(item.get("source_type") or "") == clean_source_type]
        if tag:
            clean_tag = _clip(tag, 220)
            items = [item for item in items if clean_tag in [str(value) for value in item.get("tags") or []]]
        if query:
            needle = str(query or "").strip().lower()
            items = [item for item in items if needle and needle in _search_text(item)]
        safe_limit = max(1, min(200, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_agent_labeling_queue",
            "path": str(self.path),
            "items": [self._item_summary(item) for item in items[:safe_limit]],
            "count": len(items[:safe_limit]),
            "total_count": len(self._items(payload)),
            "statuses": list(LABELING_STATUSES),
            "label_schema": DEFAULT_LABEL_SCHEMA,
            "workflow": "trace_sample -> labeling_queue -> human_label -> eval_dataset",
        }

    def read_item(self, item_id: str) -> dict[str, Any]:
        item = self._find_item(item_id)
        if item is None:
            raise ValueError(f"Unknown labeling item: {item_id}")
        return {"enabled": True, "backend": "local_agent_labeling_queue", "item": item}

    def label_item(
        self,
        item_id: str,
        *,
        labels: dict[str, Any],
        outcome: str = "accepted",
        score: int | None = None,
        comment: str = "",
        labeler: str = "human",
    ) -> dict[str, Any]:
        payload = self._load()
        item = self._find_in_items(self._items(payload), item_id)
        if item is None:
            raise ValueError(f"Unknown labeling item: {item_id}")
        clean_labels = _json_dict(labels, field_name="Labels")
        if not clean_labels:
            raise ValueError("Labels are required.")
        now = _utc_now_iso()
        annotation = {
            "id": f"annotation-{uuid.uuid4().hex[:12]}",
            "labels": clean_labels,
            "outcome": _clip(outcome or "accepted", 120) or "accepted",
            "score": _bounded_score(score),
            "comment": _clip(comment, 1200),
            "labeler": _clip(labeler, 120) or "human",
            "created_at": now,
        }
        annotations = list(item.get("annotations") or [])
        annotations.insert(0, annotation)
        item["annotations"] = annotations[:100]
        item["status"] = "labeled"
        item["updated_at"] = now
        history = self._append_history(
            item,
            "labeled",
            actor=labeler or "human",
            note=comment or "Label added.",
            extra={"annotation_id": annotation["id"]},
        )
        _write_json(self.path, payload)
        return {"item": item, "annotation": annotation, "history": history}

    def update_status(
        self,
        item_id: str,
        status: str,
        *,
        note: str = "",
        actor: str = "agent",
    ) -> dict[str, Any]:
        payload = self._load()
        item = self._find_in_items(self._items(payload), item_id)
        if item is None:
            raise ValueError(f"Unknown labeling item: {item_id}")
        old_status = str(item.get("status") or "pending")
        new_status = _normalize_status(status)
        item["status"] = new_status
        item["updated_at"] = _utc_now_iso()
        history = self._append_history(
            item,
            "status_changed",
            actor=actor,
            note=note or f"{old_status} -> {new_status}",
            extra={"from": old_status, "to": new_status},
        )
        _write_json(self.path, payload)
        return {"item": item, "history": history}

    def export_items(
        self,
        *,
        status: str = "labeled",
        limit: int = 200,
        mark_exported: bool = False,
    ) -> dict[str, Any]:
        payload = self._load()
        normalized_status = _normalize_status(status)
        safe_limit = max(1, min(500, int(limit or 200)))
        selected = [
            item
            for item in self._items(payload)
            if str(item.get("status") or "") == normalized_status
        ][:safe_limit]
        records = [self._export_record(item) for item in selected]
        exported_at = _utc_now_iso()
        if mark_exported:
            for item in selected:
                item["status"] = "exported"
                item["exported_at"] = exported_at
                item["updated_at"] = exported_at
                self._append_history(item, "exported", actor="labeling_export")
            _write_json(self.path, payload)
        return {
            "enabled": True,
            "backend": "local_agent_labeling_queue",
            "status": normalized_status,
            "count": len(records),
            "records": records,
            "jsonl": "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
            "marked_exported": mark_exported,
            "workflow": "labeled_items -> jsonl_dataset -> eval_review",
        }

    def describe(self) -> dict[str, Any]:
        items = self._items(self._load())
        by_status = {status: 0 for status in LABELING_STATUSES}
        by_source_type: dict[str, int] = {}
        annotation_count = 0
        for item in items:
            status = str(item.get("status") or "pending")
            by_status[status] = by_status.get(status, 0) + 1
            source_type = str(item.get("source_type") or "manual")
            by_source_type[source_type] = by_source_type.get(source_type, 0) + 1
            annotation_count += len(item.get("annotations") or [])
        return {
            "enabled": True,
            "backend": "local_agent_labeling_queue",
            "path": str(self.path),
            "item_count": len(items),
            "annotation_count": annotation_count,
            "by_status": by_status,
            "by_source_type": by_source_type,
            "latest_item": self._item_summary(items[0]) if items else {},
            "label_schema": DEFAULT_LABEL_SCHEMA,
            "workflow": "agent_traces -> labeling_queue -> eval_dataset_feedback",
        }

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.path, {"version": 1, "updated_at": "", "items": []})
        if not isinstance(payload.get("items"), list):
            payload["items"] = []
        return payload

    def _items(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [item for item in payload.get("items", []) if isinstance(item, dict)]

    def _find_item(self, item_id: str) -> dict[str, Any] | None:
        return self._find_in_items(self._items(self._load()), item_id)

    def _find_in_items(self, items: list[dict[str, Any]], item_id: str) -> dict[str, Any] | None:
        for item in items:
            if str(item.get("id") or "") == str(item_id or ""):
                return item
        return None

    def _find_by_source(
        self,
        items: list[dict[str, Any]],
        *,
        source_type: str,
        source_id: str,
    ) -> dict[str, Any] | None:
        if not source_id:
            return None
        clean_source_type = _clip(source_type or "manual", 80) or "manual"
        clean_source_id = _clip(source_id, 220)
        for item in items:
            if (
                str(item.get("source_type") or "") == clean_source_type
                and str(item.get("source_id") or "") == clean_source_id
            ):
                return item
        return None

    def _append_history(
        self,
        item: dict[str, Any],
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
        events = list(item.get("history") or [])
        events.append(history)
        item["history"] = events[-80:]
        return history

    def _item_summary(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": item.get("id"),
            "title": item.get("title"),
            "source_type": item.get("source_type"),
            "source_id": item.get("source_id"),
            "session_id": item.get("session_id"),
            "intent": item.get("intent"),
            "tool_names": list(item.get("tool_names") or []),
            "tags": list(item.get("tags") or []),
            "status": item.get("status"),
            "annotation_count": len(item.get("annotations") or []),
            "latest_annotation": (item.get("annotations") or [{}])[0] if item.get("annotations") else {},
            "updated_at": item.get("updated_at"),
        }

    def _export_record(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": item.get("id"),
            "source_type": item.get("source_type"),
            "source_id": item.get("source_id"),
            "session_id": item.get("session_id"),
            "input": item.get("input_text"),
            "output": item.get("agent_output"),
            "intent": item.get("intent"),
            "tool_names": list(item.get("tool_names") or []),
            "expected": dict(item.get("expected") or {}),
            "annotations": list(item.get("annotations") or []),
            "tags": list(item.get("tags") or []),
        }
