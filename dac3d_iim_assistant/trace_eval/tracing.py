"""Append-only trace logging for DAC-Agent Runtime."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SENSITIVE_KEY_PATTERN = re.compile(r"(api[_-]?key|authorization|password|secret|token)", re.IGNORECASE)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clip(value: Any, limit: int = 2000) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


class TraceLogger:
    """Append-only JSONL trace logger with lightweight secret redaction."""

    def __init__(self, trace_path: str | Path) -> None:
        self.trace_path = Path(trace_path)

    def append(self, trace: dict[str, Any]) -> dict[str, Any]:
        """Append one normalized Agent trace and return the persisted record."""
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = self._normalize(trace)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def get(self, trace_id: str) -> dict[str, Any] | None:
        """Return one trace by id."""
        wanted = str(trace_id or "")
        for record in self.iter_recent(limit=10_000):
            if str(record.get("trace_id") or "") == wanted:
                return record
        return None

    def iter_recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Read recent traces from JSONL."""
        safe_limit = max(1, min(10_000, int(limit or 20)))
        if not self.trace_path.exists():
            return []
        lines = self.trace_path.read_text(encoding="utf-8").splitlines()[-safe_limit:]
        records: list[dict[str, Any]] = []
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records

    def describe(self) -> dict[str, Any]:
        """Return trace logger diagnostics."""
        count = 0
        if self.trace_path.exists():
            count = sum(1 for _line in self.trace_path.open(encoding="utf-8"))
        recent = self.iter_recent(limit=1)
        return {
            "enabled": True,
            "backend": "jsonl_append_only",
            "path": str(self.trace_path),
            "trace_count": count,
            "latest_trace_id": recent[-1].get("trace_id") if recent else None,
            "redaction": "keys matching api_key/authorization/password/secret/token are redacted",
        }

    def _normalize(self, trace: dict[str, Any]) -> dict[str, Any]:
        record = self._redact(dict(trace))
        record.setdefault("trace_id", f"agent-trace-{uuid.uuid4().hex[:16]}")
        record.setdefault("timestamp", _utc_now_iso())
        record.setdefault("runtime", "dac-agent-runtime")
        record["session_id"] = str(record.get("session_id") or "default")
        record["user_message"] = _clip(record.get("user_message"), 2000)
        record["final_response"] = _clip(record.get("final_response") or record.get("assistant_answer"), 2000)
        record.setdefault("operator_id", None)
        record.setdefault("skill", None)
        record.setdefault("context_ids", [])
        record.setdefault("tool_calls", [])
        record.setdefault("command_preview_id", None)
        record.setdefault("risk_decision", {})
        record.setdefault("confirmation", {})
        record.setdefault("memory_patch_ids", [])
        return record

    def _redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                if SENSITIVE_KEY_PATTERN.search(str(key)):
                    redacted[str(key)] = "[REDACTED]"
                else:
                    redacted[str(key)] = self._redact(item)
            return redacted
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        return value
