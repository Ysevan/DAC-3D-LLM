"""Append-only JSONL audit tracing with redaction and hash-chain checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from tracing.redaction import redact_value, summarize_for_trace


ZERO_HASH = "0" * 64


@dataclass(slots=True)
class TraceVerificationResult:
    """Result of an audit hash-chain verification pass."""

    valid: bool
    checked: int
    error: str = ""
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checked": self.checked,
            "error": self.error,
            "line": self.line,
        }


class AuditTraceLogger:
    """Write redacted audit events to append-only JSONL with a hash chain."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()

    def append_event(
        self,
        *,
        event_type: str,
        trace_id: str | None,
        request_id: str | None = None,
        session_id: str | None = None,
        actor: dict[str, Any] | None = None,
        tool_call: dict[str, Any] | None = None,
        policy_decision: dict[str, Any] | None = None,
        confirmation: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a single redacted event and return the stored payload."""
        with self._lock:
            previous_hash = self._last_event_hash()
            event = {
                "version": 1,
                "timestamp": _utc_now_iso(),
                "event_type": str(event_type or "event"),
                "trace_id": trace_id,
                "request_id": request_id,
                "session_id": session_id,
                "actor": redact_value(actor or {}),
                "tool_call": summarize_for_trace(tool_call or {}),
                "policy_decision": summarize_for_trace(policy_decision or {}),
                "confirmation": summarize_for_trace(confirmation or {}),
                "payload": summarize_for_trace(payload or {}),
                "previous_hash": previous_hash,
            }
            event["event_hash"] = _event_hash(event)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            return event

    def query_trace(self, trace_id: str) -> list[dict[str, Any]]:
        """Return redacted events for one trace id."""
        return [event for event in self.iter_events() if event.get("trace_id") == trace_id]

    def export_redacted_trace(self, trace_id: str | None = None) -> list[dict[str, Any]]:
        """Return redacted trace events, optionally filtered by trace id."""
        events = self.query_trace(trace_id) if trace_id else self.iter_events()
        return [redact_value(event) for event in events]

    def verify_hash_chain(self) -> TraceVerificationResult:
        """Detect missing, reordered, or tampered JSONL trace events."""
        previous_hash = ZERO_HASH
        checked = 0
        for line_number, event in self._iter_events_with_line_numbers():
            stored_previous = str(event.get("previous_hash") or "")
            if stored_previous != previous_hash:
                return TraceVerificationResult(
                    valid=False,
                    checked=checked,
                    error="PREVIOUS_HASH_MISMATCH",
                    line=line_number,
                )
            stored_hash = str(event.get("event_hash") or "")
            expected_hash = _event_hash(event)
            if stored_hash != expected_hash:
                return TraceVerificationResult(
                    valid=False,
                    checked=checked,
                    error="EVENT_HASH_MISMATCH",
                    line=line_number,
                )
            previous_hash = stored_hash
            checked += 1
        return TraceVerificationResult(valid=True, checked=checked)

    def iter_events(self) -> list[dict[str, Any]]:
        """Read all valid JSON objects from the JSONL file."""
        return [event for _line, event in self._iter_events_with_line_numbers()]

    def _last_event_hash(self) -> str:
        previous_hash = ZERO_HASH
        for _line, event in self._iter_events_with_line_numbers():
            event_hash = str(event.get("event_hash") or "")
            if event_hash:
                previous_hash = event_hash
        return previous_hash

    def _iter_events_with_line_numbers(self) -> list[tuple[int, dict[str, Any]]]:
        if not self.path.exists():
            return []
        events: list[tuple[int, dict[str, Any]]] = []
        for line_number, raw_line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                events.append((line_number, {"previous_hash": "", "event_hash": ""}))
                continue
            if isinstance(event, dict):
                events.append((line_number, event))
        return events


def _event_hash(event: dict[str, Any]) -> str:
    material = {key: value for key, value in event.items() if key != "event_hash"}
    serialized = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
