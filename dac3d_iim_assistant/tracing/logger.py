"""Append-only JSONL audit tracing with redaction and hash-chain checks."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from tracing.redaction import redact_value, summarize_for_trace


ZERO_HASH = "0" * 64
SIGNATURE_ALGORITHM = "hmac-sha256"
_INTEGRITY_FIELDS = {
    "event_hash",
    "event_signature",
    "signature_algorithm",
    "signature_key_id",
}
DEFAULT_TRACE_QUERY_LIMIT = 100
MAX_TRACE_QUERY_LIMIT = 500


@dataclass(slots=True)
class TraceVerificationResult:
    """Result of an audit hash-chain verification pass."""

    valid: bool
    checked: int
    error: str = ""
    line: int | None = None
    signature_checked: int = 0
    signature_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checked": self.checked,
            "error": self.error,
            "line": self.line,
            "signature_checked": self.signature_checked,
            "signature_required": self.signature_required,
        }


@dataclass(slots=True)
class TraceQueryResult:
    """Paginated redacted audit trace query result."""

    events: list[dict[str, Any]]
    total: int
    offset: int
    limit: int
    trace_id: str | None = None

    @property
    def next_offset(self) -> int | None:
        next_value = self.offset + len(self.events)
        return next_value if next_value < self.total else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "offset": self.offset,
            "limit": self.limit,
            "returned": len(self.events),
            "total": self.total,
            "next_offset": self.next_offset,
            "events": self.events,
        }


class AuditTraceLogger:
    """Write redacted audit events to append-only JSONL with a hash chain."""

    def __init__(
        self,
        path: Path,
        *,
        signing_key: str | bytes = "",
        signing_key_id: str = "local-audit-key",
    ) -> None:
        self.path = path
        self._lock = RLock()
        self._signing_key = _normalize_signing_key(signing_key)
        self.signing_key_id = signing_key_id.strip() or "local-audit-key"

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
            if self._signing_key:
                event["signature_algorithm"] = SIGNATURE_ALGORITHM
                event["signature_key_id"] = self.signing_key_id
                event["event_signature"] = _event_signature(
                    event_hash=event["event_hash"],
                    key=self._signing_key,
                    key_id=self.signing_key_id,
                )
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

    def query_redacted_events(
        self,
        *,
        trace_id: str | None = None,
        offset: int = 0,
        limit: int = DEFAULT_TRACE_QUERY_LIMIT,
    ) -> TraceQueryResult:
        """Return a bounded, redacted page of audit events."""
        safe_offset = max(int(offset), 0)
        safe_limit = min(max(int(limit), 1), MAX_TRACE_QUERY_LIMIT)
        events: list[dict[str, Any]] = []
        total = 0
        for event in self.iter_events():
            if trace_id is not None and event.get("trace_id") != trace_id:
                continue
            total += 1
            if total <= safe_offset:
                continue
            if len(events) >= safe_limit:
                continue
            events.append(redact_value(event))
        return TraceQueryResult(
            events=events,
            total=total,
            offset=safe_offset,
            limit=safe_limit,
            trace_id=trace_id,
        )

    def verify_hash_chain(self) -> TraceVerificationResult:
        """Detect missing, reordered, or tampered JSONL trace events."""
        previous_hash = ZERO_HASH
        checked = 0
        signature_checked = 0
        for line_number, event in self._iter_events_with_line_numbers():
            stored_previous = str(event.get("previous_hash") or "")
            if stored_previous != previous_hash:
                return TraceVerificationResult(
                    valid=False,
                    checked=checked,
                    error="PREVIOUS_HASH_MISMATCH",
                    line=line_number,
                    signature_checked=signature_checked,
                    signature_required=bool(self._signing_key),
                )
            stored_hash = str(event.get("event_hash") or "")
            expected_hash = _event_hash(event)
            if stored_hash != expected_hash:
                return TraceVerificationResult(
                    valid=False,
                    checked=checked,
                    error="EVENT_HASH_MISMATCH",
                    line=line_number,
                    signature_checked=signature_checked,
                    signature_required=bool(self._signing_key),
                )
            if self._signing_key:
                signature_result = self._verify_event_signature(event, stored_hash)
                if signature_result:
                    return TraceVerificationResult(
                        valid=False,
                        checked=checked,
                        error=signature_result,
                        line=line_number,
                        signature_checked=signature_checked,
                        signature_required=True,
                    )
                signature_checked += 1
            previous_hash = stored_hash
            checked += 1
        return TraceVerificationResult(
            valid=True,
            checked=checked,
            signature_checked=signature_checked,
            signature_required=bool(self._signing_key),
        )

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

    def _verify_event_signature(self, event: dict[str, Any], event_hash: str) -> str:
        stored_signature = str(event.get("event_signature") or "")
        if not stored_signature:
            return "EVENT_SIGNATURE_MISSING"
        if str(event.get("signature_algorithm") or "") != SIGNATURE_ALGORITHM:
            return "EVENT_SIGNATURE_ALGORITHM_MISMATCH"
        key_id = str(event.get("signature_key_id") or "")
        if key_id != self.signing_key_id:
            return "EVENT_SIGNATURE_KEY_ID_MISMATCH"
        expected_signature = _event_signature(
            event_hash=event_hash,
            key=self._signing_key,
            key_id=key_id,
        )
        if not hmac.compare_digest(stored_signature, expected_signature):
            return "EVENT_SIGNATURE_MISMATCH"
        return ""

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
    material = {key: value for key, value in event.items() if key not in _INTEGRITY_FIELDS}
    serialized = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _event_signature(*, event_hash: str, key: bytes, key_id: str) -> str:
    material = json.dumps(
        {
            "algorithm": SIGNATURE_ALGORITHM,
            "event_hash": event_hash,
            "key_id": key_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hmac.new(key, material.encode("utf-8"), hashlib.sha256).hexdigest()


def _normalize_signing_key(signing_key: str | bytes) -> bytes:
    if isinstance(signing_key, bytes):
        return signing_key
    return signing_key.strip().encode("utf-8")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
