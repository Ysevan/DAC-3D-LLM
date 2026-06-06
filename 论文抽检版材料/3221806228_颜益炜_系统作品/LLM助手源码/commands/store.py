"""In-memory command lifecycle store for local DAC-Agent sessions."""

from __future__ import annotations

from threading import RLock
from typing import Any

from commands.models import (
    COMMAND_STATE_CANCELLED,
    COMMAND_STATE_CONFIRMED,
    COMMAND_STATE_CONFIRMATION_FAILED,
    COMMAND_STATE_EXPIRED,
    COMMAND_STATE_SUBMITTED,
    CommandLifecycleRecord,
    stable_preview_hash,
    utc_now_iso,
)


class CommandLifecycleStore:
    """Track pending command lifecycle state and consumed confirmation tokens."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: dict[str, CommandLifecycleRecord] = {}
        self._consumed: set[tuple[str, str, str]] = set()

    def create_pending(
        self,
        *,
        session_id: str,
        source_message: str,
        command_preview: dict[str, Any],
        ttl_seconds: int,
    ) -> CommandLifecycleRecord:
        """Create or replace the pending lifecycle record for a session."""
        record = CommandLifecycleRecord.create(
            session_id=session_id,
            source_message=source_message,
            command_preview=command_preview,
            ttl_seconds=ttl_seconds,
        )
        with self._lock:
            self._records[session_id] = record
        return record

    def get(self, session_id: str) -> CommandLifecycleRecord | None:
        """Return one pending lifecycle record."""
        with self._lock:
            return self._records.get(session_id)

    def mark_confirmation_failed(self, session_id: str, reason: str) -> CommandLifecycleRecord | None:
        """Mark confirmation failure for audit."""
        return self._transition(session_id, COMMAND_STATE_CONFIRMATION_FAILED, actor="user", reason=reason)

    def mark_confirmed(self, session_id: str) -> CommandLifecycleRecord | None:
        """Mark a lifecycle as explicitly confirmed by the user."""
        record = self._transition(session_id, COMMAND_STATE_CONFIRMED, actor="user", reason="confirmation_accepted")
        if record is not None:
            record.confirmed_at = utc_now_iso()
        return record

    def mark_submitted(self, session_id: str) -> CommandLifecycleRecord | None:
        """Mark a lifecycle as submitted by Tool Gateway."""
        record = self._transition(session_id, COMMAND_STATE_SUBMITTED, actor="tool_gateway", reason="submitted")
        if record is not None:
            record.submitted_at = utc_now_iso()
        return record

    def mark_expired(self, session_id: str) -> CommandLifecycleRecord | None:
        """Mark a pending lifecycle as expired."""
        return self._transition(session_id, COMMAND_STATE_EXPIRED, actor="tool_gateway", reason="confirmation_ttl_expired")

    def mark_cancelled(self, session_id: str) -> CommandLifecycleRecord | None:
        """Mark a pending lifecycle as cancelled."""
        return self._transition(session_id, COMMAND_STATE_CANCELLED, actor="user", reason="cancelled")

    def consume(self, session_id: str, preview_id: str, token: str) -> CommandLifecycleRecord | None:
        """Mark a confirmation token as consumed for replay protection."""
        key = self._key(session_id, preview_id, token)
        with self._lock:
            self._consumed.add(key)
            record = self._records.get(session_id)
            if record is not None:
                record.consumed_at = utc_now_iso()
                record.add_event(COMMAND_STATE_SUBMITTED, actor="tool_gateway", reason="confirmation_token_consumed")
            return record

    def is_consumed(self, session_id: str, preview_id: str, token: str) -> bool:
        """Return whether a confirmation token was already consumed."""
        with self._lock:
            return self._key(session_id, preview_id, token) in self._consumed

    def clear(self, session_id: str) -> CommandLifecycleRecord | None:
        """Remove a pending record from active pending storage."""
        with self._lock:
            return self._records.pop(session_id, None)

    def verify_preview_hash(self, session_id: str, preview: dict[str, Any]) -> bool:
        """Return whether current preview content matches the stored hash."""
        with self._lock:
            record = self._records.get(session_id)
        return record is not None and stable_preview_hash(preview) == record.preview_hash

    def _transition(
        self,
        session_id: str,
        state: str,
        *,
        actor: str,
        reason: str,
    ) -> CommandLifecycleRecord | None:
        with self._lock:
            record = self._records.get(session_id)
            if record is not None:
                record.add_event(state, actor=actor, reason=reason)
            return record

    def _key(self, session_id: str, preview_id: str, token: str) -> tuple[str, str, str]:
        return (str(session_id or ""), str(preview_id or ""), str(token or ""))
