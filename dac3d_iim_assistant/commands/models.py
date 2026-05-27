"""Typed command lifecycle state for DAC command previews."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


COMMAND_STATE_DRAFT = "draft"
COMMAND_STATE_PREVIEW_CREATED = "preview_created"
COMMAND_STATE_VALIDATION_PASSED = "validation_passed"
COMMAND_STATE_AWAITING_CONFIRMATION = "awaiting_confirmation"
COMMAND_STATE_CONFIRMED = "confirmed"
COMMAND_STATE_SUBMITTED = "submitted"
COMMAND_STATE_OBSERVED = "observed"
COMMAND_STATE_COMPLETED = "completed"

COMMAND_STATE_REJECTED = "rejected"
COMMAND_STATE_VALIDATION_FAILED = "validation_failed"
COMMAND_STATE_CONFIRMATION_FAILED = "confirmation_failed"
COMMAND_STATE_EXPIRED = "expired"
COMMAND_STATE_CANCELLED = "cancelled"


def utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return current UTC time as compact ISO string."""
    return utc_now().replace(microsecond=0).isoformat()


def parse_iso_datetime(value: str) -> datetime | None:
    """Parse an ISO timestamp, returning None on invalid values."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def stable_preview_hash(preview: dict[str, Any]) -> str:
    """Hash a command preview while excluding gateway/lifecycle metadata."""
    stable = dict(preview or {})
    stable.pop("gateway", None)
    normalized = json.dumps(stable, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def lifecycle_token(session_id: str, preview_id: str, preview_hash: str, nonce: str) -> str:
    """Create a confirmation token bound to one lifecycle record."""
    digest = hashlib.sha256(
        f"dac3d:{session_id}:{preview_id}:{preview_hash}:{nonce}".encode("utf-8")
    ).hexdigest()[:24]
    return f"confirm-{digest}"


@dataclass(slots=True)
class CommandLifecycleRecord:
    """Auditable lifecycle record for one pending command preview."""

    session_id: str
    source_message: str
    preview_id: str
    preview_hash: str
    action: str
    confirmation_token: str
    created_at: str
    expires_at: str
    ttl_seconds: int
    state: str = COMMAND_STATE_AWAITING_CONFIRMATION
    nonce: str = ""
    confirmed_at: str = ""
    submitted_at: str = ""
    consumed_at: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        source_message: str,
        command_preview: dict[str, Any],
        ttl_seconds: int,
    ) -> "CommandLifecycleRecord":
        """Create a lifecycle record for a validated preview awaiting confirmation."""
        safe_ttl = max(0, int(ttl_seconds))
        now = utc_now().replace(microsecond=0)
        expires_at = now + timedelta(seconds=safe_ttl)
        preview_hash = stable_preview_hash(command_preview)
        gateway = command_preview.get("gateway") if isinstance(command_preview.get("gateway"), dict) else {}
        preview_id = str(gateway.get("preview_id") or f"preview-{preview_hash[:16]}")
        nonce = uuid.uuid4().hex[:16]
        token = lifecycle_token(session_id, preview_id, preview_hash, nonce)
        record = cls(
            session_id=session_id,
            source_message=source_message,
            preview_id=preview_id,
            preview_hash=preview_hash,
            action=str(command_preview.get("action") or "unknown"),
            confirmation_token=token,
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            ttl_seconds=safe_ttl,
            nonce=nonce,
        )
        record.add_event(COMMAND_STATE_PREVIEW_CREATED, actor="llm", reason="preview_created")
        record.add_event(COMMAND_STATE_VALIDATION_PASSED, actor="validator", reason="validation_passed")
        record.add_event(COMMAND_STATE_AWAITING_CONFIRMATION, actor="tool_gateway", reason="awaiting_confirmation")
        return record

    def add_event(self, state: str, *, actor: str, reason: str = "") -> None:
        """Append a lifecycle state transition event."""
        self.state = state
        self.events.append(
            {
                "state": state,
                "actor": actor,
                "reason": reason,
                "at": utc_now_iso(),
            }
        )

    def is_expired(self, *, now: datetime | None = None) -> bool:
        """Return whether this confirmation token is expired."""
        expires = parse_iso_datetime(self.expires_at)
        if expires is None:
            return True
        return (now or utc_now()) >= expires

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible lifecycle data without exposing nonce."""
        payload = asdict(self)
        payload.pop("nonce", None)
        return payload
