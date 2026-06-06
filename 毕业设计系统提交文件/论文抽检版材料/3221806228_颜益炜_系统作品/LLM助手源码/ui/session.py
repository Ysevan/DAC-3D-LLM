"""Session-bound confirmation tokens for DAC-3D command execution."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from ui.auth import ApiSecurityError


DEFAULT_CONFIRMATION_TTL_SECONDS = 300


@dataclass(slots=True)
class StoredCommandPreview:
    """A command preview that can be confirmed exactly once."""

    preview_id: str
    preview_hash: str
    confirmation_token: str
    session_id: str
    operator_id: str
    command_preview: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    used_at: float | None = None

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at


class ConfirmationTokenStore:
    """In-memory confirmation token store for the FastAPI process."""

    def __init__(self, *, ttl_seconds: int = DEFAULT_CONFIRMATION_TTL_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        self._lock = RLock()
        self._previews: dict[str, StoredCommandPreview] = {}

    def create(
        self,
        *,
        session_id: str,
        operator_id: str,
        command_preview: dict[str, Any],
    ) -> StoredCommandPreview:
        """Create a preview id and one-time token bound to session/operator/hash."""
        preview_hash = compute_preview_hash(command_preview)
        token = secrets.token_urlsafe(32)
        created = time.time()
        preview = StoredCommandPreview(
            preview_id=uuid.uuid4().hex,
            preview_hash=preview_hash,
            confirmation_token=token,
            session_id=session_id,
            operator_id=operator_id,
            command_preview=dict(command_preview),
            created_at=created,
            expires_at=created + self.ttl_seconds,
        )
        with self._lock:
            self._previews[preview.preview_id] = preview
        return preview

    def consume(
        self,
        *,
        preview_id: str,
        confirmation_token: str,
        session_id: str,
        operator_id: str,
        preview_hash: str,
    ) -> StoredCommandPreview:
        """Validate and consume a confirmation token."""
        with self._lock:
            preview = self._previews.get(preview_id)
            if preview is None:
                raise ApiSecurityError(
                    code="CONFIRMATION_PREVIEW_NOT_FOUND",
                    message="The command preview was not found or has expired.",
                    status_code=403,
                )
            if preview.used_at is not None:
                raise ApiSecurityError(
                    code="CONFIRMATION_TOKEN_REPLAYED",
                    message="The confirmation token was already used.",
                    status_code=403,
                )
            if preview.expired:
                raise ApiSecurityError(
                    code="CONFIRMATION_TOKEN_EXPIRED",
                    message="The confirmation token has expired.",
                    status_code=403,
                )
            if preview.session_id != session_id:
                raise ApiSecurityError(
                    code="SESSION_ID_MISMATCH",
                    message="The confirmation token is not bound to this session.",
                    status_code=403,
                )
            if preview.operator_id != operator_id:
                raise ApiSecurityError(
                    code="OPERATOR_ID_MISMATCH",
                    message="The confirmation token is not bound to this operator.",
                    status_code=403,
                )
            if preview.preview_hash != preview_hash:
                raise ApiSecurityError(
                    code="PREVIEW_HASH_MISMATCH",
                    message="The command preview changed after confirmation token issuance.",
                    status_code=403,
                )
            if not secrets.compare_digest(preview.confirmation_token, confirmation_token):
                raise ApiSecurityError(
                    code="CONFIRMATION_TOKEN_INVALID",
                    message="The confirmation token is invalid.",
                    status_code=403,
                )

            preview.used_at = time.time()
            return preview

    def get(self, preview_id: str) -> StoredCommandPreview | None:
        """Return a stored preview without consuming it."""
        with self._lock:
            return self._previews.get(preview_id)


def compute_preview_hash(command_preview: dict[str, Any]) -> str:
    """Return a deterministic hash for a command preview."""
    canonical = json.dumps(
        command_preview,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
