"""Session helpers for SDK memory and DAC-3D pending confirmations."""

from __future__ import annotations

import re
from pathlib import Path
from threading import RLock
from typing import Any

from agent_core.schemas import AgentSessionSnapshot, PendingCommand
from commands import CommandLifecycleStore


CONFIRMATION_MARKERS = (
    "confirm",
    "confirmed",
    "yes",
    "ok",
    "执行",
    "确认",
    "确认执行",
    "确认开始",
    "立即开始",
    "马上开始",
    "开始吧",
    "可以执行",
    "同意执行",
)

ACTION_MARKERS = (
    "online",
    "offline",
    "scan",
    "detect",
    "stop",
    "result",
    "status",
    "在线",
    "离线",
    "扫描",
    "检测",
    "停止",
    "结果",
    "状态",
    "目录",
    "图片",
)


class DAC3DAgentSessionStore:
    """Keep SDK sessions and DAC-3D pending command confirmations together."""

    def __init__(self, *, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.session_dir = base_dir / ".tmp" / "agent_sessions"
        self.session_db_path = self.session_dir / "sessions.sqlite3"
        self._lock = RLock()
        self._sdk_sessions: dict[str, Any] = {}
        self._pending_commands: dict[str, PendingCommand] = {}
        self._command_history: dict[str, list[dict[str, Any]]] = {}
        self.command_lifecycle = CommandLifecycleStore()

    def normalize_session_id(self, session_id: str | None) -> str:
        """Return a safe stable session id."""
        normalized = (session_id or "default").strip() or "default"
        normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", normalized)
        return normalized[:120] or "default"

    def get_sdk_session(self, session_id: str | None) -> Any | None:
        """Return an Agents SDK SQLiteSession when the SDK is installed."""
        normalized = self.normalize_session_id(session_id)
        with self._lock:
            existing = self._sdk_sessions.get(normalized)
            if existing is not None:
                return existing

            try:
                from agents.memory import SQLiteSession
            except Exception:
                return None

            self.session_dir.mkdir(parents=True, exist_ok=True)
            sdk_session = SQLiteSession(
                normalized,
                db_path=self.session_db_path,
                sessions_table="dac3d_agent_sessions",
                messages_table="dac3d_agent_messages",
            )
            self._sdk_sessions[normalized] = sdk_session
            return sdk_session

    def remember_pending_command(
        self,
        session_id: str | None,
        *,
        source_message: str,
        command_preview: dict[str, Any],
        ttl_seconds: int = 300,
    ) -> PendingCommand:
        """Store the latest command preview that still needs confirmation."""
        normalized = self.normalize_session_id(session_id)
        preview = dict(command_preview)
        gateway = preview.get("gateway") if isinstance(preview.get("gateway"), dict) else {}
        preview["gateway"] = dict(gateway)
        lifecycle = self.command_lifecycle.create_pending(
            session_id=normalized,
            source_message=source_message,
            command_preview=preview,
            ttl_seconds=ttl_seconds,
        )
        preview["gateway"].update(
            {
                "preview_id": lifecycle.preview_id,
                "preview_hash": lifecycle.preview_hash,
                "confirmation_token": lifecycle.confirmation_token,
                "confirmation_expires_at": lifecycle.expires_at,
                "confirmation_ttl_seconds": lifecycle.ttl_seconds,
                "lifecycle_state": lifecycle.state,
                "lifecycle_events": list(lifecycle.events),
            }
        )
        pending = PendingCommand(
            session_id=normalized,
            source_message=source_message,
            command_preview=preview,
            warnings=list(preview.get("warnings") or []),
            created_at=lifecycle.created_at,
            preview_id=lifecycle.preview_id,
            preview_hash=lifecycle.preview_hash,
            confirmation_token=lifecycle.confirmation_token,
            expires_at=lifecycle.expires_at,
            ttl_seconds=lifecycle.ttl_seconds,
            lifecycle_state=lifecycle.state,
            lifecycle_events=list(lifecycle.events),
        )
        with self._lock:
            self._pending_commands[normalized] = pending
        return pending

    def get_pending_command(self, session_id: str | None) -> PendingCommand | None:
        """Return the pending command for a session, if any."""
        normalized = self.normalize_session_id(session_id)
        with self._lock:
            return self._pending_commands.get(normalized)

    def clear_pending_command(self, session_id: str | None) -> None:
        """Clear any pending command for a session."""
        normalized = self.normalize_session_id(session_id)
        with self._lock:
            self._pending_commands.pop(normalized, None)
        self.command_lifecycle.clear(normalized)

    def pending_command_expired(self, pending: PendingCommand) -> bool:
        """Return whether a pending command's confirmation token is expired."""
        lifecycle = self.command_lifecycle.get(pending.session_id)
        return lifecycle.is_expired() if lifecycle is not None else True

    def mark_pending_confirmation_failed(self, session_id: str | None, reason: str) -> None:
        """Record a failed confirmation transition for audit."""
        normalized = self.normalize_session_id(session_id)
        self.command_lifecycle.mark_confirmation_failed(normalized, reason)
        self._sync_pending_lifecycle(normalized)

    def mark_pending_confirmed(self, session_id: str | None) -> None:
        """Record a successful user confirmation transition."""
        normalized = self.normalize_session_id(session_id)
        self.command_lifecycle.mark_confirmed(normalized)
        self._sync_pending_lifecycle(normalized)

    def mark_pending_submitted(self, session_id: str | None) -> None:
        """Record a Tool Gateway submit transition."""
        normalized = self.normalize_session_id(session_id)
        self.command_lifecycle.mark_submitted(normalized)
        self._sync_pending_lifecycle(normalized)

    def mark_pending_expired(self, session_id: str | None) -> None:
        """Record and clear an expired pending command."""
        normalized = self.normalize_session_id(session_id)
        self.command_lifecycle.mark_expired(normalized)
        self._sync_pending_lifecycle(normalized)

    def mark_pending_cancelled(self, session_id: str | None) -> None:
        """Record a cancelled pending command transition."""
        normalized = self.normalize_session_id(session_id)
        self.command_lifecycle.mark_cancelled(normalized)
        self._sync_pending_lifecycle(normalized)

    def confirmation_token_consumed(self, session_id: str | None, preview_id: str, token: str) -> bool:
        """Return whether a confirmation token was already used."""
        normalized = self.normalize_session_id(session_id)
        return self.command_lifecycle.is_consumed(normalized, preview_id, token)

    def consume_confirmation_token(self, session_id: str | None, preview_id: str, token: str) -> None:
        """Mark a confirmation token as consumed."""
        normalized = self.normalize_session_id(session_id)
        self.command_lifecycle.consume(normalized, preview_id, token)
        self._sync_pending_lifecycle(normalized)

    def pending_preview_hash_matches(self, pending: PendingCommand) -> bool:
        """Return whether the pending preview still matches its original hash."""
        return self.command_lifecycle.verify_preview_hash(pending.session_id, pending.command_preview)

    def append_command_history(self, session_id: str | None, event: dict[str, Any]) -> None:
        """Append one Tool Gateway command event for audit/debugging."""
        normalized = self.normalize_session_id(session_id)
        with self._lock:
            events = self._command_history.setdefault(normalized, [])
            events.append(dict(event))
            del events[:-100]

    def read_command_history(self, session_id: str | None, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent Tool Gateway command events."""
        normalized = self.normalize_session_id(session_id)
        safe_limit = max(1, min(100, int(limit or 20)))
        with self._lock:
            events = list(self._command_history.get(normalized, []))
        return events[-safe_limit:]

    def should_use_pending_confirmation(self, text: str) -> bool:
        """Return whether the text is a confirmation-only response."""
        normalized = text.strip().lower()
        if not normalized:
            return False
        has_confirmation = any(marker in normalized for marker in CONFIRMATION_MARKERS)
        has_new_action = any(marker in normalized for marker in ACTION_MARKERS)
        return has_confirmation and not has_new_action

    def describe_session(self, session_id: str | None) -> dict[str, Any]:
        """Return UI/API diagnostics for one session."""
        normalized = self.normalize_session_id(session_id)
        pending = self.get_pending_command(normalized)
        return AgentSessionSnapshot(
            session_id=normalized,
            has_pending_command=pending is not None,
            pending_action=pending.action if pending is not None else None,
            pending_created_at=pending.created_at if pending is not None else None,
            pending_expires_at=pending.expires_at if pending is not None else None,
            pending_lifecycle_state=pending.lifecycle_state if pending is not None else None,
            pending_preview_id=pending.preview_id if pending is not None else None,
            sdk_session_enabled=True,
        ).to_dict()

    def _sync_pending_lifecycle(self, session_id: str) -> None:
        lifecycle = self.command_lifecycle.get(session_id)
        if lifecycle is None:
            return
        with self._lock:
            pending = self._pending_commands.get(session_id)
            if pending is None:
                return
            gateway = pending.command_preview.get("gateway")
            if not isinstance(gateway, dict):
                gateway = {}
                pending.command_preview["gateway"] = gateway
            gateway.update(
                {
                    "preview_id": lifecycle.preview_id,
                    "preview_hash": lifecycle.preview_hash,
                    "confirmation_token": lifecycle.confirmation_token,
                    "confirmation_expires_at": lifecycle.expires_at,
                    "confirmation_ttl_seconds": lifecycle.ttl_seconds,
                    "lifecycle_state": lifecycle.state,
                    "lifecycle_events": list(lifecycle.events),
                }
            )
            pending.lifecycle_state = lifecycle.state
            pending.lifecycle_events = list(lifecycle.events)
