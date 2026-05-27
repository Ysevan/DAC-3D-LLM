"""Session helpers for SDK memory and DAC-3D pending confirmations."""

from __future__ import annotations

import re
from pathlib import Path
from threading import RLock
from typing import Any

from agent_core.schemas import AgentSessionSnapshot, PendingCommand


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
    ) -> PendingCommand:
        """Store the latest command preview that still needs confirmation."""
        normalized = self.normalize_session_id(session_id)
        pending = PendingCommand(
            session_id=normalized,
            source_message=source_message,
            command_preview=dict(command_preview),
            warnings=list(command_preview.get("warnings") or []),
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
            sdk_session_enabled=True,
        ).to_dict()
