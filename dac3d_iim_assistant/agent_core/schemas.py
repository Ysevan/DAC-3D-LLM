"""Typed payloads used at the DAC-3D Agent/tool boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Return a compact UTC timestamp for audit/session metadata."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class AssistantResponsePayload(BaseModel):
    """Stable tool payload converted from the existing assistant response."""

    intent: str
    answer: str
    sources: list[str] = Field(default_factory=list)
    source_items: list[dict[str, Any]] = Field(default_factory=list)
    command_preview: dict[str, Any] | None = None
    status_summary: dict[str, Any] | None = None
    parsed_result: dict[str, Any] | None = None

    @classmethod
    def from_response(cls, response: Any) -> "AssistantResponsePayload":
        """Build a typed payload from app.AssistantResponse without importing app."""
        return cls(
            intent=str(getattr(response, "intent", "")),
            answer=str(getattr(response, "answer", "")),
            sources=list(getattr(response, "sources", []) or []),
            source_items=list(getattr(response, "source_items", []) or []),
            command_preview=getattr(response, "command_preview", None),
            status_summary=getattr(response, "status_summary", None),
            parsed_result=getattr(response, "parsed_result", None),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible data while supporting Pydantic v2."""
        return self.model_dump(mode="json", exclude_none=False)


class PendingCommand(BaseModel):
    """A command preview waiting for explicit user confirmation."""

    session_id: str
    source_message: str
    command_preview: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)

    @property
    def action(self) -> str:
        """Return the DAC-3D command action name."""
        return str(self.command_preview.get("action") or "unknown")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible data."""
        return self.model_dump(mode="json")


class AgentSessionSnapshot(BaseModel):
    """Public diagnostics for one Agent session."""

    session_id: str
    has_pending_command: bool = False
    pending_action: str | None = None
    pending_created_at: str | None = None
    sdk_session_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible data."""
        return self.model_dump(mode="json", exclude_none=False)
