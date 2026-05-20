"""Unified schemas for intent parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


IntentLabel = Literal["query", "operation", "interpretation", "guidance", "status"]


@dataclass(slots=True)
class ParsedCommand:
    """Normalized command fields extracted from operator language."""

    action: str | None = None
    scan_area_mm: dict[str, float] | None = None
    resolution: dict[str, float | str] | None = None
    region: str | None = None
    mode: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the parsed command into a serializable mapping."""
        return {
            "action": self.action,
            "scan_area_mm": self.scan_area_mm,
            "resolution": self.resolution,
            "region": self.region,
            "mode": self.mode,
            "payload": dict(self.payload),
            "safety": dict(self.safety),
        }


@dataclass(slots=True)
class ParsedIntent:
    """Unified result returned by the intent parsing pipeline."""

    intent: IntentLabel
    confidence: float
    hints: dict[str, object] = field(default_factory=dict)
    command: ParsedCommand | None = None
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the parsed intent into a serializable mapping."""
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "hints": dict(self.hints),
            "command": None if self.command is None else self.command.to_dict(),
            "missing_fields": list(self.missing_fields),
            "warnings": list(self.warnings),
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
        }
