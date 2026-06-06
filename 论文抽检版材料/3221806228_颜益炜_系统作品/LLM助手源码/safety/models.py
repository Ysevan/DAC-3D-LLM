"""Typed safety policy models for DAC-Agent Runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PolicyDecision:
    """Code-level decision for one security policy evaluation."""

    allowed: bool
    risk_level: str
    requires_confirmation: bool
    reasons: list[str] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    policy_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable policy decision."""
        return asdict(self)

    @classmethod
    def allow(
        cls,
        *,
        risk_level: str = "low",
        requires_confirmation: bool = False,
        reasons: list[str] | None = None,
        policy_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "PolicyDecision":
        """Create an allow decision."""
        return cls(
            allowed=True,
            risk_level=risk_level,
            requires_confirmation=requires_confirmation,
            reasons=list(reasons or []),
            blocking_reasons=[],
            policy_ids=list(policy_ids or []),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def block(
        cls,
        *,
        risk_level: str = "forbidden",
        requires_confirmation: bool = True,
        reasons: list[str] | None = None,
        blocking_reasons: list[str] | None = None,
        policy_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "PolicyDecision":
        """Create a fail-closed block decision."""
        blocks = list(blocking_reasons or reasons or ["policy_denied"])
        return cls(
            allowed=False,
            risk_level=risk_level,
            requires_confirmation=requires_confirmation,
            reasons=list(reasons or blocks),
            blocking_reasons=blocks,
            policy_ids=list(policy_ids or []),
            metadata=dict(metadata or {}),
        )
