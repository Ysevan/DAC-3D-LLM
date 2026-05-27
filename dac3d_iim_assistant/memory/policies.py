"""Memory security policy helpers for DAC Memory OS."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from safety import PolicyDecision, PolicyEngine, TrustLevel


class MemoryStatus:
    """Lifecycle labels for long-term memory records."""

    PROPOSED = "proposed"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    DELETED = "deleted"


@dataclass(slots=True)
class MemoryPolicyResult:
    """Security decision and metadata for a candidate memory write."""

    allowed: bool
    target: str
    content: str
    source: str
    memory_status: str
    trust_level: str
    provenance: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    conflict: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemorySecurityPolicy:
    """Convert raw memory-write candidates into auditable memory metadata."""

    def __init__(self, policy_engine: PolicyEngine | None = None) -> None:
        self.policy_engine = policy_engine or PolicyEngine()

    def evaluate_candidate(
        self,
        *,
        target: str,
        content: str,
        source: str = "user",
        evidence_trace_id: str = "",
        session_id: str = "",
        privileged: bool = False,
        existing_profile: dict[str, Any] | None = None,
    ) -> MemoryPolicyResult:
        """Evaluate a proposed memory write without committing it."""
        normalized_target = str(target or "memory").strip() or "memory"
        normalized_source = str(source or "user").strip() or "user"
        decision = self.policy_engine.evaluate_memory_write(
            target=normalized_target,
            content=str(content or ""),
            approved=False,
            commit=False,
            privileged=privileged,
            source=normalized_source,
            evidence_trace_id=evidence_trace_id,
        )
        conflict = self.detect_conflict(
            target=normalized_target,
            content=str(content or ""),
            existing_profile=existing_profile or {},
        )
        memory_status = MemoryStatus.PROPOSED if decision.allowed else MemoryStatus.REJECTED
        return MemoryPolicyResult(
            allowed=decision.allowed,
            target=normalized_target,
            content=str(content or ""),
            source=normalized_source,
            memory_status=memory_status,
            trust_level=TrustLevel.UNTRUSTED,
            provenance={
                "trace_id": evidence_trace_id or None,
                "session_id": session_id or None,
                "source": normalized_source,
                "approved_by": None,
            },
            policy=decision.to_dict(),
            conflict=conflict,
        )

    def approval_policy(
        self,
        *,
        target: str,
        content: str,
        source: str,
        evidence_trace_id: str = "",
        privileged: bool = False,
    ) -> PolicyDecision:
        """Evaluate the final approval gate before applying a patch."""
        return self.policy_engine.evaluate_memory_write(
            target=target,
            content=content,
            approved=True,
            commit=True,
            privileged=privileged,
            source=source,
            evidence_trace_id=evidence_trace_id,
        )

    def active_metadata(
        self,
        *,
        patch_id: str,
        target: str,
        source_trace_id: str,
        session_id: str = "",
        approved_by: str = "human_review",
    ) -> dict[str, Any]:
        """Return metadata for an approved long-term memory record."""
        return {
            "memory_id": f"memory-{patch_id}",
            "target": target,
            "status": MemoryStatus.ACTIVE,
            "trust_level": TrustLevel.APPROVED_MEMORY,
            "provenance": {
                "trace_id": source_trace_id or None,
                "session_id": session_id or None,
                "approved_by": approved_by,
            },
        }

    def detect_conflict(
        self,
        *,
        target: str,
        content: str,
        existing_profile: dict[str, Any],
    ) -> dict[str, Any]:
        """Detect simple memory conflicts that need human review."""
        target_key = "user" if target == "user" else "memory"
        existing = str(existing_profile.get(target_key) or "")
        text = str(content or "")
        if not existing or text in existing:
            return {"detected": False}
        conflict_markers = ("默认", "不要默认", "改成", "纠正", "修正", "instead", "default")
        if any(marker in text for marker in conflict_markers):
            return {
                "detected": True,
                "type": "possible_preference_conflict",
                "target": target,
                "existing_excerpt": existing[:300],
            }
        return {"detected": False}
