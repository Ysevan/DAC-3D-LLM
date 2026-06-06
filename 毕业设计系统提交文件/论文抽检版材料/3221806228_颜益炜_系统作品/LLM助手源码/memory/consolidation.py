"""Memory consolidation proposal helpers.

Consolidation intentionally creates reviewable proposals only. It never edits
long-term memory directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class MemoryConsolidationProposal:
    """Reviewable proposal generated from repeated memory evidence."""

    target: str
    content: str
    reason: str
    evidence_trace_ids: list[str] = field(default_factory=list)
    status: str = "proposed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryConsolidator:
    """Summarize repeated traces into proposed memory patches."""

    def propose(self, traces: list[dict[str, Any]], *, target: str = "memory") -> list[dict[str, Any]]:
        """Return consolidation proposals without mutating memory."""
        candidates: list[MemoryConsolidationProposal] = []
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            user_message = str(trace.get("user_message") or "")
            if not user_message:
                continue
            if any(marker in user_message for marker in ("以后", "记住", "下次", "纠正", "偏好")):
                candidates.append(
                    MemoryConsolidationProposal(
                        target=target,
                        content=f"候选长期记忆：{user_message[:500]}",
                        reason="explicit_repeated_or_corrective_memory_signal",
                        evidence_trace_ids=[str(trace.get("trace_id") or "")],
                    )
                )
        return [candidate.to_dict() for candidate in candidates]
