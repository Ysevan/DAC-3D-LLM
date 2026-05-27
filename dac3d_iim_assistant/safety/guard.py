"""Safety decisions for DAC-Agent tool execution."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from intent.structured_commands import READ_ONLY_ACTIONS
from safety.path_policy import PathPolicy


BUSY_STATES = {"running", "detecting", "scanning", "initializing"}
CONTROL_ACTIONS = {"scan", "start_online_scan", "start_offline_detection", "stop_detection"}
FORBIDDEN_ACTION_MARKERS = {"delete", "remove", "overwrite", "format", "rm", "del", "删除", "覆盖", "清空"}
INJECTION_PATTERNS = (
    r"ignore (all )?(previous|above|system) (instructions|rules)",
    r"bypass (confirmation|approval|safety)",
    r"write .*command\.json",
    r"directly write",
    r"忽略(以上|之前|系统|安全)",
    r"绕过(确认|审批|安全)",
    r"不要(确认|审批)",
    r"直接(写入|下发|执行)",
    r"写入\s*command\.json",
)


@dataclass(slots=True)
class RiskDecision:
    """Risk classification for a DAC tool or command."""

    risk_level: str
    requires_confirmation: bool
    can_execute: bool
    reason: str
    blocked_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PathDecision:
    """Path allowlist decision."""

    path: str
    allowed: bool
    reason: str
    matched_root: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ConfirmationRequest:
    """Confirmation metadata for a pending command preview."""

    preview_id: str
    confirmation_token: str
    required: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InjectionSignal:
    """Prompt-injection signal detected in untrusted text."""

    detected: bool
    severity: str = "none"
    matches: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SafetyGuard:
    """Central safety policy for Tool Gateway validation."""

    def __init__(
        self,
        *,
        allowed_roots: list[str | Path] | None = None,
        allowed_write_roots: list[str | Path] | None = None,
    ) -> None:
        self.allowed_roots = [self._resolve_path(root) for root in (allowed_roots or []) if str(root or "").strip()]
        self.allowed_write_roots = [
            self._resolve_path(root) for root in (allowed_write_roots or []) if str(root or "").strip()
        ]
        self.path_policy = PathPolicy(
            allowed_read_roots=allowed_roots or [],
            allowed_write_roots=allowed_write_roots or [],
        )

    def classify_intent_risk(
        self,
        intent: str,
        command: dict[str, Any] | None = None,
    ) -> RiskDecision:
        """Classify risk for one intent/tool operation."""
        normalized_intent = str(intent or "").strip().lower()
        command = dict(command or {})
        action = str(command.get("action") or "").strip().lower()
        safety = dict(command.get("safety") or {})
        missing_fields = list(command.get("missing_fields") or [])

        if any(marker in normalized_intent or marker in action for marker in FORBIDDEN_ACTION_MARKERS):
            return RiskDecision(
                risk_level="forbidden",
                requires_confirmation=True,
                can_execute=False,
                reason="Forbidden file or destructive action marker.",
                blocked_by=["forbidden_action"],
            )

        if normalized_intent in {"read_dac_status", "read_latest_result", "read_command_history", "list_allowed_dirs"}:
            return RiskDecision("read_only", False, True, "Read-only gateway tool.")
        if action in READ_ONLY_ACTIONS:
            return RiskDecision("read_only", False, not missing_fields, "Read-only DAC action.", list(missing_fields))
        if normalized_intent in {"preview_command", "validate_command"}:
            return RiskDecision("medium", False, not missing_fields, "Non-executing command preview.", list(missing_fields))
        if normalized_intent in {"submit_command", "execute_command"} or action in CONTROL_ACTIONS:
            requires_confirmation = bool(safety.get("needs_confirmation", True))
            return RiskDecision(
                risk_level="high",
                requires_confirmation=requires_confirmation,
                can_execute=not missing_fields,
                reason="DAC runtime state-changing command.",
                blocked_by=list(missing_fields),
            )
        return RiskDecision("low", False, not missing_fields, "Local non-executing request.", list(missing_fields))

    def validate_path(self, path: str) -> PathDecision:
        """Validate a local path against configured allowlist roots."""
        decision = self.path_policy.validate_input_dir(path)
        return PathDecision(
            path=decision.path,
            allowed=decision.allowed,
            reason=decision.reason,
            matched_root=decision.matched_root,
        )

    def validate_command_output(self, path: str) -> PathDecision:
        """Validate a DAC command output path against configured write roots."""
        decision = self.path_policy.validate_command_output(path)
        return PathDecision(
            path=decision.path,
            allowed=decision.allowed,
            reason=decision.reason,
            matched_root=decision.matched_root,
        )

    def require_confirmation(self, preview: dict[str, Any]) -> ConfirmationRequest:
        """Build confirmation metadata for one command preview."""
        preview_id = self.preview_id(preview)
        safety = dict(preview.get("safety") or {})
        required = bool(safety.get("needs_confirmation", True))
        return ConfirmationRequest(
            preview_id=preview_id,
            confirmation_token=self.confirmation_token(preview_id),
            required=required,
            message=(
                "该命令需要用户明确确认后才能通过 Tool Gateway 下发。"
                if required
                else "该命令为只读或低风险操作，不需要确认。"
            ),
        )

    def verify_confirmation(self, token: str, preview_id: str) -> bool:
        """Verify a gateway confirmation token."""
        expected = self.confirmation_token(preview_id)
        return bool(token) and str(token) == expected

    def detect_prompt_injection(self, text: str) -> InjectionSignal:
        """Detect obvious prompt-injection or safety-bypass language."""
        value = str(text or "")
        matches: list[str] = []
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, value, flags=re.IGNORECASE):
                matches.append(pattern)
        if not matches:
            return InjectionSignal(detected=False)
        severity = "high" if any("command" in match or "写入" in match for match in matches) else "medium"
        return InjectionSignal(detected=True, severity=severity, matches=matches)

    def preview_id(self, preview: dict[str, Any]) -> str:
        """Return a stable id for a command preview."""
        stable_preview = dict(preview)
        stable_preview.pop("gateway", None)
        normalized = json.dumps(stable_preview, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        return f"preview-{digest}"

    def confirmation_token(self, preview_id: str) -> str:
        """Return the local confirmation token for one preview id."""
        digest = hashlib.sha256(f"dac3d:{preview_id}".encode("utf-8")).hexdigest()[:20]
        return f"confirm-{digest}"

    def _resolve_path(self, value: str | Path) -> Path:
        return Path(value).expanduser().resolve(strict=False)

    def _is_relative_to(self, candidate: Path, root: Path) -> bool:
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        return True
