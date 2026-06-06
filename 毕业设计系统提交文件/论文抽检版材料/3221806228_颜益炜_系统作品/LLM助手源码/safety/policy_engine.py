"""Code-enforced production policy engine for DAC-Agent Runtime."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from intent.structured_commands import READ_ONLY_ACTIONS, SUPPORTED_ACTIONS, missing_required_fields
from safety.guard import CONTROL_ACTIONS, SafetyGuard
from safety.models import PolicyDecision


BYPASS_PATTERNS = (
    r"skip (confirmation|approval|safety)",
    r"bypass (confirmation|approval|safety)",
    r"without (confirmation|approval)",
    r"no (confirmation|approval) needed",
    r"ignore (all )?(previous|above|system|safety) (instructions|rules|policy)",
    r"直接执行",
    r"立即执行不用确认",
    r"不用确认",
    r"不要确认",
    r"跳过(确认|审批|安全)",
    r"绕过(确认|审批|安全)",
    r"忽略(系统|安全|以上|之前)",
)

SECRET_VALUE_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|api[_-]?key\s*[:=]\s*\S+|authorization\s*[:=]\s*\S+|"
    r"password\s*[:=]\s*\S+|secret\s*[:=]\s*\S+|token\s*[:=]\s*\S+)",
    re.IGNORECASE,
)

FORBIDDEN_TOOLS = {
    "write_command_json",
    "write_command_file",
    "direct_command_writer",
    "filesystem_write",
    "shell_exec",
    "python_exec",
}


class PolicyEngine:
    """Enforce DAC-Agent safety policy in code, independent of LLM prompts."""

    def __init__(
        self,
        *,
        allowed_roots: list[str | Path] | None = None,
        allowed_write_roots: list[str | Path] | None = None,
        registered_tools: list[str] | None = None,
    ) -> None:
        self.guard = SafetyGuard(
            allowed_roots=allowed_roots or [],
            allowed_write_roots=allowed_write_roots or [],
        )
        self.registered_tools = set(registered_tools or [])

    def evaluate_intent(
        self,
        intent: str,
        *,
        command: dict[str, Any] | None = None,
        user_text: str = "",
        retrieved_text: str = "",
    ) -> PolicyDecision:
        """Evaluate a user/model intent before any tool work happens."""
        bypass = self._find_bypass_signals(" ".join([user_text, retrieved_text]))
        if bypass:
            return PolicyDecision.block(
                risk_level="forbidden",
                requires_confirmation=True,
                reasons=["Prompt or user text attempted to bypass DAC safety policy."],
                blocking_reasons=["policy_bypass_request"],
                policy_ids=["POLICY-INJECTION-BYPASS"],
                metadata={"matches": bypass},
            )

        risk = self.guard.classify_intent_risk(intent, command)
        return PolicyDecision(
            allowed=risk.can_execute,
            risk_level=risk.risk_level,
            requires_confirmation=risk.requires_confirmation,
            reasons=[risk.reason],
            blocking_reasons=list(risk.blocked_by),
            policy_ids=["POLICY-INTENT-RISK"],
        )

    def evaluate_tool_call(
        self,
        tool_metadata: dict[str, Any] | None,
        arguments: dict[str, Any] | None = None,
        *,
        private_context: bool = False,
        schema_valid: bool = True,
        user_text: str = "",
        retrieved_text: str = "",
        confirmed: bool = False,
    ) -> PolicyDecision:
        """Evaluate one Tool Gateway call. Unknown tools fail closed."""
        metadata = dict(tool_metadata or {})
        arguments = dict(arguments or {})
        name = str(metadata.get("name") or "").strip()
        if not name:
            return PolicyDecision.block(
                reasons=["Tool metadata is missing a registered name."],
                blocking_reasons=["unknown_tool"],
                policy_ids=["POLICY-TOOL-UNKNOWN"],
            )
        if name in FORBIDDEN_TOOLS or (self.registered_tools and name not in self.registered_tools):
            return PolicyDecision.block(
                reasons=[f"Tool is not registered or is forbidden: {name}"],
                blocking_reasons=["forbidden_tool"],
                policy_ids=["POLICY-TOOL-FORBIDDEN"],
                metadata={"tool": name},
            )
        if not schema_valid or not self._arguments_match_schema(metadata.get("input_schema"), arguments):
            return PolicyDecision.block(
                risk_level=str(metadata.get("risk_level") or "forbidden"),
                requires_confirmation=bool(metadata.get("requires_confirmation", True)),
                reasons=["Tool arguments failed schema validation."],
                blocking_reasons=["schema_invalid"],
                policy_ids=["POLICY-TOOL-SCHEMA"],
                metadata={"tool": name},
            )

        bypass = self._find_bypass_signals(" ".join([user_text, retrieved_text, str(arguments)]))
        if bypass:
            return PolicyDecision.block(
                reasons=["Tool call contained safety-bypass instructions."],
                blocking_reasons=["policy_bypass_request"],
                policy_ids=["POLICY-INJECTION-BYPASS"],
                metadata={"tool": name, "matches": bypass},
            )

        risk_level = str(metadata.get("risk_level") or "low")
        requires_confirmation = bool(metadata.get("requires_confirmation"))
        destructive = bool(metadata.get("destructive"))
        open_world = bool(metadata.get("open_world"))
        high_risk = risk_level in {"high", "forbidden"} or destructive
        if open_world and private_context:
            return PolicyDecision.block(
                risk_level=risk_level,
                requires_confirmation=True,
                reasons=["Open-world tools cannot receive private context."],
                blocking_reasons=["open_world_private_context"],
                policy_ids=["POLICY-TOOL-OPEN-WORLD"],
                metadata={"tool": name},
            )
        if high_risk and not confirmed:
            return PolicyDecision.block(
                risk_level=risk_level,
                requires_confirmation=True,
                reasons=["High-risk or destructive tool requires explicit confirmation."],
                blocking_reasons=["confirmation_required"],
                policy_ids=["POLICY-TOOL-CONFIRMATION"],
                metadata={"tool": name},
            )

        return PolicyDecision.allow(
            risk_level=risk_level,
            requires_confirmation=requires_confirmation or high_risk,
            reasons=["Tool call allowed by registered metadata and policy checks."],
            policy_ids=["POLICY-TOOL-ALLOW"],
            metadata={"tool": name},
        )

    def evaluate_command_preview(
        self,
        command_preview: dict[str, Any] | None,
        *,
        source_text: str = "",
        validation_passed: bool = True,
    ) -> PolicyDecision:
        """Evaluate a command preview before it can become submit-ready."""
        preview = dict(command_preview or {})
        action = str(preview.get("action") or "").strip()
        blocks: list[str] = []
        reasons: list[str] = []
        path_decisions: list[dict[str, Any]] = []

        if action not in SUPPORTED_ACTIONS:
            blocks.append("unsupported_action")
            reasons.append(f"Unsupported DAC command action: {action or 'missing'}")
        missing = list(preview.get("missing_fields") or [])
        missing.extend(item for item in missing_required_fields(action, preview) if item not in missing)
        if missing:
            blocks.extend(f"missing:{item}" for item in missing)
            reasons.append("Command preview is missing required fields.")

        payload = dict(preview.get("payload") or {})
        read_path_fields = {"image_folder", "input_dir", "result_root"}
        write_path_fields = {"output_dir", "command_output", "command_path"}
        for field_name in (*sorted(read_path_fields), *sorted(write_path_fields)):
            value = payload.get(field_name)
            if value:
                if field_name in write_path_fields:
                    path_decision = self.guard.validate_command_output(str(value))
                else:
                    path_decision = self.guard.validate_path(str(value))
                path_decisions.append({"field": field_name, **path_decision.to_dict()})
                if not path_decision.allowed:
                    blocks.append("path_not_allowed")
                    reasons.append(f"Path is outside allowlist: {field_name}")

        injection = self.guard.detect_prompt_injection(" ".join([source_text, str(payload), str(preview.get("yaml_preview") or "")]))
        bypass = self._find_bypass_signals(source_text)
        if injection.detected or bypass:
            blocks.append("prompt_injection_detected")
            reasons.append("Command preview contains prompt-injection or safety-bypass language.")
        if not validation_passed:
            blocks.append("validation_failed")
            reasons.append("Command preview failed upstream validation.")

        risk_level = "read_only" if action in READ_ONLY_ACTIONS else "medium"
        requires_confirmation = action not in READ_ONLY_ACTIONS
        if action in CONTROL_ACTIONS:
            risk_level = "high"
            requires_confirmation = True
        if blocks:
            return PolicyDecision.block(
                risk_level="forbidden" if "prompt_injection_detected" in blocks else risk_level,
                requires_confirmation=True,
                reasons=reasons or ["Command preview denied by policy."],
                blocking_reasons=list(dict.fromkeys(blocks)),
                policy_ids=["POLICY-COMMAND-PREVIEW"],
                metadata={"path_decisions": path_decisions, "action": action},
            )
        return PolicyDecision.allow(
            risk_level=risk_level,
            requires_confirmation=requires_confirmation,
            reasons=["Command preview passed policy checks."],
            policy_ids=["POLICY-COMMAND-PREVIEW"],
            metadata={"path_decisions": path_decisions, "action": action},
        )

    def evaluate_command_submit(
        self,
        *,
        command_preview_id: str | None,
        confirmation_token: str | None,
        expected_confirmation_token: str | None,
        validation_passed: bool,
        command_preview: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        """Evaluate final command submission. Submit is fail-closed."""
        blocks: list[str] = []
        reasons: list[str] = []
        if not command_preview_id:
            blocks.append("missing_command_preview_id")
            reasons.append("Submit requires command_preview_id.")
        if not validation_passed:
            blocks.append("validation_failed")
            reasons.append("Submit requires a freshly validated command preview.")
        if not confirmation_token:
            blocks.append("missing_confirmation_token")
            reasons.append("Submit requires confirmation_token.")
        elif not expected_confirmation_token or str(confirmation_token) != str(expected_confirmation_token):
            blocks.append("confirmation_token_mismatch")
            reasons.append("Submit confirmation token does not match the pending preview.")

        preview_decision = self.evaluate_command_preview(command_preview or {}, validation_passed=validation_passed)
        if command_preview is not None and not preview_decision.allowed:
            blocks.extend(preview_decision.blocking_reasons)
            reasons.extend(preview_decision.reasons)

        if blocks:
            return PolicyDecision.block(
                risk_level="high",
                requires_confirmation=True,
                reasons=list(dict.fromkeys(reasons or ["Command submit denied by policy."])),
                blocking_reasons=list(dict.fromkeys(blocks)),
                policy_ids=["POLICY-COMMAND-SUBMIT"],
            )
        return PolicyDecision.allow(
            risk_level="high",
            requires_confirmation=True,
            reasons=["Command submit passed policy checks."],
            policy_ids=["POLICY-COMMAND-SUBMIT"],
        )

    def evaluate_memory_write(
        self,
        *,
        target: str,
        content: str,
        approved: bool = False,
        commit: bool = False,
        privileged: bool = False,
        source: str = "user",
        evidence_trace_id: str = "",
    ) -> PolicyDecision:
        """Evaluate Memory OS write or patch creation."""
        normalized_target = str(target or "memory").strip().lower()
        text = str(content or "")
        blocks: list[str] = []
        if SECRET_VALUE_PATTERN.search(text):
            blocks.append("secret_in_memory")
        if normalized_target == "policy_memory" and not privileged:
            blocks.append("policy_memory_requires_privileged_approval")
        if normalized_target == "tool_memory" and source == "tool_output":
            blocks.append("tool_output_cannot_directly_create_tool_memory")
        if normalized_target == "procedure_memory" and not evidence_trace_id:
            blocks.append("procedure_memory_requires_evidence_trace")
        if commit and not approved:
            blocks.append("memory_commit_requires_approval")
        bypass = self._find_bypass_signals(text)
        if bypass:
            blocks.append("memory_policy_bypass_text")

        if blocks:
            return PolicyDecision.block(
                risk_level="medium",
                requires_confirmation=True,
                reasons=["Memory write must remain an auditable patch and cannot weaken policy."],
                blocking_reasons=list(dict.fromkeys(blocks)),
                policy_ids=["POLICY-MEMORY-WRITE"],
            )
        return PolicyDecision.allow(
            risk_level="medium",
            requires_confirmation=not commit,
            reasons=["Memory write may proceed as a patch or approved commit."],
            policy_ids=["POLICY-MEMORY-WRITE"],
        )

    def evaluate_skill_patch(
        self,
        *,
        skill_name: str,
        content: str = "",
        apply: bool = False,
        approved: bool = False,
    ) -> PolicyDecision:
        """Evaluate skill patch proposals. Auto-apply is denied by default."""
        blocks: list[str] = []
        if not str(skill_name or "").strip():
            blocks.append("missing_skill_name")
        if self._find_bypass_signals(content):
            blocks.append("skill_patch_policy_bypass_text")
        if apply and not approved:
            blocks.append("skill_patch_apply_requires_approval")
        if blocks:
            return PolicyDecision.block(
                risk_level="medium",
                requires_confirmation=True,
                reasons=["Skill patches require human approval before apply."],
                blocking_reasons=blocks,
                policy_ids=["POLICY-SKILL-PATCH"],
            )
        return PolicyDecision.allow(
            risk_level="medium",
            requires_confirmation=not apply,
            reasons=["Skill patch may be proposed; applying still requires approval."],
            policy_ids=["POLICY-SKILL-PATCH"],
        )

    def _find_bypass_signals(self, text: str) -> list[str]:
        value = str(text or "")
        matches: list[str] = []
        for pattern in BYPASS_PATTERNS:
            if re.search(pattern, value, flags=re.IGNORECASE):
                matches.append(pattern)
        return matches

    def _arguments_match_schema(self, input_schema: Any, arguments: dict[str, Any]) -> bool:
        if not isinstance(input_schema, dict) or not input_schema:
            return True
        for field_name, field_schema in input_schema.items():
            optional = "null" in str(field_schema).lower()
            if not optional and field_name not in arguments:
                return False
            if not optional and arguments.get(field_name) in (None, ""):
                return False
        return True
