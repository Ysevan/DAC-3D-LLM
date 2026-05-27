"""MCP-style Tool Gateway for DAC-3D Agent Runtime."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_core.schemas import AssistantResponsePayload
from agent_core.sessions import DAC3DAgentSessionStore
from intent.structured_commands import READ_ONLY_ACTIONS, SUPPORTED_ACTIONS, missing_required_fields
from safety import PolicyDecision, PolicyEngine, SafetyGuard


@dataclass(slots=True)
class ToolDescriptor:
    """Stable metadata for one gateway tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    risk_level: str
    requires_confirmation: bool = False
    read_only: bool = False
    destructive: bool = False
    idempotent: bool = False
    open_world: bool = False
    allowed_scopes: list[str] = field(default_factory=list)
    read_only_hint: bool | None = None
    destructive_hint: bool | None = None
    idempotent_hint: bool | None = None
    open_world_hint: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        read_only_hint = self.read_only if self.read_only_hint is None else self.read_only_hint
        destructive_hint = self.destructive if self.destructive_hint is None else self.destructive_hint
        idempotent_hint = self.idempotent if self.idempotent_hint is None else self.idempotent_hint
        open_world_hint = self.open_world if self.open_world_hint is None else self.open_world_hint
        payload.update(
            {
                "readOnlyHint": read_only_hint,
                "destructiveHint": destructive_hint,
                "idempotentHint": idempotent_hint,
                "openWorldHint": open_world_hint,
                "annotations": {
                    "readOnlyHint": read_only_hint,
                    "destructiveHint": destructive_hint,
                    "idempotentHint": idempotent_hint,
                    "openWorldHint": open_world_hint,
                },
            }
        )
        return payload


@dataclass(slots=True)
class ToolInvocationResult:
    """Auditable tool invocation payload."""

    tool: str
    ok: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    risk: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DAC3DToolGateway:
    """Controlled boundary for DAC status, result, preview, validation, and execution tools."""

    def __init__(
        self,
        *,
        assistant: Any,
        sessions: DAC3DAgentSessionStore,
        session_id: str,
        allowed_roots: list[str | Path] | None = None,
    ) -> None:
        self.assistant = assistant
        self.sessions = sessions
        self.session_id = self.sessions.normalize_session_id(session_id)
        roots = allowed_roots or self._default_allowed_roots()
        write_roots = self._default_command_output_roots()
        self.safety_guard = SafetyGuard(allowed_roots=roots, allowed_write_roots=write_roots)
        self.policy_engine = PolicyEngine(
            allowed_roots=roots,
            allowed_write_roots=write_roots,
            registered_tools=[descriptor.name for descriptor in self.descriptors()],
        )

    def describe(self) -> dict[str, Any]:
        """Return gateway diagnostics and tool descriptors."""
        return {
            "enabled": True,
            "backend": "dac_tool_gateway",
            "tool_count": len(self.descriptors()),
            "tools": [descriptor.to_dict() for descriptor in self.descriptors()],
            "allowed_dirs": self.list_allowed_dirs().result["allowed_dirs"],
            "workflow": "preview -> validate -> confirm -> submit -> trace",
            "metadata_policy": (
                "MCP-style hints are advisory only; SafetyGuard enforces confirmation, "
                "schema validation, path allowlists, and prompt-injection blocking."
            ),
            "policy_engine": {
                "enabled": True,
                "mode": "fail_closed",
                "registered_tools": [descriptor.name for descriptor in self.descriptors()],
                "enforcement": (
                    "Every Tool Gateway call is evaluated by PolicyEngine before DAC state, "
                    "preview, validation, submit, cancel, or audit data is returned."
                ),
            },
            "command_lifecycle": {
                "enabled": True,
                "state_machine": (
                    "draft -> preview_created -> validation_passed -> awaiting_confirmation -> "
                    "confirmed -> submitted"
                ),
                "confirmation_ttl_seconds": self._confirmation_ttl_seconds(),
                "replay_protection": "confirmation tokens are unique per pending lifecycle and consumed once",
            },
        }

    def descriptors(self) -> list[ToolDescriptor]:
        """Return MCP-style tool descriptors."""
        return [
            ToolDescriptor(
                name="read_dac_status",
                description="Read current DAC-3D runtime status.",
                input_schema={},
                output_schema={"type": "object"},
                risk_level="read_only",
                read_only=True,
                idempotent=True,
                allowed_scopes=["dac_runtime_status"],
            ),
            ToolDescriptor(
                name="read_latest_result",
                description="Read latest DAC-3D inspection result summary.",
                input_schema={"sample_position": "integer | null"},
                output_schema={"type": "object"},
                risk_level="read_only",
                read_only=True,
                idempotent=True,
                allowed_scopes=["dac_results"],
            ),
            ToolDescriptor(
                name="list_allowed_dirs",
                description="List local directories that path-sensitive tools may access.",
                input_schema={},
                output_schema={"type": "object"},
                risk_level="read_only",
                read_only=True,
                idempotent=True,
                allowed_scopes=["path_policy"],
            ),
            ToolDescriptor(
                name="preview_command",
                description="Generate a structured DAC-3D command preview without executing.",
                input_schema={"command_draft": "string"},
                output_schema={"type": "assistant_payload"},
                risk_level="medium",
                read_only=False,
                idempotent=False,
                allowed_scopes=["dac_command_preview"],
                read_only_hint=False,
            ),
            ToolDescriptor(
                name="validate_command",
                description="Validate command schema, path policy, runtime risk, and confirmation need.",
                input_schema={"command_preview": "object"},
                output_schema={"type": "object"},
                risk_level="medium",
                read_only=True,
                idempotent=True,
                allowed_scopes=["dac_command_validation"],
            ),
            ToolDescriptor(
                name="submit_command",
                description="Submit a pending DAC-3D command preview after confirmation.",
                input_schema={"command_preview_id": "string", "confirmation_token": "string"},
                output_schema={"type": "assistant_payload"},
                risk_level="high",
                requires_confirmation=True,
                read_only=False,
                destructive=True,
                idempotent=False,
                allowed_scopes=["dac_command_submit"],
                destructive_hint=True,
            ),
            ToolDescriptor(
                name="cancel_pending_command",
                description="Cancel the current pending command preview.",
                input_schema={"command_preview_id": "string | null"},
                output_schema={"type": "object"},
                risk_level="low",
                read_only=False,
                idempotent=True,
                allowed_scopes=["dac_command_preview"],
            ),
            ToolDescriptor(
                name="read_command_history",
                description="Read recent gateway command preview/submission events.",
                input_schema={"limit": "integer"},
                output_schema={"type": "object"},
                risk_level="read_only",
                read_only=True,
                idempotent=True,
                allowed_scopes=["audit"],
            ),
        ]

    def read_dac_status(self) -> ToolInvocationResult:
        """Read runtime status without invoking command execution."""
        policy = self._tool_policy("read_dac_status")
        if not policy.allowed:
            return self._blocked_tool_result("read_dac_status", policy)
        status = self.assistant.dac3d_client.query_current_status()
        risk = self.safety_guard.classify_intent_risk("read_dac_status")
        return ToolInvocationResult(
            tool="read_dac_status",
            ok=True,
            result={"status": status, "agent_session": self.sessions.describe_session(self.session_id)},
            risk=risk.to_dict(),
            validation={"policy": policy.to_dict()},
        )

    def read_latest_result(self, sample_position: int = 0) -> ToolInvocationResult:
        """Read latest result summary."""
        policy = self._tool_policy("read_latest_result", {"sample_position": sample_position})
        if not policy.allowed:
            return self._blocked_tool_result("read_latest_result", policy)
        del sample_position
        result = self.assistant.dac3d_client.get_latest_result_summary()
        risk = self.safety_guard.classify_intent_risk("read_latest_result")
        return ToolInvocationResult(
            tool="read_latest_result",
            ok=True,
            result={"result": result, "agent_session": self.sessions.describe_session(self.session_id)},
            risk=risk.to_dict(),
            validation={"policy": policy.to_dict()},
        )

    def list_allowed_dirs(self) -> ToolInvocationResult:
        """List configured allowlist roots."""
        policy = self._tool_policy("list_allowed_dirs")
        if not policy.allowed:
            return self._blocked_tool_result("list_allowed_dirs", policy)
        return ToolInvocationResult(
            tool="list_allowed_dirs",
            ok=True,
            result={
                "allowed_dirs": self.safety_guard.path_policy.allowed_read_roots,
                "command_output_dirs": self.safety_guard.path_policy.allowed_write_roots,
            },
            risk=self.safety_guard.classify_intent_risk("list_allowed_dirs").to_dict(),
            validation={"policy": policy.to_dict()},
        )

    def preview_command(self, command_draft: str) -> dict[str, Any]:
        """Generate and remember a command preview without submitting it."""
        policy = self._tool_policy(
            "preview_command",
            {"command_draft": command_draft},
            user_text=command_draft,
        )
        if not policy.allowed:
            return self._blocked_payload("命令预览被安全策略拦截，未生成待执行命令。", "policy_denied", tool="preview_command", policy=policy)
        response = self.assistant.preview_operation_command(command_draft)
        payload = self._payload(response)
        preview = payload.get("command_preview")
        if isinstance(preview, dict):
            self._decorate_preview(preview)
            validation = self.validate_command(preview, source_text=command_draft).to_dict()
            payload["command_preview"] = preview
            payload["tool_gateway"] = {
                "backend": "dac_tool_gateway",
                "tool": "preview_command",
                "validation": validation.get("validation", {}),
                "risk": validation.get("risk", {}),
            }
            if not isinstance(payload.get("parsed_result"), dict):
                payload["parsed_result"] = {}
            payload["parsed_result"].setdefault("tool_gateway", payload["tool_gateway"])
            validation_status = validation.get("validation", {})
            if validation_status.get("can_submit"):
                pending = self.sessions.remember_pending_command(
                    self.session_id,
                    source_message=command_draft,
                    command_preview=preview,
                    ttl_seconds=self._confirmation_ttl_seconds(),
                )
                preview = pending.command_preview
                payload["command_preview"] = preview
            self.sessions.append_command_history(
                self.session_id,
                {
                    "event": "preview_command",
                    "preview_id": preview.get("gateway", {}).get("preview_id"),
                    "preview_hash": preview.get("gateway", {}).get("preview_hash"),
                    "action": preview.get("action"),
                    "lifecycle_state": preview.get("gateway", {}).get("lifecycle_state"),
                    "confirmation_expires_at": preview.get("gateway", {}).get("confirmation_expires_at"),
                    "validation": validation.get("validation", {}),
                    "risk": validation.get("risk", {}),
                },
            )
            payload["agent_session"] = self.sessions.describe_session(self.session_id)
        return payload

    def validate_command(self, command_preview: dict[str, Any], *, source_text: str = "") -> ToolInvocationResult:
        """Validate command preview schema, path policy, and safety policy."""
        tool_policy = self._tool_policy(
            "validate_command",
            {"command_preview": command_preview},
        )
        if not tool_policy.allowed:
            return self._blocked_tool_result("validate_command", tool_policy)
        preview = deepcopy(command_preview)
        self._decorate_preview(preview)
        action = str(preview.get("action") or "")
        errors: list[str] = []
        warnings = list(preview.get("warnings") or [])
        path_decision: dict[str, Any] | None = None
        path_decisions: list[dict[str, Any]] = []

        if action not in SUPPORTED_ACTIONS:
            errors.append(f"unsupported_action:{action or 'missing'}")

        missing_fields = list(preview.get("missing_fields") or [])
        missing_fields.extend(item for item in missing_required_fields(action, preview) if item not in missing_fields)
        if missing_fields:
            errors.extend(f"missing:{item}" for item in missing_fields)

        payload = dict(preview.get("payload") or {})
        path_fields = {
            "image_folder": "read",
            "input_dir": "read",
            "result_root": "read",
            "output_dir": "write",
            "command_output": "write",
            "command_path": "write",
        }
        for field_name, operation in path_fields.items():
            value = payload.get(field_name)
            if not value:
                continue
            if operation == "write":
                path = self.safety_guard.validate_command_output(str(value))
            else:
                path = self.safety_guard.validate_path(str(value))
            decision = {"field": field_name, "operation": operation, **path.to_dict()}
            path_decisions.append(decision)
            if field_name == "image_folder":
                path_decision = path.to_dict()
            if not path.allowed:
                errors.append("path_not_allowed")

        bridge_path = self._command_bridge_path()
        command_output_decision: dict[str, Any] | None = None
        if bridge_path is not None:
            output_path = self.safety_guard.validate_command_output(str(bridge_path))
            command_output_decision = {
                "field": "command_bridge",
                "operation": "write",
                **output_path.to_dict(),
            }
            path_decisions.append(command_output_decision)
            if not output_path.allowed:
                errors.append("command_output_not_allowed")

        injection = self.safety_guard.detect_prompt_injection(
            " ".join([str(source_text or ""), str(preview.get("yaml_preview") or ""), str(payload)])
        )
        if injection.detected:
            warnings.append("possible_prompt_injection")
            errors.append("prompt_injection_detected")

        risk = self.safety_guard.classify_intent_risk("validate_command", preview)
        runtime_status = dict(preview.get("runtime_status") or {})
        state = str(runtime_status.get("state") or "").lower()
        if action in {"scan", "start_online_scan", "start_offline_detection"} and state in {"running", "detecting", "scanning", "initializing"}:
            errors.append("runtime_busy")
        command_policy = self.policy_engine.evaluate_command_preview(
            preview,
            source_text=source_text,
            validation_passed=not errors,
        )
        if not command_policy.allowed:
            errors.extend(command_policy.blocking_reasons)
        if errors:
            risk.can_execute = False
            risk.blocked_by = list(dict.fromkeys([*risk.blocked_by, *errors]))

        validation = {
            "schema_valid": not any(error.startswith(("unsupported_action", "missing")) for error in errors),
            "path_allowed": path_decision["allowed"] if path_decision else True,
            "all_paths_allowed": not any(
                not bool(item.get("allowed", False)) for item in path_decisions
            ),
            "runtime_ready": "runtime_busy" not in errors,
            "requires_confirmation": bool(preview.get("safety", {}).get("needs_confirmation")),
            "can_submit": not errors and action not in READ_ONLY_ACTIONS,
            "read_only": action in READ_ONLY_ACTIONS,
            "errors": list(dict.fromkeys(errors)),
            "warnings": list(dict.fromkeys(warnings)),
            "path_decision": path_decision,
            "path_decisions": path_decisions,
            "command_output_decision": command_output_decision,
            "injection_signal": injection.to_dict(),
            "preview_id": preview.get("gateway", {}).get("preview_id"),
            "policy": command_policy.to_dict(),
            "tool_policy": tool_policy.to_dict(),
        }
        return ToolInvocationResult(
            tool="validate_command",
            ok=not errors,
            result={"command_preview": preview},
            risk=risk.to_dict(),
            validation=validation,
        )

    def submit_command(self, command_preview_id: str, confirmation_token: str) -> dict[str, Any]:
        """Submit the current pending command through the assistant safety path."""
        tool_policy = self._tool_policy(
            "submit_command",
            {
                "command_preview_id": str(command_preview_id or ""),
                "confirmation_token": str(confirmation_token or ""),
            },
            confirmed=bool(confirmation_token),
        )
        if not tool_policy.allowed:
            return self._blocked_payload(
                "提交命令被 Tool Gateway 安全策略拦截，未下发 DAC-3D 命令。",
                "policy_denied",
                tool="submit_command",
                policy=tool_policy,
            )
        pending = self.sessions.get_pending_command(self.session_id)
        if self.sessions.confirmation_token_consumed(self.session_id, command_preview_id, confirmation_token):
            return self._blocked_payload(
                "确认 token 已被使用过，疑似重复提交，未下发 DAC-3D 命令。",
                "confirmation_replay",
                tool="submit_command",
            )
        if pending is None:
            return self._blocked_payload("当前会话没有待确认的 DAC-3D 命令。", "no_pending_command")
        if self.sessions.pending_command_expired(pending):
            self.sessions.mark_pending_expired(self.session_id)
            expired_preview = deepcopy(pending.command_preview)
            self.sessions.append_command_history(
                self.session_id,
                {
                    "event": "submit_blocked",
                    "reason": "confirmation_expired",
                    "preview_id": pending.preview_id,
                    "action": pending.action,
                    "lifecycle_state": "expired",
                    "confirmation_expires_at": pending.expires_at,
                },
            )
            self.sessions.clear_pending_command(self.session_id)
            return self._blocked_payload(
                "确认已过期，未下发 DAC-3D 命令。请重新生成命令预览后再批准执行。",
                "confirmation_expired",
                expired_preview,
            )

        preview = deepcopy(pending.command_preview)
        self._decorate_preview(preview)
        preview_id = str(preview.get("gateway", {}).get("preview_id") or "")
        if command_preview_id != preview_id:
            self.sessions.mark_pending_confirmation_failed(self.session_id, "preview_id_mismatch")
            return self._blocked_payload("确认的命令与当前待执行命令不一致，未下发。", "preview_id_mismatch", preview)
        if confirmation_token != pending.confirmation_token:
            self.sessions.mark_pending_confirmation_failed(self.session_id, "invalid_confirmation")
            return self._blocked_payload("确认 token 无效，未下发 DAC-3D 命令。", "invalid_confirmation", preview)
        if not self.sessions.pending_preview_hash_matches(pending):
            self.sessions.mark_pending_confirmation_failed(self.session_id, "preview_hash_mismatch")
            return self._blocked_payload(
                "命令预览内容已变化，原确认失效，未下发 DAC-3D 命令。",
                "preview_hash_mismatch",
                preview,
            )
        self.sessions.mark_pending_confirmed(self.session_id)

        validation = self.validate_command(preview, source_text=pending.source_message)
        if not validation.ok:
            return self._blocked_payload("命令校验未通过，未下发 DAC-3D 命令。", "validation_failed", preview, validation)
        command_output_decision = self._command_bridge_path_decision()
        if command_output_decision is not None and not command_output_decision.get("allowed"):
            return self._blocked_payload(
                "命令输出路径未通过安全策略，未下发 DAC-3D 命令。",
                "command_output_not_allowed",
                preview,
                validation,
            )
        submit_policy = self.policy_engine.evaluate_command_submit(
            command_preview_id=command_preview_id,
            confirmation_token=confirmation_token,
            expected_confirmation_token=pending.confirmation_token,
            validation_passed=validation.ok,
            command_preview=preview,
        )
        if not submit_policy.allowed:
            self.sessions.mark_pending_confirmation_failed(self.session_id, "policy_denied")
            return self._blocked_payload(
                "命令提交未通过生产安全策略，未下发 DAC-3D 命令。",
                "policy_denied",
                preview,
                validation,
                policy=submit_policy,
            )

        response = self.assistant.execute_prepared_command(preview, confirmed_by_user=True)
        payload = self._payload(response)
        payload["tool_gateway"] = {
            "backend": "dac_tool_gateway",
            "tool": "submit_command",
            "preview_id": preview_id,
            "validation": validation.validation,
            "risk": self.safety_guard.classify_intent_risk("submit_command", preview).to_dict(),
            "policy": submit_policy.to_dict(),
            "lifecycle": self.sessions.get_pending_command(self.session_id).to_dict()
            if self.sessions.get_pending_command(self.session_id) is not None
            else {},
        }
        if not isinstance(payload.get("parsed_result"), dict):
            payload["parsed_result"] = {}
        payload["parsed_result"].setdefault("tool_gateway", payload["tool_gateway"])
        self.sessions.mark_pending_submitted(self.session_id)
        self.sessions.consume_confirmation_token(self.session_id, preview_id, confirmation_token)
        submitted_pending = self.sessions.get_pending_command(self.session_id)
        if submitted_pending is not None:
            payload["tool_gateway"]["lifecycle"] = submitted_pending.to_dict()
        self.sessions.clear_pending_command(self.session_id)
        self.sessions.append_command_history(
            self.session_id,
            {
                "event": "submit_command",
                "preview_id": preview_id,
                "preview_hash": pending.preview_hash,
                "action": preview.get("action"),
                "accepted": True,
                "lifecycle_state": "submitted",
            },
        )
        payload["agent_session"] = self.sessions.describe_session(self.session_id)
        return payload

    def cancel_pending_command(self, command_preview_id: str | None = None) -> ToolInvocationResult:
        """Cancel the pending command in this session."""
        policy = self._tool_policy(
            "cancel_pending_command",
            {"command_preview_id": command_preview_id},
        )
        if not policy.allowed:
            return self._blocked_tool_result("cancel_pending_command", policy)
        pending = self.sessions.get_pending_command(self.session_id)
        if pending is None:
            return ToolInvocationResult(
                tool="cancel_pending_command",
                ok=True,
                result={"cancelled": False, "reason": "no_pending_command"},
                risk=self.safety_guard.classify_intent_risk("cancel_pending_command").to_dict(),
                validation={"policy": policy.to_dict()},
            )
        preview = deepcopy(pending.command_preview)
        self._decorate_preview(preview)
        preview_id = str(preview.get("gateway", {}).get("preview_id") or "")
        if command_preview_id and command_preview_id != preview_id:
            return ToolInvocationResult(
                tool="cancel_pending_command",
                ok=False,
                result={"cancelled": False, "reason": "preview_id_mismatch", "current_preview_id": preview_id},
                risk=self.safety_guard.classify_intent_risk("cancel_pending_command").to_dict(),
                validation={"policy": policy.to_dict()},
            )
        self.sessions.mark_pending_cancelled(self.session_id)
        self.sessions.clear_pending_command(self.session_id)
        self.sessions.append_command_history(
            self.session_id,
            {
                "event": "cancel_pending_command",
                "preview_id": preview_id,
                "preview_hash": pending.preview_hash,
                "action": preview.get("action"),
                "lifecycle_state": "cancelled",
            },
        )
        return ToolInvocationResult(
            tool="cancel_pending_command",
            ok=True,
            result={"cancelled": True, "preview_id": preview_id},
            risk=self.safety_guard.classify_intent_risk("cancel_pending_command").to_dict(),
            validation={"policy": policy.to_dict()},
        )

    def read_command_history(self, limit: int = 20) -> ToolInvocationResult:
        """Read recent gateway command events."""
        policy = self._tool_policy("read_command_history", {"limit": limit})
        if not policy.allowed:
            return self._blocked_tool_result("read_command_history", policy)
        return ToolInvocationResult(
            tool="read_command_history",
            ok=True,
            result={
                "session_id": self.session_id,
                "events": self.sessions.read_command_history(self.session_id, limit=limit),
            },
            risk=self.safety_guard.classify_intent_risk("read_command_history").to_dict(),
            validation={"policy": policy.to_dict()},
        )

    def _decorate_preview(self, preview: dict[str, Any]) -> None:
        request = self.safety_guard.require_confirmation(preview)
        existing = preview.get("gateway") if isinstance(preview.get("gateway"), dict) else {}
        if not isinstance(preview.get("gateway"), dict):
            preview["gateway"] = {}
        preview["gateway"].update(
            {
                "preview_id": request.preview_id,
                "confirmation_required": request.required,
                "confirmation_token": existing.get("confirmation_token") or request.confirmation_token,
                "confirmation_message": request.message,
                "confirmation_expires_at": existing.get("confirmation_expires_at"),
                "confirmation_ttl_seconds": existing.get("confirmation_ttl_seconds"),
                "preview_hash": existing.get("preview_hash"),
                "lifecycle_state": existing.get("lifecycle_state"),
                "lifecycle_events": existing.get("lifecycle_events") or [],
                "submit_tool": "submit_command",
            }
        )

    def _payload(self, response: Any) -> dict[str, Any]:
        payload = AssistantResponsePayload.from_response(response).to_dict()
        payload["agent_session"] = self.sessions.describe_session(self.session_id)
        return payload

    def _tool_policy(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        confirmed: bool = False,
        private_context: bool = False,
        schema_valid: bool = True,
        user_text: str = "",
        retrieved_text: str = "",
    ) -> PolicyDecision:
        descriptor = self._descriptor_for(name)
        return self.policy_engine.evaluate_tool_call(
            descriptor.to_dict() if descriptor else None,
            arguments or {},
            private_context=private_context,
            schema_valid=schema_valid,
            user_text=user_text,
            retrieved_text=retrieved_text,
            confirmed=confirmed,
        )

    def _descriptor_for(self, name: str) -> ToolDescriptor | None:
        for descriptor in self.descriptors():
            if descriptor.name == name:
                return descriptor
        return None

    def _blocked_tool_result(self, tool: str, policy: PolicyDecision) -> ToolInvocationResult:
        return ToolInvocationResult(
            tool=tool,
            ok=False,
            result={
                "blocked": True,
                "reason": "policy_denied",
                "policy": policy.to_dict(),
            },
            error="policy_denied",
            risk={
                "risk_level": policy.risk_level,
                "requires_confirmation": policy.requires_confirmation,
                "can_execute": False,
                "reason": "; ".join(policy.reasons),
                "blocked_by": list(policy.blocking_reasons),
            },
            validation={"policy": policy.to_dict()},
        )

    def _blocked_payload(
        self,
        answer: str,
        reason: str,
        preview: dict[str, Any] | None = None,
        validation: ToolInvocationResult | None = None,
        *,
        tool: str = "submit_command",
        policy: PolicyDecision | None = None,
    ) -> dict[str, Any]:
        payload = {
            "intent": "operation",
            "answer": answer,
            "sources": [],
            "source_items": [],
            "command_preview": preview,
            "status_summary": (preview or {}).get("runtime_status") if isinstance(preview, dict) else None,
            "parsed_result": {
                "tool_gateway": {
                    "backend": "dac_tool_gateway",
                    "tool": tool,
                    "blocked": True,
                    "reason": reason,
                    "validation": validation.validation if validation else {},
                    "risk": validation.risk if validation else {},
                    "policy": policy.to_dict() if policy else {},
                }
            },
            "agent_session": self.sessions.describe_session(self.session_id),
        }
        self.sessions.append_command_history(
            self.session_id,
            {
                "event": "submit_blocked",
                "reason": reason,
                "tool": tool,
                "policy": policy.to_dict() if policy else {},
                "preview_id": (preview or {}).get("gateway", {}).get("preview_id") if isinstance(preview, dict) else None,
            },
        )
        return payload

    def _default_allowed_roots(self) -> list[Path]:
        config = getattr(self.assistant, "config", None)
        configured = list(getattr(config, "dac3d_allowed_dirs", ()) or [])
        if configured:
            return [Path(item) for item in configured]
        base_dir = Path(getattr(config, "base_dir", Path.cwd()))
        repo_root = base_dir.parent
        roots = [
            base_dir / ".tmp",
            repo_root / "福特科" / "pre_fusion_images",
            repo_root / "福特科" / "xxp_ui" / "runtime",
        ]
        return roots

    def _default_command_output_roots(self) -> list[Path]:
        config = getattr(self.assistant, "config", None)
        configured = str(getattr(config, "dac3d_command_output_dir", "") or "").strip()
        if configured:
            return [Path(configured)]
        roots: list[Path] = []
        bridge_path = self._command_bridge_path()
        if bridge_path is not None:
            roots.append(bridge_path.parent)
        base_dir = Path(getattr(config, "base_dir", Path.cwd()))
        repo_root = base_dir.parent
        roots.extend(
            [
                base_dir / ".tmp",
                repo_root / "福特科" / "xxp_ui" / "runtime",
            ]
        )
        unique: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root)
            if key not in seen:
                unique.append(root)
                seen.add(key)
        return unique

    def _command_bridge_path(self) -> Path | None:
        client = getattr(self.assistant, "dac3d_client", None)
        method = getattr(client, "command_bridge_path", None)
        if not callable(method):
            return None
        return method()

    def _command_bridge_path_decision(self) -> dict[str, Any] | None:
        bridge_path = self._command_bridge_path()
        if bridge_path is None:
            return None
        decision = self.safety_guard.validate_command_output(str(bridge_path))
        return {"field": "command_bridge", "operation": "write", **decision.to_dict()}

    def _confirmation_ttl_seconds(self) -> int:
        config = getattr(self.assistant, "config", None)
        return max(0, int(getattr(config, "command_confirmation_ttl_seconds", 300) or 0))
