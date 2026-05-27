"""Fail-closed ToolGateway and policy decisions for DAC-3D Agent tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


TOOL_NOT_REGISTERED = "TOOL_NOT_REGISTERED"
TOOL_ARGUMENTS_INVALID = "TOOL_ARGUMENTS_INVALID"
TOOL_DIRECT_USE_DISABLED = "TOOL_DIRECT_USE_DISABLED"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Security metadata for one callable Agent tool."""

    name: str
    risk_level: str
    side_effect: str
    required_args: tuple[str, ...] = ()
    optional_args: tuple[str, ...] = ()
    read_only: bool = True
    destructive: bool = False
    idempotent: bool = True
    agent_direct_allowed: bool = True
    requires_confirmation: bool = False
    requires_token_bound_confirmation: bool = False

    @property
    def allowed_args(self) -> set[str]:
        """Return the complete allowed argument set."""
        return set(self.required_args) | set(self.optional_args)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metadata."""
        return {
            "name": self.name,
            "risk_level": self.risk_level,
            "side_effect": self.side_effect,
            "required_args": list(self.required_args),
            "optional_args": list(self.optional_args),
            "read_only": self.read_only,
            "destructive": self.destructive,
            "idempotent": self.idempotent,
            "agent_direct_allowed": self.agent_direct_allowed,
            "requires_confirmation": self.requires_confirmation,
            "requires_token_bound_confirmation": self.requires_token_bound_confirmation,
        }


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Result of evaluating one tool invocation."""

    allowed: bool
    tool_name: str
    reason: str
    message: str
    risk_level: str = "unknown"
    side_effect: str = "unknown"
    read_only: bool = False
    destructive: bool = False
    idempotent: bool = False
    requires_confirmation: bool = False
    requires_token_bound_confirmation: bool = False
    gateway: str = "ToolGateway"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible policy data."""
        return {
            "allowed": self.allowed,
            "tool_name": self.tool_name,
            "reason": self.reason,
            "message": self.message,
            "risk_level": self.risk_level,
            "side_effect": self.side_effect,
            "read_only": self.read_only,
            "destructive": self.destructive,
            "idempotent": self.idempotent,
            "requires_confirmation": self.requires_confirmation,
            "requires_token_bound_confirmation": self.requires_token_bound_confirmation,
            "gateway": self.gateway,
        }


class PolicyEngine:
    """Evaluate registered Agent tools before any handler is called."""

    def __init__(self, specs: Mapping[str, ToolSpec]) -> None:
        self.specs = dict(specs)

    def evaluate(self, tool_name: str, arguments: Mapping[str, Any]) -> PolicyDecision:
        """Return whether a tool call may proceed."""
        spec = self.specs.get(tool_name)
        if spec is None:
            return PolicyDecision(
                allowed=False,
                tool_name=tool_name,
                reason=TOOL_NOT_REGISTERED,
                message=f"Tool '{tool_name}' is not registered in the ToolGateway.",
            )

        missing = [name for name in spec.required_args if name not in arguments]
        if missing:
            return self._blocked(
                spec,
                TOOL_ARGUMENTS_INVALID,
                f"Tool '{tool_name}' is missing required arguments: {', '.join(missing)}.",
            )

        unexpected = sorted(set(arguments) - spec.allowed_args)
        if unexpected:
            return self._blocked(
                spec,
                TOOL_ARGUMENTS_INVALID,
                f"Tool '{tool_name}' received unexpected arguments: {', '.join(unexpected)}.",
            )

        if not spec.agent_direct_allowed:
            return self._blocked(
                spec,
                TOOL_DIRECT_USE_DISABLED,
                f"Tool '{tool_name}' is disabled for direct Agent use.",
            )

        return PolicyDecision(
            allowed=True,
            tool_name=tool_name,
            reason="TOOL_ALLOWED",
            message=f"Tool '{tool_name}' passed ToolGateway policy checks.",
            risk_level=spec.risk_level,
            side_effect=spec.side_effect,
            read_only=spec.read_only,
            destructive=spec.destructive,
            idempotent=spec.idempotent,
            requires_confirmation=spec.requires_confirmation,
            requires_token_bound_confirmation=spec.requires_token_bound_confirmation,
        )

    def _blocked(self, spec: ToolSpec, reason: str, message: str) -> PolicyDecision:
        """Return a blocked decision populated with the tool metadata."""
        return PolicyDecision(
            allowed=False,
            tool_name=spec.name,
            reason=reason,
            message=message,
            risk_level=spec.risk_level,
            side_effect=spec.side_effect,
            read_only=spec.read_only,
            destructive=spec.destructive,
            idempotent=spec.idempotent,
            requires_confirmation=spec.requires_confirmation,
            requires_token_bound_confirmation=spec.requires_token_bound_confirmation,
        )


class ToolGateway:
    """Central fail-closed boundary for OpenAI Agents SDK tool calls."""

    def __init__(self, specs: Mapping[str, ToolSpec]) -> None:
        self.specs = dict(specs)
        self.policy_engine = PolicyEngine(self.specs)

    def invoke(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        handler: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate policy before invoking the registered tool handler."""
        decision = self.policy_engine.evaluate(tool_name, arguments)
        if not decision.allowed:
            return self._blocked_payload(decision)

        payload = handler()
        if not isinstance(payload, dict):
            payload = {"intent": "tool", "answer": str(payload)}
        return self._attach_gateway_metadata(payload, decision)

    def registered_tool_names(self) -> tuple[str, ...]:
        """Return registered tool names in stable order."""
        return tuple(sorted(self.specs))

    def describe(self) -> dict[str, Any]:
        """Return registry metadata for diagnostics."""
        return {
            "enforced": True,
            "fail_closed": True,
            "registered_tools": list(self.registered_tool_names()),
            "tools": {name: self.specs[name].to_dict() for name in self.registered_tool_names()},
        }

    def _blocked_payload(self, decision: PolicyDecision) -> dict[str, Any]:
        """Return a standard blocked tool payload without invoking the handler."""
        decision_payload = decision.to_dict()
        gateway_payload = {"enforced": True, "decision": decision_payload}
        return {
            "intent": "tool_policy",
            "answer": decision.message,
            "sources": [],
            "source_items": [],
            "command_preview": None,
            "status_summary": None,
            "parsed_result": {
                "policy_decision": decision_payload,
                "tool_gateway": gateway_payload,
            },
            "policy_decision": decision_payload,
            "tool_gateway": gateway_payload,
        }

    def _attach_gateway_metadata(
        self,
        payload: dict[str, Any],
        decision: PolicyDecision,
    ) -> dict[str, Any]:
        """Attach policy metadata without replacing stricter downstream decisions."""
        result = dict(payload)
        decision_payload = decision.to_dict()
        gateway_payload = {"enforced": True, "decision": decision_payload}
        parsed_result = result.get("parsed_result")
        if not isinstance(parsed_result, dict):
            parsed_result = {}
        parsed_result = dict(parsed_result)
        parsed_result["tool_gateway"] = gateway_payload

        existing_policy = result.get("policy_decision")
        if not isinstance(existing_policy, dict):
            result["policy_decision"] = decision_payload
            parsed_result.setdefault("policy_decision", decision_payload)
        else:
            parsed_result.setdefault("policy_decision", existing_policy)

        result["tool_gateway"] = gateway_payload
        result["parsed_result"] = parsed_result
        return result


def build_default_agent_tool_specs() -> dict[str, ToolSpec]:
    """Return the production Agent tool registry."""
    return {
        "dac3d_answer": ToolSpec(
            name="dac3d_answer",
            risk_level="low",
            side_effect="read",
            required_args=("question",),
        ),
        "dac3d_operation": ToolSpec(
            name="dac3d_operation",
            risk_level="medium",
            side_effect="preview",
            required_args=("instruction",),
            read_only=False,
            requires_confirmation=True,
        ),
        "dac3d_preview_command": ToolSpec(
            name="dac3d_preview_command",
            risk_level="medium",
            side_effect="preview",
            required_args=("instruction",),
            read_only=False,
            requires_confirmation=True,
        ),
        "dac3d_execute_command": ToolSpec(
            name="dac3d_execute_command",
            risk_level="critical",
            side_effect="control_preview",
            required_args=("instruction",),
            optional_args=("confirmed_by_user",),
            read_only=False,
            destructive=True,
            idempotent=False,
            requires_confirmation=True,
            requires_token_bound_confirmation=True,
        ),
        "dac3d_status": ToolSpec(
            name="dac3d_status",
            risk_level="low",
            side_effect="read",
        ),
        "dac3d_latest_result": ToolSpec(
            name="dac3d_latest_result",
            risk_level="low",
            side_effect="read",
            required_args=("sample_position",),
        ),
        "dac3d_rebuild_knowledge_base": ToolSpec(
            name="dac3d_rebuild_knowledge_base",
            risk_level="high",
            side_effect="filesystem_write",
            read_only=False,
            idempotent=False,
            agent_direct_allowed=False,
        ),
        "machine_agent_chat": ToolSpec(
            name="machine_agent_chat",
            risk_level="medium",
            side_effect="read",
            required_args=("message",),
        ),
        "machine_snapshot": ToolSpec(
            name="machine_snapshot",
            risk_level="low",
            side_effect="read",
        ),
        "machine_status": ToolSpec(
            name="machine_status",
            risk_level="low",
            side_effect="read",
        ),
        "machine_history": ToolSpec(
            name="machine_history",
            risk_level="low",
            side_effect="read",
            optional_args=("time_query",),
        ),
        "machine_alarms": ToolSpec(
            name="machine_alarms",
            risk_level="medium",
            side_effect="read",
            optional_args=("time_query",),
        ),
        "machine_docs": ToolSpec(
            name="machine_docs",
            risk_level="low",
            side_effect="read",
            required_args=("query",),
            optional_args=("limit",),
        ),
        "machine_condition_summary": ToolSpec(
            name="machine_condition_summary",
            risk_level="medium",
            side_effect="read",
            optional_args=("time_query",),
        ),
        "machine_abnormal_analysis": ToolSpec(
            name="machine_abnormal_analysis",
            risk_level="medium",
            side_effect="read",
            optional_args=("time_query",),
        ),
    }


DEFAULT_AGENT_TOOL_GATEWAY = ToolGateway(build_default_agent_tool_specs())
