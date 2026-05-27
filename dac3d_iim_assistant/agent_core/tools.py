"""DAC-3D Agent tool controller with typed payloads and session state."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from agent_core.schemas import AssistantResponsePayload
from agent_core.sessions import DAC3DAgentSessionStore
from agent_core.tool_gateway import DEFAULT_AGENT_TOOL_GATEWAY, ToolGateway
from machine_agent import MachineAgentService


TOKEN_BOUND_CONFIRMATION_REQUIRED = "TOKEN_BOUND_CONFIRMATION_REQUIRED"  # nosec B105
TOKEN_BOUND_CONFIRMATION_MESSAGE = (
    "Agent/CLI 直接确认执行已被安全策略阻断。真实下发必须通过 "
    "Web API 的 /api/commands/preview 与 /api/commands/confirm 完成，"
    "并校验 preview_id、preview_hash、一次性 confirmation_token、operator 和 session。"
)  # nosec B105


class DAC3DAgentToolController:
    """Bind DAC-3D assistant operations to Agent tools for one session."""

    def __init__(
        self,
        *,
        assistant: Any,
        sessions: DAC3DAgentSessionStore,
        session_id: str,
        machine_agent: MachineAgentService | None = None,
        tool_gateway: ToolGateway | None = None,
    ) -> None:
        self.assistant = assistant
        self.sessions = sessions
        self.session_id = self.sessions.normalize_session_id(session_id)
        self.machine_agent = machine_agent or MachineAgentService()
        self.tool_gateway = tool_gateway or DEFAULT_AGENT_TOOL_GATEWAY

    def handle_with_assistant(self, message: str) -> dict[str, Any]:
        """Run a message through the deterministic DAC-3D assistant router."""
        return self._invoke_tool(
            "dac3d_answer",
            {"question": message},
            lambda: self._handle_with_assistant(message),
        )

    def operation(self, instruction: str) -> dict[str, Any]:
        """Run the legacy operation parser through the ToolGateway."""
        return self._invoke_tool(
            "dac3d_operation",
            {"instruction": instruction},
            lambda: self._preview_command(instruction),
        )

    def preview_command(self, instruction: str) -> dict[str, Any]:
        """Generate and remember a command preview without submitting it."""
        return self._invoke_tool(
            "dac3d_preview_command",
            {"instruction": instruction},
            lambda: self._preview_command(instruction),
        )

    def execute_command(
        self,
        instruction: str,
        *,
        confirmed_by_user: bool = False,
    ) -> dict[str, Any]:
        """Preview command execution; block direct Agent/CLI confirmed submission."""
        return self._invoke_tool(
            "dac3d_execute_command",
            {"instruction": instruction, "confirmed_by_user": confirmed_by_user},
            lambda: self._execute_command(
                instruction,
                confirmed_by_user=confirmed_by_user,
            ),
        )

    def status(self) -> dict[str, Any]:
        """Read DAC-3D runtime status through the ToolGateway."""
        return self._invoke_tool(
            "dac3d_status",
            {},
            lambda: self._handle_with_assistant("查询当前检测状态"),
        )

    def latest_result(self, sample_position: int) -> dict[str, Any]:
        """Read a DAC-3D inspection result through the ToolGateway."""
        message = f"第{sample_position}个样品检测结果怎么样？" if sample_position > 0 else "当前检测结果怎么样？"
        return self._invoke_tool(
            "dac3d_latest_result",
            {"sample_position": sample_position},
            lambda: self._handle_with_assistant(message),
        )

    def rebuild_knowledge_base(self) -> dict[str, Any]:
        """Rebuild the local DAC-3D knowledge base."""
        return self._invoke_tool(
            "dac3d_rebuild_knowledge_base",
            {},
            lambda: self.assistant.build_knowledge_base_from_uploads(),
        )

    def machine_agent_chat(self, message: str) -> dict[str, Any]:
        """Run the industrial machine Agent with its own tool trace."""
        return self._invoke_tool(
            "machine_agent_chat",
            {"message": message},
            lambda: self._agent_payload(self.machine_agent.chat(message)),
        )

    def machine_snapshot(self) -> dict[str, Any]:
        """Return dashboard data for the industrial machine Agent."""
        return self._invoke_tool(
            "machine_snapshot",
            {},
            lambda: self._agent_payload(self.machine_agent.snapshot()),
        )

    def machine_status(self) -> dict[str, Any]:
        """Return the latest industrial equipment status."""
        return self._invoke_tool(
            "machine_status",
            {},
            lambda: self._agent_payload(self.machine_agent.get_current_machine_status()),
        )

    def machine_history(self, time_query: str = "最近30天") -> dict[str, Any]:
        """Return historical telemetry for a natural-language time range."""
        return self._invoke_tool(
            "machine_history",
            {"time_query": time_query},
            lambda: self._machine_history(time_query),
        )

    def machine_alarms(self, time_query: str = "最近30天") -> dict[str, Any]:
        """Return alarm and abnormal-event records for a time range."""
        return self._invoke_tool(
            "machine_alarms",
            {"time_query": time_query},
            lambda: self._machine_alarms(time_query),
        )

    def machine_docs(self, query: str, limit: int = 4) -> dict[str, Any]:
        """Search machine manuals, maintenance notes, and fault-code docs."""
        return self._invoke_tool(
            "machine_docs",
            {"query": query, "limit": limit},
            lambda: self._agent_payload(self.machine_agent.search_machine_docs(query, limit=limit)),
        )

    def machine_condition_summary(self, time_query: str = "最近30天") -> dict[str, Any]:
        """Summarize equipment condition over a natural-language time range."""
        return self._invoke_tool(
            "machine_condition_summary",
            {"time_query": time_query},
            lambda: self._machine_condition_summary(time_query),
        )

    def machine_abnormal_analysis(self, time_query: str = "最近30天") -> dict[str, Any]:
        """Detect explainable abnormal equipment patterns over a time range."""
        return self._invoke_tool(
            "machine_abnormal_analysis",
            {"time_query": time_query},
            lambda: self._machine_abnormal_analysis(time_query),
        )

    def _handle_with_assistant(self, message: str) -> dict[str, Any]:
        """Run a message through the deterministic DAC-3D assistant router."""
        payload = self._payload(self.assistant.handle_message(message))
        self._update_pending_state(message, payload, confirmed=False)
        return payload

    def _preview_command(self, instruction: str) -> dict[str, Any]:
        """Generate and remember a command preview without submitting it."""
        payload = self._payload(self.assistant.preview_operation_command(instruction))
        self._update_pending_state(instruction, payload, confirmed=False)
        return payload

    def _execute_command(
        self,
        instruction: str,
        *,
        confirmed_by_user: bool = False,
    ) -> dict[str, Any]:
        """Preview command execution; block direct Agent/CLI confirmed submission."""
        if confirmed_by_user and self.sessions.should_use_pending_confirmation(instruction):
            pending = self.sessions.get_pending_command(self.session_id)
            if pending is not None:
                response = self.assistant.execute_prepared_command(
                    deepcopy(pending.command_preview),
                    confirmed_by_user=False,
                )
                payload = self._payload(response)
                self._update_pending_state(instruction, payload, confirmed=False)
                return self._with_token_bound_confirmation_block(payload)

        if confirmed_by_user:
            payload = self._preview_command(instruction)
            return self._with_token_bound_confirmation_block(payload)

        payload = self._payload(
            self.assistant.execute_operation_command(
                instruction,
                confirmed_by_user=False,
            )
        )
        self._update_pending_state(instruction, payload, confirmed=False)
        return payload

    def _machine_history(self, time_query: str) -> dict[str, Any]:
        """Return historical telemetry for a natural-language time range."""
        time_range = self.machine_agent.resolve_time_range(time_query)
        return self._agent_payload(self.machine_agent.get_machine_history(time_range))

    def _machine_alarms(self, time_query: str) -> dict[str, Any]:
        """Return alarm and abnormal-event records for a time range."""
        time_range = self.machine_agent.resolve_time_range(time_query)
        return self._agent_payload(self.machine_agent.get_alarm_records(time_range))

    def _machine_condition_summary(self, time_query: str) -> dict[str, Any]:
        """Summarize equipment condition over a natural-language time range."""
        time_range = self.machine_agent.resolve_time_range(time_query)
        return self._agent_payload(self.machine_agent.summarize_machine_condition(time_range))

    def _machine_abnormal_analysis(self, time_query: str) -> dict[str, Any]:
        """Detect explainable abnormal equipment patterns over a time range."""
        time_range = self.machine_agent.resolve_time_range(time_query)
        return self._agent_payload(self.machine_agent.detect_abnormal_patterns(time_range))

    def _invoke_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        handler: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Run one Agent tool through the gateway and attach session diagnostics."""
        payload = self.tool_gateway.invoke(tool_name, arguments, handler)
        payload.setdefault("agent_session", self.sessions.describe_session(self.session_id))
        return payload

    def _payload(self, response: Any) -> dict[str, Any]:
        payload = AssistantResponsePayload.from_response(response).to_dict()
        payload["agent_session"] = self.sessions.describe_session(self.session_id)
        return payload

    def _agent_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        result["agent_session"] = self.sessions.describe_session(self.session_id)
        return result

    def _with_token_bound_confirmation_block(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Mark an Agent/CLI direct submit attempt as blocked at the tool boundary."""
        result = dict(payload)
        policy_decision = {
            "allowed": False,
            "reason": TOKEN_BOUND_CONFIRMATION_REQUIRED,
            "message": TOKEN_BOUND_CONFIRMATION_MESSAGE,
            "confirmation_flow": "api_preview_confirm_token",
            "direct_agent_submit_allowed": False,
            "requires_preview_id": True,
            "requires_preview_hash": True,
            "requires_one_time_token": True,  # nosec B105
            "requires_operator_session": True,
        }
        confirmation = dict(result.get("confirmation") or {})
        confirmation.update(
            {
                "required": True,
                "mode": "api_preview_confirm_token",
                "blocked_direct_submit": True,
                "reason": TOKEN_BOUND_CONFIRMATION_REQUIRED,
            }
        )
        parsed_result = result.get("parsed_result")
        if not isinstance(parsed_result, dict):
            parsed_result = {}
        parsed_result = dict(parsed_result)
        parsed_result["policy_decision"] = policy_decision
        parsed_result["confirmation"] = {
            key: value
            for key, value in confirmation.items()
            if key != "confirmation_token"
        }

        result["answer"] = (
            "已生成 DAC-3D 控制命令预览，但 Agent/CLI 的直接确认执行已被安全策略阻断。"
            "真实下发必须走 Web API 的 preview/confirm 一次性 token 确认链路。"
        )
        result["policy_decision"] = policy_decision
        result["confirmation"] = confirmation
        result["parsed_result"] = parsed_result
        result["agent_session"] = self.sessions.describe_session(self.session_id)
        return result

    def _update_pending_state(
        self,
        source_message: str,
        payload: dict[str, Any],
        *,
        confirmed: bool,
    ) -> None:
        command_preview = payload.get("command_preview")
        if not isinstance(command_preview, dict):
            return

        status = payload.get("status_summary")
        if confirmed and isinstance(status, dict):
            state = str(status.get("state") or "")
            if state:
                self.sessions.clear_pending_command(self.session_id)
                payload["agent_session"] = self.sessions.describe_session(self.session_id)
                return

        safety = dict(command_preview.get("safety") or {})
        missing_fields = command_preview.get("missing_fields") or []
        if safety.get("needs_confirmation") and not missing_fields:
            self.sessions.remember_pending_command(
                self.session_id,
                source_message=source_message,
                command_preview=command_preview,
            )
            payload["agent_session"] = self.sessions.describe_session(self.session_id)
