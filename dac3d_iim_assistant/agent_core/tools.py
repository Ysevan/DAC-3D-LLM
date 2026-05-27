"""DAC-3D Agent tool controller with typed payloads and session state."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent_core.schemas import AssistantResponsePayload
from agent_core.sessions import DAC3DAgentSessionStore
from machine_agent import MachineAgentService
from tool_gateway import DAC3DToolGateway


class DAC3DAgentToolController:
    """Bind DAC-3D assistant operations to Agent tools for one session."""

    def __init__(
        self,
        *,
        assistant: Any,
        sessions: DAC3DAgentSessionStore,
        session_id: str,
        machine_agent: MachineAgentService | None = None,
    ) -> None:
        self.assistant = assistant
        self.sessions = sessions
        self.session_id = self.sessions.normalize_session_id(session_id)
        self.machine_agent = machine_agent or MachineAgentService()
        self.gateway = DAC3DToolGateway(
            assistant=assistant,
            sessions=sessions,
            session_id=self.session_id,
        )

    def handle_with_assistant(self, message: str) -> dict[str, Any]:
        """Run a message through the deterministic DAC-3D assistant router."""
        payload = self._payload(self.assistant.handle_message(message))
        self._update_pending_state(message, payload, confirmed=False)
        return payload

    def preview_command(self, instruction: str) -> dict[str, Any]:
        """Generate and remember a command preview without submitting it."""
        return self.gateway.preview_command(instruction)

    def execute_command(
        self,
        instruction: str,
        *,
        confirmed_by_user: bool = False,
    ) -> dict[str, Any]:
        """Execute a new command or the latest pending command in this session."""
        if confirmed_by_user and self.sessions.should_use_pending_confirmation(instruction):
            pending = self.sessions.get_pending_command(self.session_id)
            if pending is not None:
                preview = deepcopy(pending.command_preview)
                gateway = dict(preview.get("gateway") or {})
                return self.gateway.submit_command(
                    str(gateway.get("preview_id") or ""),
                    str(gateway.get("confirmation_token") or ""),
                )

        preview_payload = self.gateway.preview_command(instruction)
        preview = preview_payload.get("command_preview")
        if confirmed_by_user and isinstance(preview, dict) and not preview.get("missing_fields"):
            gateway = dict(preview.get("gateway") or {})
            return self.gateway.submit_command(
                str(gateway.get("preview_id") or ""),
                str(gateway.get("confirmation_token") or ""),
            )
        if isinstance(preview, dict) and dict(preview.get("safety") or {}).get("needs_confirmation"):
            preview_payload["answer"] = (
                "已生成 DAC-3D 控制命令，但该命令需要用户明确确认后才会下发。"
                "如果确认执行，请明确说明“确认执行”或“立即开始”。"
            )
        return preview_payload

    def read_dac_status(self) -> dict[str, Any]:
        """Read current DAC status through the Tool Gateway."""
        payload = self.gateway.read_dac_status().to_dict()
        return {
            "intent": "status",
            "answer": self._status_answer(payload["result"]["status"]),
            "sources": [],
            "source_items": [],
            "command_preview": None,
            "status_summary": payload["result"]["status"],
            "parsed_result": {"tool_gateway": payload},
            "agent_session": self.sessions.describe_session(self.session_id),
        }

    def read_latest_result(self, sample_position: int = 0) -> dict[str, Any]:
        """Read latest DAC result through the Tool Gateway."""
        payload = self.gateway.read_latest_result(sample_position=sample_position).to_dict()
        return {
            "intent": "interpretation",
            "answer": "已通过 Tool Gateway 读取 DAC-3D 最近检测结果。",
            "sources": [],
            "source_items": [],
            "command_preview": None,
            "status_summary": None,
            "parsed_result": payload["result"].get("result"),
            "tool_gateway": payload,
            "agent_session": self.sessions.describe_session(self.session_id),
        }

    def list_allowed_dirs(self) -> dict[str, Any]:
        """List Tool Gateway path allowlist roots."""
        return self.gateway.list_allowed_dirs().to_dict()

    def validate_command(self, command_preview: dict[str, Any]) -> dict[str, Any]:
        """Validate a command preview through the Tool Gateway."""
        return self.gateway.validate_command(command_preview).to_dict()

    def cancel_pending_command(self, command_preview_id: str = "") -> dict[str, Any]:
        """Cancel the pending command through the Tool Gateway."""
        return self.gateway.cancel_pending_command(command_preview_id or None).to_dict()

    def read_command_history(self, limit: int = 20) -> dict[str, Any]:
        """Read Tool Gateway command history."""
        return self.gateway.read_command_history(limit=limit).to_dict()

    def tool_gateway_manifest(self) -> dict[str, Any]:
        """Return Tool Gateway descriptors."""
        return self.gateway.describe()

    def rebuild_knowledge_base(self) -> dict[str, Any]:
        """Rebuild the local DAC-3D knowledge base."""
        return self.assistant.build_knowledge_base_from_uploads()

    def machine_agent_chat(self, message: str) -> dict[str, Any]:
        """Run the industrial machine Agent with its own tool trace."""
        payload = self.machine_agent.chat(message)
        return self._agent_payload(payload)

    def machine_snapshot(self) -> dict[str, Any]:
        """Return dashboard data for the industrial machine Agent."""
        return self._agent_payload(self.machine_agent.snapshot())

    def machine_status(self) -> dict[str, Any]:
        """Return the latest industrial equipment status."""
        return self._agent_payload(self.machine_agent.get_current_machine_status())

    def machine_history(self, time_query: str = "最近30天") -> dict[str, Any]:
        """Return historical telemetry for a natural-language time range."""
        time_range = self.machine_agent.resolve_time_range(time_query)
        return self._agent_payload(self.machine_agent.get_machine_history(time_range))

    def machine_alarms(self, time_query: str = "最近30天") -> dict[str, Any]:
        """Return alarm and abnormal-event records for a time range."""
        time_range = self.machine_agent.resolve_time_range(time_query)
        return self._agent_payload(self.machine_agent.get_alarm_records(time_range))

    def machine_docs(self, query: str, limit: int = 4) -> dict[str, Any]:
        """Search machine manuals, maintenance notes, and fault-code docs."""
        return self._agent_payload(self.machine_agent.search_machine_docs(query, limit=limit))

    def machine_condition_summary(self, time_query: str = "最近30天") -> dict[str, Any]:
        """Summarize equipment condition over a natural-language time range."""
        time_range = self.machine_agent.resolve_time_range(time_query)
        return self._agent_payload(self.machine_agent.summarize_machine_condition(time_range))

    def machine_abnormal_analysis(self, time_query: str = "最近30天") -> dict[str, Any]:
        """Detect explainable abnormal equipment patterns over a time range."""
        time_range = self.machine_agent.resolve_time_range(time_query)
        return self._agent_payload(self.machine_agent.detect_abnormal_patterns(time_range))

    def _payload(self, response: Any) -> dict[str, Any]:
        payload = AssistantResponsePayload.from_response(response).to_dict()
        payload["agent_session"] = self.sessions.describe_session(self.session_id)
        return payload

    def _agent_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
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

    def _status_answer(self, status: dict[str, Any]) -> str:
        state = status.get("state", "unknown")
        progress = status.get("progress", "unknown")
        message = status.get("message", "无状态消息")
        return f"当前 DAC-3D 状态为 {state}，进度 {progress}%，最新消息: {message}"
