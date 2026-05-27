"""OpenAI Agents SDK runtime for DAC-3D assistant workflows."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from agent_core import (
    AGENT_INSTRUCTIONS,
    AGENT_TOOL_NAMES,
    AssistantResponsePayload,
    DAC3DAgentSessionStore,
    DAC3DAgentToolController,
    DEFAULT_AGENT_TOOL_GATEWAY,
)
from app import AssistantResponse, DAC3DAssistant
from config import AppConfig
from memory import ConversationMemoryStore


VALID_AGENT_API_TYPES = {"auto", "responses", "chat_completions"}
AGENT_OUTPUT_PARSE_ERROR = "agent_output_parse_error"
LOCAL_VALIDATION_MODEL_NAME = "local-validation"


def response_to_payload(response: AssistantResponse) -> dict[str, Any]:
    """Convert the existing assistant response to a tool-friendly payload."""
    return AssistantResponsePayload.from_response(response).to_dict()


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from the Agent final output."""
    raw = str(text or "").strip()
    if not raw:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _looks_like_local_validation_model(model_name: str) -> bool:
    """Return whether the Agent should use the local validation model."""
    normalized = str(model_name or "").strip().lower()
    return normalized in {LOCAL_VALIDATION_MODEL_NAME, "local_validation", "validation"}


class LocalValidationAgentModel:
    """Deterministic Agents SDK model for local Chrome validation."""

    def __init__(self) -> None:
        self.calls = 0
        self.tool_name = "dac3d_answer"
        self.arguments: dict[str, Any] = {"question": ""}
        self.final_payload: dict[str, Any] = {
            "answer": "我已通过本地 Agent 验证模型处理请求。",
            "structured_data": {
                "intent": "agent_validation",
                "tool_calls": [{"name": "dac3d_answer", "purpose": "本地验证默认问答"}],
            },
        }

    async def get_response(
        self,
        system_instructions,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing,
        *,
        previous_response_id,
        conversation_id,
        prompt,
    ):
        del (
            system_instructions,
            model_settings,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
            conversation_id,
            prompt,
        )
        from agents import ModelResponse
        from agents.usage import Usage
        from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
        from openai.types.responses.response_output_message import ResponseOutputMessage
        from openai.types.responses.response_output_text import ResponseOutputText

        self.calls += 1
        if self.calls == 1:
            message = self._input_text(input)
            self.tool_name, self.arguments, self.final_payload = self._plan(message)
            tool_names = {tool.name for tool in tools}
            if self.tool_name not in tool_names:
                self.tool_name = "dac3d_answer"
                self.arguments = {"question": message}
            return ModelResponse(
                output=[
                    ResponseFunctionToolCall(
                        arguments=json.dumps(self.arguments, ensure_ascii=False),
                        call_id=f"local_{self.tool_name}",
                        name=self.tool_name,
                        type="function_call",
                        status="completed",
                    )
                ],
                usage=Usage(),
                response_id=None,
            )

        return ModelResponse(
            output=[
                ResponseOutputMessage(
                    id="local_validation_message",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        ResponseOutputText(
                            annotations=[],
                            text=json.dumps(self.final_payload, ensure_ascii=False),
                            type="output_text",
                        )
                    ],
                )
            ],
            usage=Usage(),
            response_id=None,
        )

    async def stream_response(self, *args, **kwargs):
        del args, kwargs
        if False:
            yield None

    def _input_text(self, input_value: Any) -> str:
        if isinstance(input_value, str):
            return input_value
        try:
            return json.dumps(input_value, ensure_ascii=False)
        except TypeError:
            return str(input_value)

    def _plan(self, message: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
        lowered = message.lower()
        if any(marker in message for marker in ("温度报警", "报警变多", "设备状态")):
            return (
                "machine_agent_chat",
                {"message": message},
                {
                    "answer": "最近温度报警增多主要与连续高负载和散热效率下降有关。建议优先检查冷却水路、风扇和滤网。",
                    "structured_data": {
                        "intent": "machine_alarm_analysis",
                        "tool_calls": [
                            {"name": "machine_agent_chat", "purpose": "读取历史、报警和异常模式"}
                        ],
                        "findings": [
                            {"metric": "temperature_high", "value": "最近30天 6 次", "meaning": "温度报警是最高频异常"},
                            {"metric": "temperature_delta", "value": "+6.5C", "meaning": "异常采样温度明显高于正常采样"},
                        ],
                        "recommendations": ["检查冷却水路", "检查风扇和滤网", "复核连续高负载工况"],
                        "limits": ["本回答来自本地验证模型，用于 Chrome 功能验证，不访问外部 LLM"],
                    },
                },
            )
        if any(marker in message for marker in ("反光", "曝光", "怎么办", "guidance")) or "reflect" in lowered:
            return (
                "dac3d_answer",
                {"question": message},
                {
                    "answer": "样品表面反光很强时，建议先检查三相机原图是否过曝，再逐项降低曝光或光源强度，调整后复扫确认。",
                    "structured_data": {
                        "intent": "guidance",
                        "tool_calls": [{"name": "dac3d_answer", "purpose": "查询 DAC-3D 操作建议"}],
                        "recommendations": ["检查原图过曝", "降低曝光或光源强度", "复扫确认融合图和检测图"],
                    },
                },
            )
        if any(marker in message for marker in ("停止检测", "停止", "stop")):
            return (
                "dac3d_execute_command",
                {"instruction": "停止检测", "confirmed_by_user": True},
                {
                    "answer": "已生成停止检测命令预览，但 Agent/CLI 直接下发已被安全策略阻断。请通过 Web API preview/confirm 一次性 token 链路确认。",
                    "structured_data": {
                        "intent": "operation_preview",
                        "tool_calls": [{"name": "dac3d_execute_command", "purpose": "下发停止检测命令"}],
                        "command": {"action": "stop_detection", "payload": {"func": "Stop"}},
                        "policy_decision": {
                            "allowed": False,
                            "reason": "TOKEN_BOUND_CONFIRMATION_REQUIRED",
                        },
                        "requires_confirmation": True,
                    },
                },
            )
        if any(marker in message for marker in ("执行扫描", "开始扫描", "立即开始", "确认执行")):
            return (
                "dac3d_execute_command",
                {"instruction": "执行扫描", "confirmed_by_user": True},
                {
                    "answer": "已生成在线扫描命令预览，但 Agent/CLI 直接下发已被安全策略阻断。请通过 Web API preview/confirm 一次性 token 链路确认。",
                    "structured_data": {
                        "intent": "operation_preview",
                        "tool_calls": [{"name": "dac3d_execute_command", "purpose": "提交在线扫描命令"}],
                        "command": {"action": "start_online_scan", "payload": {"func": "Scan", "total_positions": 144}},
                        "policy_decision": {
                            "allowed": False,
                            "reason": "TOKEN_BOUND_CONFIRMATION_REQUIRED",
                        },
                        "requires_confirmation": True,
                    },
                },
            )
        if any(marker in message for marker in ("扫描", "scan", "10mm")):
            return (
                "dac3d_preview_command",
                {"instruction": message},
                {
                    "answer": "已生成扫描命令预览，执行前需要你确认。",
                    "structured_data": {
                        "intent": "operation_preview",
                        "tool_calls": [{"name": "dac3d_preview_command", "purpose": "生成扫描命令预览"}],
                        "command_preview": {
                            "action": "scan",
                            "scan_area_mm": {"width": 10, "height": 10},
                            "safety": {"needs_confirmation": True},
                        },
                        "requires_confirmation": True,
                    },
                },
            )
        if any(marker in message for marker in ("结果", "样品", "result")):
            return (
                "dac3d_latest_result",
                {"sample_position": 0},
                {
                    "answer": "已读取最近一次 DAC-3D 检测结果。当前示例结果存在 scratch 类缺陷，建议复核检测图和 CSV 明细。",
                    "structured_data": {
                        "intent": "interpretation",
                        "tool_calls": [{"name": "dac3d_latest_result", "purpose": "读取最近检测结果"}],
                        "result_summary": {"defect_type": "scratch", "recommendation": "复核"},
                    },
                },
            )
        if any(marker in message for marker in ("状态", "进度", "status")):
            return (
                "dac3d_status",
                {},
                {
                    "answer": "当前 DAC-3D 处于空闲/停止状态，进度 0%。",
                    "structured_data": {
                        "intent": "status",
                        "tool_calls": [{"name": "dac3d_status", "purpose": "读取当前检测状态"}],
                        "status_summary": {"state": "stopped", "progress": 0, "message": "当前没有正在执行的检测任务"},
                    },
                },
            )
        return (
            "dac3d_answer",
            {"question": message},
            {
                "answer": "我已通过 DAC-3D Agent 查询本地知识库并生成回复。",
                "structured_data": {
                    "intent": "query",
                    "tool_calls": [{"name": "dac3d_answer", "purpose": "查询 DAC-3D 知识库"}],
                },
            },
        )


@dataclass(slots=True)
class DAC3DAgentRuntime:
    """Build and run the DAC-3D assistant as an OpenAI Agents SDK agent."""

    assistant: DAC3DAssistant
    config: AppConfig
    sessions: DAC3DAgentSessionStore | None = None

    def __post_init__(self) -> None:
        if self.sessions is None:
            self.sessions = DAC3DAgentSessionStore(base_dir=self.config.base_dir)

    @classmethod
    def create(
        cls,
        config: AppConfig | None = None,
        *,
        rebuild_kb: bool = False,
    ) -> "DAC3DAgentRuntime":
        """Create the underlying assistant and wrap it as an agent runtime."""
        assistant = DAC3DAssistant.create(config=config, rebuild_kb=rebuild_kb)
        return cls(assistant=assistant, config=assistant.config)

    def handle_with_assistant(self, message: str) -> dict[str, Any]:
        """Run a message through the deterministic DAC-3D assistant router."""
        return self.tool_controller().handle_with_assistant(message)

    def preview_command(self, instruction: str, *, session_id: str = "default") -> dict[str, Any]:
        """Generate a DAC-3D command preview without submitting it."""
        return self.tool_controller(session_id).preview_command(instruction)

    def execute_command(
        self,
        instruction: str,
        *,
        confirmed_by_user: bool = False,
        session_id: str = "default",
    ) -> dict[str, Any]:
        """Generate a DAC-3D command preview and block direct Agent/CLI confirmed submit."""
        return self.tool_controller(session_id).execute_command(
            instruction,
            confirmed_by_user=confirmed_by_user,
        )

    def run_chat_payload(self, message: str, *, session_id: str = "default") -> dict[str, Any]:
        """Run the unified Agent entrypoint and return a UI-friendly payload."""
        answer = self.run_sync(message, session_id=session_id)
        return self._payload_from_agent_output(answer, session_id=session_id)

    def run_text(self, message: str, *, session_id: str = "default") -> str:
        """Return only the final Agent text for CLI entrypoints."""
        return str(self.run_chat_payload(message, session_id=session_id).get("answer") or "")

    def run_local_tool_plan(self, message: str, *, session_id: str = "default") -> dict[str, Any]:
        """Explicit local diagnostic planner; production chat enters the LLM first."""
        tools = self.tool_controller(session_id)
        if self.sessions is None:
            raise RuntimeError("Agent session store is not initialized.")
        if self.sessions.should_use_pending_confirmation(message):
            payload = tools.execute_command(message, confirmed_by_user=True)
        else:
            payload = tools.handle_with_assistant(message)

        payload = dict(payload)
        payload.setdefault("sources", [])
        payload.setdefault("source_items", [])
        payload.setdefault("command_preview", None)
        payload.setdefault("status_summary", None)
        payload.setdefault("parsed_result", None)
        payload["agent_mode"] = "local-tool-router"
        return payload

    def _payload_from_agent_output(self, output: str, *, session_id: str) -> dict[str, Any]:
        """Convert the LLM-authored final output into the UI payload contract."""
        session_snapshot = self.sessions.describe_session(session_id) if self.sessions else {}
        raw_output = str(output or "").strip()
        parsed = _parse_json_object(raw_output)
        if parsed is None:
            return {
                "intent": "agent",
                "answer": raw_output,
                "sources": [],
                "source_items": [],
                "command_preview": None,
                "status_summary": None,
                "parsed_result": {
                    "type": AGENT_OUTPUT_PARSE_ERROR,
                    "raw_output": raw_output,
                    "expected_format": "JSON object with answer and structured_data",
                },
                "agent_session": session_snapshot,
                "agent_mode": "openai-agents-sdk",
            }

        structured = parsed.get("structured_data")
        if not isinstance(structured, dict):
            structured = {}
        structured = dict(structured)
        structured.setdefault("agent_mode", "openai-agents-sdk")
        structured.setdefault("agent_session", session_snapshot)
        command_preview = structured.get("command_preview") or structured.get("command")
        status_summary = structured.get("status_summary")
        if not isinstance(command_preview, dict):
            command_preview = None
        if not isinstance(status_summary, dict):
            status_summary = None
        return {
            "intent": str(structured.get("intent") or parsed.get("intent") or "agent"),
            "answer": str(parsed.get("answer") or raw_output),
            "sources": list(parsed.get("sources") or structured.get("sources") or []),
            "source_items": list(parsed.get("source_items") or structured.get("source_items") or []),
            "command_preview": command_preview,
            "status_summary": status_summary,
            "parsed_result": structured,
            "agent_session": session_snapshot,
            "agent_mode": "openai-agents-sdk",
        }

    def tool_controller(self, session_id: str = "default") -> DAC3DAgentToolController:
        """Return a DAC-3D tool controller bound to one Agent session."""
        if self.sessions is None:
            raise RuntimeError("Agent session store is not initialized.")
        return DAC3DAgentToolController(
            assistant=self.assistant,
            sessions=self.sessions,
            session_id=session_id,
        )

    def describe(self) -> dict[str, Any]:
        """Return a stable description of this Agent project."""
        resolved_api_type = self.resolved_agent_api_type()
        local_validation = _looks_like_local_validation_model(self.config.agent_model_name)
        return {
            "name": "DAC-3D Inspection Agent",
            "sdk": "openai-agents",
            "model": self.config.agent_model_name or "SDK default model",
            "model_provider": {
                "api_base_url": self.config.agent_api_base_url or "SDK default endpoint",
                "api_key_configured": bool(self.config.agent_api_key),
                "api_type": self.config.agent_api_type,
                "resolved_api_type": resolved_api_type,
                "openai_compatible": bool(self.config.agent_api_base_url),
                "local_validation": local_validation,
            },
            "max_turns": self.config.agent_max_turns,
            "tracing_disabled": self.config.agent_tracing_disabled,
            "tools": list(AGENT_TOOL_NAMES),
            "underlying_runtime": "DAC3DAssistant",
            "capability_runtimes": ["DAC3DAssistant", "MachineAgentService"],
            "mock_mode": self.config.mock_mode,
            "session_memory": {
                "sdk_session_store": "SQLiteSession",
                "session_db_path": str(self.sessions.session_db_path) if self.sessions else "",
                "default_session": (
                    self.sessions.describe_session("default") if self.sessions else {}
                ),
            },
            "conversation_memory": {
                "enabled": self.config.memory_enabled,
                "backend": "json",
                "path": str(self.config.conversation_memory_dir),
                "layers": [
                    "short_term_history",
                    "session_recent_json",
                    "session_summary",
                    "long_term_json_search",
                ],
            },
            "control": {
                "preview_tool": "dac3d_preview_command",
                "execute_tool": "dac3d_execute_command",
                "confirmation_required_for_risky_commands": True,
                "direct_agent_submit_allowed": False,
                "tool_gateway_enforced": True,
                "confirmation_flow": "api_preview_confirm_token",
                "confirmation_requirements": [
                    "preview_id",
                    "preview_hash",
                    "one_time_confirmation_token",
                    "operator_id",
                    "session_id",
                ],
                "bridge_modes": ["embedded", "command_file_bridge", "mock"],
            },
            "tool_gateway": DEFAULT_AGENT_TOOL_GATEWAY.describe(),
        }

    def resolved_agent_api_type(self) -> str:
        """Resolve the Agent model API mode for OpenAI or OpenAI-compatible providers."""
        api_type = (self.config.agent_api_type or "auto").strip().lower()
        if api_type not in VALID_AGENT_API_TYPES:
            raise ValueError(
                "DAC3D_AGENT_API_TYPE must be one of: auto, responses, chat_completions."
            )
        if api_type != "auto":
            return api_type
        if self.config.agent_api_base_url:
            return "chat_completions"
        return "responses"

    def build_model_provider(self) -> Any:
        """Build an Agents SDK provider for OpenAI or third-party OpenAI-compatible models."""
        try:
            from openai import AsyncOpenAI
            from agents.models.multi_provider import MultiProvider
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "OpenAI Agents SDK is not installed. Install it with `pip install openai-agents`."
            ) from exc

        api_type = self.resolved_agent_api_type()
        use_responses = api_type == "responses"
        openai_client = AsyncOpenAI(
            api_key=self.config.agent_api_key or None,
            base_url=self.config.agent_api_base_url or None,
            timeout=self.config.timeout_seconds,
        )
        return MultiProvider(
            openai_client=openai_client,
            openai_use_responses=use_responses,
            openai_prefix_mode="model_id" if self.config.agent_api_base_url else "alias",
            unknown_prefix_mode="model_id" if self.config.agent_api_base_url else "error",
        )

    def build_run_config(self) -> Any:
        """Build the Agents SDK run config with DAC-3D model-provider settings."""
        try:
            from agents.run import RunConfig
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "OpenAI Agents SDK is not installed. Install it with `pip install openai-agents`."
            ) from exc

        if _looks_like_local_validation_model(self.config.agent_model_name):
            return RunConfig(
                workflow_name="DAC-3D Agent",
                tracing_disabled=self.config.agent_tracing_disabled,
            )

        return RunConfig(
            workflow_name="DAC-3D Agent",
            tracing_disabled=self.config.agent_tracing_disabled,
            model_provider=self.build_model_provider(),
        )

    def build_agent(self, *, session_id: str = "default") -> Any:
        """Create an OpenAI Agents SDK Agent with DAC-3D tools."""
        try:
            from agents import Agent, Model, function_tool
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "OpenAI Agents SDK is not installed. Install it with `pip install openai-agents`."
            ) from exc

        tools = self.tool_controller(session_id)

        @function_tool(
            name_override="dac3d_answer",
            description_override=(
                "Answer DAC-3D document, parameter, workflow, troubleshooting, "
                "and operator guidance questions using the local DAC-3D assistant."
            ),
        )
        def dac3d_answer(question: str) -> dict[str, Any]:
            """Answer a DAC-3D knowledge or guidance question."""
            return tools.handle_with_assistant(question)

        @function_tool(
            name_override="dac3d_operation",
            description_override=(
                "Legacy DAC-3D operation parser. It returns a structured command preview "
                "without submitting the command; use dac3d_execute_command for control."
            ),
        )
        def dac3d_operation(instruction: str) -> dict[str, Any]:
            """Preview a DAC-3D operation request through the compatibility tool name."""
            return tools.operation(instruction)

        @function_tool(
            name_override="dac3d_preview_command",
            description_override=(
                "Parse a natural-language DAC-3D control request into a structured command "
                "preview with runtime status and safety information. This never submits the command."
            ),
        )
        def dac3d_preview_command(instruction: str) -> dict[str, Any]:
            """Preview a DAC-3D control command without submitting it."""
            return tools.preview_command(instruction)

        @function_tool(
            name_override="dac3d_execute_command",
            description_override=(
                "Parse a DAC-3D control request and return a command preview plus security "
                "decision. Agent/CLI direct submission is disabled: confirmed_by_user=true "
                "does not submit to the runtime bridge, and high-risk execution must go "
                "through the Web API /api/commands/preview and /api/commands/confirm "
                "token-bound flow."
            ),
        )
        def dac3d_execute_command(
            instruction: str,
            confirmed_by_user: bool = False,
        ) -> dict[str, Any]:
            """Return a DAC-3D control preview; direct Agent/CLI submit is blocked."""
            return tools.execute_command(
                instruction,
                confirmed_by_user=confirmed_by_user,
            )

        @function_tool(
            name_override="dac3d_status",
            description_override="Read the current DAC-3D runtime status and progress.",
        )
        def dac3d_status() -> dict[str, Any]:
            """Read current DAC-3D runtime status."""
            return tools.status()

        @function_tool(
            name_override="dac3d_latest_result",
            description_override=(
                "Read and interpret the latest DAC-3D inspection result. Pass sample_position=0 "
                "for the latest/all recorded results, or a 1-based sample position."
            ),
        )
        def dac3d_latest_result(sample_position: int) -> dict[str, Any]:
            """Read a DAC-3D inspection result summary."""
            return tools.latest_result(sample_position)

        @function_tool(
            name_override="dac3d_rebuild_knowledge_base",
            description_override="Rebuild the local DAC-3D knowledge base from current documents.",
        )
        def dac3d_rebuild_knowledge_base() -> dict[str, Any]:
            """Rebuild the local DAC-3D knowledge base."""
            return tools.rebuild_knowledge_base()

        @function_tool(
            name_override="machine_agent_chat",
            description_override=(
                "Answer industrial equipment status, history, alarm, fault-code, maintenance, "
                "and abnormal-pattern questions with a tool-call trace."
            ),
        )
        def machine_agent_chat(message: str) -> dict[str, Any]:
            """Run the industrial machine-management Agent."""
            return tools.machine_agent_chat(message)

        @function_tool(
            name_override="machine_snapshot",
            description_override="Return the machine Agent dashboard snapshot.",
        )
        def machine_snapshot() -> dict[str, Any]:
            """Return current machine dashboard data."""
            return tools.machine_snapshot()

        @function_tool(
            name_override="machine_status",
            description_override="Read current industrial equipment status and active alarms.",
        )
        def machine_status() -> dict[str, Any]:
            """Return current machine status."""
            return tools.machine_status()

        @function_tool(
            name_override="machine_history",
            description_override="Read equipment telemetry history for a Chinese time range such as 最近30天 or 上个月.",
        )
        def machine_history(time_query: str = "最近30天") -> dict[str, Any]:
            """Return machine telemetry history."""
            return tools.machine_history(time_query)

        @function_tool(
            name_override="machine_alarms",
            description_override="Read alarm and abnormal-event records for a Chinese time range.",
        )
        def machine_alarms(time_query: str = "最近30天") -> dict[str, Any]:
            """Return machine alarm records."""
            return tools.machine_alarms(time_query)

        @function_tool(
            name_override="machine_docs",
            description_override="Search machine manuals, maintenance notes, and fault-code documentation.",
        )
        def machine_docs(query: str, limit: int = 4) -> dict[str, Any]:
            """Search machine documents."""
            return tools.machine_docs(query, limit=limit)

        @function_tool(
            name_override="machine_condition_summary",
            description_override="Summarize equipment operating condition over a Chinese time range.",
        )
        def machine_condition_summary(time_query: str = "最近30天") -> dict[str, Any]:
            """Summarize machine condition."""
            return tools.machine_condition_summary(time_query)

        @function_tool(
            name_override="machine_abnormal_analysis",
            description_override="Analyze high-frequency alarms and telemetry differences over a Chinese time range.",
        )
        def machine_abnormal_analysis(time_query: str = "最近30天") -> dict[str, Any]:
            """Analyze abnormal machine patterns."""
            return tools.machine_abnormal_analysis(time_query)

        model: Any = self.config.agent_model_name or None
        if _looks_like_local_validation_model(self.config.agent_model_name):
            class RuntimeLocalValidationAgentModel(LocalValidationAgentModel, Model):
                pass

            model = RuntimeLocalValidationAgentModel()

        return Agent(
            name="DAC-3D Inspection Agent",
            instructions=AGENT_INSTRUCTIONS,
            tools=[
                dac3d_answer,
                dac3d_operation,
                dac3d_preview_command,
                dac3d_execute_command,
                dac3d_status,
                dac3d_latest_result,
                dac3d_rebuild_knowledge_base,
                machine_agent_chat,
                machine_snapshot,
                machine_status,
                machine_history,
                machine_alarms,
                machine_docs,
                machine_condition_summary,
                machine_abnormal_analysis,
            ],
            model=model,
        )

    def run_sync(self, message: str, *, session_id: str = "default") -> str:
        """Run a user message through the OpenAI Agents SDK."""
        try:
            from agents import Runner
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "OpenAI Agents SDK is not installed. Install it with `pip install openai-agents`."
            ) from exc

        try:
            sdk_session = self.sessions.get_sdk_session(session_id) if self.sessions else None
            result = Runner.run_sync(
                self.build_agent(session_id=session_id),
                message,
                max_turns=self.config.agent_max_turns,
                run_config=self.build_run_config(),
                session=sdk_session,
            )
        except Exception as exc:
            if "Missing credentials" in str(exc):
                raise RuntimeError(
                    "OpenAI Agents SDK credentials are missing. Set DAC3D_AGENT_API_KEY "
                    "or OPENAI_API_KEY before running `python app.py --agent ...`."
                ) from exc
            raise
        return str(result.final_output)


@dataclass(slots=True)
class DAC3DAgentChatAdapter:
    """Expose the OpenAI Agents SDK runtime through the existing chat app contract."""

    runtime: DAC3DAgentRuntime
    memory_store: ConversationMemoryStore | None = None

    def __post_init__(self) -> None:
        if self.memory_store is None and self.config.memory_enabled:
            self.memory_store = ConversationMemoryStore.from_config(self.config)
            self.memory_store.ensure_directories()

    @property
    def config(self) -> AppConfig:
        """Return the underlying app config expected by the web UI."""
        return self.runtime.config

    def handle_message(
        self,
        message: str,
        history: Sequence[tuple[str, str]] | None = None,
        *,
        session_id: str = "web",
    ) -> AssistantResponse:
        """Run web/Gradio chat messages through the Agent runtime."""
        agent_input, memory_hits = self._format_agent_input(
            message,
            history,
            session_id=session_id,
        )
        try:
            payload = self.runtime.run_chat_payload(agent_input, session_id=session_id)
        except Exception as exc:
            return AssistantResponse(intent="agent", answer=f"Agent run failed: {exc}")
        self._attach_memory_metadata(payload, memory_hits)
        response = self._response_from_payload(payload)
        self._remember_turn(session_id, message, response)
        return response

    def stream_message(
        self,
        message: str,
        history: Sequence[tuple[str, str]] | None = None,
        *,
        session_id: str = "web",
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Provide a buffered streaming-compatible surface for Agent responses."""
        response = self.handle_message(message, history, session_id=session_id)
        yield "meta", response.to_ui_payload() | {"answer": ""}
        yield "status", {"stage": "agent", "label": "Agent 执行中"}
        for index in range(0, len(response.answer), 12):
            yield "delta", {"chunk": response.answer[index : index + 12]}
        yield "done", response.to_ui_payload()

    def runtime_summary(self) -> dict[str, Any]:
        """Return runtime diagnostics with Agent metadata."""
        summary = self.runtime.assistant.runtime_summary()
        summary["chat_runtime"] = "openai-agents"
        summary["agent"] = self.runtime.describe()
        summary["memory"] = (
            self.memory_store.describe()
            if self.memory_store is not None
            else {"enabled": False, "backend": "json"}
        )
        return summary

    def knowledge_base_summary(self) -> dict[str, Any]:
        """Delegate knowledge-base diagnostics to the underlying assistant."""
        return self.runtime.assistant.knowledge_base_summary()

    def build_knowledge_base_from_uploads(self, uploaded_files: Sequence[Any] | None = None) -> dict[str, Any]:
        """Delegate knowledge-base rebuilds to the underlying assistant."""
        return self.runtime.assistant.build_knowledge_base_from_uploads(uploaded_files)

    def _response_from_payload(self, payload: dict[str, Any]) -> AssistantResponse:
        parsed_result = payload.get("parsed_result")
        if parsed_result is None and str(payload.get("intent") or "") == "machine_agent":
            parsed_result = {
                "tool_calls": payload.get("tool_calls", []),
                "time_range": payload.get("time_range"),
                "current_status": payload.get("current_status"),
                "alarm_records": payload.get("alarm_records", []),
                "summary_result": payload.get("summary_result"),
                "abnormal_result": payload.get("abnormal_result"),
                "doc_results": payload.get("doc_results", []),
                "agent_mode": payload.get("agent_mode"),
                "agent_session": payload.get("agent_session"),
            }
        return AssistantResponse(
            intent=str(payload.get("intent") or "agent"),
            answer=str(payload.get("answer") or ""),
            sources=list(payload.get("sources") or []),
            source_items=list(payload.get("source_items") or []),
            command_preview=payload.get("command_preview"),
            status_summary=payload.get("status_summary"),
            parsed_result=parsed_result,
        )

    def _format_agent_input(
        self,
        message: str,
        history: Sequence[tuple[str, str]] | None,
        *,
        session_id: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        history_tail = list(history or [])[-self.config.history_window :]
        memory_context = ""
        memory_hits: list[dict[str, Any]] = []
        if self.memory_store is not None:
            memory_context, memory_hits = self.memory_store.format_context(
                message,
                session_id=session_id,
                history=history_tail,
                recent_limit=self.config.memory_recent_turns,
                search_limit=self.config.memory_search_limit,
            )
        elif history_tail:
            turns: list[str] = []
            for user_message, assistant_message in history_tail:
                turns.append(f"用户: {user_message}")
                turns.append(f"助手: {assistant_message}")
            memory_context = "短期记忆（当前页面历史）:\n" + "\n".join(turns)

        if not memory_context:
            return message, []
        return (
            "\n\n".join(
                [
                    memory_context,
                    f"当前用户问题:\n{message}",
                    "请先基于系统规则判断是否需要调用工具，再输出规定 JSON。",
                ]
            ),
            memory_hits,
        )

    def _attach_memory_metadata(
        self,
        payload: dict[str, Any],
        memory_hits: list[dict[str, Any]],
    ) -> None:
        if not memory_hits:
            return

        parsed_result = payload.get("parsed_result")
        if not isinstance(parsed_result, dict):
            parsed_result = {}
        memory_payload = {
            "backend": "json",
            "layers": list(
                dict.fromkeys(
                    str(hit.get("layer") or "")
                    for hit in memory_hits
                    if hit.get("layer")
                )
            ),
            "hits": memory_hits,
        }
        parsed_result.setdefault("memory", memory_payload)
        payload["parsed_result"] = parsed_result

        source_items = list(payload.get("source_items") or [])
        sources = list(payload.get("sources") or [])
        for hit in memory_hits:
            source = f"conversation_memory:{hit.get('session_id') or 'unknown'}"
            if source not in sources:
                sources.append(source)
            source_items.append(
                {
                    "source": source,
                    "title": "历史对话记忆",
                    "section": str(hit.get("layer") or "memory"),
                    "document_type": "conversation_memory",
                    "chunk_id": str(hit.get("turn_id") or ""),
                    "score": float(hit.get("score") or 0.0),
                    "text": str(hit.get("snippet") or ""),
                    "metadata": hit,
                }
            )
        payload["sources"] = sources
        payload["source_items"] = source_items

    def _remember_turn(
        self,
        session_id: str,
        message: str,
        response: AssistantResponse,
    ) -> None:
        if self.memory_store is None:
            return
        structured_data = response.parsed_result if isinstance(response.parsed_result, dict) else {}
        self.memory_store.append_turn(
            session_id=session_id,
            user=message,
            assistant=response.answer,
            intent=response.intent,
            structured_data=structured_data,
        )


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the Agent-first command-line parser."""
    parser = argparse.ArgumentParser(description="DAC-3D OpenAI Agents SDK runtime")
    parser.add_argument("--message", help="Run one message through the DAC-3D Agent.")
    parser.add_argument(
        "--preview-command",
        help="Parse one DAC-3D operation into a command preview without submitting it.",
    )
    parser.add_argument(
        "--execute-command",
        help=(
            "Parse one DAC-3D operation through the Agent control path. Direct submit is "
            "blocked; use the Web API preview/confirm token flow for real execution."
        ),
    )
    parser.add_argument(
        "--confirmed",
        action="store_true",
        help=(
            "Legacy compatibility flag. Agent/CLI direct submit remains blocked and "
            "returns TOKEN_BOUND_CONFIRMATION_REQUIRED."
        ),
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print the registered DAC-3D Agent tools and exit.",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print the DAC-3D Agent project description as JSON and exit.",
    )
    parser.add_argument(
        "--rebuild-kb",
        action="store_true",
        help="Rebuild the local knowledge base before creating the Agent.",
    )
    parser.add_argument("--agent-model", help="Override the OpenAI Agents SDK model name.")
    parser.add_argument(
        "--agent-api-base-url",
        help="Override the OpenAI-compatible Agent model API base URL.",
    )
    parser.add_argument(
        "--agent-api-key",
        help="Override the Agent model API key. Prefer DAC3D_AGENT_API_KEY in .env.",
    )
    parser.add_argument(
        "--agent-api-type",
        choices=sorted(VALID_AGENT_API_TYPES),
        help="Agent model API mode. Use chat_completions for most third-party OpenAI-compatible providers.",
    )
    return parser


def main() -> None:
    """Run the Agent-first command-line interface."""
    args = build_argument_parser().parse_args()
    try:
        runtime = DAC3DAgentRuntime.create(rebuild_kb=args.rebuild_kb)
    except Exception as exc:  # pragma: no cover - startup safety net
        print(f"Agent startup failed: {exc}")
        return

    if args.agent_model:
        runtime.config.agent_model_name = args.agent_model
    if args.agent_api_base_url:
        runtime.config.agent_api_base_url = args.agent_api_base_url
    if args.agent_api_key:
        runtime.config.agent_api_key = args.agent_api_key
    if args.agent_api_type:
        runtime.config.agent_api_type = args.agent_api_type

    if args.list_tools:
        for tool_name in AGENT_TOOL_NAMES:
            print(tool_name)
        return

    if args.describe:
        print(json.dumps(runtime.describe(), ensure_ascii=False, indent=2))
        return

    if args.preview_command:
        payload = runtime.preview_command(args.preview_command)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.execute_command:
        payload = runtime.execute_command(
            args.execute_command,
            confirmed_by_user=args.confirmed,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.message:
        try:
            print(runtime.run_text(args.message))
        except Exception as exc:
            print(f"Agent run failed: {exc}")
        return

    print("Use --message, --preview-command, --execute-command, --list-tools, or --describe.")


if __name__ == "__main__":
    main()
