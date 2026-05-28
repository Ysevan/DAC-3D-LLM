"""OpenAI Agents SDK runtime for DAC-3D assistant workflows."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_core import (
    AGENT_HANDOFF_NAMES,
    AGENT_INSTRUCTIONS,
    AGENT_SPECIALIST_NAMES,
    AGENT_TOOL_NAMES,
    AssistantResponsePayload,
    DAC3DAgentSessionStore,
    DAC3DAgentToolController,
    DAC3D_CONTROL_AGENT_INSTRUCTIONS,
    DAC3D_QA_AGENT_INSTRUCTIONS,
    DAC3D_RESULT_AGENT_INSTRUCTIONS,
    MACHINE_AGENT_INSTRUCTIONS,
    MEMORY_AGENT_INSTRUCTIONS,
    SAFETY_AGENT_INSTRUCTIONS,
    SKILL_AGENT_INSTRUCTIONS,
)
from app import AssistantResponse, DAC3DAssistant
from config import AppConfig
from context_engineering import ContextBuilder, FileBackedContextTree, GitWorkspaceContext, RepoContextMapStore
from goals import (
    ArtifactStore,
    AgentDeploymentStore,
    AgentFleetStore,
    AgentLabelingStore,
    AgentRegistryStore,
    AutomationPlannerStore,
    BrowserContextStore,
    CheckpointStore,
    ConversationThreadStore,
    EventQueueStore,
    GoalStore,
    ObservabilityReporter,
    AgentPerformanceStore,
    ReviewHandoffStore,
    SharedStateStore,
    TaskBoardStore,
    ToolMarketplaceStore,
    VerificationRunnerStore,
    WorkflowTemplateStore,
)
from memory import ConversationMemoryStore, LocalMemoryProvider
from skill_system import SkillPatchStore, SkillRegistry
from trace_eval import CodexHandoffGenerator, EvalDraftGenerator, EvalRunner, TraceLogger


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


def _default_agent_registry_entries() -> list[dict[str, Any]]:
    """Return built-in DAC-Agent registry entries for local discovery."""
    return [
        {
            "role": "coordinator",
            "name": "DAC-3D Multi-Agent Coordinator",
            "agent_type": "coordinator",
            "description": "统一入口 Agent，负责理解用户问题、选择 specialist、组合上下文和工具结果。",
            "capabilities": ["intent_routing", "handoff", "context_planning"],
            "tools": [],
            "triggers": ["任何问题", "统一入口", "router", "handoff", "agent"],
            "tags": ["entrypoint", "multi-agent"],
        },
        {
            "role": "dac3d_qa",
            "name": "DAC-3D QA Agent",
            "handoff_name": "handoff_dac3d_qa_agent",
            "description": "回答 DAC-3D 文档、参数、流程、故障排查和操作建议问题。",
            "capabilities": ["document_qa", "parameter_explain", "operator_guidance"],
            "tools": ["dac3d_answer", "dac3d_rebuild_knowledge_base"],
            "triggers": ["参数", "文档", "手册", "反光", "怎么办", "知识库"],
            "tags": ["qa", "rag"],
        },
        {
            "role": "dac3d_control",
            "name": "DAC-3D Control Agent",
            "handoff_name": "handoff_dac3d_control_agent",
            "description": "生成 DAC-3D 命令预览，读取状态，并协调受控命令提交。",
            "capabilities": ["command_preview", "runtime_status", "operation_control"],
            "tools": [
                "dac3d_operation",
                "dac3d_preview_command",
                "dac3d_execute_command",
                "dac3d_status",
            ],
            "triggers": ["扫描", "离线检测", "停止", "状态", "进度", "执行", "command"],
            "tags": ["control", "dac3d"],
        },
        {
            "role": "dac3d_result",
            "name": "DAC-3D Result Agent",
            "handoff_name": "handoff_dac3d_result_agent",
            "description": "读取和解释 DAC-3D 最新检测结果、样品缺陷摘要和复核建议。",
            "capabilities": ["result_lookup", "defect_explain", "sample_summary"],
            "tools": ["dac3d_latest_result"],
            "triggers": ["结果", "样品", "缺陷", "复检", "result"],
            "tags": ["result", "inspection"],
        },
        {
            "role": "machine",
            "name": "Machine Agent",
            "handoff_name": "handoff_machine_agent",
            "description": "分析设备状态、报警趋势、历史采样、维护文档和异常归因。",
            "capabilities": ["machine_status", "alarm_analysis", "maintenance_guidance"],
            "tools": [
                "machine_agent_chat",
                "machine_snapshot",
                "machine_status",
                "machine_history",
                "machine_alarms",
                "machine_docs",
                "machine_condition_summary",
                "machine_abnormal_analysis",
            ],
            "triggers": ["设备", "报警", "温度", "维护", "异常", "machine"],
            "tags": ["machine", "ops"],
        },
        {
            "role": "memory",
            "name": "Memory Agent",
            "handoff_name": "handoff_memory_agent",
            "description": "检索和维护 JSON/Markdown 多层记忆、历史对话、知识笔记和流程经验。",
            "capabilities": ["memory_search", "profile_memory", "procedure_memory"],
            "tools": [
                "conversation_memory_search",
                "conversation_memory_recent",
                "conversation_memory_profile",
                "conversation_memory_update",
                "conversation_knowledge_notes",
                "conversation_procedure_memories",
            ],
            "triggers": ["记忆", "历史", "上次", "刚才", "以后", "流程经验", "memory"],
            "tags": ["memory-os", "context"],
        },
        {
            "role": "skill",
            "name": "Skill Agent",
            "handoff_name": "handoff_skill_agent",
            "description": "发现、选择、读取和提出 DAC-Agent skills 的可审核改进建议。",
            "capabilities": ["skill_selection", "skill_read", "skill_patch_proposal"],
            "tools": [
                "dac_skill_list",
                "dac_skill_select",
                "dac_skill_read",
                "dac_skill_propose_patch",
                "dac_skill_patches",
            ],
            "triggers": ["技能", "skill", "流程模板", "可用流程", "加载什么"],
            "tags": ["skills", "workflow"],
        },
        {
            "role": "safety",
            "name": "Safety Agent",
            "handoff_name": "handoff_safety_agent",
            "description": "提供命令风险、工具网关和执行前检查的解释型审查。",
            "capabilities": ["safety_review", "tool_manifest", "command_validation"],
            "tools": [
                "dac3d_safety_review",
                "dac_tool_manifest",
                "dac_mcp_manifest",
                "dac_tool_allowed_dirs",
                "dac_tool_validate_command",
            ],
            "triggers": ["安全", "风险", "能不能执行", "执行前检查", "工具网关"],
            "tags": ["safety", "tool-gateway"],
        },
    ]


def _default_tool_marketplace_entries() -> list[dict[str, Any]]:
    """Return built-in tool packs for the local workflow marketplace."""
    return [
        {
            "slug": "dac3d-control-pack",
            "name": "DAC-3D Control Tools",
            "description": "状态读取、命令预览、命令提交和停止检测相关的 DAC-3D 控制工具包。",
            "category": "dac3d_operations",
            "provider": "dac-agent",
            "status": "installed",
            "tools": [
                "dac3d_operation",
                "dac3d_preview_command",
                "dac3d_execute_command",
                "dac3d_status",
                "dac_tool_command_history",
            ],
            "required_context": ["runtime_status", "tool_gateway_manifest", "command_schema"],
            "prompt_examples": ["扫描 10mm x 10mm 区域", "停止当前检测", "当前检测状态是什么？"],
            "tags": ["dac3d", "control", "workflow"],
        },
        {
            "slug": "dac3d-qa-pack",
            "name": "DAC-3D QA Tools",
            "description": "面向 DAC-3D 文档、参数说明、操作建议和知识库重建的问答工具包。",
            "category": "knowledge",
            "provider": "dac-agent",
            "status": "installed",
            "tools": ["dac3d_answer", "dac3d_rebuild_knowledge_base"],
            "required_context": ["document_memory", "skill_context"],
            "prompt_examples": ["样品表面反光很强怎么办？", "这个参数是什么意思？"],
            "tags": ["qa", "rag", "documents"],
        },
        {
            "slug": "dac3d-result-pack",
            "name": "DAC-3D Result Tools",
            "description": "读取、归一化和解释 DAC-3D 最新检测结果与样品缺陷摘要。",
            "category": "inspection_results",
            "provider": "dac-agent",
            "status": "installed",
            "tools": ["dac3d_latest_result"],
            "required_context": ["latest_result", "result_history"],
            "prompt_examples": ["第三个样品检测结果怎么样？", "最近一次检测有什么缺陷？"],
            "tags": ["result", "inspection"],
        },
        {
            "slug": "machine-diagnostics-pack",
            "name": "Machine Diagnostics Tools",
            "description": "设备状态、历史采样、报警记录、维护文档和异常归因工具包。",
            "category": "machine_ops",
            "provider": "dac-agent",
            "status": "installed",
            "tools": [
                "machine_agent_chat",
                "machine_snapshot",
                "machine_status",
                "machine_history",
                "machine_alarms",
                "machine_docs",
                "machine_condition_summary",
                "machine_abnormal_analysis",
            ],
            "required_context": ["machine_status", "alarm_history"],
            "prompt_examples": ["为什么最近温度报警变多了？", "现在设备状态怎么样？"],
            "tags": ["machine", "diagnostics"],
        },
        {
            "slug": "memory-os-pack",
            "name": "Memory OS Tools",
            "description": "会话检索、长期记忆、知识笔记、流程记忆和可审核记忆补丁工具包。",
            "category": "memory",
            "provider": "dac-agent",
            "status": "installed",
            "tools": [
                "conversation_memory_search",
                "conversation_memory_recent",
                "conversation_memory_profile",
                "conversation_memory_update",
                "conversation_knowledge_notes",
                "conversation_knowledge_read",
                "conversation_knowledge_write",
                "conversation_procedure_memories",
                "conversation_procedure_read",
                "conversation_procedure_write",
            ],
            "required_context": ["session_memory", "curated_memory", "procedure_memory"],
            "prompt_examples": ["刚才讨论过什么？", "以后默认先展示命令预览。"],
            "tags": ["memory-os", "context"],
        },
        {
            "slug": "skill-system-pack",
            "name": "Skill System Tools",
            "description": "发现、选择、读取和提出 DAC-Agent skill 改进建议的工具包。",
            "category": "skills",
            "provider": "dac-agent",
            "status": "installed",
            "tools": [
                "dac_skill_list",
                "dac_skill_select",
                "dac_skill_read",
                "dac_skill_propose_patch",
                "dac_skill_patches",
            ],
            "required_context": ["skill_registry", "active_task"],
            "prompt_examples": ["这个任务应该加载什么技能？", "列出可用流程。"],
            "tags": ["skills", "workflow"],
        },
        {
            "slug": "workspace-collaboration-pack",
            "name": "Workspace Collaboration Tools",
            "description": "本地 Agent 工作台产物、事件、线程、浏览上下文、恢复点和可观测摘要工具目录。",
            "category": "workspace",
            "provider": "dac-agent",
            "status": "available",
            "tools": [
                "agent_artifacts",
                "agent_events",
                "agent_threads",
                "agent_browser_contexts",
                "agent_checkpoints",
                "agent_observability",
            ],
            "required_context": ["agent_workspace", "conversation_thread", "shared_state"],
            "prompt_examples": ["把这次协作保存成 thread。", "记录当前页面观察。"],
            "tags": ["workspace", "collaboration", "low-code"],
        },
    ]


class LocalValidationAgentModel:
    """Deterministic Agents SDK model for local Chrome validation."""

    def __init__(self) -> None:
        self.calls = 0
        self.tool_name = "dac3d_answer"
        self.arguments: dict[str, Any] = {"question": ""}
        self._pending_tool_after_handoff = False
        self._pending_tool_name = ""
        self._pending_arguments: dict[str, Any] = {}
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
            handoff_names = {handoff.tool_name for handoff in handoffs}
            call_name = self.tool_name
            call_arguments = dict(self.arguments)
            if call_name not in tool_names and call_name not in handoff_names:
                planned_handoff = self._handoff_for_tool(call_name)
                if planned_handoff in handoff_names:
                    self._pending_tool_after_handoff = True
                    self._pending_tool_name = call_name
                    self._pending_arguments = call_arguments
                    call_name = planned_handoff
                    call_arguments = {}
                elif "dac3d_answer" in tool_names:
                    call_name = "dac3d_answer"
                    call_arguments = {"question": message}
                elif "handoff_dac3d_qa_agent" in handoff_names:
                    self._pending_tool_after_handoff = True
                    self._pending_tool_name = "dac3d_answer"
                    self._pending_arguments = {"question": message}
                    call_name = "handoff_dac3d_qa_agent"
                    call_arguments = {}
            elif call_name in handoff_names:
                self._pending_tool_after_handoff = True
                self._pending_tool_name = "dac3d_answer"
                self._pending_arguments = {"question": message}
                call_arguments = {}
            elif call_name not in tool_names:
                call_name = "dac3d_answer"
                call_arguments = {"question": message}
            return ModelResponse(
                output=[
                    ResponseFunctionToolCall(
                        arguments=json.dumps(call_arguments, ensure_ascii=False),
                        call_id=f"local_{call_name}",
                        name=call_name,
                        type="function_call",
                        status="completed",
                    )
                ],
                usage=Usage(),
                response_id=None,
            )

        tool_names = {tool.name for tool in tools}
        if self._pending_tool_after_handoff and self._pending_tool_name in tool_names:
            self._pending_tool_after_handoff = False
            call_name = self._pending_tool_name
            call_arguments = dict(self._pending_arguments)
            self._pending_tool_name = ""
            self._pending_arguments = {}
            return ModelResponse(
                output=[
                    ResponseFunctionToolCall(
                        arguments=json.dumps(call_arguments, ensure_ascii=False),
                        call_id=f"local_{call_name}",
                        name=call_name,
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
        current_message = self._current_user_message(message)
        lowered = current_message.lower()
        if any(marker in current_message for marker in ("刚才", "上次", "历史", "记得", "前面")):
            return (
                "conversation_memory_search",
                {"query": current_message, "include_global": True, "limit": 5},
                {
                    "answer": "我已检索本地 JSON 历史对话，并结合命中的记忆回答。",
                    "structured_data": {
                        "intent": "memory_search",
                        "agent_path": ["coordinator", "memory_agent"],
                        "tool_calls": [
                            {"name": "conversation_memory_search", "purpose": "检索 JSON 历史对话"}
                        ],
                        "memory": {"backend": "json"},
                        "limits": ["本回答来自本地验证模型，用于功能验证"],
                    },
                },
            )
        scan_like = any(marker in current_message for marker in ("扫描", "scan", "10mm"))
        bypass_like = any(
            marker in current_message
            for marker in ("绕过", "忽略", "跳过", "不要确认", "不用确认", "直接写入", "直接下发")
        ) or any(marker in lowered for marker in ("bypass", "ignore", "without approval", "no confirmation", "command.json"))
        if scan_like and bypass_like:
            return (
                "dac3d_preview_command",
                {"instruction": current_message},
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
        if any(marker in current_message for marker in ("安全", "风险", "能不能执行", "执行前检查")):
            return (
                "dac3d_safety_review",
                {"instruction": current_message, "confirmed_by_user": False},
                {
                    "answer": "我已完成执行前安全审查；需要确认的命令必须等待用户明确确认后再下发。",
                    "structured_data": {
                        "intent": "safety_review",
                        "agent_path": ["coordinator", "safety_agent"],
                        "tool_calls": [
                            {"name": "dac3d_safety_review", "purpose": "审查 DAC-3D 控制命令风险"}
                        ],
                        "safety_review": {"decision": "requires_confirmation", "can_execute": False},
                    },
                },
            )
        if any(marker in current_message for marker in ("技能", "skill", "可用流程", "加载什么")):
            return (
                "dac_skill_select",
                {"task": current_message, "limit": 3},
                {
                    "answer": "我已根据当前任务选择最相关的 DAC-Agent skills，并会按需加载技能说明。",
                    "structured_data": {
                        "intent": "skill_select",
                        "agent_path": ["coordinator", "skill_agent"],
                        "tool_calls": [
                            {"name": "dac_skill_select", "purpose": "选择当前任务相关技能"}
                        ],
                        "skill_system": {"backend": "local_agent_skills"},
                    },
                },
            )
        if any(marker in lowered for marker in ("mcp", "agents sdk", "capability manifest")) or any(
            marker in current_message for marker in ("资源目录", "prompt 模板", "工具目录")
        ):
            return (
                "dac_mcp_manifest",
                {},
                {
                    "answer": "我已读取 MCP-compatible capability manifest，包含 DAC 工具、资源、prompt 模板和部署模式。",
                    "structured_data": {
                        "intent": "mcp_manifest",
                        "agent_path": ["coordinator", "dac3d_control_agent"],
                        "tool_calls": [
                            {"name": "dac_mcp_manifest", "purpose": "读取 MCP-compatible capability manifest"}
                        ],
                        "mcp": {"backend": "adapter_manifest_only"},
                    },
                },
            )
        if any(marker in current_message for marker in ("工具网关", "tool gateway", "gateway", "白名单")):
            return (
                "dac_tool_manifest",
                {},
                {
                    "answer": "我已读取 DAC Tool Gateway 元数据，包括工具风险等级、确认要求和路径白名单。",
                    "structured_data": {
                        "intent": "tool_gateway_diagnostics",
                        "agent_path": ["coordinator", "safety_agent"],
                        "tool_calls": [
                            {"name": "dac_tool_manifest", "purpose": "读取受控工具网关元数据"}
                        ],
                        "tool_gateway": {"backend": "dac_tool_gateway"},
                    },
                },
            )
        if any(marker in current_message for marker in ("温度报警", "报警变多", "设备状态")):
            return (
                "machine_agent_chat",
                {"message": current_message},
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
        if any(marker in current_message for marker in ("反光", "曝光", "怎么办", "guidance")) or "reflect" in lowered:
            return (
                "dac3d_answer",
                {"question": current_message},
                {
                    "answer": "样品表面反光很强时，建议先检查三相机原图是否过曝，再逐项降低曝光或光源强度，调整后复扫确认。",
                    "structured_data": {
                        "intent": "guidance",
                        "tool_calls": [{"name": "dac3d_answer", "purpose": "查询 DAC-3D 操作建议"}],
                        "recommendations": ["检查原图过曝", "降低曝光或光源强度", "复扫确认融合图和检测图"],
                    },
                },
            )
        if any(marker in current_message for marker in ("停止检测", "停止", "stop")):
            return (
                "dac3d_execute_command",
                {"instruction": "停止检测", "confirmed_by_user": True},
                {
                    "answer": "已提交停止检测命令，DAC-3D 已收到停止请求。",
                    "structured_data": {
                        "intent": "operation_execute",
                        "tool_calls": [{"name": "dac3d_execute_command", "purpose": "下发停止检测命令"}],
                        "command": {"action": "stop_detection", "payload": {"func": "Stop"}},
                        "status_summary": {"state": "stopped", "progress": 0, "message": "Mock 检测任务已收到停止请求。"},
                    },
                },
            )
        if any(marker in current_message for marker in ("执行扫描", "开始扫描", "立即开始", "确认执行")):
            return (
                "dac3d_execute_command",
                {"instruction": "执行扫描", "confirmed_by_user": True},
                {
                    "answer": "在线扫描命令已提交，当前任务已排队，进度 0%。",
                    "structured_data": {
                        "intent": "operation_execute",
                        "tool_calls": [{"name": "dac3d_execute_command", "purpose": "提交在线扫描命令"}],
                        "command": {"action": "start_online_scan", "payload": {"func": "Scan", "total_positions": 144}},
                        "status_summary": {"state": "queued", "progress": 0, "message": "在线扫描任务已排队"},
                    },
                },
            )
        if any(marker in current_message for marker in ("扫描", "scan", "10mm")):
            return (
                "dac3d_preview_command",
                {"instruction": current_message},
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
        if any(marker in current_message for marker in ("结果", "样品", "result")):
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
        if any(marker in current_message for marker in ("状态", "进度", "status")):
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
            {"question": current_message},
            {
                "answer": "我已通过 DAC-3D Agent 查询本地知识库并生成回复。",
                "structured_data": {
                    "intent": "query",
                    "tool_calls": [{"name": "dac3d_answer", "purpose": "查询 DAC-3D 知识库"}],
                },
            },
        )

    def _current_user_message(self, message: str) -> str:
        """Extract the actual user request from prompt-wrapped memory context."""
        text = str(message or "")
        candidates = [text]
        if "\\n" in text:
            candidates.append(text.replace("\\n", "\n"))
        try:
            decoded = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            decoded = None
        if decoded is not None:
            candidates.append("\n".join(self._collect_text_fields(decoded)))

        pattern = r"当前用户问题:\s*(.*?)(?:\n请先基于系统规则判断是否需要调用工具|$)"
        for candidate in reversed(candidates):
            matches = list(re.finditer(pattern, candidate, flags=re.DOTALL))
            match = matches[-1] if matches else None
            if match:
                current = match.group(1).strip()
                if current:
                    return current
        return text

    def _collect_text_fields(self, value: Any) -> list[str]:
        """Collect text fragments from Agents SDK input structures."""
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            texts: list[str] = []
            for key, item in value.items():
                if key in {"content", "text", "input", "message"} or isinstance(item, (dict, list)):
                    texts.extend(self._collect_text_fields(item))
            return texts
        if isinstance(value, list):
            texts = []
            for item in value:
                texts.extend(self._collect_text_fields(item))
            return texts
        return []

    def _handoff_for_tool(self, tool_name: str) -> str:
        if tool_name in {
            "dac3d_answer",
            "dac3d_rebuild_knowledge_base",
        }:
            return "handoff_dac3d_qa_agent"
        if tool_name in {
            "dac3d_operation",
            "dac3d_preview_command",
            "dac3d_execute_command",
            "dac3d_status",
        }:
            return "handoff_dac3d_control_agent"
        if tool_name == "dac3d_latest_result":
            return "handoff_dac3d_result_agent"
        if tool_name.startswith("machine_"):
            return "handoff_machine_agent"
        if tool_name.startswith("conversation_memory_") or tool_name.startswith("conversation_knowledge_"):
            return "handoff_memory_agent"
        if tool_name.startswith("dac_skill_"):
            return "handoff_skill_agent"
        if tool_name == "dac3d_safety_review":
            return "handoff_safety_agent"
        if tool_name.startswith("dac_tool_"):
            return "handoff_safety_agent"
        return ""


@dataclass(slots=True)
class DAC3DAgentRuntime:
    """Build and run the DAC-3D assistant as an OpenAI Agents SDK agent."""

    assistant: DAC3DAssistant
    config: AppConfig
    sessions: DAC3DAgentSessionStore | None = None
    memory_store: ConversationMemoryStore | None = None
    memory_provider: LocalMemoryProvider | None = None
    skill_registry: SkillRegistry | None = None
    skill_patch_store: SkillPatchStore | None = None
    context_builder: ContextBuilder | None = None

    def __post_init__(self) -> None:
        if self.sessions is None:
            self.sessions = DAC3DAgentSessionStore(base_dir=self.config.base_dir)
        if self.memory_store is None and self.config.memory_enabled:
            self.memory_store = ConversationMemoryStore.from_config(self.config)
            self.memory_store.ensure_directories()
        if self.memory_provider is None and self.memory_store is not None:
            self.memory_provider = LocalMemoryProvider(self.memory_store)
        if self.skill_registry is None:
            self.skill_registry = SkillRegistry(self.config.agent_skills_dir)
        if self.skill_patch_store is None:
            self.skill_patch_store = SkillPatchStore.from_root(
                self.config.conversation_memory_dir,
                self.skill_registry,
            )

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
        """Generate and submit a DAC-3D command when confirmation and safety checks pass."""
        return self.tool_controller(session_id).execute_command(
            instruction,
            confirmed_by_user=confirmed_by_user,
        )

    def search_conversation_memory(
        self,
        query: str,
        *,
        session_id: str = "default",
        include_global: bool = True,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Search JSON conversation memory for a session-aware Agent tool."""
        if self.memory_store is None:
            return {
                "enabled": False,
                "backend": "json+markdown",
                "query": query,
                "hits": [],
                "message": "Conversation memory is disabled.",
            }

        search_limit = limit or self.config.memory_search_limit
        hits = self.memory_store.search(
            query,
            session_id=session_id,
            limit=search_limit,
            include_global=include_global,
        )
        return {
            "enabled": True,
            "backend": "json+markdown",
            "query": query,
            "session_id": session_id,
            "include_global": include_global,
            "hit_count": len(hits),
            "hits": [hit.to_dict() for hit in hits],
        }

    def recent_conversation_memory(
        self,
        *,
        session_id: str = "default",
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Return recent JSON conversation turns for a session-aware Agent tool."""
        if self.memory_store is None:
            return {
                "enabled": False,
                "backend": "json",
                "session_id": session_id,
                "turns": [],
                "message": "Conversation memory is disabled.",
            }

        turn_limit = limit or self.config.memory_recent_turns
        turns = self.memory_store.recent_turns(session_id, limit=turn_limit)
        return {
            "enabled": True,
            "backend": "json",
            "session_id": session_id,
            "turn_count": len(turns),
            "turns": turns,
        }

    def conversation_memory_profile(self) -> dict[str, Any]:
        """Return Hermes-style curated memory and routed knowledge-note index."""
        if self.memory_store is None:
            return {
                "enabled": False,
                "backend": "json+markdown",
                "message": "Conversation memory is disabled.",
            }
        return {
            "enabled": True,
            "profile": self.memory_store.load_curated_memory(),
            "knowledge_notes": self.memory_store.list_knowledge_notes(),
        }

    def update_conversation_memory(
        self,
        *,
        target: str,
        content: str,
        mode: str = "append",
    ) -> dict[str, Any]:
        """Create an auditable core/user memory patch instead of direct write."""
        if self.memory_provider is None:
            return {
                "enabled": False,
                "backend": "json+markdown",
                "message": "Conversation memory is disabled.",
            }
        trace = self.memory_provider.record_trace(
            {
                "session_id": "memory-tool",
                "user_message": f"memory update requested: {content}",
                "assistant_answer": "已生成记忆补丁，等待审核。",
                "intent": "memory_update_request",
                "parsed_result": {
                    "memory_write_candidates": [
                        {
                            "target": target,
                            "content": content,
                            "mode": mode,
                            "reason": "conversation_memory_update_tool",
                            "metadata": {"source": "memory_agent_tool"},
                        }
                    ]
                },
            }
        )
        patches = self.memory_provider.propose_writes(trace)
        return {
            "enabled": True,
            "backend": "json+markdown",
            "applied": False,
            "requires_approval": True,
            "patches": patches,
            "count": len(patches),
            "profile": self.memory_store.load_curated_memory() if self.memory_store else {},
        }

    def list_conversation_knowledge_notes(self) -> dict[str, Any]:
        """List routed topic knowledge notes."""
        if self.memory_store is None:
            return {"enabled": False, "topics": []}
        return {"enabled": True, **self.memory_store.list_knowledge_notes()}

    def read_conversation_knowledge_note(self, topic: str) -> dict[str, Any]:
        """Read one routed topic knowledge note."""
        if self.memory_store is None:
            return {"enabled": False, "topic": topic, "text": ""}
        return {"enabled": True, **self.memory_store.read_knowledge_note(topic)}

    def write_conversation_knowledge_note(
        self,
        *,
        topic: str,
        content: str,
        mode: str = "append",
    ) -> dict[str, Any]:
        """Create an auditable topic-note patch instead of direct write."""
        if self.memory_provider is None:
            return {"enabled": False, "topic": topic, "message": "Memory disabled."}
        trace = self.memory_provider.record_trace(
            {
                "session_id": "memory-tool",
                "user_message": f"knowledge note update requested: {topic}",
                "assistant_answer": "已生成知识笔记补丁，等待审核。",
                "intent": "memory_update_request",
                "parsed_result": {
                    "memory_write_candidates": [
                        {
                            "target": "knowledge",
                            "topic": topic,
                            "content": content,
                            "mode": mode,
                            "reason": "conversation_knowledge_write_tool",
                            "metadata": {"source": "memory_agent_tool"},
                        }
                    ]
                },
            }
        )
        patches = self.memory_provider.propose_writes(trace)
        return {
            "enabled": True,
            "backend": "json+markdown",
            "topic": topic,
            "applied": False,
            "requires_approval": True,
            "patches": patches,
            "count": len(patches),
        }

    def list_conversation_procedure_memories(self) -> dict[str, Any]:
        """List approved Markdown procedure memories."""
        if self.memory_provider is None:
            return {"enabled": False, "procedures": [], "count": 0}
        return {"enabled": True, **self.memory_provider.list_procedures()}

    def read_conversation_procedure_memory(self, name: str) -> dict[str, Any]:
        """Read one approved Markdown procedure memory."""
        if self.memory_provider is None:
            return {"enabled": False, "name": name, "text": ""}
        return {"enabled": True, **self.memory_provider.read_procedure(name)}

    def write_conversation_procedure_memory(
        self,
        *,
        name: str,
        content: str,
        reason: str = "procedure_memory_candidate",
        mode: str = "append",
    ) -> dict[str, Any]:
        """Create a procedure-memory patch instead of editing Markdown directly."""
        if self.memory_provider is None:
            return {"enabled": False, "name": name, "message": "Memory disabled."}
        trace = self.memory_provider.record_trace(
            {
                "session_id": "memory-tool",
                "user_message": f"procedure memory update requested: {name}",
                "assistant_answer": "已生成流程记忆补丁，等待审核。",
                "intent": "procedure_memory_update_request",
                "parsed_result": {
                    "memory_write_candidates": [
                        {
                            "target": "procedure_memory",
                            "topic": name,
                            "content": content,
                            "mode": mode,
                            "reason": reason,
                            "metadata": {
                                "source": "memory_agent_tool",
                                "procedure_name": name,
                                "title": name,
                            },
                        }
                    ]
                },
            }
        )
        patches = self.memory_provider.propose_writes(trace)
        return {
            "enabled": True,
            "backend": "json+markdown",
            "name": name,
            "applied": False,
            "requires_approval": True,
            "patches": patches,
            "count": len(patches),
            "procedures": self.memory_provider.list_procedures(),
        }

    def list_conversation_memory_patches(self, status: str = "pending") -> dict[str, Any]:
        """List auditable Memory OS patches."""
        if self.memory_provider is None:
            return {"enabled": False, "patches": [], "message": "Memory disabled."}
        normalized_status = str(status or "").strip() or None
        return {"enabled": True, **self.memory_provider.list_patches(status=normalized_status)}

    def approve_conversation_memory_patch(self, patch_id: str) -> dict[str, Any]:
        """Approve one pending Memory OS patch."""
        if self.memory_provider is None:
            return {"enabled": False, "patch_id": patch_id, "message": "Memory disabled."}
        return {"enabled": True, **self.memory_provider.approve_write(patch_id)}

    def reject_conversation_memory_patch(
        self,
        patch_id: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Reject one pending Memory OS patch."""
        if self.memory_provider is None:
            return {"enabled": False, "patch_id": patch_id, "message": "Memory disabled."}
        return {"enabled": True, **self.memory_provider.reject_write(patch_id, reason=reason)}

    def list_dac_skills(self) -> dict[str, Any]:
        """List local DAC-Agent skills."""
        if self.skill_registry is None:
            return {"enabled": False, "skills": []}
        return {"enabled": True, **self.skill_registry.describe()}

    def select_dac_skills(self, task: str, limit: int = 3) -> dict[str, Any]:
        """Select relevant DAC-Agent skills for one task."""
        if self.skill_registry is None:
            return {"enabled": False, "matches": []}
        matches = self.skill_registry.select(task, limit=limit)
        return {
            "enabled": True,
            "task": task,
            "matches": [match.to_dict(include_content=False) for match in matches],
        }

    def read_dac_skill(self, name: str, include_assets: bool = False) -> dict[str, Any]:
        """Read one local DAC-Agent skill on demand."""
        if self.skill_registry is None:
            return {"enabled": False, "name": name}
        return {"enabled": True, **self.skill_registry.read(name, include_assets=include_assets)}

    def propose_dac_skill_patch(
        self,
        *,
        target_skill: str,
        reason: str,
        diff: str = "",
        replacement_section: str = "",
        evidence_trace_ids: list[str] | None = None,
        risk_level: str = "medium",
        proposed_by: str = "agent",
    ) -> dict[str, Any]:
        """Create a reviewable skill patch proposal without editing SKILL.md."""
        if self.skill_patch_store is None:
            return {"enabled": False, "backend": "json_skill_patch_queue", "patch": None}
        return {
            "enabled": True,
            **self.skill_patch_store.propose_patch(
                target_skill=target_skill,
                reason=reason,
                diff=diff,
                replacement_section=replacement_section,
                evidence_trace_ids=evidence_trace_ids or [],
                risk_level=risk_level,
                proposed_by=proposed_by,
            ),
        }

    def list_dac_skill_patches(self, status: str = "pending") -> dict[str, Any]:
        """List reviewable skill patch proposals."""
        if self.skill_patch_store is None:
            return {"enabled": False, "backend": "json_skill_patch_queue", "patches": [], "count": 0}
        normalized_status = str(status or "").strip() or None
        return {"enabled": True, **self.skill_patch_store.list_patches(status=normalized_status)}

    def approve_dac_skill_patch(self, patch_id: str) -> dict[str, Any]:
        """Approve a pending skill patch proposal without auto-applying it."""
        if self.skill_patch_store is None:
            return {"enabled": False, "patch_id": patch_id, "message": "Skill patch store is disabled."}
        return {"enabled": True, **self.skill_patch_store.approve_patch(patch_id)}

    def reject_dac_skill_patch(self, patch_id: str, reason: str = "") -> dict[str, Any]:
        """Reject a pending skill patch proposal."""
        if self.skill_patch_store is None:
            return {"enabled": False, "patch_id": patch_id, "message": "Skill patch store is disabled."}
        return {"enabled": True, **self.skill_patch_store.reject_patch(patch_id, reason=reason)}

    def review_command_safety(
        self,
        instruction: str,
        *,
        confirmed_by_user: bool = False,
        session_id: str = "default",
    ) -> dict[str, Any]:
        """Preview and classify DAC-3D command safety without executing."""
        payload = self.preview_command(instruction, session_id=session_id)
        command_preview = payload.get("command_preview")
        if not isinstance(command_preview, dict):
            return {
                **payload,
                "safety_review": {
                    "decision": "not_a_control_command",
                    "can_execute": False,
                    "reason": "No DAC-3D command preview was produced.",
                },
            }

        safety = dict(command_preview.get("safety") or {})
        status = dict(payload.get("status_summary") or command_preview.get("runtime_status") or {})
        state = str(status.get("state") or "").lower()
        busy_states = {"running", "detecting", "scanning", "initializing"}
        missing_fields = list(command_preview.get("missing_fields") or [])
        warnings = list(command_preview.get("warnings") or [])
        needs_confirmation = bool(safety.get("needs_confirmation"))
        hardware_required = bool(safety.get("hardware_required"))

        decision = "review_passed"
        can_execute = True
        reason = "Command preview passed safety review."
        if missing_fields:
            decision = "needs_clarification"
            can_execute = False
            reason = "Command preview has missing required fields."
        elif state in busy_states:
            decision = "blocked_runtime_busy"
            can_execute = False
            reason = f"DAC-3D runtime is busy: {state}."
        elif needs_confirmation and not confirmed_by_user:
            decision = "requires_confirmation"
            can_execute = False
            reason = "This command requires explicit user confirmation."

        review = {
            "decision": decision,
            "can_execute": can_execute,
            "requires_confirmation": needs_confirmation,
            "confirmed_by_user": confirmed_by_user,
            "hardware_required": hardware_required,
            "missing_fields": missing_fields,
            "warnings": warnings,
            "runtime_state": state or "unknown",
            "reason": reason,
        }
        return {
            **payload,
            "safety_review": review,
            "agent_session": self.sessions.describe_session(session_id) if self.sessions else {},
        }

    def run_chat_payload(self, message: str, *, session_id: str = "default") -> dict[str, Any]:
        """Run the unified Agent entrypoint and return a UI-friendly payload."""
        answer = self.run_sync(message, session_id=session_id)
        return self._payload_from_agent_output(answer, session_id=session_id, source_message=message)

    def run_text(self, message: str, *, session_id: str = "default") -> str:
        """Return only the final Agent text for CLI entrypoints."""
        return str(self.run_chat_payload(message, session_id=session_id).get("answer") or "")

    def run_local_tool_plan(self, message: str, *, session_id: str = "default") -> dict[str, Any]:
        """Explicit local diagnostic planner; production chat enters the LLM first."""
        tools = self.tool_controller(session_id)
        assert self.sessions is not None
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

    def approve_pending_command(
        self,
        *,
        session_id: str = "default",
        preview_id: str | None = None,
        confirmation_token: str | None = None,
    ) -> dict[str, Any]:
        """Submit the current pending DAC-3D command after a UI approval click."""
        assert self.sessions is not None
        pending = self.sessions.get_pending_command(session_id)
        if pending is not None:
            gateway = pending.command_preview.get("gateway") if isinstance(pending.command_preview, dict) else {}
            if not isinstance(gateway, dict):
                gateway = {}
            preview_id = preview_id or str(gateway.get("preview_id") or "")
            confirmation_token = confirmation_token or str(gateway.get("confirmation_token") or "")

        payload = self.tool_controller(session_id).gateway.submit_command(
            str(preview_id or ""),
            str(confirmation_token or ""),
        )
        payload = dict(payload)
        payload.setdefault("sources", [])
        payload.setdefault("source_items", [])
        payload.setdefault("command_preview", None)
        payload.setdefault("status_summary", None)
        payload.setdefault("parsed_result", None)
        payload["agent_mode"] = "openai-agents-sdk"
        return payload

    def _payload_from_agent_output(self, output: str, *, session_id: str, source_message: str = "") -> dict[str, Any]:
        """Convert the LLM-authored final output into the UI payload contract."""
        session_snapshot = self.sessions.describe_session(session_id) if self.sessions else {}
        current_user_message = self._current_user_message(source_message)
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
        answer_text = str(parsed.get("answer") or raw_output)
        command_preview = structured.get("command_preview") or structured.get("command")
        status_summary = structured.get("status_summary")
        if not isinstance(command_preview, dict):
            command_preview = None
        if not isinstance(status_summary, dict):
            status_summary = None
        if self.sessions is not None and command_preview is not None and not isinstance(
            command_preview.get("gateway"), dict
        ):
            validation_payload = self.tool_controller(session_id).gateway.validate_command(
                command_preview,
                source_text=current_user_message,
            ).to_dict()
            validation = dict(validation_payload.get("validation") or {})
            result = dict(validation_payload.get("result") or {})
            validated_preview = result.get("command_preview")
            if isinstance(validated_preview, dict):
                command_preview = validated_preview
                structured["command_preview"] = command_preview
                structured.setdefault("requires_confirmation", True)
                structured.setdefault(
                    "approval",
                    {
                        "type": "pending_command",
                        "source": "dac_tool_gateway",
                        "requires_user_click": bool(validation.get("can_submit")),
                    },
                )
                structured.setdefault(
                    "tool_gateway",
                    {
                        "backend": "dac_tool_gateway",
                        "tool": "validate_command",
                        "validation": validation,
                        "risk": validation_payload.get("risk") or {},
                    },
                )
                if validation.get("can_submit"):
                    pending = self.sessions.remember_pending_command(
                        session_id,
                        source_message="llm_structured_command",
                        command_preview=command_preview,
                        ttl_seconds=max(0, int(getattr(self.config, "command_confirmation_ttl_seconds", 300) or 0)),
                    )
                    command_preview = pending.command_preview
                    structured["command_preview"] = command_preview
                else:
                    self.sessions.clear_pending_command(session_id)
                    structured["intent"] = "operation_preview"
                    answer_text = "已生成 DAC-3D 命令预览，但安全策略阻止提交；请修改请求后重新预览。"
        if self.sessions is not None and (
            command_preview is None or not isinstance(command_preview.get("gateway"), dict)
        ):
            pending = self.sessions.get_pending_command(session_id)
            if pending is not None:
                command_preview = dict(pending.command_preview)
                structured["command_preview"] = command_preview
        return {
            "intent": str(structured.get("intent") or parsed.get("intent") or "agent"),
            "answer": answer_text,
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
        assert self.sessions is not None
        return DAC3DAgentToolController(
            assistant=self.assistant,
            sessions=self.sessions,
            session_id=session_id,
        )

    def _current_user_message(self, message: str) -> str:
        """Extract the user request from context-wrapped Agent input."""
        text = str(message or "")
        candidates = [text]
        if "\\n" in text:
            candidates.append(text.replace("\\n", "\n"))
        try:
            decoded = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            decoded = None
        if decoded is not None:
            candidates.append("\n".join(self._collect_text_fields(decoded)))

        pattern = r"当前用户问题:\s*(.*?)(?:\n请先基于系统规则判断是否需要调用工具|$)"
        for candidate in reversed(candidates):
            matches = list(re.finditer(pattern, candidate, flags=re.DOTALL))
            match = matches[-1] if matches else None
            if match:
                current = match.group(1).strip()
                if current:
                    return current
        return text

    def _collect_text_fields(self, value: Any) -> list[str]:
        """Collect text fragments from Agents SDK input structures."""
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            texts: list[str] = []
            for key, item in value.items():
                if key in {"content", "text", "input", "message"} or isinstance(item, (dict, list)):
                    texts.extend(self._collect_text_fields(item))
            return texts
        if isinstance(value, list):
            texts = []
            for item in value:
                texts.extend(self._collect_text_fields(item))
            return texts
        return []

    def describe(self) -> dict[str, Any]:
        """Return a stable description of this Agent project."""
        resolved_api_type = self.resolved_agent_api_type()
        local_validation = _looks_like_local_validation_model(self.config.agent_model_name)
        return {
            "name": "DAC-3D Multi-Agent Inspection System",
            "sdk": "openai-agents",
            "architecture": "coordinator_with_specialist_handoffs",
            "entry_agent": "DAC-3D Multi-Agent Coordinator",
            "specialist_agents": list(AGENT_SPECIALIST_NAMES),
            "handoffs": list(AGENT_HANDOFF_NAMES),
            "network_capabilities": [
                "domain_routing",
                "json_conversation_memory_search",
                "hermes_style_curated_memory",
                "topic_knowledge_notes",
                "auditable_memory_patches",
                "reviewed_procedure_memory_markdown",
                "trace_based_memory_feedback",
                "append_only_trace_logger",
                "local_eval_runner",
                "progressive_skill_selection",
                "reviewable_skill_patch_queue",
                "context_engineering",
                "runtime_status_context",
                "shared_workspace_artifacts",
                "durable_event_queue",
                "static_repo_context_map",
                "git_workspace_context",
                "verification_feedback_runner",
                "review_handoff_queue",
                "code_symbol_navigator",
                "workflow_checkpoint_store",
                "agent_observability_snapshot",
                "scoped_shared_state",
                "agent_registry_discovery",
                "agent_fleet_control_plane",
                "agent_deployment_catalog",
                "agent_labeling_queue",
                "agent_performance_analysis",
                "threaded_agent_conversation",
                "shared_browser_context",
                "local_tool_marketplace",
                "mcp_style_tool_gateway",
                "mcp_capability_manifest",
                "path_allowlist_validation",
                "command_safety_review",
                "specialist_tool_isolation",
                "runtime_diagnostics",
            ],
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
            "tool_groups": {
                "dac3d_qa_agent": ["dac3d_answer", "dac3d_rebuild_knowledge_base"],
                "dac3d_control_agent": [
                    "dac3d_operation",
                    "dac3d_preview_command",
                    "dac3d_execute_command",
                    "dac3d_status",
                    "dac3d_safety_review",
                    "dac_tool_manifest",
                    "dac_mcp_manifest",
                    "dac_tool_allowed_dirs",
                    "dac_tool_validate_command",
                    "dac_tool_cancel_pending_command",
                    "dac_tool_command_history",
                ],
                "dac3d_result_agent": ["dac3d_latest_result", "dac3d_answer"],
                "machine_agent": [
                    "machine_agent_chat",
                    "machine_snapshot",
                    "machine_status",
                    "machine_history",
                    "machine_alarms",
                    "machine_docs",
                    "machine_condition_summary",
                    "machine_abnormal_analysis",
                ],
                "memory_agent": [
                    "conversation_memory_search",
                    "conversation_memory_recent",
                    "conversation_memory_profile",
                    "conversation_memory_update",
                    "conversation_knowledge_notes",
                    "conversation_knowledge_read",
                    "conversation_knowledge_write",
                    "conversation_procedure_memories",
                    "conversation_procedure_read",
                    "conversation_procedure_write",
                    "conversation_memory_patches",
                    "conversation_memory_approve_patch",
                    "conversation_memory_reject_patch",
                ],
                "skill_system": [
                    "dac_skill_list",
                    "dac_skill_select",
                    "dac_skill_read",
                    "dac_skill_propose_patch",
                    "dac_skill_patches",
                    "dac_skill_approve_patch",
                    "dac_skill_reject_patch",
                ],
                "safety_agent": [
                    "dac3d_safety_review",
                    "dac3d_preview_command",
                    "dac3d_status",
                    "dac_tool_manifest",
                    "dac_mcp_manifest",
                    "dac_tool_allowed_dirs",
                    "dac_tool_validate_command",
                    "dac_tool_cancel_pending_command",
                    "dac_tool_command_history",
                ],
            },
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
                "backend": "json+markdown",
                "path": str(self.config.conversation_memory_dir),
                "layers": [
                    "short_term_history",
                    "core_markdown_memory",
                    "session_recent_json",
                    "session_summary",
                    "topic_knowledge_notes",
                    "procedure_markdown_memory",
                    "long_term_json_search",
                ],
                "curated_files": ["MEMORY.md", "USER.md"],
                "topic_notes_dir": "knowledge_notes",
            },
            "skill_system": (
                self.skill_registry.describe()
                if self.skill_registry is not None
                else {"enabled": False, "backend": "local_agent_skills"}
            ),
            "skill_patches": (
                self.skill_patch_store.describe()
                if self.skill_patch_store is not None
                else {"enabled": False, "backend": "json_skill_patch_queue"}
            ),
            "control": {
                "preview_tool": "dac3d_preview_command",
                "execute_tool": "dac3d_execute_command",
                "safety_review_tool": "dac3d_safety_review",
                "tool_gateway": "dac_tool_manifest",
                "confirmation_required_for_risky_commands": True,
                "bridge_modes": ["embedded", "command_file_bridge", "mock"],
            },
            "tool_gateway": self.tool_controller("default").tool_gateway_manifest(),
            "mcp": self.mcp_capability_manifest(session_id="default"),
        }

    def mcp_capability_manifest(self, *, session_id: str = "default") -> dict[str, Any]:
        """Return MCP-compatible capability discovery metadata for this runtime."""
        return self.tool_controller(session_id).mcp_capability_manifest()

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
                workflow_name="DAC-3D Multi-Agent",
                tracing_disabled=self.config.agent_tracing_disabled,
            )

        return RunConfig(
            workflow_name="DAC-3D Multi-Agent",
            tracing_disabled=self.config.agent_tracing_disabled,
            model_provider=self.build_model_provider(),
        )

    def build_agent(self, *, session_id: str = "default", role: str = "coordinator") -> Any:
        """Create the coordinator or one specialist Agent in the DAC-3D network."""
        try:
            from agents import Agent, Model, function_tool, handoff
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
            return tools.preview_command(instruction)

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
                "Parse and submit a DAC-3D control request to the active runtime bridge. "
                "Set confirmed_by_user=true only when the user explicitly confirmed execution "
                "or directly asked to start/stop/execute the operation, including Chinese "
                "requests such as 执行扫描、开始扫描、确认执行、立即开始、停止检测."
            ),
        )
        def dac3d_execute_command(
            instruction: str,
            confirmed_by_user: bool = False,
        ) -> dict[str, Any]:
            """Execute a DAC-3D control command after confirmation and safety checks."""
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
            return tools.read_dac_status()

        @function_tool(
            name_override="dac3d_latest_result",
            description_override=(
                "Read and interpret the latest DAC-3D inspection result. Pass sample_position=0 "
                "for the latest/all recorded results, or a 1-based sample position."
            ),
        )
        def dac3d_latest_result(sample_position: int) -> dict[str, Any]:
            """Read a DAC-3D inspection result summary."""
            return tools.read_latest_result(sample_position=sample_position)

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

        @function_tool(
            name_override="conversation_memory_search",
            description_override=(
                "Search JSON conversation memory for prior user goals, preferences, "
                "previous answers, and cross-turn references."
            ),
        )
        def conversation_memory_search(
            query: str,
            include_global: bool = True,
            limit: int = 5,
        ) -> dict[str, Any]:
            """Search JSON conversation memory."""
            return self.search_conversation_memory(
                query,
                session_id=session_id,
                include_global=include_global,
                limit=limit,
            )

        @function_tool(
            name_override="conversation_memory_recent",
            description_override="Return recent JSON-backed conversation turns for this session.",
        )
        def conversation_memory_recent(limit: int = 4) -> dict[str, Any]:
            """Return recent JSON conversation memory turns."""
            return self.recent_conversation_memory(
                session_id=session_id,
                limit=limit,
            )

        @function_tool(
            name_override="conversation_memory_profile",
            description_override=(
                "Read Hermes-style curated MEMORY.md and USER.md plus the routed "
                "knowledge-note index."
            ),
        )
        def conversation_memory_profile() -> dict[str, Any]:
            """Read curated core/user memory and topic index."""
            return self.conversation_memory_profile()

        @function_tool(
            name_override="conversation_memory_update",
            description_override=(
                "Propose a bounded curated memory patch. target must be memory or user; "
                "mode can be append or replace. This does not apply memory until approved."
            ),
        )
        def conversation_memory_update(
            target: str,
            content: str,
            mode: str = "append",
        ) -> dict[str, Any]:
            """Propose a curated core/user memory patch."""
            return self.update_conversation_memory(
                target=target,
                content=content,
                mode=mode,
            )

        @function_tool(
            name_override="conversation_knowledge_notes",
            description_override="List routed topic knowledge notes available to Memory Agent.",
        )
        def conversation_knowledge_notes() -> dict[str, Any]:
            """List topic-routed knowledge notes."""
            return self.list_conversation_knowledge_notes()

        @function_tool(
            name_override="conversation_knowledge_read",
            description_override="Read a topic-routed Markdown knowledge note by topic.",
        )
        def conversation_knowledge_read(topic: str) -> dict[str, Any]:
            """Read one topic-routed knowledge note."""
            return self.read_conversation_knowledge_note(topic)

        @function_tool(
            name_override="conversation_knowledge_write",
            description_override=(
                "Propose a topic-routed Markdown knowledge-note patch. "
                "mode can be append or replace. This does not apply memory until approved."
            ),
        )
        def conversation_knowledge_write(
            topic: str,
            content: str,
            mode: str = "append",
        ) -> dict[str, Any]:
            """Propose a topic-routed knowledge-note patch."""
            return self.write_conversation_knowledge_note(
                topic=topic,
                content=content,
                mode=mode,
            )

        @function_tool(
            name_override="conversation_procedure_memories",
            description_override="List reviewed Markdown procedure memories available to Memory Agent.",
        )
        def conversation_procedure_memories() -> dict[str, Any]:
            """List approved procedure memories."""
            return self.list_conversation_procedure_memories()

        @function_tool(
            name_override="conversation_procedure_read",
            description_override="Read one reviewed Markdown procedure memory by name.",
        )
        def conversation_procedure_read(name: str) -> dict[str, Any]:
            """Read one approved procedure memory."""
            return self.read_conversation_procedure_memory(name)

        @function_tool(
            name_override="conversation_procedure_write",
            description_override=(
                "Propose a reviewed Markdown procedure memory patch. This records a "
                "pending procedure_memory patch only; it never edits procedure files "
                "until the patch is approved."
            ),
        )
        def conversation_procedure_write(
            name: str,
            content: str,
            reason: str = "procedure_memory_candidate",
            mode: str = "append",
        ) -> dict[str, Any]:
            """Propose a procedure memory patch."""
            return self.write_conversation_procedure_memory(
                name=name,
                content=content,
                reason=reason,
                mode=mode,
            )

        @function_tool(
            name_override="conversation_memory_patches",
            description_override="List auditable pending/approved/rejected Memory OS patches.",
        )
        def conversation_memory_patches(status: str = "pending") -> dict[str, Any]:
            """List Memory OS patches."""
            return self.list_conversation_memory_patches(status=status)

        @function_tool(
            name_override="conversation_memory_approve_patch",
            description_override=(
                "Approve one pending Memory OS patch by id. Use only when the user "
                "explicitly asks to approve or save that patch."
            ),
        )
        def conversation_memory_approve_patch(patch_id: str) -> dict[str, Any]:
            """Approve a pending Memory OS patch."""
            return self.approve_conversation_memory_patch(patch_id)

        @function_tool(
            name_override="conversation_memory_reject_patch",
            description_override="Reject one pending Memory OS patch by id with an optional reason.",
        )
        def conversation_memory_reject_patch(
            patch_id: str,
            reason: str = "",
        ) -> dict[str, Any]:
            """Reject a pending Memory OS patch."""
            return self.reject_conversation_memory_patch(patch_id, reason=reason)

        @function_tool(
            name_override="dac_skill_list",
            description_override="List registered DAC-Agent skills and their metadata.",
        )
        def dac_skill_list() -> dict[str, Any]:
            """List local DAC-Agent skills."""
            return self.list_dac_skills()

        @function_tool(
            name_override="dac_skill_select",
            description_override="Select relevant DAC-Agent skills for the current task.",
        )
        def dac_skill_select(task: str, limit: int = 3) -> dict[str, Any]:
            """Select skills for a task."""
            return self.select_dac_skills(task, limit=limit)

        @function_tool(
            name_override="dac_skill_read",
            description_override=(
                "Read one DAC-Agent skill by name. Set include_assets=true only when "
                "the task needs schema, examples, templates, or reference files."
            ),
        )
        def dac_skill_read(name: str, include_assets: bool = False) -> dict[str, Any]:
            """Read one skill on demand."""
            return self.read_dac_skill(name, include_assets=include_assets)

        @function_tool(
            name_override="dac_skill_propose_patch",
            description_override=(
                "Propose a reviewable change to a DAC-Agent skill. This records a "
                "pending patch only; it never edits SKILL.md automatically."
            ),
        )
        def dac_skill_propose_patch(
            target_skill: str,
            reason: str,
            diff: str = "",
            replacement_section: str = "",
            evidence_trace_ids: str = "",
            risk_level: str = "medium",
        ) -> dict[str, Any]:
            """Propose a skill patch without applying it."""
            trace_ids = [
                item.strip()
                for item in str(evidence_trace_ids or "").split(",")
                if item.strip()
            ]
            return self.propose_dac_skill_patch(
                target_skill=target_skill,
                reason=reason,
                diff=diff,
                replacement_section=replacement_section,
                evidence_trace_ids=trace_ids,
                risk_level=risk_level,
                proposed_by="skill_agent",
            )

        @function_tool(
            name_override="dac_skill_patches",
            description_override="List pending/approved/rejected DAC-Agent skill patch proposals.",
        )
        def dac_skill_patches(status: str = "pending") -> dict[str, Any]:
            """List reviewable skill patches."""
            return self.list_dac_skill_patches(status=status)

        @function_tool(
            name_override="dac_skill_approve_patch",
            description_override=(
                "Mark a pending skill patch approved for human review records. "
                "This does not edit SKILL.md automatically."
            ),
        )
        def dac_skill_approve_patch(patch_id: str) -> dict[str, Any]:
            """Approve one skill patch proposal."""
            return self.approve_dac_skill_patch(patch_id)

        @function_tool(
            name_override="dac_skill_reject_patch",
            description_override="Reject one DAC-Agent skill patch proposal by id.",
        )
        def dac_skill_reject_patch(patch_id: str, reason: str = "") -> dict[str, Any]:
            """Reject one skill patch proposal."""
            return self.reject_dac_skill_patch(patch_id, reason=reason)

        @function_tool(
            name_override="dac3d_safety_review",
            description_override=(
                "Review a DAC-3D control instruction before execution. It previews the "
                "command, checks missing fields, confirmation requirements, runtime busy "
                "state, hardware risk, and warnings. It never executes the command."
            ),
        )
        def dac3d_safety_review(
            instruction: str,
            confirmed_by_user: bool = False,
        ) -> dict[str, Any]:
            """Review DAC-3D command safety without executing."""
            return self.review_command_safety(
                instruction,
                confirmed_by_user=confirmed_by_user,
                session_id=session_id,
            )

        @function_tool(
            name_override="dac_tool_manifest",
            description_override="Return Tool Gateway descriptors, risk metadata, and path policy.",
        )
        def dac_tool_manifest() -> dict[str, Any]:
            """Describe the controlled DAC Tool Gateway."""
            return tools.tool_gateway_manifest()

        @function_tool(
            name_override="dac_mcp_manifest",
            description_override=(
                "Return MCP-compatible capability metadata for DAC tools, resources, "
                "prompts, roots, and deployment modes."
            ),
        )
        def dac_mcp_manifest() -> dict[str, Any]:
            """Describe future MCP/Agents SDK integration surfaces."""
            return tools.mcp_capability_manifest()

        @function_tool(
            name_override="dac_tool_allowed_dirs",
            description_override="List DAC directories that path-sensitive tools may access.",
        )
        def dac_tool_allowed_dirs() -> dict[str, Any]:
            """List path allowlist roots."""
            return tools.list_allowed_dirs()

        @function_tool(
            name_override="dac_tool_validate_command",
            description_override="Validate a command preview for schema, path, runtime, and safety policy.",
        )
        def dac_tool_validate_command(command_preview_json: str) -> dict[str, Any]:
            """Validate a command preview through the Tool Gateway."""
            parsed_preview = _parse_json_object(command_preview_json) or {}
            if not parsed_preview:
                return {
                    "tool": "dac_tool_validate_command",
                    "ok": False,
                    "error": "command_preview_json must be a JSON object string.",
                }
            command_preview = parsed_preview
            return tools.validate_command(command_preview)

        @function_tool(
            name_override="dac_tool_cancel_pending_command",
            description_override="Cancel the current pending DAC command preview by optional preview id.",
        )
        def dac_tool_cancel_pending_command(command_preview_id: str = "") -> dict[str, Any]:
            """Cancel a pending command preview."""
            return tools.cancel_pending_command(command_preview_id)

        @function_tool(
            name_override="dac_tool_command_history",
            description_override="Read recent Tool Gateway command preview, submit, and cancel events.",
        )
        def dac_tool_command_history(limit: int = 20) -> dict[str, Any]:
            """Read command history for this session."""
            return tools.read_command_history(limit=limit)

        model: Any = self.config.agent_model_name or None
        if _looks_like_local_validation_model(self.config.agent_model_name):
            class RuntimeLocalValidationAgentModel(LocalValidationAgentModel, Model):
                pass

            model = RuntimeLocalValidationAgentModel()

        qa_tools = [
            dac3d_answer,
            dac3d_rebuild_knowledge_base,
        ]
        control_tools = [
            dac3d_operation,
            dac3d_preview_command,
            dac3d_execute_command,
            dac3d_status,
            dac3d_safety_review,
            dac_tool_manifest,
            dac_mcp_manifest,
            dac_tool_allowed_dirs,
            dac_tool_validate_command,
            dac_tool_cancel_pending_command,
            dac_tool_command_history,
        ]
        result_tools = [
            dac3d_latest_result,
            dac3d_answer,
        ]
        machine_tools = [
            machine_agent_chat,
            machine_snapshot,
            machine_status,
            machine_history,
            machine_alarms,
            machine_docs,
            machine_condition_summary,
            machine_abnormal_analysis,
        ]
        memory_tools = [
            conversation_memory_search,
            conversation_memory_recent,
            conversation_memory_profile,
            conversation_memory_update,
            conversation_knowledge_notes,
            conversation_knowledge_read,
            conversation_knowledge_write,
            conversation_procedure_memories,
            conversation_procedure_read,
            conversation_procedure_write,
            conversation_memory_patches,
            conversation_memory_approve_patch,
            conversation_memory_reject_patch,
        ]
        safety_tools = [
            dac3d_safety_review,
            dac3d_preview_command,
            dac3d_status,
            dac_tool_manifest,
            dac_mcp_manifest,
            dac_tool_allowed_dirs,
            dac_tool_validate_command,
            dac_tool_cancel_pending_command,
            dac_tool_command_history,
        ]
        skill_tools = [
            dac_skill_list,
            dac_skill_select,
            dac_skill_read,
            dac_skill_propose_patch,
            dac_skill_patches,
            dac_skill_approve_patch,
            dac_skill_reject_patch,
        ]
        all_tools = [
            dac3d_answer,
            dac3d_rebuild_knowledge_base,
            *control_tools,
            dac3d_latest_result,
            *machine_tools,
            *memory_tools,
            *skill_tools,
        ]

        qa_agent = Agent(
            name="DAC-3D QA Agent",
            handoff_description=(
                "DAC-3D documents, parameters, workflow, troubleshooting, and guidance."
            ),
            instructions=DAC3D_QA_AGENT_INSTRUCTIONS,
            tools=qa_tools,
            model=model,
        )
        control_agent = Agent(
            name="DAC-3D Control Agent",
            handoff_description=(
                "DAC-3D status, scan/offline/stop command preview, and confirmed execution."
            ),
            instructions=DAC3D_CONTROL_AGENT_INSTRUCTIONS,
            tools=control_tools,
            model=model,
        )
        result_agent = Agent(
            name="DAC-3D Result Agent",
            handoff_description=(
                "DAC-3D latest result, sample result, defect severity, and interpretation."
            ),
            instructions=DAC3D_RESULT_AGENT_INSTRUCTIONS,
            tools=result_tools,
            model=model,
        )
        machine_agent = Agent(
            name="Machine Agent",
            handoff_description=(
                "Industrial equipment status, telemetry history, alarms, maintenance docs, "
                "and abnormal-pattern analysis."
            ),
            instructions=MACHINE_AGENT_INSTRUCTIONS,
            tools=machine_tools,
            model=model,
        )
        memory_agent = Agent(
            name="Memory Agent",
            handoff_description=(
                "JSON conversation memory, previous user goals, prior answers, and preferences."
            ),
            instructions=MEMORY_AGENT_INSTRUCTIONS,
            tools=memory_tools,
            model=model,
        )
        skill_agent = Agent(
            name="Skill Agent",
            handoff_description=(
                "DAC-Agent skill discovery, task skill selection, and progressive skill loading."
            ),
            instructions=SKILL_AGENT_INSTRUCTIONS,
            tools=skill_tools,
            model=model,
        )
        safety_agent = Agent(
            name="Safety Agent",
            handoff_description=(
                "DAC-3D command safety, confirmation requirements, runtime busy checks, "
                "and execution risk review."
            ),
            instructions=SAFETY_AGENT_INSTRUCTIONS,
            tools=safety_tools,
            model=model,
        )
        coordinator = Agent(
            name="DAC-3D Multi-Agent Coordinator",
            instructions=AGENT_INSTRUCTIONS,
            handoffs=[
                handoff(
                    qa_agent,
                    tool_name_override="handoff_dac3d_qa_agent",
                    tool_description_override=(
                        "Send DAC-3D document, parameter, workflow, troubleshooting, "
                        "and operator guidance questions to the QA specialist."
                    ),
                ),
                handoff(
                    control_agent,
                    tool_name_override="handoff_dac3d_control_agent",
                    tool_description_override=(
                        "Send DAC-3D status, command preview, confirmed execution, scan, "
                        "offline detection, and stop requests to the control specialist."
                    ),
                ),
                handoff(
                    result_agent,
                    tool_name_override="handoff_dac3d_result_agent",
                    tool_description_override=(
                        "Send latest inspection result, sample result, defect severity, "
                        "and result interpretation questions to the result specialist."
                    ),
                ),
                handoff(
                    machine_agent,
                    tool_name_override="handoff_machine_agent",
                    tool_description_override=(
                        "Send equipment status, history, alarms, fault-code, maintenance, "
                        "and abnormal-pattern questions to the machine specialist."
                    ),
                ),
                handoff(
                    memory_agent,
                    tool_name_override="handoff_memory_agent",
                    tool_description_override=(
                        "Send questions about prior conversation, memory, user preferences, "
                        "or cross-turn references to the memory specialist."
                    ),
                ),
                handoff(
                    skill_agent,
                    tool_name_override="handoff_skill_agent",
                    tool_description_override=(
                        "Send skill discovery, skill selection, and skill loading questions "
                        "to the skill specialist."
                    ),
                ),
                handoff(
                    safety_agent,
                    tool_name_override="handoff_safety_agent",
                    tool_description_override=(
                        "Send command risk, execution readiness, confirmation, and safety "
                        "review questions to the safety specialist."
                    ),
                ),
            ],
            model=model,
        )

        agents_by_role = {
            "coordinator": coordinator,
            "root": coordinator,
            "qa": qa_agent,
            "dac3d_qa": qa_agent,
            "control": control_agent,
            "dac3d_control": control_agent,
            "result": result_agent,
            "dac3d_result": result_agent,
            "machine": machine_agent,
            "memory": memory_agent,
            "skill": skill_agent,
            "safety": safety_agent,
            "all_tools": Agent(
                name="DAC-3D Flat Tool Compatibility Agent",
                instructions=AGENT_INSTRUCTIONS,
                tools=all_tools,
                model=model,
            ),
        }
        normalized_role = role.strip().lower().replace("-", "_")
        if normalized_role not in agents_by_role:
            raise ValueError(
                "Unknown DAC-3D Agent role. Expected one of: "
                + ", ".join(sorted(agents_by_role))
            )
        return agents_by_role[normalized_role]

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
    memory_provider: LocalMemoryProvider | None = None
    skill_registry: SkillRegistry | None = None
    skill_patch_store: SkillPatchStore | None = None
    context_builder: ContextBuilder | None = None
    context_tree: FileBackedContextTree | None = None
    repo_context_map: RepoContextMapStore | None = None
    git_workspace_context: GitWorkspaceContext | None = None
    agent_registry_store: AgentRegistryStore | None = None
    agent_fleet_store: AgentFleetStore | None = None
    agent_deployment_store: AgentDeploymentStore | None = None
    agent_labeling_store: AgentLabelingStore | None = None
    agent_performance_store: AgentPerformanceStore | None = None
    conversation_thread_store: ConversationThreadStore | None = None
    browser_context_store: BrowserContextStore | None = None
    goal_store: GoalStore | None = None
    task_board_store: TaskBoardStore | None = None
    automation_store: AutomationPlannerStore | None = None
    workflow_store: WorkflowTemplateStore | None = None
    artifact_store: ArtifactStore | None = None
    event_queue_store: EventQueueStore | None = None
    verification_store: VerificationRunnerStore | None = None
    review_handoff_store: ReviewHandoffStore | None = None
    checkpoint_store: CheckpointStore | None = None
    shared_state_store: SharedStateStore | None = None
    trace_logger: TraceLogger | None = None
    observability_reporter: ObservabilityReporter | None = None
    tool_marketplace_store: ToolMarketplaceStore | None = None

    def __post_init__(self) -> None:
        if self.memory_store is None and self.runtime.memory_store is not None:
            self.memory_store = self.runtime.memory_store
        if self.memory_provider is None and self.runtime.memory_provider is not None:
            self.memory_provider = self.runtime.memory_provider
        if self.memory_store is None and self.config.memory_enabled:
            self.memory_store = ConversationMemoryStore.from_config(self.config)
            self.memory_store.ensure_directories()
            self.runtime.memory_store = self.memory_store
        if self.memory_provider is None and self.memory_store is not None:
            self.memory_provider = LocalMemoryProvider(self.memory_store)
            self.runtime.memory_provider = self.memory_provider
        if self.skill_registry is None and self.runtime.skill_registry is not None:
            self.skill_registry = self.runtime.skill_registry
        if self.skill_registry is None:
            self.skill_registry = SkillRegistry(self.config.agent_skills_dir)
            self.runtime.skill_registry = self.skill_registry
        if self.skill_patch_store is None and self.runtime.skill_patch_store is not None:
            self.skill_patch_store = self.runtime.skill_patch_store
        if self.skill_patch_store is None:
            self.skill_patch_store = SkillPatchStore.from_root(
                self.config.conversation_memory_dir,
                self.skill_registry,
            )
            self.runtime.skill_patch_store = self.skill_patch_store
        if self.context_tree is None:
            self.context_tree = FileBackedContextTree(self.config.conversation_memory_dir / "context_tree")
            self.context_tree.ensure_defaults()
        if self.repo_context_map is None:
            self.repo_context_map = RepoContextMapStore.from_config_root(
                self.config.base_dir,
                self.config.conversation_memory_dir,
            )
        if self.git_workspace_context is None:
            self.git_workspace_context = GitWorkspaceContext.from_config_root(self.config.base_dir)
        if self.agent_registry_store is None:
            self.agent_registry_store = AgentRegistryStore.from_root(self.config.conversation_memory_dir)
            self.agent_registry_store.seed_defaults(_default_agent_registry_entries())
        if self.agent_fleet_store is None:
            self.agent_fleet_store = AgentFleetStore.from_root(self.config.conversation_memory_dir)
        if self.agent_deployment_store is None:
            self.agent_deployment_store = AgentDeploymentStore.from_root(
                self.config.conversation_memory_dir
            )
        if self.agent_labeling_store is None:
            self.agent_labeling_store = AgentLabelingStore.from_root(self.config.conversation_memory_dir)
        if self.agent_performance_store is None:
            self.agent_performance_store = AgentPerformanceStore.from_root(
                self.config.conversation_memory_dir
            )
        if self.conversation_thread_store is None:
            self.conversation_thread_store = ConversationThreadStore.from_root(
                self.config.conversation_memory_dir
            )
        if self.browser_context_store is None:
            self.browser_context_store = BrowserContextStore.from_root(self.config.conversation_memory_dir)
        if self.goal_store is None:
            self.goal_store = GoalStore.from_root(self.config.conversation_memory_dir)
        if self.task_board_store is None:
            self.task_board_store = TaskBoardStore.from_root(self.config.conversation_memory_dir)
        if self.automation_store is None:
            self.automation_store = AutomationPlannerStore.from_root(self.config.conversation_memory_dir)
        if self.workflow_store is None:
            self.workflow_store = WorkflowTemplateStore.from_root(self.config.conversation_memory_dir)
        if self.artifact_store is None:
            self.artifact_store = ArtifactStore.from_root(self.config.conversation_memory_dir)
        if self.event_queue_store is None:
            self.event_queue_store = EventQueueStore.from_root(self.config.conversation_memory_dir)
        if self.verification_store is None:
            self.verification_store = VerificationRunnerStore.from_root(
                self.config.conversation_memory_dir,
                self.config.base_dir,
            )
        if self.review_handoff_store is None:
            self.review_handoff_store = ReviewHandoffStore.from_root(self.config.conversation_memory_dir)
        if self.checkpoint_store is None:
            self.checkpoint_store = CheckpointStore.from_root(self.config.conversation_memory_dir)
        if self.shared_state_store is None:
            self.shared_state_store = SharedStateStore.from_root(self.config.conversation_memory_dir)
        if self.tool_marketplace_store is None:
            self.tool_marketplace_store = ToolMarketplaceStore.from_root(self.config.conversation_memory_dir)
            self.tool_marketplace_store.seed_defaults(_default_tool_marketplace_entries())
        if self.context_builder is None:
            self.context_builder = ContextBuilder(
                memory_provider=self.memory_provider,
                memory_store=self.memory_store,
                skill_registry=self.skill_registry,
                context_tree=self.context_tree,
                runtime_status_getter=self.runtime.assistant.dac3d_client.runtime_snapshot,
                skill_limit=self.config.context_skill_limit,
                char_limit=self.config.context_char_limit,
            )
        if self.trace_logger is None:
            self.trace_logger = TraceLogger(self.config.conversation_memory_dir / "agent_traces.jsonl")
        if self.observability_reporter is None:
            self.observability_reporter = ObservabilityReporter(
                trace_logger=self.trace_logger,
                event_queue_store=self.event_queue_store,
                verification_store=self.verification_store,
                review_handoff_store=self.review_handoff_store,
                checkpoint_store=self.checkpoint_store,
            )

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
        agent_input, memory_bundle = self._format_agent_input(
            message,
            history,
            session_id=session_id,
        )
        try:
            payload = self.runtime.run_chat_payload(agent_input, session_id=session_id)
        except Exception as exc:
            return AssistantResponse(intent="agent", answer=f"Agent run failed: {exc}")
        self._attach_memory_metadata(payload, memory_bundle)
        response = self._response_from_payload(payload)
        self._record_memory_trace(
            session_id=session_id,
            message=message,
            agent_input=agent_input,
            context_bundle=memory_bundle,
            response=response,
        )
        self._record_agent_trace(
            session_id=session_id,
            message=message,
            response=response,
            context_bundle=memory_bundle,
            event_type="agent_chat",
        )
        self._maybe_capture_goal(session_id=session_id, message=message, response=response)
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
        if self.memory_store is not None:
            memory_summary = self.memory_store.describe()
            if self.memory_provider is not None:
                memory_summary["memory_os"] = self.memory_provider.describe()
            summary["memory"] = memory_summary
        else:
            summary["memory"] = {"enabled": False, "backend": "json+markdown"}
        summary["skills"] = (
            self.skill_registry.describe()
            if self.skill_registry is not None
            else {"enabled": False, "backend": "local_agent_skills"}
        )
        summary["skill_patches"] = (
            self.skill_patch_store.describe()
            if self.skill_patch_store is not None
            else {"enabled": False, "backend": "json_skill_patch_queue"}
        )
        summary["context_builder"] = (
            self.context_builder.describe()
            if self.context_builder is not None
            else {"enabled": False, "backend": "context_builder"}
        )
        summary["context_tree"] = (
            self.context_tree.describe()
            if self.context_tree is not None
            else {"enabled": False, "backend": "file_context_tree"}
        )
        summary["repo_context_map"] = (
            self.repo_context_map.describe()
            if self.repo_context_map is not None
            else {"enabled": False, "backend": "static_repo_context_map"}
        )
        summary["code_symbols"] = (
            self.repo_context_map.describe_symbols()
            if self.repo_context_map is not None
            else {"enabled": False, "backend": "code_symbol_navigator"}
        )
        summary["git_workspace"] = (
            self.git_workspace_context.describe()
            if self.git_workspace_context is not None
            else {"enabled": False, "backend": "git_workspace_context"}
        )
        summary["agent_registry"] = (
            self.agent_registry_store.describe()
            if self.agent_registry_store is not None
            else {"enabled": False, "backend": "local_agent_registry"}
        )
        summary["agent_fleet"] = (
            self.agent_fleet_store.describe()
            if self.agent_fleet_store is not None
            else {"enabled": False, "backend": "local_agent_fleet"}
        )
        summary["agent_deployments"] = (
            self.agent_deployment_store.describe()
            if self.agent_deployment_store is not None
            else {"enabled": False, "backend": "local_agent_deployment_catalog"}
        )
        summary["agent_labeling"] = (
            self.agent_labeling_store.describe()
            if self.agent_labeling_store is not None
            else {"enabled": False, "backend": "local_agent_labeling_queue"}
        )
        summary["agent_performance"] = (
            self.agent_performance_store.describe()
            if self.agent_performance_store is not None
            else {"enabled": False, "backend": "local_agent_performance_store"}
        )
        summary["conversation_threads"] = (
            self.conversation_thread_store.describe()
            if self.conversation_thread_store is not None
            else {"enabled": False, "backend": "local_agent_threads"}
        )
        summary["browser_contexts"] = (
            self.browser_context_store.describe()
            if self.browser_context_store is not None
            else {"enabled": False, "backend": "local_browser_context_store"}
        )
        summary["goals"] = (
            self.goal_store.describe()
            if self.goal_store is not None
            else {"enabled": False, "backend": "local_goal_store"}
        )
        summary["task_board"] = (
            self.task_board_store.describe()
            if self.task_board_store is not None
            else {"enabled": False, "backend": "local_agent_task_board"}
        )
        summary["automations"] = (
            self.automation_store.describe()
            if self.automation_store is not None
            else {"enabled": False, "backend": "local_automation_planner"}
        )
        summary["workflow_templates"] = (
            self.workflow_store.describe()
            if self.workflow_store is not None
            else {"enabled": False, "backend": "local_workflow_templates"}
        )
        summary["artifacts"] = (
            self.artifact_store.describe()
            if self.artifact_store is not None
            else {"enabled": False, "backend": "local_agent_artifact_store"}
        )
        summary["event_queue"] = (
            self.event_queue_store.describe()
            if self.event_queue_store is not None
            else {"enabled": False, "backend": "local_agent_event_queue"}
        )
        summary["verification_feedback"] = (
            self.verification_store.describe()
            if self.verification_store is not None
            else {"enabled": False, "backend": "local_verification_runner"}
        )
        summary["review_handoffs"] = (
            self.review_handoff_store.describe()
            if self.review_handoff_store is not None
            else {"enabled": False, "backend": "local_review_handoff_queue"}
        )
        summary["checkpoints"] = (
            self.checkpoint_store.describe()
            if self.checkpoint_store is not None
            else {"enabled": False, "backend": "local_agent_checkpoint_store"}
        )
        summary["shared_state"] = (
            self.shared_state_store.describe()
            if self.shared_state_store is not None
            else {"enabled": False, "backend": "local_agent_shared_state"}
        )
        summary["tool_marketplace"] = (
            self.tool_marketplace_store.describe()
            if self.tool_marketplace_store is not None
            else {"enabled": False, "backend": "local_tool_marketplace"}
        )
        summary["observability"] = (
            self.observability_reporter.describe()
            if self.observability_reporter is not None
            else {"enabled": False, "backend": "local_agent_observability"}
        )
        summary["trace_eval"] = {
            "trace_logger": self.trace_logger.describe()
            if self.trace_logger is not None
            else {"enabled": False},
            "eval_runner": {
                "enabled": True,
                "backend": "local_deterministic_eval",
                "cases_dir": str(self.config.base_dir / "evals" / "cases"),
                "workflow": "traces -> eval cases -> regression report",
            },
            "eval_draft_generator": {
                "enabled": True,
                "backend": "trace_to_eval_draft",
                "drafts_dir": str(self.config.base_dir / "evals" / "drafts"),
                "workflow": "trace -> eval draft -> human review -> evals/cases",
                "auto_approved": False,
            },
            "codex_handoff_generator": {
                "enabled": True,
                "backend": "codex_handoff_generator",
                "path": str(self.config.base_dir / "docs" / "generated" / "codex_handoff_next.md"),
                "workflow": "failing evals + traces -> Codex handoff -> next implementation pass",
                "auto_applied": False,
            },
        }
        return summary

    def mcp_capability_manifest(self, *, session_id: str = "web") -> dict[str, Any]:
        """Expose MCP-compatible capability discovery through the chat adapter."""
        return self.runtime.mcp_capability_manifest(session_id=session_id)

    def knowledge_base_summary(self) -> dict[str, Any]:
        """Delegate knowledge-base diagnostics to the underlying assistant."""
        return self.runtime.assistant.knowledge_base_summary()

    def build_knowledge_base_from_uploads(self, uploaded_files: Sequence[Any] | None = None) -> dict[str, Any]:
        """Delegate knowledge-base rebuilds to the underlying assistant."""
        return self.runtime.assistant.build_knowledge_base_from_uploads(uploaded_files)

    def run_evals(self, *, categories: list[str] | None = None) -> dict[str, Any]:
        """Run bundled deterministic eval cases against this Agent adapter."""
        runner = EvalRunner(
            cases_dir=self.config.base_dir / "evals" / "cases",
            chat_handler=lambda message, session_id: self.handle_message(
                message,
                [],
                session_id=session_id,
            ),
            approve_handler=lambda session_id, preview_id, confirmation_token: self.approve_pending_command(
                session_id=session_id,
                preview_id=preview_id,
                confirmation_token=confirmation_token,
            ),
        )
        original_agent_model = self.runtime.config.agent_model_name
        self.runtime.config.agent_model_name = LOCAL_VALIDATION_MODEL_NAME
        try:
            result = runner.run(categories=categories)
        finally:
            self.runtime.config.agent_model_name = original_agent_model
        result["model"] = LOCAL_VALIDATION_MODEL_NAME
        if self.trace_logger is not None:
            result["trace_logger"] = self.trace_logger.describe()
        return result

    def generate_eval_drafts(
        self,
        *,
        trace_ids: list[str] | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Generate reviewable eval case drafts from selected or recent traces."""
        if self.trace_logger is None:
            raise ValueError("Trace logger is not enabled.")
        generator = EvalDraftGenerator(
            trace_logger=self.trace_logger,
            drafts_dir=self.config.base_dir / "evals" / "drafts",
        )
        return generator.generate(trace_ids=trace_ids, limit=limit)

    def list_eval_drafts(self) -> dict[str, Any]:
        """List reviewable eval drafts without running them as approved cases."""
        if self.trace_logger is None:
            raise ValueError("Trace logger is not enabled.")
        generator = EvalDraftGenerator(
            trace_logger=self.trace_logger,
            drafts_dir=self.config.base_dir / "evals" / "drafts",
        )
        return generator.list_drafts()

    def generate_codex_handoff(
        self,
        *,
        categories: list[str] | None = None,
        recent_trace_limit: int = 8,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run local evals and generate a Codex handoff markdown document."""
        if self.trace_logger is None:
            raise ValueError("Trace logger is not enabled.")
        eval_result = self.run_evals(categories=categories)
        generator = CodexHandoffGenerator(
            trace_logger=self.trace_logger,
            output_path=output_path
            or self.config.base_dir / "docs" / "generated" / "codex_handoff_next.md",
        )
        return generator.generate(
            eval_result=eval_result,
            recent_trace_limit=recent_trace_limit,
        )

    def agent_workspace(self) -> dict[str, Any]:
        """Return a UI-ready overview of the local multi-agent workspace."""
        agent = self.runtime.describe()
        context_tree = (
            self.context_tree.describe()
            if self.context_tree is not None
            else {"enabled": False, "backend": "file_context_tree", "nodes": []}
        )
        repo_context_map = (
            self.repo_context_map.describe()
            if self.repo_context_map is not None
            else {"enabled": False, "backend": "static_repo_context_map"}
        )
        code_symbols = (
            self.repo_context_map.describe_symbols()
            if self.repo_context_map is not None
            else {"enabled": False, "backend": "code_symbol_navigator"}
        )
        git_workspace = (
            self.git_workspace_context.describe()
            if self.git_workspace_context is not None
            else {"enabled": False, "backend": "git_workspace_context"}
        )
        agent_registry = (
            self.agent_registry_store.describe()
            if self.agent_registry_store is not None
            else {"enabled": False, "backend": "local_agent_registry"}
        )
        agent_fleet = (
            self.agent_fleet_store.describe()
            if self.agent_fleet_store is not None
            else {"enabled": False, "backend": "local_agent_fleet"}
        )
        agent_deployments = (
            self.agent_deployment_store.describe()
            if self.agent_deployment_store is not None
            else {"enabled": False, "backend": "local_agent_deployment_catalog"}
        )
        agent_labeling = (
            self.agent_labeling_store.describe()
            if self.agent_labeling_store is not None
            else {"enabled": False, "backend": "local_agent_labeling_queue"}
        )
        agent_performance = (
            self.agent_performance_store.describe()
            if self.agent_performance_store is not None
            else {"enabled": False, "backend": "local_agent_performance_store"}
        )
        conversation_threads = (
            self.conversation_thread_store.describe()
            if self.conversation_thread_store is not None
            else {"enabled": False, "backend": "local_agent_threads"}
        )
        browser_contexts = (
            self.browser_context_store.describe()
            if self.browser_context_store is not None
            else {"enabled": False, "backend": "local_browser_context_store"}
        )
        skills = (
            self.skill_registry.describe()
            if self.skill_registry is not None
            else {"enabled": False, "backend": "local_agent_skills", "skills": []}
        )
        skill_patches = (
            self.skill_patch_store.describe()
            if self.skill_patch_store is not None
            else {"enabled": False, "backend": "json_skill_patch_queue", "patch_count": 0}
        )
        memory = (
            self.memory_provider.describe()
            if self.memory_provider is not None
            else {"enabled": False, "backend": "local_memory_os"}
        )
        goals = (
            self.goal_store.describe()
            if self.goal_store is not None
            else {"enabled": False, "backend": "local_goal_store"}
        )
        task_board = (
            self.task_board_store.describe()
            if self.task_board_store is not None
            else {"enabled": False, "backend": "local_agent_task_board"}
        )
        automations = (
            self.automation_store.describe()
            if self.automation_store is not None
            else {"enabled": False, "backend": "local_automation_planner"}
        )
        workflow_templates = (
            self.workflow_store.describe()
            if self.workflow_store is not None
            else {"enabled": False, "backend": "local_workflow_templates"}
        )
        artifacts = (
            self.artifact_store.describe()
            if self.artifact_store is not None
            else {"enabled": False, "backend": "local_agent_artifact_store"}
        )
        event_queue = (
            self.event_queue_store.describe()
            if self.event_queue_store is not None
            else {"enabled": False, "backend": "local_agent_event_queue"}
        )
        verification_feedback = (
            self.verification_store.describe()
            if self.verification_store is not None
            else {"enabled": False, "backend": "local_verification_runner"}
        )
        review_handoffs = (
            self.review_handoff_store.describe()
            if self.review_handoff_store is not None
            else {"enabled": False, "backend": "local_review_handoff_queue"}
        )
        checkpoints = (
            self.checkpoint_store.describe()
            if self.checkpoint_store is not None
            else {"enabled": False, "backend": "local_agent_checkpoint_store"}
        )
        shared_state = (
            self.shared_state_store.describe()
            if self.shared_state_store is not None
            else {"enabled": False, "backend": "local_agent_shared_state"}
        )
        tool_marketplace = (
            self.tool_marketplace_store.describe()
            if self.tool_marketplace_store is not None
            else {"enabled": False, "backend": "local_tool_marketplace"}
        )
        observability = (
            self.observability_reporter.describe()
            if self.observability_reporter is not None
            else {"enabled": False, "backend": "local_agent_observability"}
        )
        return {
            "enabled": True,
            "backend": "dac_agent_workspace",
            "entry_agent": agent.get("entry_agent"),
            "specialist_agents": agent.get("specialist_agents", []),
            "tool_groups": agent.get("tool_groups", {}),
            "capabilities": agent.get("network_capabilities", []),
            "skills": skills,
            "skill_patches": skill_patches,
            "context_tree": context_tree,
            "repo_context_map": repo_context_map,
            "code_symbols": code_symbols,
            "git_workspace": git_workspace,
            "agent_registry": agent_registry,
            "agent_fleet": agent_fleet,
            "agent_deployments": agent_deployments,
            "agent_labeling": agent_labeling,
            "agent_performance": agent_performance,
            "conversation_threads": conversation_threads,
            "browser_contexts": browser_contexts,
            "memory_os": memory,
            "goals": goals,
            "task_board": task_board,
            "automations": automations,
            "workflow_templates": workflow_templates,
            "artifacts": artifacts,
            "event_queue": event_queue,
            "verification_feedback": verification_feedback,
            "review_handoffs": review_handoffs,
            "checkpoints": checkpoints,
            "shared_state": shared_state,
            "tool_marketplace": tool_marketplace,
            "observability": observability,
            "workflow": [
                "user_task",
                "goal_tracking",
                "task_board_card",
                "automation_planning",
                "workflow_template",
                "artifact_store",
                "event_queue",
                "verification_feedback",
                "review_handoff",
                "workflow_checkpoint",
                "shared_state",
                "tool_marketplace",
                "agent_deployment_catalog",
                "agent_labeling_queue",
                "agent_performance_analysis",
                "observability_snapshot",
                "coordinator_route",
                "skill_selection",
                "context_tree_search",
                "repo_context_map",
                "symbol_navigation",
                "git_workspace_context",
                "agent_registry",
                "agent_fleet",
                "agent_deployments",
                "agent_labeling",
                "agent_performance",
                "conversation_thread",
                "shared_browser_context",
                "memory_prefetch",
                "specialist_agent",
                "tool_loop",
                "trace_feedback",
            ],
        }

    def _checkpoint_workspace_state(self) -> dict[str, Any]:
        """Capture a compact workspace snapshot for a checkpoint state."""
        workspace = self.agent_workspace()
        return {
            "backend": "dac_agent_workspace_snapshot",
            "entry_agent": workspace.get("entry_agent"),
            "workflow": workspace.get("workflow", []),
            "counts": {
                "agents": (workspace.get("agent_registry") or {}).get("agent_count", 0),
                "fleet_instances": (workspace.get("agent_fleet") or {}).get("instance_count", 0),
                "deployments": (workspace.get("agent_deployments") or {}).get("deployment_count", 0),
                "labeling_items": (workspace.get("agent_labeling") or {}).get("item_count", 0),
                "performance_metrics": (workspace.get("agent_performance") or {}).get("metric_count", 0),
                "threads": (workspace.get("conversation_threads") or {}).get("thread_count", 0),
                "browser_contexts": (workspace.get("browser_contexts") or {}).get("context_count", 0),
                "goals": (workspace.get("goals") or {}).get("goal_count", 0),
                "tasks": (workspace.get("task_board") or {}).get("task_count", 0),
                "events": (workspace.get("event_queue") or {}).get("event_count", 0),
                "artifacts": (workspace.get("artifacts") or {}).get("artifact_count", 0),
                "reviews": (workspace.get("review_handoffs") or {}).get("review_count", 0),
                "checkpoints": (workspace.get("checkpoints") or {}).get("checkpoint_count", 0),
                "tool_packs": (workspace.get("tool_marketplace") or {}).get("entry_count", 0),
            },
        }

    def agent_observability(self, *, recent_trace_limit: int = 20) -> dict[str, Any]:
        """Return a read-only observability snapshot across Agent workspace signals."""
        if self.observability_reporter is None:
            raise ValueError("Agent observability is not enabled.")
        return self.observability_reporter.snapshot(recent_trace_limit=recent_trace_limit)

    def list_agent_registry(
        self,
        *,
        status: str | None = None,
        role: str | None = None,
        capability: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List local Agent registry entries for workspace discovery."""
        if self.agent_registry_store is None:
            return {"enabled": False, "backend": "local_agent_registry", "agents": [], "count": 0}
        return self.agent_registry_store.list_agents(
            status=status,
            role=role,
            capability=capability,
            query=query,
            limit=limit,
        )

    def read_agent_registry_entry(self, agent_id_or_role: str) -> dict[str, Any]:
        """Read one Agent registry entry by id or role."""
        if self.agent_registry_store is None:
            raise ValueError("Agent registry is not enabled.")
        return self.agent_registry_store.read_agent(agent_id_or_role)

    def register_agent_entry(
        self,
        name: str,
        *,
        role: str,
        description: str = "",
        status: str = "active",
        handoff_name: str = "",
        agent_type: str = "specialist",
        capabilities: list[Any] | None = None,
        tools: list[Any] | None = None,
        triggers: list[Any] | None = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        owner_agent: str = "web",
    ) -> dict[str, Any]:
        """Create or update one Agent registry entry."""
        if self.agent_registry_store is None:
            raise ValueError("Agent registry is not enabled.")
        return {
            "enabled": True,
            **self.agent_registry_store.register_agent(
                name,
                role=role,
                description=description,
                status=status,
                handoff_name=handoff_name,
                agent_type=agent_type,
                capabilities=capabilities,
                tools=tools,
                triggers=triggers,
                tags=tags,
                metadata=metadata,
                owner_agent=owner_agent,
            ),
        }

    def route_agent_candidates(self, task: str, *, limit: int = 5) -> dict[str, Any]:
        """Return registry-ranked specialist Agent candidates for a task."""
        if self.agent_registry_store is None:
            raise ValueError("Agent registry is not enabled.")
        return self.agent_registry_store.route_candidates(task, limit=limit)

    def list_agent_fleet(
        self,
        *,
        status: str | None = None,
        agent_role: str | None = None,
        environment: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List local Agent fleet instances for workload coordination."""
        if self.agent_fleet_store is None:
            return {"enabled": False, "backend": "local_agent_fleet", "instances": [], "count": 0}
        return self.agent_fleet_store.list_instances(
            status=status,
            agent_role=agent_role,
            environment=environment,
            tag=tag,
            query=query,
            limit=limit,
        )

    def read_agent_fleet_instance(self, instance_id_or_name: str) -> dict[str, Any]:
        """Read one local Agent fleet instance."""
        if self.agent_fleet_store is None:
            raise ValueError("Agent fleet is not enabled.")
        return self.agent_fleet_store.read_instance(instance_id_or_name)

    def register_agent_fleet_instance(
        self,
        name: str,
        *,
        agent_role: str,
        environment: str = "local",
        endpoint: str = "",
        status: str = "ready",
        capabilities: list[Any] | None = None,
        max_concurrency: int = 1,
        current_load: int = 0,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        owner_agent: str = "agent",
    ) -> dict[str, Any]:
        """Create or update one local Agent fleet instance."""
        if self.agent_fleet_store is None:
            raise ValueError("Agent fleet is not enabled.")
        return {
            "enabled": True,
            **self.agent_fleet_store.register_instance(
                name,
                agent_role=agent_role,
                environment=environment,
                endpoint=endpoint,
                status=status,
                capabilities=capabilities,
                max_concurrency=max_concurrency,
                current_load=current_load,
                tags=tags,
                metadata=metadata,
                owner_agent=owner_agent,
            ),
        }

    def heartbeat_agent_fleet_instance(
        self,
        instance_id_or_name: str,
        *,
        status: str = "ready",
        current_load: int | None = None,
        metrics: dict[str, Any] | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Record a heartbeat and load snapshot for one fleet instance."""
        if self.agent_fleet_store is None:
            raise ValueError("Agent fleet is not enabled.")
        return {
            "enabled": True,
            **self.agent_fleet_store.heartbeat(
                instance_id_or_name,
                status=status,
                current_load=current_load,
                metrics=metrics,
                note=note,
            ),
        }

    def assign_agent_fleet_task(
        self,
        instance_id_or_name: str,
        *,
        task_id: str,
        summary: str = "",
        thread_id: str = "",
        workflow_id: str = "",
        priority: str = "normal",
        metadata: dict[str, Any] | None = None,
        assigned_by: str = "coordinator",
    ) -> dict[str, Any]:
        """Assign one task/workflow item to a local fleet instance."""
        if self.agent_fleet_store is None:
            raise ValueError("Agent fleet is not enabled.")
        return {
            "enabled": True,
            **self.agent_fleet_store.assign_task(
                instance_id_or_name,
                task_id=task_id,
                summary=summary,
                thread_id=thread_id,
                workflow_id=workflow_id,
                priority=priority,
                metadata=metadata,
                assigned_by=assigned_by,
            ),
        }

    def update_agent_fleet_assignment_status(
        self,
        instance_id_or_name: str,
        assignment_id: str,
        status: str,
        *,
        note: str = "",
        actor: str = "agent",
    ) -> dict[str, Any]:
        """Update one fleet assignment status."""
        if self.agent_fleet_store is None:
            raise ValueError("Agent fleet is not enabled.")
        return {
            "enabled": True,
            **self.agent_fleet_store.update_assignment_status(
                instance_id_or_name,
                assignment_id,
                status,
                note=note,
                actor=actor,
            ),
        }

    def list_agent_deployments(
        self,
        *,
        status: str | None = None,
        environment: str | None = None,
        app_type: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List local Agent deployment catalog entries."""
        if self.agent_deployment_store is None:
            return {
                "enabled": False,
                "backend": "local_agent_deployment_catalog",
                "deployments": [],
                "count": 0,
            }
        return self.agent_deployment_store.list_deployments(
            status=status,
            environment=environment,
            app_type=app_type,
            tag=tag,
            query=query,
            limit=limit,
        )

    def read_agent_deployment(self, deployment_id_or_slug: str) -> dict[str, Any]:
        """Read one Agent deployment by id or slug."""
        if self.agent_deployment_store is None:
            raise ValueError("Agent deployment catalog is not enabled.")
        return self.agent_deployment_store.read_deployment(deployment_id_or_slug)

    def create_agent_deployment(
        self,
        name: str,
        *,
        entrypoint: str,
        slug: str = "",
        app_type: str = "agent_app",
        version: str = "0.1.0",
        environment: str = "local",
        route_path: str = "",
        status: str = "draft",
        workflow_ids: list[Any] | None = None,
        tool_pack_slugs: list[Any] | None = None,
        agent_roles: list[Any] | None = None,
        config_refs: list[Any] | None = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        created_by: str = "agent",
    ) -> dict[str, Any]:
        """Create or update one local deployable Agent app entry."""
        if self.agent_deployment_store is None:
            raise ValueError("Agent deployment catalog is not enabled.")
        return {
            "enabled": True,
            **self.agent_deployment_store.create_deployment(
                name,
                slug=slug,
                app_type=app_type,
                entrypoint=entrypoint,
                version=version,
                environment=environment,
                route_path=route_path,
                status=status,
                workflow_ids=workflow_ids,
                tool_pack_slugs=tool_pack_slugs,
                agent_roles=agent_roles,
                config_refs=config_refs,
                tags=tags,
                metadata=metadata,
                created_by=created_by,
            ),
        }

    def update_agent_deployment_status(
        self,
        deployment_id_or_slug: str,
        status: str,
        *,
        note: str = "",
        actor: str = "agent",
    ) -> dict[str, Any]:
        """Move one Agent deployment between local lifecycle statuses."""
        if self.agent_deployment_store is None:
            raise ValueError("Agent deployment catalog is not enabled.")
        return {
            "enabled": True,
            **self.agent_deployment_store.update_status(
                deployment_id_or_slug,
                status,
                note=note,
                actor=actor,
            ),
        }

    def record_agent_deployment_release(
        self,
        deployment_id_or_slug: str,
        *,
        version: str,
        summary: str = "",
        artifact_ids: list[Any] | None = None,
        verification_run_ids: list[Any] | None = None,
        released_by: str = "agent",
    ) -> dict[str, Any]:
        """Record a release snapshot for one Agent deployment entry."""
        if self.agent_deployment_store is None:
            raise ValueError("Agent deployment catalog is not enabled.")
        return {
            "enabled": True,
            **self.agent_deployment_store.record_release(
                deployment_id_or_slug,
                version=version,
                summary=summary,
                artifact_ids=artifact_ids,
                verification_run_ids=verification_run_ids,
                released_by=released_by,
            ),
        }

    def list_agent_labeling_items(
        self,
        *,
        status: str | None = None,
        source_type: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List local LLMOps labeling queue items."""
        if self.agent_labeling_store is None:
            return {
                "enabled": False,
                "backend": "local_agent_labeling_queue",
                "items": [],
                "count": 0,
            }
        return self.agent_labeling_store.list_items(
            status=status,
            source_type=source_type,
            tag=tag,
            query=query,
            limit=limit,
        )

    def read_agent_labeling_item(self, item_id: str) -> dict[str, Any]:
        """Read one LLMOps labeling queue item."""
        if self.agent_labeling_store is None:
            raise ValueError("Agent labeling queue is not enabled.")
        return self.agent_labeling_store.read_item(item_id)

    def create_agent_labeling_item(
        self,
        title: str,
        *,
        source_type: str = "manual",
        source_id: str = "",
        session_id: str = "",
        input_text: str = "",
        agent_output: str = "",
        intent: str = "",
        tool_calls: list[Any] | None = None,
        expected: dict[str, Any] | None = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        created_by: str = "agent",
    ) -> dict[str, Any]:
        """Create or update one local labeling sample."""
        if self.agent_labeling_store is None:
            raise ValueError("Agent labeling queue is not enabled.")
        return {
            "enabled": True,
            **self.agent_labeling_store.create_item(
                title,
                source_type=source_type,
                source_id=source_id,
                session_id=session_id,
                input_text=input_text,
                agent_output=agent_output,
                intent=intent,
                tool_calls=tool_calls,
                expected=expected,
                tags=tags,
                metadata=metadata,
                created_by=created_by,
            ),
        }

    def create_agent_labeling_item_from_trace(
        self,
        trace_id: str,
        *,
        title: str = "",
        tags: list[Any] | None = None,
        created_by: str = "agent",
    ) -> dict[str, Any]:
        """Create or refresh a labeling item from a persisted Agent trace."""
        if self.agent_labeling_store is None:
            raise ValueError("Agent labeling queue is not enabled.")
        if self.trace_logger is None:
            raise ValueError("Trace logger is not enabled.")
        trace = self.trace_logger.get(trace_id)
        if trace is None:
            raise ValueError(f"Unknown trace: {trace_id}")
        return {
            "enabled": True,
            **self.agent_labeling_store.create_from_trace(
                trace,
                title=title,
                tags=tags,
                created_by=created_by,
            ),
        }

    def label_agent_labeling_item(
        self,
        item_id: str,
        *,
        labels: dict[str, Any],
        outcome: str = "accepted",
        score: int | None = None,
        comment: str = "",
        labeler: str = "human",
    ) -> dict[str, Any]:
        """Attach one human label annotation to a queue item."""
        if self.agent_labeling_store is None:
            raise ValueError("Agent labeling queue is not enabled.")
        return {
            "enabled": True,
            **self.agent_labeling_store.label_item(
                item_id,
                labels=labels,
                outcome=outcome,
                score=score,
                comment=comment,
                labeler=labeler,
            ),
        }

    def update_agent_labeling_item_status(
        self,
        item_id: str,
        status: str,
        *,
        note: str = "",
        actor: str = "agent",
    ) -> dict[str, Any]:
        """Move one labeling item between queue statuses."""
        if self.agent_labeling_store is None:
            raise ValueError("Agent labeling queue is not enabled.")
        return {
            "enabled": True,
            **self.agent_labeling_store.update_status(
                item_id,
                status,
                note=note,
                actor=actor,
            ),
        }

    def export_agent_labeling_items(
        self,
        *,
        status: str = "labeled",
        limit: int = 200,
        mark_exported: bool = False,
    ) -> dict[str, Any]:
        """Export labeled samples as JSONL-ready records."""
        if self.agent_labeling_store is None:
            raise ValueError("Agent labeling queue is not enabled.")
        return self.agent_labeling_store.export_items(
            status=status,
            limit=limit,
            mark_exported=mark_exported,
        )

    def list_agent_performance_metrics(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        metric_name: str | None = None,
        source_type: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List local LLMOps performance metrics."""
        if self.agent_performance_store is None:
            return {
                "enabled": False,
                "backend": "local_agent_performance_store",
                "metrics": [],
                "count": 0,
            }
        return self.agent_performance_store.list_metrics(
            status=status,
            category=category,
            metric_name=metric_name,
            source_type=source_type,
            tag=tag,
            query=query,
            limit=limit,
        )

    def record_agent_performance_metric(
        self,
        metric_name: str,
        metric_value: Any,
        *,
        unit: str = "",
        category: str = "custom",
        target: str = "",
        source_type: str = "manual",
        source_id: str = "",
        session_id: str = "",
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        recorded_by: str = "agent",
    ) -> dict[str, Any]:
        """Record one local LLMOps performance metric."""
        if self.agent_performance_store is None:
            raise ValueError("Agent performance store is not enabled.")
        return {
            "enabled": True,
            **self.agent_performance_store.record_metric(
                metric_name,
                metric_value,
                unit=unit,
                category=category,
                target=target,
                source_type=source_type,
                source_id=source_id,
                session_id=session_id,
                tags=tags,
                metadata=metadata,
                recorded_by=recorded_by,
            ),
        }

    def record_agent_performance_from_trace(
        self,
        trace_id: str,
        *,
        tags: list[Any] | None = None,
        recorded_by: str = "agent",
    ) -> dict[str, Any]:
        """Extract performance metrics from a persisted Agent trace."""
        if self.agent_performance_store is None:
            raise ValueError("Agent performance store is not enabled.")
        if self.trace_logger is None:
            raise ValueError("Trace logger is not enabled.")
        trace = self.trace_logger.get(trace_id)
        if trace is None:
            raise ValueError(f"Unknown trace: {trace_id}")
        return self.agent_performance_store.record_from_trace(
            trace,
            tags=tags,
            recorded_by=recorded_by,
        )

    def agent_performance_summary(
        self,
        *,
        status: str = "active",
        category: str | None = None,
        metric_name: str | None = None,
        source_type: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate local LLMOps performance metrics."""
        if self.agent_performance_store is None:
            raise ValueError("Agent performance store is not enabled.")
        return self.agent_performance_store.summarize(
            status=status,
            category=category,
            metric_name=metric_name,
            source_type=source_type,
        )

    def update_agent_performance_metric_status(
        self,
        metric_id: str,
        status: str,
        *,
        note: str = "",
        actor: str = "agent",
    ) -> dict[str, Any]:
        """Archive or reactivate one performance metric."""
        if self.agent_performance_store is None:
            raise ValueError("Agent performance store is not enabled.")
        return {
            "enabled": True,
            **self.agent_performance_store.update_status(
                metric_id,
                status,
                note=note,
                actor=actor,
            ),
        }

    def list_agent_threads(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        participant: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List local multi-agent conversation threads."""
        if self.conversation_thread_store is None:
            return {"enabled": False, "backend": "local_agent_threads", "threads": [], "count": 0}
        return self.conversation_thread_store.list_threads(
            session_id=session_id,
            status=status,
            participant=participant,
            tag=tag,
            query=query,
            limit=limit,
        )

    def read_agent_thread(self, thread_id: str) -> dict[str, Any]:
        """Read one multi-agent conversation thread."""
        if self.conversation_thread_store is None:
            raise ValueError("Conversation thread store is not enabled.")
        return self.conversation_thread_store.read_thread(thread_id)

    def create_agent_thread(
        self,
        title: str,
        *,
        session_id: str = "web",
        summary: str = "",
        participants: list[Any] | None = None,
        status: str = "active",
        tags: list[Any] | None = None,
        shared_state_ids: list[Any] | None = None,
        artifact_ids: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        created_by: str = "agent",
    ) -> dict[str, Any]:
        """Create a local conversation thread for multi-agent handoff context."""
        if self.conversation_thread_store is None:
            raise ValueError("Conversation thread store is not enabled.")
        return {
            "enabled": True,
            **self.conversation_thread_store.create_thread(
                title,
                session_id=session_id,
                summary=summary,
                participants=participants,
                status=status,
                tags=tags,
                shared_state_ids=shared_state_ids,
                artifact_ids=artifact_ids,
                metadata=metadata,
                created_by=created_by,
            ),
        }

    def append_agent_thread_message(
        self,
        thread_id: str,
        *,
        role: str,
        content: str,
        agent_role: str = "",
        tool_calls: list[Any] | None = None,
        artifact_ids: list[Any] | None = None,
        shared_state_ids: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one message to a local multi-agent conversation thread."""
        if self.conversation_thread_store is None:
            raise ValueError("Conversation thread store is not enabled.")
        return {
            "enabled": True,
            **self.conversation_thread_store.append_message(
                thread_id,
                role=role,
                content=content,
                agent_role=agent_role,
                tool_calls=tool_calls,
                artifact_ids=artifact_ids,
                shared_state_ids=shared_state_ids,
                metadata=metadata,
            ),
        }

    def update_agent_thread_status(
        self,
        thread_id: str,
        status: str,
        *,
        note: str = "",
        actor: str = "agent",
    ) -> dict[str, Any]:
        """Move one multi-agent conversation thread between local statuses."""
        if self.conversation_thread_store is None:
            raise ValueError("Conversation thread store is not enabled.")
        return {
            "enabled": True,
            **self.conversation_thread_store.update_status(
                thread_id,
                status,
                note=note,
                actor=actor,
            ),
        }

    def list_agent_browser_contexts(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        thread_id: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List shared browser context records for Agent collaboration."""
        if self.browser_context_store is None:
            return {
                "enabled": False,
                "backend": "local_browser_context_store",
                "contexts": [],
                "count": 0,
            }
        return self.browser_context_store.list_contexts(
            session_id=session_id,
            status=status,
            thread_id=thread_id,
            tag=tag,
            query=query,
            limit=limit,
        )

    def read_agent_browser_context(self, context_id: str) -> dict[str, Any]:
        """Read one shared browser context record."""
        if self.browser_context_store is None:
            raise ValueError("Browser context store is not enabled.")
        return self.browser_context_store.read_context(context_id)

    def create_agent_browser_context(
        self,
        title: str,
        *,
        url: str = "",
        session_id: str = "web",
        thread_id: str = "",
        owner_agent: str = "agent",
        summary: str = "",
        status: str = "active",
        tags: list[Any] | None = None,
        artifact_ids: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a shared browser context record without launching a browser."""
        if self.browser_context_store is None:
            raise ValueError("Browser context store is not enabled.")
        return {
            "enabled": True,
            **self.browser_context_store.create_context(
                title,
                url=url,
                session_id=session_id,
                thread_id=thread_id,
                owner_agent=owner_agent,
                summary=summary,
                status=status,
                tags=tags,
                artifact_ids=artifact_ids,
                metadata=metadata,
            ),
        }

    def append_agent_browser_observation(
        self,
        context_id: str,
        *,
        url: str = "",
        title: str = "",
        text: str = "",
        agent_role: str = "agent",
        selector: str = "",
        screenshot_path: str = "",
        artifact_ids: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one observation to a shared browser context record."""
        if self.browser_context_store is None:
            raise ValueError("Browser context store is not enabled.")
        return {
            "enabled": True,
            **self.browser_context_store.append_observation(
                context_id,
                url=url,
                title=title,
                text=text,
                agent_role=agent_role,
                selector=selector,
                screenshot_path=screenshot_path,
                artifact_ids=artifact_ids,
                metadata=metadata,
            ),
        }

    def update_agent_browser_context_status(
        self,
        context_id: str,
        status: str,
        *,
        note: str = "",
        actor: str = "agent",
    ) -> dict[str, Any]:
        """Move one shared browser context between local statuses."""
        if self.browser_context_store is None:
            raise ValueError("Browser context store is not enabled.")
        return {
            "enabled": True,
            **self.browser_context_store.update_status(
                context_id,
                status,
                note=note,
                actor=actor,
            ),
        }

    def list_agent_tool_marketplace(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        provider: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List local marketplace tool packs available to Agent workflows."""
        if self.tool_marketplace_store is None:
            return {"enabled": False, "backend": "local_tool_marketplace", "entries": [], "count": 0}
        return self.tool_marketplace_store.list_entries(
            status=status,
            category=category,
            provider=provider,
            tag=tag,
            query=query,
            limit=limit,
        )

    def read_agent_tool_marketplace_entry(self, entry_id_or_slug: str) -> dict[str, Any]:
        """Read one local marketplace tool pack by id or slug."""
        if self.tool_marketplace_store is None:
            raise ValueError("Tool marketplace is not enabled.")
        return self.tool_marketplace_store.read_entry(entry_id_or_slug)

    def register_agent_tool_marketplace_entry(
        self,
        name: str,
        *,
        slug: str = "",
        description: str = "",
        category: str = "agent_tools",
        provider: str = "dac-agent",
        status: str = "available",
        tools: list[Any] | None = None,
        required_context: list[Any] | None = None,
        prompt_examples: list[Any] | None = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        owner_agent: str = "agent",
    ) -> dict[str, Any]:
        """Create or update one local marketplace tool pack."""
        if self.tool_marketplace_store is None:
            raise ValueError("Tool marketplace is not enabled.")
        return {
            "enabled": True,
            **self.tool_marketplace_store.register_entry(
                name,
                slug=slug,
                description=description,
                category=category,
                provider=provider,
                status=status,
                tools=tools,
                required_context=required_context,
                prompt_examples=prompt_examples,
                tags=tags,
                metadata=metadata,
                owner_agent=owner_agent,
            ),
        }

    def update_agent_tool_marketplace_status(
        self,
        entry_id_or_slug: str,
        status: str,
        *,
        note: str = "",
        actor: str = "agent",
    ) -> dict[str, Any]:
        """Move one local marketplace tool pack between catalog statuses."""
        if self.tool_marketplace_store is None:
            raise ValueError("Tool marketplace is not enabled.")
        return {
            "enabled": True,
            **self.tool_marketplace_store.update_status(
                entry_id_or_slug,
                status,
                note=note,
                actor=actor,
            ),
        }

    def list_agent_shared_state(
        self,
        *,
        session_id: str | None = None,
        scope: str | None = None,
        namespace: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List scoped shared state entries visible to Agent workflows."""
        if self.shared_state_store is None:
            return {
                "enabled": False,
                "backend": "local_agent_shared_state",
                "states": [],
                "count": 0,
            }
        return self.shared_state_store.list_states(
            session_id=session_id,
            scope=scope,
            namespace=namespace,
            tag=tag,
            query=query,
            limit=limit,
        )

    def read_agent_shared_state(
        self,
        state_id_or_key: str,
        *,
        session_id: str = "web",
        scope: str = "session",
        namespace: str = "default",
    ) -> dict[str, Any]:
        """Read one scoped shared state entry by id or key."""
        if self.shared_state_store is None:
            raise ValueError("Shared state store is not enabled.")
        return self.shared_state_store.read_state(
            state_id_or_key,
            session_id=session_id,
            scope=scope,
            namespace=namespace,
        )

    def set_agent_shared_state(
        self,
        key: str,
        value: Any,
        *,
        session_id: str = "web",
        scope: str = "session",
        namespace: str = "default",
        owner_agent: str = "",
        task_id: str = "",
        workflow_id: str = "",
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Create or update one scoped shared state entry."""
        if self.shared_state_store is None:
            raise ValueError("Shared state store is not enabled.")
        return {
            "enabled": True,
            **self.shared_state_store.set_state(
                key,
                value,
                session_id=session_id,
                scope=scope,
                namespace=namespace,
                owner_agent=owner_agent,
                task_id=task_id,
                workflow_id=workflow_id,
                tags=tags,
                metadata=metadata,
                note=note,
            ),
        }

    def list_agent_verification_presets(self) -> dict[str, Any]:
        """List runnable local verification feedback presets."""
        if self.verification_store is None:
            return {
                "enabled": False,
                "backend": "local_verification_runner",
                "presets": [],
                "count": 0,
            }
        return self.verification_store.presets()

    def list_agent_verification_runs(
        self,
        *,
        status: str | None = None,
        preset_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List recent local verification run records."""
        if self.verification_store is None:
            return {
                "enabled": False,
                "backend": "local_verification_runner",
                "runs": [],
                "count": 0,
            }
        return self.verification_store.list_runs(status=status, preset_id=preset_id, limit=limit)

    def run_agent_verification(
        self,
        preset_id: str,
        *,
        timeout_seconds: int = 120,
    ) -> dict[str, Any]:
        """Run one known local verification preset and persist its feedback."""
        if self.verification_store is None:
            raise ValueError("Verification feedback runner is not enabled.")
        return self.verification_store.run_preset(preset_id, timeout_seconds=timeout_seconds)

    def list_agent_review_handoffs(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List local Agent review handoff packets."""
        if self.review_handoff_store is None:
            return {
                "enabled": False,
                "backend": "local_review_handoff_queue",
                "reviews": [],
                "count": 0,
            }
        return self.review_handoff_store.list_reviews(
            session_id=session_id,
            status=status,
            priority=priority,
            limit=limit,
        )

    def read_agent_review_handoff(self, review_id: str) -> dict[str, Any]:
        """Read one local Agent review handoff packet."""
        if self.review_handoff_store is None:
            raise ValueError("Review handoff queue is not enabled.")
        return self.review_handoff_store.read_review(review_id)

    def create_agent_review_handoff(
        self,
        title: str,
        *,
        summary: str = "",
        session_id: str = "web",
        status: str = "pending",
        priority: str = "normal",
        task_id: str = "",
        workflow_id: str = "",
        trace_id: str = "",
        files: list[Any] | None = None,
        verification_run_ids: list[Any] | None = None,
        checklist: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a review handoff packet for a human or reviewer agent."""
        if self.review_handoff_store is None:
            raise ValueError("Review handoff queue is not enabled.")
        return {
            "enabled": True,
            **self.review_handoff_store.create_review(
                title,
                summary=summary,
                session_id=session_id,
                status=status,
                priority=priority,
                task_id=task_id,
                workflow_id=workflow_id,
                trace_id=trace_id,
                files=files,
                verification_run_ids=verification_run_ids,
                checklist=checklist,
                metadata=metadata,
            ),
        }

    def add_agent_review_comment(
        self,
        review_id: str,
        body: str,
        *,
        reviewer: str = "human",
    ) -> dict[str, Any]:
        """Append a reviewer comment to one handoff packet."""
        if self.review_handoff_store is None:
            raise ValueError("Review handoff queue is not enabled.")
        return {
            "enabled": True,
            **self.review_handoff_store.add_comment(review_id, body, reviewer=reviewer),
        }

    def update_agent_review_status(
        self,
        review_id: str,
        status: str,
        *,
        note: str = "",
        reviewer: str = "human",
    ) -> dict[str, Any]:
        """Move one review handoff through its decision states."""
        if self.review_handoff_store is None:
            raise ValueError("Review handoff queue is not enabled.")
        return {
            "enabled": True,
            **self.review_handoff_store.update_status(
                review_id,
                status,
                note=note,
                reviewer=reviewer,
            ),
        }

    def list_agent_checkpoints(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        tag: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List local Agent workflow checkpoints."""
        if self.checkpoint_store is None:
            return {
                "enabled": False,
                "backend": "local_agent_checkpoint_store",
                "checkpoints": [],
                "count": 0,
            }
        return self.checkpoint_store.list_checkpoints(
            session_id=session_id,
            status=status,
            tag=tag,
            limit=limit,
        )

    def read_agent_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        """Read one local Agent workflow checkpoint."""
        if self.checkpoint_store is None:
            raise ValueError("Checkpoint store is not enabled.")
        return self.checkpoint_store.read_checkpoint(checkpoint_id)

    def create_agent_checkpoint(
        self,
        title: str,
        *,
        state: dict[str, Any] | None = None,
        session_id: str = "web",
        summary: str = "",
        status: str = "active",
        task_id: str = "",
        workflow_id: str = "",
        event_id: str = "",
        review_id: str = "",
        trace_id: str = "",
        parent_checkpoint_id: str = "",
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a durable workflow checkpoint for later resume context."""
        if self.checkpoint_store is None:
            raise ValueError("Checkpoint store is not enabled.")
        checkpoint_state = state if state is not None else self._checkpoint_workspace_state()
        return {
            "enabled": True,
            **self.checkpoint_store.create_checkpoint(
                title,
                state=checkpoint_state,
                session_id=session_id,
                summary=summary,
                status=status,
                task_id=task_id,
                workflow_id=workflow_id,
                event_id=event_id,
                review_id=review_id,
                trace_id=trace_id,
                parent_checkpoint_id=parent_checkpoint_id,
                tags=tags,
                metadata=metadata,
            ),
        }

    def restore_agent_checkpoint(
        self,
        checkpoint_id: str,
        *,
        note: str = "",
        actor: str = "agent",
    ) -> dict[str, Any]:
        """Mark one checkpoint as the selected resume point."""
        if self.checkpoint_store is None:
            raise ValueError("Checkpoint store is not enabled.")
        return {
            "enabled": True,
            **self.checkpoint_store.restore_checkpoint(checkpoint_id, note=note, actor=actor),
        }

    def list_goals(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List tracked DAC-Agent goals."""
        if self.goal_store is None:
            return {"enabled": False, "backend": "local_goal_store", "goals": [], "count": 0}
        return self.goal_store.list_goals(session_id=session_id, status=status, limit=limit)

    def create_goal(self, objective: str, *, session_id: str = "web") -> dict[str, Any]:
        """Create a durable Agent goal."""
        if self.goal_store is None:
            raise ValueError("Goal store is not enabled.")
        return {"enabled": True, **self.goal_store.create_goal(objective, session_id=session_id)}

    def append_goal_progress(
        self,
        goal_id: str,
        note: str,
        *,
        status: str | None = None,
    ) -> dict[str, Any]:
        """Append progress to one Agent goal."""
        if self.goal_store is None:
            raise ValueError("Goal store is not enabled.")
        return {
            "enabled": True,
            **self.goal_store.append_progress(goal_id, note, status=status),
        }

    def complete_goal(self, goal_id: str, *, note: str = "") -> dict[str, Any]:
        """Mark one Agent goal complete."""
        if self.goal_store is None:
            raise ValueError("Goal store is not enabled.")
        return {"enabled": True, **self.goal_store.complete_goal(goal_id, note=note)}

    def list_agent_tasks(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        goal_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List local Agent task-board cards."""
        if self.task_board_store is None:
            return {"enabled": False, "backend": "local_agent_task_board", "tasks": [], "count": 0}
        return self.task_board_store.list_tasks(
            session_id=session_id,
            status=status,
            goal_id=goal_id,
            limit=limit,
        )

    def create_agent_task(
        self,
        title: str,
        *,
        session_id: str = "web",
        description: str = "",
        status: str = "backlog",
        priority: str = "normal",
        goal_id: str = "",
        agent_path: list[Any] | None = None,
        tool_candidates: list[Any] | None = None,
        dependencies: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a local Agent task-board card."""
        if self.task_board_store is None:
            raise ValueError("Task board is not enabled.")
        return {
            "enabled": True,
            **self.task_board_store.create_task(
                title,
                session_id=session_id,
                description=description,
                status=status,
                priority=priority,
                goal_id=goal_id,
                agent_path=agent_path,
                tool_candidates=tool_candidates,
                dependencies=dependencies,
                metadata=metadata,
            ),
        }

    def update_agent_task_status(self, task_id: str, status: str, *, note: str = "") -> dict[str, Any]:
        """Move one Agent task-board card to another status column."""
        if self.task_board_store is None:
            raise ValueError("Task board is not enabled.")
        return {"enabled": True, **self.task_board_store.update_status(task_id, status, note=note)}

    def create_agent_task_from_workflow(
        self,
        task: str,
        *,
        session_id: str = "web",
        goal_id: str = "",
        status: str = "ready",
        priority: str = "normal",
    ) -> dict[str, Any]:
        """Create a task-board card from the current workflow preview."""
        if self.task_board_store is None:
            raise ValueError("Task board is not enabled.")
        preview = self.preview_agent_workflow(task, session_id=session_id)
        created = self.task_board_store.create_from_workflow_preview(
            preview,
            session_id=session_id,
            goal_id=goal_id,
            status=status,
            priority=priority,
        )
        return {"enabled": True, "preview": preview, **created}

    def list_agent_automations(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List local Agent automation plans."""
        if self.automation_store is None:
            return {"enabled": False, "backend": "local_automation_planner", "automations": [], "count": 0}
        return self.automation_store.list_automations(
            session_id=session_id,
            status=status,
            limit=limit,
        )

    def list_due_agent_automations(self, *, limit: int = 20) -> dict[str, Any]:
        """List active automation plans whose next_run_at is due."""
        if self.automation_store is None:
            return {"enabled": False, "backend": "local_automation_planner", "automations": [], "count": 0}
        return self.automation_store.due_automations(limit=limit)

    def create_agent_automation(
        self,
        name: str,
        prompt: str,
        *,
        session_id: str = "web",
        schedule: dict[str, Any] | None = None,
        status: str = "active",
        task_id: str = "",
        goal_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a local scheduled automation definition."""
        if self.automation_store is None:
            raise ValueError("Automation planner is not enabled.")
        return {
            "enabled": True,
            **self.automation_store.create_automation(
                name,
                prompt,
                session_id=session_id,
                schedule=schedule,
                status=status,
                task_id=task_id,
                goal_id=goal_id,
                metadata=metadata,
            ),
        }

    def update_agent_automation_status(
        self,
        automation_id: str,
        status: str,
        *,
        note: str = "",
    ) -> dict[str, Any]:
        """Pause, resume, or archive one local automation definition."""
        if self.automation_store is None:
            raise ValueError("Automation planner is not enabled.")
        return {"enabled": True, **self.automation_store.update_status(automation_id, status, note=note)}

    def record_agent_automation_run(
        self,
        automation_id: str,
        *,
        result: str = "",
        status: str = "completed",
        trace_id: str = "",
    ) -> dict[str, Any]:
        """Record one external/future-worker automation run result."""
        if self.automation_store is None:
            raise ValueError("Automation planner is not enabled.")
        return {
            "enabled": True,
            **self.automation_store.record_run(
                automation_id,
                result=result,
                status=status,
                trace_id=trace_id,
            ),
        }

    def list_agent_workflows(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List reusable Agent workflow templates."""
        if self.workflow_store is None:
            return {"enabled": False, "backend": "local_workflow_templates", "workflows": [], "count": 0}
        return self.workflow_store.list_workflows(
            session_id=session_id,
            status=status,
            limit=limit,
        )

    def read_agent_workflow(self, workflow_id: str) -> dict[str, Any]:
        """Read one reusable Agent workflow template."""
        if self.workflow_store is None:
            raise ValueError("Workflow template store is not enabled.")
        return self.workflow_store.read_workflow(workflow_id)

    def create_agent_workflow(
        self,
        name: str,
        *,
        session_id: str = "web",
        description: str = "",
        nodes: list[Any] | None = None,
        edges: list[Any] | None = None,
        status: str = "draft",
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a reusable Agent workflow template from a DAG payload."""
        if self.workflow_store is None:
            raise ValueError("Workflow template store is not enabled.")
        return {
            "enabled": True,
            **self.workflow_store.create_workflow(
                name,
                session_id=session_id,
                description=description,
                nodes=nodes,
                edges=edges,
                status=status,
                tags=tags,
                metadata=metadata,
            ),
        }

    def create_agent_workflow_from_preview(
        self,
        task: str,
        *,
        name: str = "",
        session_id: str = "web",
        status: str = "draft",
        tags: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Create a reusable workflow template from the current workflow preview."""
        if self.workflow_store is None:
            raise ValueError("Workflow template store is not enabled.")
        preview = self.preview_agent_workflow(task, session_id=session_id)
        created = self.workflow_store.create_from_preview(
            preview,
            name=name,
            session_id=session_id,
            status=status,
            tags=tags,
        )
        return {"enabled": True, "preview": preview, **created}

    def update_agent_workflow_status(self, workflow_id: str, status: str) -> dict[str, Any]:
        """Move one workflow template between draft, active, and archived states."""
        if self.workflow_store is None:
            raise ValueError("Workflow template store is not enabled.")
        return {"enabled": True, **self.workflow_store.update_status(workflow_id, status)}

    def list_agent_artifacts(
        self,
        *,
        session_id: str | None = None,
        artifact_type: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List local Agent workspace artifacts."""
        if self.artifact_store is None:
            return {"enabled": False, "backend": "local_agent_artifact_store", "artifacts": [], "count": 0}
        return self.artifact_store.list_artifacts(
            session_id=session_id,
            artifact_type=artifact_type,
            tag=tag,
            query=query,
            limit=limit,
        )

    def create_agent_artifact(
        self,
        title: str,
        content: Any,
        *,
        artifact_type: str = "markdown",
        session_id: str = "web",
        task_id: str = "",
        workflow_id: str = "",
        trace_id: str = "",
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a file-backed Agent workspace artifact."""
        if self.artifact_store is None:
            raise ValueError("Artifact store is not enabled.")
        return {
            "enabled": True,
            **self.artifact_store.create_artifact(
                title,
                content,
                artifact_type=artifact_type,
                session_id=session_id,
                task_id=task_id,
                workflow_id=workflow_id,
                trace_id=trace_id,
                tags=tags,
                metadata=metadata,
            ),
        }

    def read_agent_artifact(self, artifact_id: str) -> dict[str, Any]:
        """Read one Agent workspace artifact and its file content."""
        if self.artifact_store is None:
            raise ValueError("Artifact store is not enabled.")
        return self.artifact_store.read_artifact(artifact_id)

    def list_agent_events(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        event_type: str | None = None,
        priority: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List durable Agent workflow events."""
        if self.event_queue_store is None:
            return {"enabled": False, "backend": "local_agent_event_queue", "events": [], "count": 0}
        return self.event_queue_store.list_events(
            session_id=session_id,
            status=status,
            event_type=event_type,
            priority=priority,
            limit=limit,
        )

    def enqueue_agent_event(
        self,
        event_type: str,
        *,
        payload: dict[str, Any] | None = None,
        session_id: str = "web",
        priority: str = "normal",
        scheduled_for: str = "",
        task_id: str = "",
        workflow_id: str = "",
        automation_id: str = "",
        goal_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a queued event for a future Agent worker or UI action."""
        if self.event_queue_store is None:
            raise ValueError("Event queue is not enabled.")
        return {
            "enabled": True,
            **self.event_queue_store.enqueue_event(
                event_type,
                payload=payload,
                session_id=session_id,
                priority=priority,
                scheduled_for=scheduled_for,
                task_id=task_id,
                workflow_id=workflow_id,
                automation_id=automation_id,
                goal_id=goal_id,
                metadata=metadata,
            ),
        }

    def list_due_agent_events(self, *, limit: int = 20) -> dict[str, Any]:
        """List queued events that are ready to be claimed."""
        if self.event_queue_store is None:
            return {"enabled": False, "backend": "local_agent_event_queue", "events": [], "count": 0}
        return self.event_queue_store.due_events(limit=limit)

    def claim_next_agent_event(
        self,
        *,
        worker_id: str = "agent-worker",
        session_id: str | None = None,
        event_type: str | None = None,
    ) -> dict[str, Any]:
        """Claim the next due queued event for a worker."""
        if self.event_queue_store is None:
            raise ValueError("Event queue is not enabled.")
        return self.event_queue_store.claim_next(
            worker_id=worker_id,
            session_id=session_id,
            event_type=event_type,
        )

    def update_agent_event_status(
        self,
        event_id: str,
        status: str,
        *,
        note: str = "",
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update one Agent event after a worker or UI action."""
        if self.event_queue_store is None:
            raise ValueError("Event queue is not enabled.")
        return {
            "enabled": True,
            **self.event_queue_store.update_status(
                event_id,
                status,
                note=note,
                result=result,
            ),
        }

    def build_repo_context_map(self, *, max_files: int = 1200) -> dict[str, Any]:
        """Build or refresh the local static repository context map."""
        if self.repo_context_map is None:
            raise ValueError("Repo context map is not enabled.")
        return self.repo_context_map.build_map(max_files=max_files)

    def read_repo_context_map(self) -> dict[str, Any]:
        """Read the latest static repository context map."""
        if self.repo_context_map is None:
            raise ValueError("Repo context map is not enabled.")
        return self.repo_context_map.read_map()

    def search_repo_context_map(self, query: str, *, limit: int = 20) -> dict[str, Any]:
        """Search files, roles, symbols, and API routes in the repo context map."""
        if self.repo_context_map is None:
            raise ValueError("Repo context map is not enabled.")
        return self.repo_context_map.search(query, limit=limit)

    def list_code_symbols(
        self,
        query: str | None = None,
        *,
        kind: str | None = None,
        module: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List Python code symbols discovered by the static repo context map."""
        if self.repo_context_map is None:
            raise ValueError("Code symbol navigator is not enabled.")
        return self.repo_context_map.list_symbols(
            query=query,
            kind=kind,
            module=module,
            limit=limit,
        )

    def git_workspace_status(
        self,
        *,
        recent_limit: int = 5,
        include_diff_stat: bool = True,
        include_worktrees: bool = True,
    ) -> dict[str, Any]:
        """Return read-only Git branch, diff, and worktree context."""
        if self.git_workspace_context is None:
            raise ValueError("Git workspace context is not enabled.")
        return self.git_workspace_context.snapshot(
            recent_limit=recent_limit,
            include_diff_stat=include_diff_stat,
            include_worktrees=include_worktrees,
        )

    def preview_agent_workflow(
        self,
        task: str,
        *,
        session_id: str = "web-preview",
    ) -> dict[str, Any]:
        """Preview which agents, skills, tools, and context nodes a task would use."""
        clean_task = str(task or "").strip()
        if not clean_task:
            raise ValueError("The `task` field is required.")
        context_payload: dict[str, Any] = {}
        if self.context_builder is not None:
            context_payload = self.context_builder.build(
                clean_task,
                session_id=session_id,
                history=[],
                recent_limit=self.config.memory_recent_turns,
                search_limit=self.config.memory_search_limit,
            ).to_dict()
        skill_matches = list(
            (context_payload.get("skills") or {}).get("matches", [])
            if isinstance(context_payload.get("skills"), dict)
            else []
        )
        tree_matches = list(
            (context_payload.get("context_tree") or {}).get("matches", [])
            if isinstance(context_payload.get("context_tree"), dict)
            else []
        )
        memory_hits = list(
            (context_payload.get("memory") or {}).get("hits", [])
            if isinstance(context_payload.get("memory"), dict)
            else []
        )
        agent_path = self._preview_agent_path(clean_task, skill_matches)
        tool_candidates = self._preview_tool_candidates(clean_task, skill_matches)
        nodes = [
            {"id": "input", "label": "用户任务", "kind": "input", "status": "ready"},
            {"id": "coordinator", "label": "Coordinator", "kind": "agent", "status": "selected"},
            {
                "id": "skills",
                "label": "Skill Selector",
                "kind": "skill",
                "status": "selected" if skill_matches else "skipped",
                "count": len(skill_matches),
            },
            {
                "id": "context_tree",
                "label": "Context Tree",
                "kind": "context",
                "status": "selected" if tree_matches else "skipped",
                "count": len(tree_matches),
            },
            {
                "id": "memory",
                "label": "Memory OS",
                "kind": "memory",
                "status": "selected" if memory_hits else "skipped",
                "count": len(memory_hits),
            },
            {
                "id": "specialist",
                "label": agent_path[-1] if len(agent_path) > 1 else "DAC-3D QA Agent",
                "kind": "agent",
                "status": "selected",
            },
            {
                "id": "tools",
                "label": "Tool Loop",
                "kind": "tool",
                "status": "selected" if tool_candidates else "skipped",
                "count": len(tool_candidates),
            },
        ]
        return {
            "enabled": True,
            "backend": "agent_workflow_preview",
            "task": clean_task,
            "session_id": session_id,
            "agent_path": agent_path,
            "tool_candidates": tool_candidates,
            "skill_matches": skill_matches,
            "context_tree_matches": tree_matches,
            "memory_hits": memory_hits,
            "context_sections": context_payload.get("sections", []),
            "nodes": nodes,
            "workflow": "task -> coordinator -> skills/context_tree/memory -> specialist -> tools -> response",
        }

    def _preview_agent_path(self, task: str, skill_matches: list[dict[str, Any]]) -> list[str]:
        lowered = task.lower()
        path = ["coordinator"]
        if any(marker in task for marker in ("技能", "skill", "可用流程", "加载什么")):
            path.append("skill_agent")
        elif any(marker in task for marker in ("刚才", "上次", "历史", "记得", "前面")):
            path.append("memory_agent")
        elif any(marker in task for marker in ("温度报警", "报警", "设备状态", "异常")):
            path.append("machine_agent")
        elif any(marker in task for marker in ("结果", "样品")) or "result" in lowered:
            path.append("dac3d_result_agent")
        elif any(marker in task for marker in ("扫描", "停止", "离线检测", "执行", "状态", "进度")) or any(
            marker in lowered for marker in ("scan", "stop", "offline", "status", "progress")
        ):
            path.append("dac3d_control_agent")
        else:
            path.append("dac3d_qa_agent")
        if skill_matches and "skill_agent" not in path:
            path.insert(1, "skill_agent")
        return path

    def _preview_tool_candidates(self, task: str, skill_matches: list[dict[str, Any]]) -> list[str]:
        lowered = task.lower()
        tools: list[str] = []
        if any(marker in task for marker in ("状态", "进度")) or "status" in lowered:
            tools.append("dac3d_status")
        if any(marker in task for marker in ("结果", "样品")) or "result" in lowered:
            tools.append("dac3d_latest_result")
        if any(marker in task for marker in ("扫描", "离线检测")) or any(
            marker in lowered for marker in ("scan", "offline")
        ):
            tools.append("dac3d_preview_command")
        if any(marker in task for marker in ("停止", "执行")) or "stop" in lowered:
            tools.append("dac3d_execute_command")
        if any(marker in task for marker in ("技能", "skill", "可用流程", "加载什么")):
            tools.append("dac_skill_select")
        if any(marker in task for marker in ("刚才", "上次", "历史", "记得", "前面")):
            tools.append("conversation_memory_search")
        if any(marker in task for marker in ("报警", "设备状态", "异常")):
            tools.append("machine_agent_chat")
        for match in skill_matches:
            skill = match.get("skill") if isinstance(match, dict) else {}
            for tool in skill.get("tools", []) if isinstance(skill, dict) else []:
                if tool not in tools:
                    tools.append(str(tool))
        if not tools:
            tools.append("dac3d_answer")
        return tools

    def propose_skill_patch(
        self,
        *,
        target_skill: str,
        reason: str,
        diff: str = "",
        replacement_section: str = "",
        evidence_trace_ids: list[str] | None = None,
        risk_level: str = "medium",
    ) -> dict[str, Any]:
        """Create a reviewable skill patch proposal from the UI/API path."""
        return self.runtime.propose_dac_skill_patch(
            target_skill=target_skill,
            reason=reason,
            diff=diff,
            replacement_section=replacement_section,
            evidence_trace_ids=evidence_trace_ids or [],
            risk_level=risk_level,
            proposed_by="ui",
        )

    def list_skill_patches(self, status: str = "pending") -> dict[str, Any]:
        """List skill patch proposals for human review."""
        return self.runtime.list_dac_skill_patches(status=status)

    def approve_skill_patch(self, patch_id: str) -> dict[str, Any]:
        """Approve one pending skill patch without auto-applying it."""
        return self.runtime.approve_dac_skill_patch(patch_id)

    def reject_skill_patch(self, patch_id: str, reason: str = "") -> dict[str, Any]:
        """Reject one skill patch proposal."""
        return self.runtime.reject_dac_skill_patch(patch_id, reason=reason)

    def list_procedure_memories(self) -> dict[str, Any]:
        """List approved Markdown procedure memories."""
        return self.runtime.list_conversation_procedure_memories()

    def read_procedure_memory(self, name: str) -> dict[str, Any]:
        """Read one approved Markdown procedure memory."""
        return self.runtime.read_conversation_procedure_memory(name)

    def propose_procedure_memory(
        self,
        *,
        name: str,
        content: str,
        reason: str = "procedure_memory_candidate",
        mode: str = "append",
    ) -> dict[str, Any]:
        """Create a reviewable procedure-memory patch."""
        return self.runtime.write_conversation_procedure_memory(
            name=name,
            content=content,
            reason=reason,
            mode=mode,
        )

    def list_memory_patches(self, status: str = "pending") -> dict[str, Any]:
        """List Memory OS patches for human review."""
        if self.memory_provider is None:
            return {"enabled": False, "backend": "json+markdown", "patches": [], "count": 0}
        normalized_status = str(status or "").strip() or None
        return {"enabled": True, **self.memory_provider.list_patches(status=normalized_status)}

    def approve_memory_patch(self, patch_id: str) -> dict[str, Any]:
        """Approve one pending Memory OS patch from the UI/API review path."""
        if self.memory_provider is None:
            raise ValueError("Memory provider is not enabled.")
        return {"enabled": True, **self.memory_provider.approve_write(patch_id)}

    def reject_memory_patch(self, patch_id: str, reason: str = "") -> dict[str, Any]:
        """Reject one pending Memory OS patch from the UI/API review path."""
        if self.memory_provider is None:
            raise ValueError("Memory provider is not enabled.")
        return {"enabled": True, **self.memory_provider.reject_write(patch_id, reason=reason)}

    def read_trace(self, trace_id: str) -> dict[str, Any]:
        """Read one append-only Agent trace."""
        if self.trace_logger is None:
            raise ValueError("Trace logger is not enabled.")
        record = self.trace_logger.get(trace_id)
        if record is None:
            raise ValueError(f"Unknown trace id: {trace_id}")
        return record

    def approve_pending_command(
        self,
        *,
        session_id: str = "web",
        preview_id: str | None = None,
        confirmation_token: str | None = None,
    ) -> AssistantResponse:
        """Continue a pending command after the user clicks the UI approval button."""
        try:
            payload = self.runtime.approve_pending_command(
                session_id=session_id,
                preview_id=preview_id,
                confirmation_token=confirmation_token,
            )
        except Exception as exc:
            return AssistantResponse(intent="operation", answer=f"批准执行失败: {exc}")
        response = self._response_from_payload(payload)
        self._record_agent_trace(
            session_id=session_id,
            message="批准执行",
            response=response,
            context_bundle={},
            event_type="approval_submit",
            confirmation={"approved": True, "source": "ui_button", "preview_id": preview_id},
        )
        self._remember_turn(session_id, "批准执行", response)
        return response

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

    def _record_agent_trace(
        self,
        *,
        session_id: str,
        message: str,
        response: AssistantResponse,
        context_bundle: dict[str, Any],
        event_type: str,
        confirmation: dict[str, Any] | None = None,
    ) -> None:
        if self.trace_logger is None:
            return
        parsed_result = response.parsed_result if isinstance(response.parsed_result, dict) else {}
        command_preview = response.command_preview if isinstance(response.command_preview, dict) else {}
        gateway = command_preview.get("gateway") if isinstance(command_preview.get("gateway"), dict) else {}
        tool_gateway = parsed_result.get("tool_gateway") if isinstance(parsed_result.get("tool_gateway"), dict) else {}
        risk_decision = tool_gateway.get("risk") if isinstance(tool_gateway.get("risk"), dict) else {}
        memory_os = parsed_result.get("memory_os") if isinstance(parsed_result.get("memory_os"), dict) else {}
        memory_patch_ids = [
            str(item.get("id"))
            for item in memory_os.get("pending_patches", [])
            if isinstance(item, dict) and item.get("id")
        ]
        context_ids = [
            str(section.get("name"))
            for section in context_bundle.get("sections", [])
            if isinstance(section, dict) and section.get("name")
        ]
        skill_matches = []
        skills = context_bundle.get("skills") if isinstance(context_bundle.get("skills"), dict) else {}
        for item in skills.get("matches", []):
            if isinstance(item, dict):
                skill = item.get("skill")
                if isinstance(skill, dict) and skill.get("name"):
                    skill_matches.append(str(skill.get("name")))
        trace = self.trace_logger.append(
            {
                "event_type": event_type,
                "session_id": session_id,
                "user_message": message,
                "intent": response.intent,
                "skill": skill_matches[0] if skill_matches else None,
                "context_ids": context_ids,
                "tool_calls": parsed_result.get("tool_calls", []),
                "command_preview_id": gateway.get("preview_id"),
                "risk_decision": risk_decision,
                "confirmation": confirmation
                or {
                    "required": gateway.get("confirmation_required"),
                    "approved": event_type == "approval_submit",
                },
                "final_response": response.answer,
                "memory_patch_ids": memory_patch_ids,
                "memory_trace_id": memory_os.get("trace_id"),
                "status_summary": response.status_summary,
            }
        )
        parsed_result = dict(parsed_result)
        parsed_result["trace_eval"] = {
            "trace_id": trace.get("trace_id"),
            "event_type": event_type,
            "workflow": "append_only_trace -> eval_runner -> regression_report",
        }
        response.parsed_result = parsed_result

    def _format_agent_input(
        self,
        message: str,
        history: Sequence[tuple[str, str]] | None,
        *,
        session_id: str,
    ) -> tuple[str, dict[str, Any]]:
        history_tail = list(history or [])[-self.config.history_window :]
        if self.context_builder is not None:
            context_bundle = self.context_builder.build(
                message,
                session_id=session_id,
                history=history_tail,
                recent_limit=self.config.memory_recent_turns,
                search_limit=self.config.memory_search_limit,
            )
            context_payload = context_bundle.to_dict()
            if not context_bundle.prompt_context:
                return message, context_payload
            return (
                "\n\n".join(
                    [
                        context_bundle.prompt_context,
                        f"当前用户问题:\n{message}",
                        "请先基于系统规则判断是否需要调用工具，再输出规定 JSON。",
                    ]
                ),
                context_payload,
            )

        skill_context = ""
        skill_matches: list[dict[str, Any]] = []
        if self.skill_registry is not None:
            skill_context, skill_matches = self.skill_registry.build_context(
                message,
                limit=self.config.context_skill_limit,
            )
        memory_context = ""
        legacy_bundle: dict[str, Any] = {"backend": "context_builder_legacy", "memory": {"hits": []}}
        if self.memory_provider is not None:
            bundle = self.memory_provider.prefetch(
                message,
                user=session_id,
                context={
                    "session_id": session_id,
                    "history": history_tail,
                    "recent_limit": self.config.memory_recent_turns,
                    "search_limit": self.config.memory_search_limit,
                },
            )
            memory_context = bundle.context_text
            legacy_bundle["memory"] = bundle.to_dict()
        elif self.memory_store is not None:
            memory_context, memory_hits = self.memory_store.format_context(
                message,
                session_id=session_id,
                history=history_tail,
                recent_limit=self.config.memory_recent_turns,
                search_limit=self.config.memory_search_limit,
            )
            legacy_bundle["memory"] = {"backend": "json+markdown", "hits": memory_hits}
        elif history_tail:
            turns: list[str] = []
            for user_message, assistant_message in history_tail:
                turns.append(f"用户: {user_message}")
                turns.append(f"助手: {assistant_message}")
            memory_context = "短期记忆（当前页面历史）:\n" + "\n".join(turns)
            legacy_bundle["memory"] = {"backend": "short_term", "hits": [], "selected_layers": ["short_term_history"]}
        legacy_bundle["skills"] = {"backend": "local_agent_skills", "matches": skill_matches}

        if not memory_context and not skill_context:
            return message, legacy_bundle
        context_sections = [section for section in [skill_context, memory_context] if section]
        return (
            "\n\n".join(
                context_sections
                + [
                    f"当前用户问题:\n{message}",
                    "请先基于系统规则判断是否需要调用工具，再输出规定 JSON。",
                ]
            ),
            legacy_bundle,
        )

    def _attach_memory_metadata(
        self,
        payload: dict[str, Any],
        context_bundle: dict[str, Any],
    ) -> None:
        memory_payload_source = context_bundle.get("memory") if isinstance(context_bundle.get("memory"), dict) else {}
        skills_payload_source = context_bundle.get("skills") if isinstance(context_bundle.get("skills"), dict) else {}
        tree_payload_source = (
            context_bundle.get("context_tree")
            if isinstance(context_bundle.get("context_tree"), dict)
            else {}
        )
        memory_hits = list(memory_payload_source.get("hits") or [])
        skill_matches = list(skills_payload_source.get("matches") or [])
        tree_matches = list(tree_payload_source.get("matches") or [])
        has_context_metadata = bool(context_bundle.get("sections") or context_bundle.get("runtime_status"))
        if not memory_hits and not skill_matches and not tree_matches and not has_context_metadata:
            return

        parsed_result = payload.get("parsed_result")
        if not isinstance(parsed_result, dict):
            parsed_result = {}
        if memory_hits:
            memory_payload = {
                "backend": "json+markdown",
                "provider": "local_memory_os" if self.memory_provider is not None else "conversation_store",
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
        if skill_matches:
            parsed_result.setdefault(
                "skill_system",
                {
                    "backend": "local_agent_skills",
                    "matches": skill_matches,
                    "selected": [
                        item.get("skill", {}).get("name")
                        for item in skill_matches
                        if isinstance(item, dict)
                    ],
                },
            )
        if has_context_metadata:
            parsed_result.setdefault(
                "context_engineering",
                {
                    "backend": "context_builder",
                    "sections": [
                        section.get("name")
                        for section in context_bundle.get("sections", [])
                        if isinstance(section, dict)
                    ],
                    "excluded_context": context_bundle.get("excluded_context", []),
                    "limits": context_bundle.get("limits", {}),
                    "runtime_status": context_bundle.get("runtime_status", {}),
                    "trust": context_bundle.get("trust", {}),
                    "context_tree": {
                        "backend": "file_context_tree",
                        "selected": [
                            item.get("node", {}).get("id")
                            for item in tree_matches
                            if isinstance(item, dict)
                        ],
                        "matches": tree_matches,
                    },
                    "safety_policy_included": bool(
                        (context_bundle.get("safety_policy") or {}).get("included")
                        if isinstance(context_bundle.get("safety_policy"), dict)
                        else False
                    ),
                },
            )
        if tree_matches:
            parsed_result.setdefault(
                "context_tree",
                {
                    "backend": "file_context_tree",
                    "selected": [
                        item.get("node", {}).get("id")
                        for item in tree_matches
                        if isinstance(item, dict)
                    ],
                    "matches": tree_matches,
                },
            )
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
                    "trust_level": "approved_memory",
                    "can_instruct_agent": False,
                    "can_influence_tools": False,
                    "chunk_id": str(hit.get("turn_id") or ""),
                    "score": float(hit.get("score") or 0.0),
                    "text": str(hit.get("snippet") or ""),
                    "metadata": hit,
                }
            )
        for match in tree_matches:
            node = match.get("node") if isinstance(match, dict) else {}
            if not isinstance(node, dict):
                continue
            source = f"context_tree:{node.get('id') or 'unknown'}"
            if source not in sources:
                sources.append(source)
            source_items.append(
                {
                    "source": source,
                    "title": str(node.get("title") or "Context Tree"),
                    "section": str(node.get("kind") or "context"),
                    "document_type": "context_tree",
                    "trust_level": "approved_memory",
                    "can_instruct_agent": False,
                    "can_influence_tools": False,
                    "chunk_id": str(node.get("id") or ""),
                    "score": float(match.get("score") or 0.0) if isinstance(match, dict) else 0.0,
                    "text": str(node.get("content_preview") or ""),
                    "metadata": node,
                }
            )
        payload["sources"] = sources
        payload["source_items"] = source_items

    def _maybe_capture_goal(
        self,
        *,
        session_id: str,
        message: str,
        response: AssistantResponse,
    ) -> None:
        if self.goal_store is None:
            return
        try:
            result = self.goal_store.maybe_create_from_message(message, session_id=session_id)
        except ValueError:
            return
        if not result:
            return
        parsed_result = response.parsed_result if isinstance(response.parsed_result, dict) else {}
        parsed_result["goals"] = {
            "backend": "local_goal_store",
            "captured": result.get("created") is True,
            "goal": result.get("goal"),
            "workflow": "chat_goal_signal -> goal_store -> agent_workspace",
        }
        response.parsed_result = parsed_result

    def _record_memory_trace(
        self,
        *,
        session_id: str,
        message: str,
        agent_input: str,
        context_bundle: dict[str, Any],
        response: AssistantResponse,
    ) -> None:
        if self.memory_provider is None:
            return
        parsed_result = response.parsed_result if isinstance(response.parsed_result, dict) else {}
        context_memory = context_bundle.get("memory") if isinstance(context_bundle.get("memory"), dict) else {}
        context_skills = context_bundle.get("skills") if isinstance(context_bundle.get("skills"), dict) else {}
        context_tree = context_bundle.get("context_tree") if isinstance(context_bundle.get("context_tree"), dict) else {}
        trace = {
            "session_id": session_id,
            "user_message": message,
            "agent_input_preview": agent_input[:2000],
            "assistant_answer": response.answer,
            "intent": response.intent,
            "tool_calls": parsed_result.get("tool_calls", []),
            "command_preview": response.command_preview,
            "status_summary": response.status_summary,
            "safety_review": parsed_result.get("safety_review"),
            "memory_bundle": {
                "backend": context_memory.get("backend"),
                "selected_layers": context_memory.get("selected_layers", []),
                "hit_count": len(context_memory.get("hits") or []),
                "skill_matches": [
                    item.get("skill", {}).get("name")
                    for item in context_skills.get("matches", [])
                    if isinstance(item, dict)
                ],
                "context_tree_matches": [
                    item.get("node", {}).get("id")
                    for item in context_tree.get("matches", [])
                    if isinstance(item, dict)
                ],
                "context_sections": [
                    section.get("name")
                    for section in context_bundle.get("sections", [])
                    if isinstance(section, dict)
                ],
                "runtime_state": (context_bundle.get("runtime_status") or {}).get("state")
                if isinstance(context_bundle.get("runtime_status"), dict)
                else None,
            },
            "parsed_result": parsed_result,
        }
        try:
            record = self.memory_provider.record_trace(trace)
            patches = self.memory_provider.propose_writes(record)
        except Exception as exc:
            memory_os = {"error": str(exc), "workflow": "trace -> memory_patch -> approval"}
        else:
            memory_os = {
                "trace_id": record.get("trace_id"),
                "proposed_patch_count": len(patches),
                "pending_patches": [
                    {
                        "id": patch.get("id"),
                        "target": patch.get("target"),
                        "topic": patch.get("topic"),
                        "reason": patch.get("reason"),
                        "status": patch.get("status"),
                    }
                    for patch in patches
                ],
                "workflow": "trace -> memory_patch -> approval -> long_term_memory",
            }
        parsed_result = dict(parsed_result)
        parsed_result["memory_os"] = memory_os
        response.parsed_result = parsed_result

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
        help="Parse and submit one DAC-3D operation through the Agent control path.",
    )
    parser.add_argument(
        "--confirmed",
        action="store_true",
        help="Mark --execute-command as explicitly confirmed by the user.",
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
