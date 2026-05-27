"""Tests for the OpenAI Agents SDK wrapper."""

from __future__ import annotations

import json
import os
from typing import Any

from agents import Model, ModelResponse, Runner
from agents.run import RunConfig
from agents.usage import Usage
from app import DAC3DAssistant
from agent_runtime import (
    AGENT_OUTPUT_PARSE_ERROR,
    AGENT_TOOL_NAMES,
    DAC3DAgentChatAdapter,
    DAC3DAgentRuntime,
    LOCAL_VALIDATION_MODEL_NAME,
    build_argument_parser,
)
from config import AppConfig
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText


class FakeControlModel(Model):
    """Local model stub that emits a DAC-3D execute tool call, then a final message."""

    def __init__(self) -> None:
        self.calls = 0

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
    ) -> ModelResponse:
        del (
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
            conversation_id,
            prompt,
        )
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                output=[
                    ResponseFunctionToolCall(
                        arguments=json.dumps(
                            {
                                "instruction": "start online scan",
                                "confirmed_by_user": True,
                            }
                        ),
                        call_id="call_1",
                        name="dac3d_execute_command",
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
                    id="msg_1",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        ResponseOutputText(
                            annotations=[],
                            text="Agent 直接下发已被安全策略阻断，需要通过 Web API token 确认。",
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


class FakeMachineModel(Model):
    """Local model stub that chooses the Machine Agent tool before answering."""

    def __init__(self) -> None:
        self.calls = 0

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
    ) -> ModelResponse:
        del (
            system_instructions,
            input,
            model_settings,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
            conversation_id,
            prompt,
        )
        self.calls += 1
        tool_names = {tool.name for tool in tools}
        assert "machine_agent_chat" in tool_names
        if self.calls == 1:
            return ModelResponse(
                output=[
                    ResponseFunctionToolCall(
                        arguments=json.dumps(
                            {"message": "为什么最近温度报警变多了？"}
                        ),
                        call_id="call_machine",
                        name="machine_agent_chat",
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
                    id="msg_machine",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        ResponseOutputText(
                            annotations=[],
                            text="LLM 已调用 machine_agent_chat 分析温度报警。",
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


class FakeJsonToolModel(Model):
    """Local model stub that calls one Agent tool, then emits the final JSON contract."""

    def __init__(self, tool_name: str, arguments: dict[str, Any], final_payload: dict[str, Any]) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.final_payload = final_payload
        self.calls = 0

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
    ) -> ModelResponse:
        del (
            system_instructions,
            input,
            model_settings,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
            conversation_id,
            prompt,
        )
        self.calls += 1
        tool_names = {tool.name for tool in tools}
        assert self.tool_name in tool_names
        if self.calls == 1:
            return ModelResponse(
                output=[
                    ResponseFunctionToolCall(
                        arguments=json.dumps(self.arguments, ensure_ascii=False),
                        call_id=f"call_{self.tool_name}",
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
                    id=f"msg_{self.tool_name}",
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


def make_agent_config(tmp_path) -> AppConfig:
    config = AppConfig(
        base_dir=tmp_path,
        mock_mode=True,
        provider="mock",
        vector_store_type="manifest",
        embedding_download_allowed=False,
    )
    config.ensure_directories()
    (config.documents_dir / "manual.md").write_text(
        "# DAC-3D 手册\n\n## 状态\nDAC-3D 状态包括 idle、running 和 error。\n",
        encoding="utf-8",
    )
    return config


def test_agent_runtime_builds_dac3d_agent(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    agent = runtime.build_agent()

    assert agent.name == "DAC-3D Inspection Agent"
    tool_names = {tool.name for tool in agent.tools}
    assert {
        "dac3d_answer",
        "dac3d_operation",
        "dac3d_preview_command",
        "dac3d_execute_command",
        "dac3d_status",
        "dac3d_latest_result",
        "dac3d_rebuild_knowledge_base",
        "machine_agent_chat",
        "machine_status",
        "machine_alarms",
        "machine_abnormal_analysis",
    }.issubset(tool_names)
    assert set(AGENT_TOOL_NAMES).issubset(tool_names)


def test_agent_runtime_can_call_existing_assistant_router(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    payload = runtime.handle_with_assistant("当前检测状态是什么？")

    assert payload["intent"] == "status"
    assert "DAC-3D" in payload["answer"]
    assert payload["status_summary"]["state"] == "idle"
    assert payload["tool_gateway"]["decision"]["tool_name"] == "dac3d_answer"
    assert payload["policy_decision"]["allowed"] is True


def test_agent_runtime_uses_llm_authored_structured_output(tmp_path, monkeypatch) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    def fake_run_sync(self, message: str, *, session_id: str = "default") -> str:
        del self
        assert session_id == "web"
        assert message == "当前检测状态是什么？"
        return json.dumps(
            {
                "answer": "当前 DAC-3D 处于空闲状态，进度 0%。",
                "structured_data": {
                    "intent": "status",
                    "tool_calls": [
                        {"name": "dac3d_status", "purpose": "读取当前检测状态"}
                    ],
                    "status_summary": {
                        "state": "idle",
                        "progress": 0,
                        "message": "当前没有正在执行的检测任务",
                    },
                    "next_action": "可以开始新的检测任务",
                },
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(DAC3DAgentRuntime, "run_sync", fake_run_sync)

    payload = runtime.run_chat_payload("当前检测状态是什么？", session_id="web")

    assert payload["intent"] == "status"
    assert payload["answer"] == "当前 DAC-3D 处于空闲状态，进度 0%。"
    assert payload["status_summary"]["state"] == "idle"
    assert payload["parsed_result"]["tool_calls"][0]["name"] == "dac3d_status"
    assert payload["parsed_result"]["agent_mode"] == "openai-agents-sdk"


def test_agent_runtime_keeps_raw_answer_when_structured_output_is_missing(
    tmp_path,
    monkeypatch,
) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    monkeypatch.setattr(
        DAC3DAgentRuntime,
        "run_sync",
        lambda self, message, *, session_id="default": "普通自然语言回复",
    )

    payload = runtime.run_chat_payload("你好", session_id="web")

    assert payload["intent"] == "agent"
    assert payload["answer"] == "普通自然语言回复"
    assert payload["parsed_result"]["type"] == AGENT_OUTPUT_PARSE_ERROR


def test_agent_runtime_previews_control_command_without_submitting(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    payload = runtime.preview_command("start online scan")

    assert payload["intent"] == "operation"
    assert payload["command_preview"]["action"] == "start_online_scan"
    assert payload["status_summary"]["state"] == "idle"
    assert payload["tool_gateway"]["decision"]["tool_name"] == "dac3d_preview_command"
    assert payload["policy_decision"]["requires_confirmation"] is True
    assert assistant.dac3d_client.runtime_snapshot()["last_command_action"] is None


def test_agent_runtime_blocks_agent_direct_knowledge_base_rebuild(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    payload = runtime.tool_controller().rebuild_knowledge_base()

    assert payload["intent"] == "tool_policy"
    assert payload["policy_decision"]["allowed"] is False
    assert payload["policy_decision"]["reason"] == "TOOL_DIRECT_USE_DISABLED"
    assert payload["policy_decision"]["tool_name"] == "dac3d_rebuild_knowledge_base"


def test_agent_runtime_machine_tools_use_gateway_policy(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    payload = runtime.tool_controller().machine_status()

    assert payload["machine_id"] == "IM-Press-01"
    assert payload["tool_gateway"]["decision"]["tool_name"] == "machine_status"
    assert payload["policy_decision"]["allowed"] is True
    assert payload["policy_decision"]["read_only"] is True


def test_agents_sdk_dialogue_tool_call_covers_machine_agent(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    agent = runtime.build_agent()
    agent.model = FakeMachineModel()

    result = Runner.run_sync(
        agent,
        "为什么最近温度报警变多了？",
        max_turns=3,
        run_config=RunConfig(tracing_disabled=True),
    )

    assert result.final_output == "LLM 已调用 machine_agent_chat 分析温度报警。"


def test_agent_runtime_chat_payload_covers_core_tool_paths_without_network(
    tmp_path,
    monkeypatch,
) -> None:
    cases = [
        {
            "message": "当前检测状态是什么？",
            "tool_name": "dac3d_status",
            "arguments": {},
            "final_payload": {
                "answer": "当前 DAC-3D 处于空闲状态。",
                "structured_data": {
                    "intent": "status",
                    "tool_calls": [{"name": "dac3d_status", "purpose": "读取状态"}],
                    "status_summary": {"state": "idle", "progress": 0},
                },
            },
            "expected_intent": "status",
        },
        {
            "message": "样品表面反光很强怎么办？",
            "tool_name": "dac3d_answer",
            "arguments": {"question": "样品表面反光很强怎么办？"},
            "final_payload": {
                "answer": "建议降低曝光、调整入射角并重新标定。",
                "structured_data": {
                    "intent": "guidance",
                    "tool_calls": [{"name": "dac3d_answer", "purpose": "查询操作建议"}],
                    "recommendations": ["降低曝光", "调整入射角", "重新标定"],
                },
            },
            "expected_intent": "guidance",
        },
        {
            "message": "扫描 10mm x 10mm 区域",
            "tool_name": "dac3d_preview_command",
            "arguments": {"instruction": "扫描 10mm x 10mm 区域"},
            "final_payload": {
                "answer": "已生成扫描命令预览，执行前需要确认。",
                "structured_data": {
                    "intent": "operation_preview",
                    "tool_calls": [
                        {"name": "dac3d_preview_command", "purpose": "生成命令预览"}
                    ],
                    "command_preview": {
                        "action": "start_online_scan",
                        "payload": {"width_mm": 10, "height_mm": 10},
                        "safety": {"needs_confirmation": True},
                    },
                    "requires_confirmation": True,
                },
            },
            "expected_intent": "operation_preview",
        },
        {
            "message": "第三个样品检测结果怎么样？",
            "tool_name": "dac3d_latest_result",
            "arguments": {"sample_position": 3},
            "final_payload": {
                "answer": "第三个样品存在较高风险缺陷，建议复检。",
                "structured_data": {
                    "intent": "interpretation",
                    "tool_calls": [{"name": "dac3d_latest_result", "purpose": "读取样品结果"}],
                    "result_summary": {"sample_position": 3, "recommendation": "复检"},
                },
            },
            "expected_intent": "interpretation",
        },
        {
            "message": "为什么最近温度报警变多了？",
            "tool_name": "machine_agent_chat",
            "arguments": {"message": "为什么最近温度报警变多了？"},
            "final_payload": {
                "answer": "最近温度报警增多，优先检查冷却水路和风扇滤网。",
                "structured_data": {
                    "intent": "machine_alarm_analysis",
                    "tool_calls": [
                        {"name": "machine_agent_chat", "purpose": "分析报警趋势"}
                    ],
                    "recommendations": ["检查冷却水路", "检查风扇滤网"],
                },
            },
            "expected_intent": "machine_alarm_analysis",
        },
    ]
    original_build_agent = DAC3DAgentRuntime.build_agent

    for index, case in enumerate(cases):
        config = make_agent_config(tmp_path / f"case-{index}")
        config.agent_api_key = "test-key"
        assistant = DAC3DAssistant.create(config=config)
        runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

        def fake_build_agent(self, *, session_id: str = "default", selected_case=case):
            agent = original_build_agent(self, session_id=session_id)
            agent.model = FakeJsonToolModel(
                selected_case["tool_name"],
                selected_case["arguments"],
                selected_case["final_payload"],
            )
            return agent

        monkeypatch.setattr(DAC3DAgentRuntime, "build_agent", fake_build_agent)

        payload = runtime.run_chat_payload(case["message"], session_id=f"web-{index}")

        assert payload["intent"] == case["expected_intent"]
        assert payload["answer"] == case["final_payload"]["answer"]
        assert payload["parsed_result"]["tool_calls"][0]["name"] == case["tool_name"]
        assert payload["parsed_result"]["agent_mode"] == "openai-agents-sdk"


def test_agent_runtime_requires_confirmation_for_control_command(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    payload = runtime.execute_command("start online scan", confirmed_by_user=False)

    assert payload["intent"] == "operation"
    assert payload["command_preview"]["action"] == "start_online_scan"
    assert "需要用户明确确认" in payload["answer"]
    assert assistant.dac3d_client.runtime_snapshot()["last_command_action"] is None


def test_agent_runtime_blocks_confirmed_control_command_without_api_token(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    payload = runtime.execute_command("start online scan", confirmed_by_user=True)

    assert payload["intent"] == "operation"
    assert payload["command_preview"]["action"] == "start_online_scan"
    assert payload["policy_decision"]["allowed"] is False
    assert payload["policy_decision"]["reason"] == "TOKEN_BOUND_CONFIRMATION_REQUIRED"
    assert payload["tool_gateway"]["decision"]["tool_name"] == "dac3d_execute_command"
    assert payload["tool_gateway"]["decision"]["requires_token_bound_confirmation"] is True
    assert payload["confirmation"]["blocked_direct_submit"] is True
    assert payload["agent_session"]["has_pending_command"] is True
    assert assistant.dac3d_client.runtime_snapshot()["last_command_action"] is None


def test_agent_runtime_blocks_chinese_direct_scan_command_without_api_token(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    payload = runtime.execute_command("执行扫描", confirmed_by_user=True)

    assert payload["intent"] == "operation"
    assert payload["command_preview"]["action"] == "start_online_scan"
    assert payload["command_preview"]["payload"]["func"] == "Scan"
    assert payload["policy_decision"]["reason"] == "TOKEN_BOUND_CONFIRMATION_REQUIRED"
    assert payload["confirmation"]["mode"] == "api_preview_confirm_token"
    assert assistant.dac3d_client.runtime_snapshot()["last_command_action"] is None


def test_agent_runtime_blocks_pending_confirmation_without_api_token(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    preview = runtime.preview_command("start online scan", session_id="operator-session")
    payload = runtime.execute_command(
        "确认执行",
        confirmed_by_user=True,
        session_id="operator-session",
    )

    assert preview["agent_session"]["has_pending_command"] is True
    assert payload["command_preview"]["action"] == "start_online_scan"
    assert payload["policy_decision"]["reason"] == "TOKEN_BOUND_CONFIRMATION_REQUIRED"
    assert payload["policy_decision"]["requires_one_time_token"] is True
    assert payload["agent_session"]["has_pending_command"] is True
    assert assistant.dac3d_client.runtime_snapshot()["last_command_action"] is None


def test_agent_runtime_blocks_confirmed_command_file_bridge_without_api_token(tmp_path) -> None:
    status_file = tmp_path / "dac3d_runtime_status.json"
    command_file = tmp_path / "dac3d_assistant_command.json"
    status_file.write_text(
        json.dumps({"status": {"state": "idle", "progress": 0}}, ensure_ascii=False),
        encoding="utf-8",
    )
    config = make_agent_config(tmp_path)
    config.dac3d_endpoint = status_file.as_uri()
    old_command_path = os.environ.get("DAC3D_COMMAND_PATH")
    os.environ["DAC3D_COMMAND_PATH"] = str(command_file)
    try:
        assistant = DAC3DAssistant.create(config=config)
        runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

        payload = runtime.execute_command("start online scan", confirmed_by_user=True)
    finally:
        if old_command_path is None:
            os.environ.pop("DAC3D_COMMAND_PATH", None)
        else:
            os.environ["DAC3D_COMMAND_PATH"] = old_command_path

    assert payload["command_preview"]["action"] == "start_online_scan"
    assert payload["policy_decision"]["reason"] == "TOKEN_BOUND_CONFIRMATION_REQUIRED"
    assert not command_file.exists()


def test_agent_runtime_blocks_chinese_direct_scan_file_bridge_without_api_token(tmp_path) -> None:
    status_file = tmp_path / "dac3d_runtime_status.json"
    command_file = tmp_path / "dac3d_assistant_command.json"
    status_file.write_text(
        json.dumps({"status": {"state": "idle", "progress": 0}}, ensure_ascii=False),
        encoding="utf-8",
    )
    config = make_agent_config(tmp_path)
    config.dac3d_endpoint = status_file.as_uri()
    old_command_path = os.environ.get("DAC3D_COMMAND_PATH")
    os.environ["DAC3D_COMMAND_PATH"] = str(command_file)
    try:
        assistant = DAC3DAssistant.create(config=config)
        runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

        payload = runtime.execute_command("执行扫描", confirmed_by_user=True)
    finally:
        if old_command_path is None:
            os.environ.pop("DAC3D_COMMAND_PATH", None)
        else:
            os.environ["DAC3D_COMMAND_PATH"] = old_command_path

    assert payload["command_preview"]["action"] == "start_online_scan"
    assert payload["command_preview"]["payload"]["func"] == "Scan"
    assert payload["policy_decision"]["reason"] == "TOKEN_BOUND_CONFIRMATION_REQUIRED"
    assert not command_file.exists()


def test_agents_sdk_dialogue_tool_call_controls_runtime_without_network(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    agent = runtime.build_agent()
    agent.model = FakeControlModel()

    result = Runner.run_sync(
        agent,
        "确认立即开始在线扫描",
        max_turns=3,
        run_config=RunConfig(tracing_disabled=True),
    )

    assert result.final_output == "Agent 直接下发已被安全策略阻断，需要通过 Web API token 确认。"
    assert assistant.dac3d_client.runtime_snapshot()["last_command_action"] is None


def test_agent_chat_adapter_exposes_agent_runtime_summary(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    adapter = DAC3DAgentChatAdapter(runtime)

    summary = adapter.runtime_summary()

    assert summary["chat_runtime"] == "openai-agents"
    assert summary["agent"]["control"]["execute_tool"] == "dac3d_execute_command"
    assert summary["memory"]["backend"] == "json"


def test_agent_chat_adapter_persists_and_injects_json_memory(
    tmp_path,
    monkeypatch,
) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    seen_messages: list[str] = []

    def fake_run_sync(self, message: str, *, session_id: str = "default") -> str:
        del self
        assert session_id == "memory-session"
        seen_messages.append(message)
        return json.dumps(
            {
                "answer": "已处理。",
                "structured_data": {
                    "intent": "guidance",
                    "tool_calls": [{"name": "dac3d_answer", "purpose": "查询建议"}],
                },
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(DAC3DAgentRuntime, "run_sync", fake_run_sync)
    adapter = DAC3DAgentChatAdapter(runtime)

    first = adapter.handle_message(
        "样品表面反光很强怎么办？",
        session_id="memory-session",
    )
    second = adapter.handle_message(
        "反光还要注意什么？",
        session_id="memory-session",
    )

    assert first.answer == "已处理。"
    assert seen_messages[0] == "样品表面反光很强怎么办？"
    assert "会话记忆（JSON 最近对话）" in seen_messages[1]
    assert "长期记忆检索（JSON 索引命中）" in seen_messages[1]
    assert "样品表面反光很强怎么办？" in seen_messages[1]
    assert second.parsed_result["memory"]["backend"] == "json"
    session_file = config.conversation_memory_dir / "sessions" / "memory-session.json"
    assert session_file.exists()


def test_agent_runtime_uses_openai_compatible_chat_completions_provider(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    config.agent_model_name = "vendor/dac3d-control-model"
    config.agent_api_key = "third-party-key"
    config.agent_api_base_url = "https://llm.example.test/v1"
    config.timeout_seconds = 12.5
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    run_config = runtime.build_run_config()
    model = run_config.model_provider.get_model(config.agent_model_name)

    assert runtime.resolved_agent_api_type() == "chat_completions"
    assert model.__class__.__name__ == "OpenAIChatCompletionsModel"
    assert model.model == "vendor/dac3d-control-model"
    assert "llm.example.test" in str(model._client.base_url)
    assert model._client.timeout == 12.5


def test_agent_runtime_local_validation_model_runs_without_credentials(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    config.agent_model_name = LOCAL_VALIDATION_MODEL_NAME
    config.agent_api_key = ""
    config.agent_api_base_url = ""
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    payload = runtime.run_chat_payload("为什么最近温度报警变多了？", session_id="chrome-test")

    assert payload["intent"] == "machine_alarm_analysis"
    assert "温度报警" in payload["answer"]
    assert payload["parsed_result"]["tool_calls"][0]["name"] == "machine_agent_chat"
    assert runtime.describe()["model_provider"]["local_validation"] is True


def test_agent_runtime_describes_agent_project(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    description = runtime.describe()

    assert description["name"] == "DAC-3D Inspection Agent"
    assert description["sdk"] == "openai-agents"
    assert description["underlying_runtime"] == "DAC3DAssistant"
    assert "MachineAgentService" in description["capability_runtimes"]
    assert description["tools"] == list(AGENT_TOOL_NAMES)
    assert description["control"]["execute_tool"] == "dac3d_execute_command"
    assert description["control"]["direct_agent_submit_allowed"] is False
    assert description["control"]["tool_gateway_enforced"] is True
    assert description["control"]["confirmation_flow"] == "api_preview_confirm_token"
    assert description["tool_gateway"]["fail_closed"] is True
    assert set(description["tool_gateway"]["registered_tools"]) == set(AGENT_TOOL_NAMES)
    assert description["model_provider"]["resolved_api_type"] == "responses"


def test_agent_runtime_parser_supports_agent_project_commands() -> None:
    parser = build_argument_parser()

    args = parser.parse_args(
        [
            "--describe",
            "--list-tools",
            "--preview-command",
            "start online scan",
            "--execute-command",
            "stop detection",
            "--confirmed",
            "--agent-model",
            "gpt-5.4-mini",
            "--agent-api-base-url",
            "https://llm.example.test/v1",
            "--agent-api-key",
            "third-party-key",
            "--agent-api-type",
            "chat_completions",
        ]
    )

    assert args.describe is True
    assert args.list_tools is True
    assert args.preview_command == "start online scan"
    assert args.execute_command == "stop detection"
    assert args.confirmed is True
    assert args.agent_model == "gpt-5.4-mini"
    assert args.agent_api_base_url == "https://llm.example.test/v1"
    assert args.agent_api_key == "third-party-key"
    assert args.agent_api_type == "chat_completions"
