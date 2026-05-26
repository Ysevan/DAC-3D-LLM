"""Tests for the OpenAI Agents SDK wrapper."""

from __future__ import annotations

import json
import os

from agents import Model, ModelResponse, Runner
from agents.run import RunConfig
from agents.usage import Usage
from app import DAC3DAssistant
from agent_runtime import (
    AGENT_TOOL_NAMES,
    DAC3DAgentChatAdapter,
    DAC3DAgentRuntime,
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
                            text="已通过 Agent 下发在线扫描。",
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


def test_agent_runtime_previews_control_command_without_submitting(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    payload = runtime.preview_command("start online scan")

    assert payload["intent"] == "operation"
    assert payload["command_preview"]["action"] == "start_online_scan"
    assert payload["status_summary"]["state"] == "idle"
    assert assistant.dac3d_client.runtime_snapshot()["last_command_action"] is None


def test_agent_runtime_requires_confirmation_for_control_command(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    payload = runtime.execute_command("start online scan", confirmed_by_user=False)

    assert payload["intent"] == "operation"
    assert payload["command_preview"]["action"] == "start_online_scan"
    assert "需要用户明确确认" in payload["answer"]
    assert assistant.dac3d_client.runtime_snapshot()["last_command_action"] is None


def test_agent_runtime_executes_confirmed_control_command(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    payload = runtime.execute_command("start online scan", confirmed_by_user=True)

    assert payload["intent"] == "operation"
    assert payload["command_preview"]["action"] == "start_online_scan"
    assert payload["status_summary"]["state"] == "queued"
    assert assistant.dac3d_client.runtime_snapshot()["last_command_action"] == "start_online_scan"


def test_agent_runtime_writes_confirmed_command_to_file_bridge(tmp_path) -> None:
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

    command_payload = json.loads(command_file.read_text(encoding="utf-8"))
    assert payload["status_summary"]["state"] == "command_sent"
    assert payload["status_summary"]["source"] == "command_file_bridge"
    assert command_payload["status"] == "pending"
    assert command_payload["command"]["action"] == "start_online_scan"


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

    assert result.final_output == "已通过 Agent 下发在线扫描。"
    assert assistant.dac3d_client.runtime_snapshot()["last_command_action"] == "start_online_scan"


def test_agent_chat_adapter_exposes_agent_runtime_summary(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    adapter = DAC3DAgentChatAdapter(runtime)

    summary = adapter.runtime_summary()

    assert summary["chat_runtime"] == "openai-agents"
    assert summary["agent"]["control"]["execute_tool"] == "dac3d_execute_command"


def test_agent_runtime_uses_openai_compatible_chat_completions_provider(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    config.agent_model_name = "vendor/dac3d-control-model"
    config.agent_api_key = "third-party-key"
    config.agent_api_base_url = "https://llm.example.test/v1"
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    run_config = runtime.build_run_config()
    model = run_config.model_provider.get_model(config.agent_model_name)

    assert runtime.resolved_agent_api_type() == "chat_completions"
    assert model.__class__.__name__ == "OpenAIChatCompletionsModel"
    assert model.model == "vendor/dac3d-control-model"
    assert "llm.example.test" in str(model._client.base_url)


def test_agent_runtime_describes_agent_project(tmp_path) -> None:
    config = make_agent_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)

    description = runtime.describe()

    assert description["name"] == "DAC-3D Inspection Agent"
    assert description["sdk"] == "openai-agents"
    assert description["underlying_runtime"] == "DAC3DAssistant"
    assert description["tools"] == list(AGENT_TOOL_NAMES)
    assert description["control"]["execute_tool"] == "dac3d_execute_command"
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
