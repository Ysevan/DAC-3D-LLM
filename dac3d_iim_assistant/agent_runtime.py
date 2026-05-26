"""OpenAI Agents SDK runtime for DAC-3D assistant workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from app import AssistantResponse, DAC3DAssistant
from config import AppConfig


AGENT_INSTRUCTIONS = """你是 DAC-3D 智能检测 Agent。

职责:
- 只围绕 DAC-3D 检测、文档问答、运行状态、结构化命令和结果解读工作。
- 对参数、流程、故障建议和缺陷严重度问题，必须调用 dac3d_answer 工具获取基于知识库的答案。
- 对扫描、离线检测、停止检测等操作请求，先调用 dac3d_preview_command 生成结构化命令。
- 如果用户已经明确要求执行、确认执行、立即开始或停止，调用 dac3d_execute_command 下发命令。
- 对当前状态问题，必须调用 dac3d_status 工具。
- 对检测结果或第几个样品的问题，必须调用 dac3d_latest_result 工具。
- 不要编造 DAC-3D 文档、状态、结果或设备能力。工具返回不确定时，要明确说明限制。
- 执行类命令必须遵守工具返回的安全提示；不要绕过人工确认、运行时忙碌检查或离线目录校验。
- needs_confirmation=true 的命令只有在用户同一轮或前文明确确认时，confirmed_by_user 才能设为 true。
"""

AGENT_TOOL_NAMES = (
    "dac3d_answer",
    "dac3d_operation",
    "dac3d_preview_command",
    "dac3d_execute_command",
    "dac3d_status",
    "dac3d_latest_result",
    "dac3d_rebuild_knowledge_base",
)
VALID_AGENT_API_TYPES = {"auto", "responses", "chat_completions"}


def response_to_payload(response: AssistantResponse) -> dict[str, Any]:
    """Convert the existing assistant response to a tool-friendly payload."""
    return {
        "intent": response.intent,
        "answer": response.answer,
        "sources": response.sources,
        "source_items": response.source_items,
        "command_preview": response.command_preview,
        "status_summary": response.status_summary,
        "parsed_result": response.parsed_result,
    }


@dataclass(slots=True)
class DAC3DAgentRuntime:
    """Build and run the DAC-3D assistant as an OpenAI Agents SDK agent."""

    assistant: DAC3DAssistant
    config: AppConfig

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
        return response_to_payload(self.assistant.handle_message(message))

    def preview_command(self, instruction: str) -> dict[str, Any]:
        """Generate a DAC-3D command preview without submitting it."""
        return response_to_payload(self.assistant.preview_operation_command(instruction))

    def execute_command(
        self,
        instruction: str,
        *,
        confirmed_by_user: bool = False,
    ) -> dict[str, Any]:
        """Generate and submit a DAC-3D command when confirmation and safety checks pass."""
        return response_to_payload(
            self.assistant.execute_operation_command(
                instruction,
                confirmed_by_user=confirmed_by_user,
            )
        )

    def describe(self) -> dict[str, Any]:
        """Return a stable description of this Agent project."""
        resolved_api_type = self.resolved_agent_api_type()
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
            },
            "max_turns": self.config.agent_max_turns,
            "tracing_disabled": self.config.agent_tracing_disabled,
            "tools": list(AGENT_TOOL_NAMES),
            "underlying_runtime": "DAC3DAssistant",
            "mock_mode": self.config.mock_mode,
            "control": {
                "preview_tool": "dac3d_preview_command",
                "execute_tool": "dac3d_execute_command",
                "confirmation_required_for_risky_commands": True,
                "bridge_modes": ["embedded", "command_file_bridge", "mock"],
            },
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
            from agents.models.multi_provider import MultiProvider
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "OpenAI Agents SDK is not installed. Install it with `pip install openai-agents`."
            ) from exc

        api_type = self.resolved_agent_api_type()
        use_responses = api_type == "responses"
        return MultiProvider(
            openai_api_key=self.config.agent_api_key or None,
            openai_base_url=self.config.agent_api_base_url or None,
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

        return RunConfig(
            workflow_name="DAC-3D Agent",
            tracing_disabled=self.config.agent_tracing_disabled,
            model_provider=self.build_model_provider(),
        )

    def build_agent(self) -> Any:
        """Create an OpenAI Agents SDK Agent with DAC-3D tools."""
        try:
            from agents import Agent, function_tool
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "OpenAI Agents SDK is not installed. Install it with `pip install openai-agents`."
            ) from exc

        assistant = self.assistant

        @function_tool(
            name_override="dac3d_answer",
            description_override=(
                "Answer DAC-3D document, parameter, workflow, troubleshooting, "
                "and operator guidance questions using the local DAC-3D assistant."
            ),
        )
        def dac3d_answer(question: str) -> dict[str, Any]:
            """Answer a DAC-3D knowledge or guidance question."""
            return response_to_payload(assistant.handle_message(question))

        @function_tool(
            name_override="dac3d_operation",
            description_override=(
                "Legacy DAC-3D operation parser. It returns a structured command preview "
                "without submitting the command; use dac3d_execute_command for control."
            ),
        )
        def dac3d_operation(instruction: str) -> dict[str, Any]:
            """Preview a DAC-3D operation request through the compatibility tool name."""
            return self.preview_command(instruction)

        @function_tool(
            name_override="dac3d_preview_command",
            description_override=(
                "Parse a natural-language DAC-3D control request into a structured command "
                "preview with runtime status and safety information. This never submits the command."
            ),
        )
        def dac3d_preview_command(instruction: str) -> dict[str, Any]:
            """Preview a DAC-3D control command without submitting it."""
            return self.preview_command(instruction)

        @function_tool(
            name_override="dac3d_execute_command",
            description_override=(
                "Parse and submit a DAC-3D control request to the active runtime bridge. "
                "Set confirmed_by_user=true only when the user explicitly confirmed execution "
                "or directly asked to start/stop/execute the operation."
            ),
        )
        def dac3d_execute_command(
            instruction: str,
            confirmed_by_user: bool = False,
        ) -> dict[str, Any]:
            """Execute a DAC-3D control command after confirmation and safety checks."""
            return self.execute_command(
                instruction,
                confirmed_by_user=confirmed_by_user,
            )

        @function_tool(
            name_override="dac3d_status",
            description_override="Read the current DAC-3D runtime status and progress.",
        )
        def dac3d_status() -> dict[str, Any]:
            """Read current DAC-3D runtime status."""
            return response_to_payload(assistant.handle_message("查询当前检测状态"))

        @function_tool(
            name_override="dac3d_latest_result",
            description_override=(
                "Read and interpret the latest DAC-3D inspection result. Pass sample_position=0 "
                "for the latest/all recorded results, or a 1-based sample position."
            ),
        )
        def dac3d_latest_result(sample_position: int) -> dict[str, Any]:
            """Read a DAC-3D inspection result summary."""
            if sample_position > 0:
                message = f"第{sample_position}个样品检测结果怎么样？"
            else:
                message = "当前检测结果怎么样？"
            return response_to_payload(assistant.handle_message(message))

        @function_tool(
            name_override="dac3d_rebuild_knowledge_base",
            description_override="Rebuild the local DAC-3D knowledge base from current documents.",
        )
        def dac3d_rebuild_knowledge_base() -> dict[str, Any]:
            """Rebuild the local DAC-3D knowledge base."""
            return assistant.build_knowledge_base_from_uploads()

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
            ],
            model=self.config.agent_model_name or None,
        )

    def run_sync(self, message: str) -> str:
        """Run a user message through the OpenAI Agents SDK."""
        try:
            from agents import Runner
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "OpenAI Agents SDK is not installed. Install it with `pip install openai-agents`."
            ) from exc

        try:
            result = Runner.run_sync(
                self.build_agent(),
                message,
                max_turns=self.config.agent_max_turns,
                run_config=self.build_run_config(),
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

    @property
    def config(self) -> AppConfig:
        """Return the underlying app config expected by the web UI."""
        return self.runtime.config

    def handle_message(
        self,
        message: str,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> AssistantResponse:
        """Run web/Gradio chat messages through the Agent runtime."""
        try:
            answer = self.runtime.run_sync(self._format_agent_input(message, history))
        except Exception as exc:
            return AssistantResponse(intent="agent", answer=f"Agent run failed: {exc}")
        return AssistantResponse(intent="agent", answer=answer)

    def stream_message(
        self,
        message: str,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Provide a buffered streaming-compatible surface for Agent responses."""
        response = self.handle_message(message, history)
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
        return summary

    def knowledge_base_summary(self) -> dict[str, Any]:
        """Delegate knowledge-base diagnostics to the underlying assistant."""
        return self.runtime.assistant.knowledge_base_summary()

    def build_knowledge_base_from_uploads(self, uploaded_files: Sequence[Any] | None = None) -> dict[str, Any]:
        """Delegate knowledge-base rebuilds to the underlying assistant."""
        return self.runtime.assistant.build_knowledge_base_from_uploads(uploaded_files)

    def _format_agent_input(
        self,
        message: str,
        history: Sequence[tuple[str, str]] | None,
    ) -> str:
        if not history:
            return message
        turns: list[str] = []
        for user_message, assistant_message in history[-self.config.history_window :]:
            turns.append(f"用户: {user_message}")
            turns.append(f"助手: {assistant_message}")
        turns.append(f"当前用户: {message}")
        return "\n".join(turns)


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
            print(runtime.run_sync(args.message))
        except Exception as exc:
            print(f"Agent run failed: {exc}")
        return

    print("Use --message, --preview-command, --execute-command, --list-tools, or --describe.")


if __name__ == "__main__":
    main()
