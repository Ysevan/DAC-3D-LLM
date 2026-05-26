"""Main entry point and top-level routing for the DAC-3D assistant."""

from __future__ import annotations

import argparse
import json
import shutil
import threading
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterator, Sequence
from typing import Any

from config import AppConfig
from intent.classifier import IntentClassifier, IntentResult
from intent.command_generator import CommandGenerator
from integration.dac3d_client import DAC3DClient, DAC3DUnavailableError, DAC3DValidationError
from integration.result_parser import parse_result, parse_result_from_text
from knowledge_base.build_kb import (
    build_knowledge_base,
    describe_document_format,
    detect_document_format,
    summarize_knowledge_base,
)
from rag.llm_client import LLMClient, LLMConfigurationError, LLMProviderError
from rag.prompts import (
    build_command_clarification_prompt,
    build_greeting_prompt,
    build_guidance_prompt,
    build_interpretation_prompt,
    build_qa_prompt,
)
from rag.retriever import RetrievalItem, Retriever
from ui.chat_widget import ChatWidget
from ui.web_api import create_api_app


@dataclass(slots=True)
class AssistantResponse:
    """Top-level response returned by the application router."""

    intent: str
    answer: str
    sources: list[str] = field(default_factory=list)
    source_items: list[dict[str, Any]] = field(default_factory=list)
    command_preview: dict[str, Any] | None = None
    status_summary: dict[str, Any] | None = None
    parsed_result: dict[str, Any] | None = None

    def render_text(self) -> str:
        """Render the response into a readable terminal block."""
        lines = [f"Intent: {self.intent}", self.answer]
        if self.command_preview is not None:
            lines.append("Command Preview:")
            lines.append(json.dumps(self.command_preview, indent=2, ensure_ascii=False))
        if self.status_summary is not None:
            lines.append("Status:")
            lines.append(json.dumps(self.status_summary, indent=2, ensure_ascii=False))
        if self.parsed_result is not None:
            lines.append("Parsed Result:")
            lines.append(json.dumps(self.parsed_result, indent=2, ensure_ascii=False))
        if self.sources:
            lines.append("Sources:")
            lines.extend(f"- {source}" for source in self.sources)
        return "\n".join(lines)

    def to_ui_payload(self) -> dict[str, Any]:
        """Return UI-friendly structured data."""
        return {
            "intent": self.intent,
            "answer": self.answer,
            "sources": self.source_items,
            "command_preview": self.command_preview,
            "status_summary": self.status_summary,
            "parsed_result": self.parsed_result,
        }


class DAC3DAssistant:
    """Application orchestrator for all DAC-3D assistant flows."""

    _retrieval_expansions = {
        "反光": "样品反光 reflective samples 照明 曝光 消光 倾角",
        "不稳定": "扫描不稳定 unstable scans 样品固定 选区 standard mode precision mode",
        "预览": "preview unstable scans standard mode precision mode",
        "命名区域": "region selection current selection ROI",
        "整片扫描": "scan area 扩大扫描范围 current selection ROI",
        "重新扫描": "inspection status 状态消息 进度百分比",
        "状态": "inspection status 状态消息 进度百分比",
        "precision mode": "scan modes standard mode precision mode defect confirmation",
    }

    def __init__(
        self,
        *,
        config: AppConfig,
        retriever: Retriever,
        llm_client: LLMClient,
        intent_classifier: IntentClassifier,
        command_generator: CommandGenerator,
        dac3d_client: DAC3DClient,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.llm_client = llm_client
        self.intent_classifier = intent_classifier
        self.command_generator = command_generator
        self.dac3d_client = dac3d_client

    @classmethod
    def create(
        cls,
        config: AppConfig | None = None,
        *,
        rebuild_kb: bool = False,
        dac3d_client: DAC3DClient | None = None,
        runtime_bridge: Any | None = None,
    ) -> "DAC3DAssistant":
        """Create an application instance and ensure the vector store exists."""
        active_config = config or AppConfig.from_env()
        active_config.ensure_directories()
        if rebuild_kb or not active_config.vector_store_ready:
            build_knowledge_base(active_config)

        return cls(
            config=active_config,
            retriever=Retriever.from_config(active_config),
            llm_client=LLMClient(active_config),
            intent_classifier=IntentClassifier(),
            command_generator=CommandGenerator(),
            dac3d_client=dac3d_client
            or DAC3DClient(
                mock_mode=active_config.mock_mode,
                endpoint=active_config.dac3d_endpoint,
                runtime_bridge=runtime_bridge,
            ),
        )

    def handle_message(
        self,
        message: str,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> AssistantResponse:
        """Route a user message into the matching DAC-3D flow."""
        try:
            clean_message = message.strip()
            if not clean_message:
                return AssistantResponse(intent="error", answer="请输入有效问题。")

            hello_response = self._hello_model_response(clean_message)
            if hello_response is not None:
                return hello_response

            intent = self.intent_classifier.classify(clean_message)
            history_tail = self._trim_history(history)

            if intent.label == "status":
                return self._handle_status_query(intent)
            if intent.label == "operation":
                return self._handle_operation(clean_message, history_tail)
            if intent.label == "interpretation":
                return self._handle_interpretation(clean_message, history_tail, intent)
            if intent.label == "guidance":
                return self._handle_guidance(clean_message, history_tail, intent)
            return self._handle_query(clean_message, history_tail, intent)
        except (
            FileNotFoundError,
            LLMConfigurationError,
            LLMProviderError,
            DAC3DValidationError,
            DAC3DUnavailableError,
        ) as exc:
            return self._error_response(exc)
        except Exception as exc:  # pragma: no cover - defensive top-level guard
            return self._error_response(exc)

    def stream_message(
        self,
        message: str,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield UI streaming events for a user message."""
        try:
            clean_message = message.strip()
            if not clean_message:
                yield "error", {"message": "请输入有效问题。"}
                return

            hello_response = self._hello_model_response(clean_message)
            if hello_response is not None:
                yield from self._emit_buffered_response(hello_response)
                return

            intent = self.intent_classifier.classify(clean_message)
            history_tail = self._trim_history(history)
            yield self._status_event("intent", "分析请求中")

            if intent.label == "status":
                response = self._handle_status_query(intent)
                yield from self._emit_buffered_response(response)
                return
            if intent.label == "operation":
                yield from self._stream_operation(clean_message, history_tail)
                return
            if intent.label == "interpretation":
                yield from self._stream_interpretation(clean_message, history_tail, intent)
                return
            if intent.label == "guidance":
                yield from self._stream_grounded_answer(clean_message, history_tail, intent, task="guidance")
                return
            yield from self._stream_grounded_answer(clean_message, history_tail, intent, task="query")
        except Exception as exc:  # pragma: no cover - defensive top-level guard
            error_response = self._error_response(exc)
            yield "error", {"message": error_response.answer}

    def runtime_summary(self) -> dict[str, Any]:
        """Return a diagnostic summary for the UI."""
        manifest = getattr(self.retriever, "_manifest", {})
        storage_backend = str(manifest.get("storage_backend", self.config.vector_store_type))
        embedding_backend = str(manifest.get("embedding_backend", "unknown"))
        knowledge_base = self.knowledge_base_summary()
        return {
            "provider": self.config.provider,
            "model": self.config.model_name,
            "vector_store": storage_backend,
            "embedding_backend": embedding_backend,
            "embedding_model_name": self.config.embedding_model_name,
            "embedding_download_allowed": self.config.embedding_download_allowed,
            "retrieval_top_k": self.config.retrieval_top_k,
            "mock_mode": self.config.mock_mode,
            "dac3d": self.dac3d_client.runtime_snapshot(),
            "document_count": knowledge_base.get("document_count", 0),
            "chunk_count": knowledge_base.get("chunk_count", len(list(getattr(self.retriever, "_chunks", [])))),
            "latest_build_at": knowledge_base.get("latest_build_at"),
            "knowledge_base": knowledge_base,
        }

    def _hello_model_response(self, message: str) -> AssistantResponse | None:
        """Return an LLM-generated model introduction for plain hello messages."""
        if message.casefold() != "hello":
            return None

        provider = self.config.provider.strip() or "unknown provider"
        model_name = self.config.model_name.strip() or "unknown model"
        prompt = build_greeting_prompt(provider=provider, model_name=model_name)
        answer = self.llm_client.generate(
            prompt,
            task="greeting",
            question=f"{provider}||{model_name}",
        )
        return AssistantResponse(
            intent="query",
            answer=answer,
        )

    def knowledge_base_summary(self) -> dict[str, Any]:
        """Return persisted knowledge-base metadata for the Web control panel."""
        return summarize_knowledge_base(self.config)

    def build_knowledge_base_from_uploads(self, uploaded_files: Sequence[Any] | None = None) -> dict[str, Any]:
        """Copy uploaded files into the knowledge-base directory and rebuild the vector store."""
        self.config.ensure_directories()
        saved_files: list[str] = []
        normalized_uploads: list[Any]
        if uploaded_files is None:
            normalized_uploads = []
        elif isinstance(uploaded_files, (str, Path)):
            normalized_uploads = [uploaded_files]
        else:
            normalized_uploads = list(uploaded_files)

        for file_ref in normalized_uploads:
            source_path = self._resolve_uploaded_path(file_ref)
            detected_format = detect_document_format(source_path)
            if detected_format not in {"text", "pdf", "docx"}:
                raise ValueError(
                    f"Unsupported knowledge-base file: {source_path.name}. {describe_document_format(source_path)}"
                )

            destination_path = self.config.documents_dir / source_path.name
            if source_path.resolve() != destination_path.resolve():
                shutil.copy2(source_path, destination_path)
            saved_files.append(destination_path.name)

        trigger = "web_upload" if saved_files else "web_rebuild"
        build_knowledge_base(
            self.config,
            build_context={"trigger": trigger, "uploaded_files": saved_files},
        )
        self.retriever = Retriever.from_config(self.config)
        summary = self.knowledge_base_summary()
        if saved_files:
            summary["status_message"] = (
                f"已上传 {len(saved_files)} 个文件并重建知识库: {', '.join(saved_files)}"
            )
        else:
            summary["status_message"] = "已使用当前文档重建知识库。"
        return summary

    def _handle_query(
        self,
        message: str,
        history: Sequence[tuple[str, str]],
        intent: IntentResult,
    ) -> AssistantResponse:
        retrieval_query = self._build_retrieval_query(message, history, intent)
        items = self.retriever.retrieve(retrieval_query, include_all=True)
        display_items = items[: self.config.retrieval_top_k]
        prompt = build_qa_prompt(message, items, history)
        answer = self.llm_client.generate(
            prompt,
            task="query",
            question=message,
            retrieval_items=items,
        )
        return AssistantResponse(
            intent="query",
            answer=answer,
            sources=self._unique_sources(display_items),
            source_items=self._source_items(display_items),
        )

    def _handle_guidance(
        self,
        message: str,
        history: Sequence[tuple[str, str]],
        intent: IntentResult,
    ) -> AssistantResponse:
        retrieval_query = self._build_retrieval_query(message, history, intent)
        items = self.retriever.retrieve(retrieval_query, include_all=True)
        display_items = items[: self.config.retrieval_top_k]
        prompt = build_guidance_prompt(message, items, history)
        answer = self.llm_client.generate(
            prompt,
            task="guidance",
            question=message,
            retrieval_items=items,
        )
        return AssistantResponse(
            intent="guidance",
            answer=answer,
            sources=self._unique_sources(display_items),
            source_items=self._source_items(display_items),
        )

    def _stream_grounded_answer(
        self,
        message: str,
        history: Sequence[tuple[str, str]],
        intent: IntentResult,
        *,
        task: str,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        yield self._status_event("retrieval", "检索资料中")
        retrieval_query = self._build_retrieval_query(message, history, intent)
        items = self.retriever.retrieve(retrieval_query, include_all=True)
        display_items = items[: self.config.retrieval_top_k]
        prompt = (
            build_guidance_prompt(message, items, history)
            if task == "guidance"
            else build_qa_prompt(message, items, history)
        )
        response = AssistantResponse(
            intent=task,
            answer="",
            sources=self._unique_sources(display_items),
            source_items=self._source_items(display_items),
        )
        yield "meta", self._meta_payload(response)
        yield self._status_event("thinking", "整理答案中")

        chunks = self.llm_client.stream_generate(
            prompt,
            task=task,
            question=message,
            retrieval_items=items,
        )
        yield from self._emit_done_with_chunks(response, chunks)

    def _handle_operation(
        self,
        message: str,
        history: Sequence[tuple[str, str]],
    ) -> AssistantResponse:
        command, preview = self._build_operation_preview(message)
        if command.missing_fields:
            prompt = build_command_clarification_prompt(preview, history)
            answer = self.llm_client.generate(
                prompt,
                task="command_clarification",
                question=message,
                command=preview,
            )
            return AssistantResponse(
                intent="operation",
                answer=answer,
                command_preview=preview,
            )

        read_only_response = self._handle_read_only_operation(preview)
        if read_only_response is not None:
            return read_only_response

        if self._should_auto_execute(preview):
            result = self.dac3d_client.submit_scan_command(preview)
            return AssistantResponse(
                intent="operation",
                answer=self._command_result_answer(preview, result),
                command_preview=preview,
                status_summary=result.get("status"),
                parsed_result=result.get("result"),
            )

        if self._should_submit(message):
            busy_message = self._busy_runtime_message(preview)
            if busy_message:
                return AssistantResponse(
                    intent="operation",
                    answer=busy_message,
                    command_preview=preview,
                    status_summary=preview.get("runtime_status"),
                )
            result = self.dac3d_client.submit_scan_command(preview)
            warning_text = ""
            if command.warnings:
                warning_text = f" 风险提示: {'；'.join(command.warnings)}"
            submit_message = result.get("message") or "结构化 DAC-3D 命令已通过校验，并提交到运行时。"
            return AssistantResponse(
                intent="operation",
                answer=f"{submit_message}{warning_text}".strip(),
                command_preview=preview,
                status_summary=result["status"],
            )

        answer = self._operation_preview_answer(preview)
        if command.warnings:
            answer = f"{answer} 风险提示: {'；'.join(command.warnings)}"
        return AssistantResponse(
            intent="operation",
            answer=answer,
            command_preview=preview,
        )

    def _stream_operation(
        self,
        message: str,
        history: Sequence[tuple[str, str]],
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        yield self._status_event("command", "解析指令中")
        command, preview = self._build_operation_preview(message)
        if command.missing_fields:
            prompt = build_command_clarification_prompt(preview, history)
            response = AssistantResponse(
                intent="operation",
                answer="",
                command_preview=preview,
            )
            yield "meta", self._meta_payload(response)
            yield self._status_event("thinking", "补全指令中")
            chunks = self.llm_client.stream_generate(
                prompt,
                task="command_clarification",
                question=message,
                command=preview,
            )
            yield from self._emit_done_with_chunks(response, chunks)
            return

        read_only_response = self._handle_read_only_operation(preview)
        if read_only_response is not None:
            yield from self._emit_buffered_response(read_only_response)
            return

        if self._should_auto_execute(preview):
            result = self.dac3d_client.submit_scan_command(preview)
            response = AssistantResponse(
                intent="operation",
                answer=self._command_result_answer(preview, result),
                command_preview=preview,
                status_summary=result.get("status"),
                parsed_result=result.get("result"),
            )
            yield from self._emit_buffered_response(response)
            return

        if self._should_submit(message):
            busy_message = self._busy_runtime_message(preview)
            if busy_message:
                response = AssistantResponse(
                    intent="operation",
                    answer=busy_message,
                    command_preview=preview,
                    status_summary=preview.get("runtime_status"),
                )
                yield from self._emit_buffered_response(response)
                return
            result = self.dac3d_client.submit_scan_command(preview)
            warning_text = ""
            if command.warnings:
                warning_text = f" 风险提示: {'；'.join(command.warnings)}"
            submit_message = result.get("message") or "结构化 DAC-3D 命令已通过校验，并提交到运行时。"
            response = AssistantResponse(
                intent="operation",
                answer=f"{submit_message}{warning_text}".strip(),
                command_preview=preview,
                status_summary=result["status"],
            )
            yield from self._emit_buffered_response(response)
            return

        answer = self._operation_preview_answer(preview)
        if command.warnings:
            answer = f"{answer} 风险提示: {'；'.join(command.warnings)}"
        response = AssistantResponse(
            intent="operation",
            answer=answer,
            command_preview=preview,
        )
        yield from self._emit_buffered_response(response)

    def preview_operation_command(
        self,
        message: str,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> AssistantResponse:
        """Return a structured DAC-3D command preview without submitting it."""
        history_tail = self._trim_history(history)
        command, preview = self._build_operation_preview(message)
        if command.missing_fields:
            prompt = build_command_clarification_prompt(preview, history_tail)
            answer = self.llm_client.generate(
                prompt,
                task="command_clarification",
                question=message,
                command=preview,
            )
            return AssistantResponse(
                intent="operation",
                answer=answer,
                command_preview=preview,
                status_summary=preview.get("runtime_status"),
            )

        answer = self._operation_preview_answer(preview)
        if command.warnings:
            answer = f"{answer} 风险提示: {'；'.join(command.warnings)}"
        return AssistantResponse(
            intent="operation",
            answer=answer,
            command_preview=preview,
            status_summary=preview.get("runtime_status"),
        )

    def execute_operation_command(
        self,
        message: str,
        *,
        confirmed_by_user: bool = False,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> AssistantResponse:
        """Generate and, when allowed, submit a DAC-3D command to the runtime."""
        history_tail = self._trim_history(history)
        command, preview = self._build_operation_preview(message)
        if command.missing_fields:
            prompt = build_command_clarification_prompt(preview, history_tail)
            answer = self.llm_client.generate(
                prompt,
                task="command_clarification",
                question=message,
                command=preview,
            )
            return AssistantResponse(
                intent="operation",
                answer=answer,
                command_preview=preview,
                status_summary=preview.get("runtime_status"),
            )

        return self._execute_operation_preview(
            preview,
            command.warnings,
            confirmed_by_user=confirmed_by_user,
        )

    def _build_operation_preview(self, message: str) -> tuple[Any, dict[str, Any]]:
        """Build a structured command and attach current runtime context."""
        command = self.command_generator.generate(message)
        preview = command.to_dict()
        self._attach_runtime_context_to_command(preview, command.warnings)
        return command, preview

    def _execute_operation_preview(
        self,
        preview: dict[str, Any],
        command_warnings: Sequence[str],
        *,
        confirmed_by_user: bool,
    ) -> AssistantResponse:
        """Submit a complete command preview, enforcing runtime and confirmation checks."""
        read_only_response = self._handle_read_only_operation(preview)
        if read_only_response is not None:
            return read_only_response

        safety = dict(preview.get("safety") or {})
        if safety.get("needs_confirmation") and not confirmed_by_user:
            warnings = f" 风险提示: {'；'.join(command_warnings)}" if command_warnings else ""
            return AssistantResponse(
                intent="operation",
                answer=(
                    "已生成 DAC-3D 控制命令，但该命令需要用户明确确认后才会下发。"
                    "如果确认执行，请明确说明“确认执行”或“立即开始”。"
                    f"{warnings}"
                ).strip(),
                command_preview=preview,
                status_summary=preview.get("runtime_status"),
            )

        busy_message = self._busy_runtime_message(preview)
        if busy_message:
            return AssistantResponse(
                intent="operation",
                answer=busy_message,
                command_preview=preview,
                status_summary=preview.get("runtime_status"),
            )

        validation_response = self._validate_offline_command_before_submit(preview)
        if validation_response is not None:
            return validation_response

        result = self.dac3d_client.submit_scan_command(preview)
        warning_text = ""
        if command_warnings:
            warning_text = f" 风险提示: {'；'.join(command_warnings)}"
        if str(preview.get("action") or "") in {"validate_offline_folder"}:
            answer = self._command_result_answer(preview, result)
        else:
            submit_message = result.get("message") or "结构化 DAC-3D 命令已通过校验，并提交到运行时。"
            answer = f"{submit_message}{warning_text}".strip()
        return AssistantResponse(
            intent="operation",
            answer=answer,
            command_preview=preview,
            status_summary=result.get("status"),
            parsed_result=result.get("result") or result.get("validation"),
        )

    def _validate_offline_command_before_submit(
        self,
        preview: dict[str, Any],
    ) -> AssistantResponse | None:
        """Block offline detection startup when the selected image folder is incomplete."""
        if str(preview.get("action") or "") != "start_offline_detection":
            return None

        payload = dict(preview.get("payload") or {})
        if not payload.get("validate_before_run", True):
            return None

        validate_command = deepcopy(preview)
        validate_command["action"] = "validate_offline_folder"
        validate_command["safety"] = {
            "needs_confirmation": False,
            "hardware_required": False,
            "safe_to_auto_execute": True,
        }
        if self.dac3d_client.runtime_bridge is not None:
            validation_result = self.dac3d_client.submit_scan_command(validate_command)
            validation = dict(validation_result.get("validation") or {})
            status = validation_result.get("status") or preview.get("runtime_status")
        else:
            validation = self.dac3d_client.validate_offline_folder(validate_command)
            status = preview.get("runtime_status")

        if validation.get("ready"):
            return None

        missing = validation.get("missing_requirements") or []
        return AssistantResponse(
            intent="operation",
            answer=(
                "离线图片目录校验未通过，未启动 DAC-3D 离线检测。"
                f"缺少或异常项：{', '.join(map(str, missing)) or 'unknown'}。"
            ),
            command_preview=preview,
            status_summary=status,
            parsed_result=validation,
        )

    def _handle_read_only_operation(self, preview: dict[str, Any]) -> AssistantResponse | None:
        action = str(preview.get("action") or "")
        if action == "get_latest_result":
            try:
                if self.dac3d_client.runtime_bridge is not None:
                    bridge_result = self.dac3d_client.submit_scan_command(preview)
                    result_summary = dict(bridge_result.get("result") or {})
                    status = dict(bridge_result.get("status") or preview.get("runtime_status") or {})
                else:
                    result_summary = self.dac3d_client.get_latest_result_summary()
                    status = self.dac3d_client.query_current_status()
            except DAC3DUnavailableError as exc:
                return AssistantResponse(
                    intent="operation",
                    answer=str(exc),
                    command_preview=preview,
                    status_summary=preview.get("runtime_status"),
                )
            result = {
                "accepted": True,
                "mode": "local_status_file",
                "command": preview,
                "status": status,
                "result": result_summary,
            }
            return AssistantResponse(
                intent="operation",
                answer=self._command_result_answer(preview, result),
                command_preview=preview,
                status_summary=status,
                parsed_result=result_summary,
            )
        if action == "query_status":
            status = self.dac3d_client.query_current_status()
            result = {"accepted": True, "status": status, "command": preview}
            return AssistantResponse(
                intent="operation",
                answer=self._command_result_answer(preview, result),
                command_preview=preview,
                status_summary=status,
            )
        return None

    def _attach_runtime_context_to_command(
        self,
        preview: dict[str, Any],
        command_warnings: list[str],
    ) -> None:
        """Attach live DAC-3D runtime state to every operation command preview."""
        try:
            status = self.dac3d_client.query_current_status()
        except Exception as exc:  # noqa: BLE001 - keep command preview available.
            status = {
                "state": "unknown",
                "message": f"Unable to read DAC-3D runtime status: {exc}",
            }

        state = str(status.get("state") or "unknown")
        preview["runtime_status"] = status

        busy_states = {"running", "detecting", "scanning", "initializing"}
        if state.lower() in busy_states:
            warning = (
                f"DAC-3D 当前状态为 {state}，已有任务可能正在执行；"
                "执行新命令前请先确认当前任务是否可中断或已完成。"
            )
            if warning not in command_warnings:
                command_warnings.append(warning)
            preview.setdefault("warnings", [])
            if warning not in preview["warnings"]:
                preview["warnings"].append(warning)

        yaml_preview = preview.get("yaml_preview")
        if isinstance(yaml_preview, str):
            preview["yaml_preview"] = (
                f"{yaml_preview}\n"
                "runtime_status:\n"
                f"  state: {state}\n"
                f"  progress: {status.get('progress', 'null')}\n"
                f"  step: {status.get('step', 'null')}\n"
                f"  source: {status.get('source', 'null')}"
            )

    def _handle_interpretation(
        self,
        message: str,
        history: Sequence[tuple[str, str]],
        intent: IntentResult,
    ) -> AssistantResponse:
        parsed_result = parse_result_from_text(message)
        if parsed_result is None:
            raw_result = self.dac3d_client.get_recent_inspection_result()
            parsed_result = parse_result(raw_result)
        retrieval_query = f"{parsed_result.defect_type} 严重 阈值 severity threshold {parsed_result.rule_reason}"
        items = self.retriever.retrieve(
            retrieval_query,
            document_type="defect",
            include_all=True,
        ) or self.retriever.retrieve(retrieval_query, include_all=True)
        display_items = items[: self.config.retrieval_top_k]
        prompt = build_interpretation_prompt(message, items, parsed_result.to_dict(), history)
        answer = self.llm_client.generate(
            prompt,
            task="interpretation",
            question=message,
            retrieval_items=items,
            parsed_result=parsed_result,
        )
        return AssistantResponse(
            intent="interpretation",
            answer=answer,
            sources=self._unique_sources(display_items),
            source_items=self._source_items(display_items),
            parsed_result=parsed_result.to_dict(),
        )

    def _stream_interpretation(
        self,
        message: str,
        history: Sequence[tuple[str, str]],
        intent: IntentResult,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        del intent
        yield self._status_event("parsing", "解析检测结果中")
        parsed_result = parse_result_from_text(message)
        if parsed_result is None:
            raw_result = self.dac3d_client.get_recent_inspection_result()
            parsed_result = parse_result(raw_result)
        yield self._status_event("retrieval", "检索判定依据中")
        retrieval_query = f"{parsed_result.defect_type} 严重 阈值 severity threshold {parsed_result.rule_reason}"
        items = self.retriever.retrieve(
            retrieval_query,
            document_type="defect",
            include_all=True,
        ) or self.retriever.retrieve(retrieval_query, include_all=True)
        display_items = items[: self.config.retrieval_top_k]
        prompt = build_interpretation_prompt(message, items, parsed_result.to_dict(), history)
        response = AssistantResponse(
            intent="interpretation",
            answer="",
            sources=self._unique_sources(display_items),
            source_items=self._source_items(display_items),
            parsed_result=parsed_result.to_dict(),
        )
        yield "meta", self._meta_payload(response)
        yield self._status_event("thinking", "生成解读中")
        chunks = self.llm_client.stream_generate(
            prompt,
            task="interpretation",
            question=message,
            retrieval_items=items,
            parsed_result=parsed_result,
        )
        yield from self._emit_done_with_chunks(response, chunks)

    def _handle_status_query(self, intent: IntentResult) -> AssistantResponse:
        del intent
        status = self.dac3d_client.query_current_status()
        runtime = self.dac3d_client.runtime_snapshot()
        state = status.get("state", "unknown")
        progress = status.get("progress", "unknown")
        message = status.get("message", "无状态消息")
        step = (
            status.get("step")
            or status.get("stage")
            or status.get("current_step")
            or status.get("phase")
        )
        last_action = runtime.get("last_command_action") or "无"
        mode = runtime.get("mode", "unknown")
        bridge_snapshot = runtime.get("runtime_bridge")

        lines = [
            f"当前 DAC-3D 状态为 {state}，进度 {progress}%，最新消息: {message}",
            f"运行模式: {mode}；最近一次结构化命令: {last_action}。",
        ]
        if step:
            lines.append(f"当前步骤: {step}。")
        if isinstance(bridge_snapshot, dict) and bridge_snapshot:
            bridge_state = bridge_snapshot.get("state") or bridge_snapshot.get("status")
            bridge_step = (
                bridge_snapshot.get("step")
                or bridge_snapshot.get("stage")
                or bridge_snapshot.get("current_step")
                or bridge_snapshot.get("phase")
            )
            bridge_message = bridge_snapshot.get("message") or bridge_snapshot.get("log")
            if bridge_state or bridge_step or bridge_message:
                details = []
                if bridge_state:
                    details.append(f"状态={bridge_state}")
                if bridge_step:
                    details.append(f"步骤={bridge_step}")
                if bridge_message:
                    details.append(f"消息={bridge_message}")
                lines.append("主系统实时桥接信息: " + "；".join(details) + "。")
        elif mode == "mock":
            lines.append("当前为本地 mock 模式；接入 DAC-3D 主系统 runtime bridge 后，此处会读取主系统实时状态、步骤和最新消息。")

        answer = "\n".join(lines)
        return AssistantResponse(
            intent="status",
            answer=answer,
            status_summary=status,
        )

    def _trim_history(
        self,
        history: Sequence[tuple[str, str]] | None,
    ) -> list[tuple[str, str]]:
        if not history:
            return []
        return list(history[-self.config.history_window :])

    def _build_retrieval_query(
        self,
        message: str,
        history: Sequence[tuple[str, str]],
        intent: IntentResult | None = None,
    ) -> str:
        retrieval_query = message
        lowered = message.lower()
        deictic_markers = ("this", "that", "it", "这个", "这个参数", "它", "上述", "刚才")
        if history and any(marker in lowered for marker in deictic_markers):
            recent_user_context = " ".join(user for user, _ in history[-2:])
            retrieval_query = f"{recent_user_context} {retrieval_query}".strip()

        expansions: list[str] = []
        if intent is not None:
            if intent.label == "guidance":
                expansions.append("guidance troubleshooting 操作建议 风险控制")
            if intent.label == "interpretation":
                expansions.append("defect severity threshold measurement reporting guidance")

        for marker, expansion in self._retrieval_expansions.items():
            if marker in lowered:
                expansions.append(expansion)

        if "还要看什么" in lowered and "状态" in lowered:
            expansions.append("inspection status 状态消息 进度百分比")
        if "命名区域" in lowered or "整片扫描" in lowered:
            expansions.append("region selection current selection ROI scan area")

        if expansions:
            retrieval_query = " ".join([retrieval_query, *expansions]).strip()
        return retrieval_query

    def _should_submit(self, message: str) -> bool:
        lowered = message.lower()
        submit_keywords = (
            "execute",
            "submit",
            "start now",
            "run now",
            "stop",
            "abort",
            "cancel",
            "开始扫描",
            "执行扫描",
            "离线测试",
            "离线检测",
            "停止",
            "停止检测",
            "停止离线检测",
            "中止",
            "取消",
            "执行",
            "提交",
            "立即开始",
            "马上扫描",
        )
        return any(keyword in lowered for keyword in submit_keywords)

    def _operation_preview_answer(self, command: dict[str, Any]) -> str:
        action = str(command.get("action") or "")
        if action != "scan":
            return "已生成 DAC-3D 结构化命令预览，请确认后再执行。"

        area = dict(command.get("scan_area_mm") or {})
        resolution = dict(command.get("resolution") or {})
        width = area.get("width")
        height = area.get("height")
        step_um = resolution.get("value")
        mode = command.get("mode") or "standard"
        scan_path = "蛇形扫描"

        lines = ["好的，我为您生成了扫描配置："]
        if width and height:
            lines.append(f"- 扫描范围：{width:g}mm × {height:g}mm")
        else:
            lines.append("- 扫描范围：待补充")

        if step_um:
            lines.append(f"- 步长：{float(step_um):g}μm")
        else:
            lines.append("- 步长：待确认")

        lines.append(f"- 扫描模式：{scan_path}（推荐，效率更高；当前模式 {mode}）")

        if width and height and step_um:
            points = int((float(width) * 1000.0 / float(step_um)) * (float(height) * 1000.0 / float(step_um)))
            estimated_minutes = max(1, round(points * 0.000336 / 60))
            lines.append(f"- 预计采集点数：{points:,}")
            lines.append(f"- 预计扫描时间：约{estimated_minutes}分钟")
        else:
            lines.append("- 预计采集点数：需要补充步长后计算")
            lines.append("- 预计扫描时间：需要补充步长后计算")

        lines.append("")
        lines.append("是否需要我生成配置文件并开始扫描？如需执行，请回复“执行扫描”。")
        return "\n".join(lines)

    def _busy_runtime_message(self, command: dict[str, Any]) -> str | None:
        action = str(command.get("action") or "")
        if action not in {"scan", "start_online_scan", "start_offline_detection"}:
            return None
        status = dict(command.get("runtime_status") or {})
        state = str(status.get("state") or "").lower()
        if state not in {"running", "detecting", "scanning", "initializing"}:
            return None
        progress = status.get("progress", "unknown")
        message = status.get("message") or "当前已有检测任务正在执行。"
        return (
            f"当前 DAC-3D 正在运行，进度 {progress}%，最新状态: {message} "
            "为避免重复启动检测，本次命令未下发。请先等待完成或发送“停止检测”。"
        )

    def _should_auto_execute(self, command: dict[str, Any]) -> bool:
        safety = dict(command.get("safety") or {})
        return bool(safety.get("safe_to_auto_execute")) and not command.get("missing_fields")

    def _command_result_answer(self, command: dict[str, Any], result: dict[str, Any]) -> str:
        action = str(command.get("action", "command"))
        if action == "validate_offline_folder":
            validation = dict(result.get("validation") or {})
            if validation.get("ready"):
                return "离线图片目录校验通过：三相机图像和 surface1/surface2 要求已满足。"
            missing = validation.get("missing_requirements") or []
            return f"离线图片目录校验未通过，缺少或异常项：{', '.join(map(str, missing)) or 'unknown'}。"
        if action == "get_latest_result":
            result_summary = dict(result.get("result") or {})
            if not result_summary:
                return "当前没有读取到 DAC-3D 最近检测结果。请确认主系统已经完成至少一个样品检测。"

            history = [dict(item) for item in (result_summary.get("result_history") or []) if isinstance(item, dict)]
            requested_position = (command.get("payload") or {}).get("sample_position")
            if requested_position is not None:
                try:
                    requested_position = int(requested_position)
                except (TypeError, ValueError):
                    requested_position = None
            if requested_position is not None:
                matched = None
                for item in history:
                    try:
                        item_position = int(item.get("position") or -1)
                    except (TypeError, ValueError):
                        continue
                    if item_position == requested_position:
                        matched = item
                        break
                if matched is None:
                    checked_count = len(history)
                    return (
                        f"当前真实状态文件中还没有第 {requested_position} 个样品的检测结果。"
                        f"目前已记录 {checked_count} 个样品。"
                    )
                return self._single_sample_result_answer(matched)

            quality_label = result_summary.get("quality_label")
            if quality_label is None and "quality" in result_summary:
                quality_label = "合格" if result_summary.get("quality") else "不合格"
            defects = list(result_summary.get("defects") or [])
            files = [str(file) for file in (result_summary.get("files") or []) if file]
            parsed = dict(result_summary.get("parsed_result") or {})
            reason = parsed.get("rule_reason") or result_summary.get("message") or ""
            position = result_summary.get("position")
            tray_id = result_summary.get("tray_id")

            if not result_summary.get("files") and not result_summary.get("parsed_result") and result_summary.get("message"):
                return str(result_summary["message"])

            if history:
                total = len(history)
                failed = [item for item in history if item.get("quality") is False or item.get("quality_label") == "不合格"]
                passed = total - len(failed)
                defect_total = sum(int(item.get("defects_num") or 0) for item in history)
                defect_counter: Counter[str] = Counter()
                for item in history:
                    for defect in item.get("defects") or []:
                        if isinstance(defect, dict):
                            defect_counter[str(defect.get("defect_type") or "unknown")] += 1

                lines = [
                    f"已读取 DAC-3D 已检测样品结果：当前已完成 {total} 个样品，合格 {passed} 个，不合格 {len(failed)} 个，累计缺陷 {defect_total} 个。"
                ]
                if defect_counter:
                    top_defects = "；".join(
                        f"{name} {count} 个" for name, count in defect_counter.most_common(4)
                    )
                    lines.append(f"主要缺陷分布: {top_defects}。")
                recent_items = history[-5:]
                lines.append("最近样品情况:")
                for item in recent_items:
                    item_quality = item.get("quality_label")
                    if item_quality is None and "quality" in item:
                        item_quality = "合格" if item.get("quality") else "不合格"
                    item_defects = int(item.get("defects_num") or 0)
                    item_pos = item.get("position", "未知")
                    item_reason = ""
                    parsed_item = item.get("parsed_result")
                    if isinstance(parsed_item, dict):
                        item_reason = str(parsed_item.get("rule_reason") or "")
                    if not item_reason and item.get("defects"):
                        first_defect = (item.get("defects") or [{}])[0]
                        if isinstance(first_defect, dict):
                            item_reason = str(first_defect.get("reason") or "")
                    detail = f"- 位置 {item_pos}: {item_quality or '未知'}，缺陷 {item_defects} 个"
                    if item_reason:
                        detail += f"，依据: {item_reason}"
                    lines.append(detail + "。")
                if failed:
                    failed_positions = ", ".join(str(item.get("position", "未知")) for item in failed[:8])
                    lines.append(f"分析结论: 当前异常集中在位置 {failed_positions}；建议优先打开这些位置的结果图核查缺陷框、区域和阈值原因。")
                else:
                    lines.append("分析结论: 当前已检测样品均为合格，建议继续观察后续样品是否出现缺陷聚集。")
                return "\n".join(lines)

            lines = ["已读取最近一次 DAC-3D 检测结果。"]
            if tray_id is not None or position is not None:
                lines.append(f"样品信息: 托盘 {tray_id if tray_id is not None else '未知'}，位置 {position if position is not None else '未知'}。")
            if quality_label:
                lines.append(f"判定结果: {quality_label}。")
            lines.append(f"缺陷数量: {result_summary.get('defects_num', len(defects))}。")
            if defects:
                preview = []
                for defect in defects[:3]:
                    preview.append(
                        f"{defect.get('defect_type', 'unknown')}@{defect.get('position', 'unknown')}"
                    )
                lines.append(f"主要缺陷: {'；'.join(preview)}。")
            if reason:
                lines.append(f"判定依据: {reason}")
            if files:
                lines.append("结果图: " + "；".join(files[:4]))
            return "\n".join(lines)
        if action == "query_status":
            status = dict(result.get("status") or {})
            return (
                f"当前 DAC-3D 状态为 {status.get('state')}，进度 {status.get('progress')}%，"
                f"最新消息：{status.get('message')}"
            )
        return f"已处理 DAC-3D 结构化命令：{action}。"

    def _single_sample_result_answer(self, sample: dict[str, Any]) -> str:
        position = sample.get("position", "未知")
        tray_id = sample.get("tray_id", "未知")
        quality_label = sample.get("quality_label")
        if quality_label is None and "quality" in sample:
            quality_label = "合格" if sample.get("quality") else "不合格"
        defects = [dict(item) for item in (sample.get("defects") or []) if isinstance(item, dict)]
        parsed = dict(sample.get("parsed_result") or {})
        reason = parsed.get("rule_reason") or ""
        files = [str(file) for file in (sample.get("files") or []) if file]

        lines = [
            f"第 {position} 个样品检测结果如下。",
            f"托盘: {tray_id}；判定: {quality_label or '未知'}；缺陷数量: {sample.get('defects_num', len(defects))}。",
        ]
        if defects:
            lines.append("缺陷明细:")
            for defect in defects[:8]:
                defect_type = defect.get("defect_type", "unknown")
                defect_position = defect.get("position", "unknown")
                defect_size = defect.get("size")
                defect_reason = defect.get("reason") or ""
                detail = f"- {defect_type}，位置 {defect_position}"
                if defect_size is not None:
                    detail += f"，尺寸 {defect_size}"
                if defect_reason:
                    detail += f"，原因: {defect_reason}"
                lines.append(detail + "。")
            if len(defects) > 8:
                lines.append(f"- 其余 {len(defects) - 8} 个缺陷已省略，可继续追问该样品的完整缺陷明细。")
        else:
            lines.append("该样品未记录不合格缺陷。")
        if reason:
            lines.append(f"主要判定依据: {reason}")
        if files:
            lines.append("结果图: " + "；".join(files[:4]))
        if sample.get("quality") is False:
            lines.append("分析建议: 优先打开结果图复核缺陷框位置和阈值原因，必要时复扫该点位确认是否为真实缺陷。")
        else:
            lines.append("分析建议: 当前样品结果正常，可继续观察后续点位是否出现连续异常。")
        return "\n".join(lines)

    def _unique_sources(self, items: Sequence[RetrievalItem]) -> list[str]:
        sources: list[str] = []
        for item in items:
            source = f"{item.source} ({item.section})"
            if source not in sources:
                sources.append(source)
        return sources

    def _source_items(self, items: Sequence[RetrievalItem]) -> list[dict[str, Any]]:
        source_items: list[dict[str, Any]] = []
        for item in items:
            source_items.append(
                {
                    "source": item.source,
                    "title": item.title,
                    "section": item.section,
                    "document_type": item.document_type,
                    "score": round(item.score, 4),
                    "chunk_id": item.chunk_id,
                }
            )
        return source_items

    def _resolve_uploaded_path(self, file_ref: Any) -> Path:
        if isinstance(file_ref, Path):
            return file_ref
        if isinstance(file_ref, str):
            return Path(file_ref)
        file_name = getattr(file_ref, "name", None)
        if isinstance(file_name, str):
            return Path(file_name)
        raise ValueError("Unsupported uploaded file reference.")

    def _emit_buffered_response(
        self,
        response: AssistantResponse,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        yield "meta", self._meta_payload(response)
        yield self._status_event("generating", "生成回复中")
        for chunk in self._buffer_text(response.answer):
            yield "delta", {"chunk": chunk}
        yield "done", response.to_ui_payload()

    def _emit_done_with_chunks(
        self,
        response: AssistantResponse,
        chunks: Iterator[str],
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        answer_parts: list[str] = []
        emitted_generating = False
        for chunk in chunks:
            if not chunk:
                continue
            if not emitted_generating:
                yield self._status_event("generating", "生成回复中")
                emitted_generating = True
            answer_parts.append(chunk)
            yield "delta", {"chunk": chunk}
        response.answer = "".join(answer_parts).strip()
        yield "done", response.to_ui_payload()

    def _meta_payload(self, response: AssistantResponse) -> dict[str, Any]:
        payload = response.to_ui_payload()
        payload["answer"] = ""
        return payload

    def _status_event(self, stage: str, label: str) -> tuple[str, dict[str, Any]]:
        return "status", {"stage": stage, "label": label}

    def _buffer_text(self, text: str, *, chunk_size: int = 12) -> Iterator[str]:
        if not text:
            return
        for index in range(0, len(text), chunk_size):
            yield text[index : index + chunk_size]

    def _error_response(self, exc: Exception) -> AssistantResponse:
        if isinstance(exc, FileNotFoundError):
            return AssistantResponse(intent="error", answer=f"知识库不可用: {exc}")
        if isinstance(exc, LLMConfigurationError):
            return AssistantResponse(intent="error", answer=f"模型配置错误: {exc}")
        if isinstance(exc, LLMProviderError):
            return AssistantResponse(intent="error", answer=f"模型调用失败: {exc}")
        if isinstance(exc, DAC3DValidationError):
            return AssistantResponse(intent="error", answer=f"DAC-3D 命令校验失败: {exc}")
        if isinstance(exc, DAC3DUnavailableError):
            return AssistantResponse(intent="error", answer=f"DAC-3D 当前不可用: {exc}")
        return AssistantResponse(intent="error", answer=f"系统处理失败: {exc}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(description="DAC-3D IIM Assistant")
    parser.add_argument("--message", help="Run one message through the assistant and exit.")
    parser.add_argument(
        "--rebuild-kb",
        action="store_true",
        help="Rebuild the local knowledge base before starting the assistant.",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Force the terminal chat instead of the web UI.",
    )
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Launch the legacy Gradio UI instead of the FastAPI/React web app.",
    )
    parser.add_argument(
        "--web-only",
        action="store_true",
        help="Launch only the FastAPI/React web UI.",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Run through the OpenAI Agents SDK DAC-3D agent.",
    )
    parser.add_argument(
        "--agent-model",
        help="Override the OpenAI Agents SDK model name for this run.",
    )
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
        choices=["auto", "chat_completions", "responses"],
        help="Agent model API mode. Use chat_completions for most third-party OpenAI-compatible providers.",
    )
    parser.add_argument(
        "--agent-web",
        action="store_true",
        help="Use the OpenAI Agents SDK runtime as the web/Gradio chat backend.",
    )
    parser.add_argument("--host", help="Override the web server host.")
    parser.add_argument("--port", type=int, help="Override the web server port.")
    return parser


def _build_gradio_widget(assistant: DAC3DAssistant, *, host: str, port: int) -> ChatWidget:
    return ChatWidget(
        assistant.handle_message,
        runtime_summary_getter=assistant.runtime_summary,
        knowledge_base_builder=assistant.build_knowledge_base_from_uploads,
        knowledge_base_summary_getter=assistant.knowledge_base_summary,
        streaming=assistant.config.streaming,
        host=host,
        port=port,
        share=assistant.config.gradio_share,
    )


def _launch_gradio_widget(widget: ChatWidget) -> None:
    try:
        widget.launch_background()
    except RuntimeError as exc:
        print(f"Legacy web UI unavailable: {exc}")


def _launch_legacy_web_ui(assistant: DAC3DAssistant) -> None:
    try:
        import uvicorn
    except Exception as exc:
        print(f"Legacy source UI unavailable: {exc}")
        return

    legacy_app = create_api_app(assistant, frontend_dist_dir=assistant.config.legacy_frontend_dist_dir)
    uvicorn.run(
        legacy_app,
        host=assistant.config.gradio_host,
        port=assistant.config.gradio_port,
    )


def _launch_agent_cli(agent_runtime: Any) -> None:
    """Launch a small terminal loop backed by the OpenAI Agents SDK."""
    print("DAC-3D Agent CLI. Type `exit` to quit.")
    while True:
        try:
            message = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            return
        try:
            print(agent_runtime.run_sync(message))
        except Exception as exc:  # pragma: no cover - interactive safety net
            print(f"Agent run failed: {exc}")


def main() -> None:
    """Start the assistant application."""
    args = build_argument_parser().parse_args()
    try:
        assistant = DAC3DAssistant.create(rebuild_kb=args.rebuild_kb)
    except Exception as exc:  # pragma: no cover - startup safety net
        print(f"Assistant startup failed: {exc}")
        return

    if args.agent_model:
        assistant.config.agent_model_name = args.agent_model
    if args.agent_api_base_url:
        assistant.config.agent_api_base_url = args.agent_api_base_url
    if args.agent_api_key:
        assistant.config.agent_api_key = args.agent_api_key
    if args.agent_api_type:
        assistant.config.agent_api_type = args.agent_api_type

    if args.agent:
        try:
            from agent_runtime import DAC3DAgentRuntime
        except Exception as exc:  # pragma: no cover - dependency guard
            print(f"Agent runtime unavailable: {exc}")
            return
        agent_runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
        if args.message:
            try:
                print(agent_runtime.run_sync(args.message))
            except Exception as exc:
                print(f"Agent run failed: {exc}")
            return
        _launch_agent_cli(agent_runtime)
        return

    if args.message:
        print(assistant.handle_message(args.message).render_text())
        return

    chat_runtime: Any = assistant
    if args.agent_web:
        try:
            from agent_runtime import DAC3DAgentChatAdapter, DAC3DAgentRuntime
        except Exception as exc:  # pragma: no cover - dependency guard
            print(f"Agent runtime unavailable: {exc}")
            return
        chat_runtime = DAC3DAgentChatAdapter(
            DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
        )

    if args.cli:
        widget = _build_gradio_widget(
            chat_runtime,
            host=chat_runtime.config.gradio_host,
            port=chat_runtime.config.gradio_port,
        )
        widget.launch_cli()
        return

    if args.gradio:
        widget = _build_gradio_widget(
            chat_runtime,
            host=args.host or chat_runtime.config.gradio_host,
            port=args.port or chat_runtime.config.gradio_port,
        )
        try:
            widget.launch()
        except RuntimeError as exc:
            print(f"Legacy web UI unavailable: {exc}")
            print("Falling back to terminal chat. Use `python app.py --cli` to skip this message.")
            widget.launch_cli()
        return

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - optional dependency guard
        print(f"FastAPI web server unavailable: {exc}")
        print("Use `python app.py --gradio` for the legacy UI or `python app.py --cli`.")
        return

    if not args.web_only:
        legacy_ui_thread = threading.Thread(
            target=_launch_legacy_web_ui,
            args=(chat_runtime,),
            name="dac3d-legacy-source-ui",
            daemon=True,
        )
        legacy_ui_thread.start()
        print(
            f"Running dual UIs: React/FastAPI at http://{args.host or chat_runtime.config.web_host}:{args.port or chat_runtime.config.web_port} "
            f"and legacy source UI at http://{chat_runtime.config.gradio_host}:{chat_runtime.config.gradio_port}"
        )

    web_app = create_api_app(chat_runtime)
    uvicorn.run(
        web_app,
        host=args.host or chat_runtime.config.web_host,
        port=args.port or chat_runtime.config.web_port,
    )


if __name__ == "__main__":
    main()
