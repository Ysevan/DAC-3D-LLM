"""Main entry point and top-level routing for the DAC-3D assistant."""

from __future__ import annotations

import argparse
import json
import shutil
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
    def create(cls, config: AppConfig | None = None, *, rebuild_kb: bool = False) -> "DAC3DAssistant":
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
            dac3d_client=DAC3DClient(
                mock_mode=active_config.mock_mode,
                endpoint=active_config.dac3d_endpoint,
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
        command = self.command_generator.generate(message)
        preview = command.to_dict()
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

        if self._should_submit(message):
            result = self.dac3d_client.submit_scan_command(preview)
            warning_text = ""
            if command.warnings:
                warning_text = f" 风险提示: {'；'.join(command.warnings)}"
            return AssistantResponse(
                intent="operation",
                answer=(
                    "结构化 DAC-3D 命令已通过校验，并提交到 mock 运行时。"
                    f"{warning_text}"
                ).strip(),
                command_preview=preview,
                status_summary=result["status"],
            )

        answer = "已生成 DAC-3D 扫描命令预览，请确认后再执行。"
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
        command = self.command_generator.generate(message)
        preview = command.to_dict()
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

        if self._should_submit(message):
            result = self.dac3d_client.submit_scan_command(preview)
            warning_text = ""
            if command.warnings:
                warning_text = f" 风险提示: {'；'.join(command.warnings)}"
            response = AssistantResponse(
                intent="operation",
                answer=(
                    "结构化 DAC-3D 命令已通过校验，并提交到 mock 运行时。"
                    f"{warning_text}"
                ).strip(),
                command_preview=preview,
                status_summary=result["status"],
            )
            yield from self._emit_buffered_response(response)
            return

        answer = "已生成 DAC-3D 扫描命令预览，请确认后再执行。"
        if command.warnings:
            answer = f"{answer} 风险提示: {'；'.join(command.warnings)}"
        response = AssistantResponse(
            intent="operation",
            answer=answer,
            command_preview=preview,
        )
        yield from self._emit_buffered_response(response)

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
        answer = (
            f"当前 DAC-3D 状态为 {status['state']}，进度 {status['progress']}%，"
            f"最新消息: {status['message']}"
        )
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
            "执行",
            "提交",
            "立即开始",
            "马上扫描",
        )
        return any(keyword in lowered for keyword in submit_keywords)

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
    parser.add_argument("--host", help="Override the web server host.")
    parser.add_argument("--port", type=int, help="Override the web server port.")
    return parser


def main() -> None:
    """Start the assistant application."""
    args = build_argument_parser().parse_args()
    try:
        assistant = DAC3DAssistant.create(rebuild_kb=args.rebuild_kb)
    except Exception as exc:  # pragma: no cover - startup safety net
        print(f"Assistant startup failed: {exc}")
        return

    if args.message:
        print(assistant.handle_message(args.message).render_text())
        return

    if args.cli:
        widget = ChatWidget(
            assistant.handle_message,
            runtime_summary_getter=assistant.runtime_summary,
            knowledge_base_builder=assistant.build_knowledge_base_from_uploads,
            knowledge_base_summary_getter=assistant.knowledge_base_summary,
            streaming=assistant.config.streaming,
            host=assistant.config.gradio_host,
            port=assistant.config.gradio_port,
            share=assistant.config.gradio_share,
        )
        widget.launch_cli()
        return

    if args.gradio:
        widget = ChatWidget(
            assistant.handle_message,
            runtime_summary_getter=assistant.runtime_summary,
            knowledge_base_builder=assistant.build_knowledge_base_from_uploads,
            knowledge_base_summary_getter=assistant.knowledge_base_summary,
            streaming=assistant.config.streaming,
            host=args.host or assistant.config.gradio_host,
            port=args.port or assistant.config.gradio_port,
            share=assistant.config.gradio_share,
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

    web_app = create_api_app(assistant)
    uvicorn.run(
        web_app,
        host=args.host or assistant.config.web_host,
        port=args.port or assistant.config.web_port,
    )


if __name__ == "__main__":
    main()
