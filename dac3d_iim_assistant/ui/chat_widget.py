"""Web and terminal chat widgets for the DAC-3D assistant."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from typing import Any


class ChatWidget:
    """Chat widget with Gradio-first UX and a terminal fallback."""

    def __init__(
        self,
        handler: Callable[[str, Sequence[tuple[str, str]] | None], Any],
        *,
        runtime_summary_getter: Callable[[], dict[str, Any]] | None = None,
        knowledge_base_builder: Callable[[Sequence[Any] | None], dict[str, Any]] | None = None,
        knowledge_base_summary_getter: Callable[[], dict[str, Any]] | None = None,
        streaming: bool = False,
        stream_delay_seconds: float = 0.008,
        host: str = "127.0.0.1",
        port: int = 7860,
        share: bool = False,
    ) -> None:
        self.handler = handler
        self.runtime_summary_getter = runtime_summary_getter
        self.knowledge_base_builder = knowledge_base_builder
        self.knowledge_base_summary_getter = knowledge_base_summary_getter
        self.streaming = streaming
        self.stream_delay_seconds = stream_delay_seconds
        self.host = host
        self.port = port
        self.share = share
        self.history: list[tuple[str, str]] = []

    def launch(self) -> None:
        """Launch the Gradio web app."""
        demo = self.build_demo()
        demo.launch(server_name=self.host, server_port=self.port, share=self.share)

    def launch_background(self) -> None:
        """Launch the Gradio web app without blocking the current thread."""
        demo = self.build_demo()
        demo.launch(
            server_name=self.host,
            server_port=self.port,
            share=self.share,
            prevent_thread_lock=True,
        )

    def build_demo(self) -> Any:
        """Build the Gradio Blocks app."""
        try:
            import gradio as gr
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "Gradio is not installed. Install project dependencies or use --cli."
            ) from exc

        with gr.Blocks(title="DAC-3D IIM Assistant", fill_height=True) as demo:
            gr.HTML(f"<style>{self._demo_css()}</style>")
            history_state = gr.State([])
            pending_message = gr.State("")

            with gr.Row(elem_classes=["app-shell"]):
                with gr.Column(scale=5, min_width=720, elem_classes=["chat-column"]):
                    with gr.Row(elem_classes=["topbar"]):
                        gr.HTML(self._topbar_html(), elem_classes=["topbar-copy"])
                        details_button = gr.Button("详细资料", elem_classes=["toolbar-button"])
                        settings_button = gr.Button("设置", elem_classes=["toolbar-button"])
                        clear_button = gr.Button("新对话", elem_classes=["toolbar-button", "ghost-toolbar"])

                    chatbot = gr.Chatbot(
                        show_label=False,
                        height=720,
                        layout="bubble",
                        placeholder="询问参数、扫描命令、缺陷判定或现场问题。",
                        elem_id="chat-stage",
                        elem_classes=["chat-shell"],
                    )

                    with gr.Row(elem_classes=["composer-row"]):
                        message = gr.Textbox(
                            show_label=False,
                            placeholder="给 DAC-3D IIM Assistant 发送消息",
                            lines=1,
                            max_lines=8,
                            container=False,
                            autofocus=True,
                            elem_id="chat-composer",
                            elem_classes=["composer-box"],
                        )
                        send = gr.Button("↑", elem_classes=["send-button"])

                with gr.Column(
                    scale=2,
                    min_width=360,
                    visible=False,
                    elem_classes=["side-panel"],
                ) as side_panel:
                    with gr.Row(elem_classes=["side-header"]):
                        panel_title = gr.Markdown("**详细资料**", elem_classes=["side-title"])
                        close_panel = gr.Button("收起", elem_classes=["toolbar-button", "ghost-toolbar"])

                    with gr.Group(elem_classes=["side-section"], visible=True) as details_panel:
                        sources = gr.Markdown(
                            self._empty_sources_markdown(),
                            elem_classes=["side-card"],
                        )
                        details = gr.Markdown(
                            self._empty_details_markdown(),
                            elem_classes=["side-card"],
                        )
                        with gr.Accordion("结构化数据", open=False, elem_classes=["json-shell"]):
                            command_preview = gr.JSON(label="命令预览")
                            status_summary = gr.JSON(label="状态摘要")
                            parsed_result = gr.JSON(label="结果解析")

                    with gr.Group(elem_classes=["side-section"], visible=False) as settings_panel:
                        runtime_markdown = gr.Markdown(
                            self.format_runtime_markdown(),
                            elem_classes=["side-card"],
                        )
                        with gr.Group(elem_classes=["side-card", "settings-card"]):
                            gr.Markdown(
                                "**知识库构建**\n\n知识库上传、重建和构建历史已收进设置面板，主界面只保留对话。",
                                elem_classes=["settings-copy"],
                            )
                            upload_files = gr.File(
                                label="上传知识库文档",
                                file_count="multiple",
                                type="filepath",
                            )
                            build_kb_button = gr.Button(
                                "创建 / 重建知识库",
                                variant="primary",
                                elem_classes=["settings-build-button"],
                            )
                        knowledge_base_markdown = gr.Markdown(
                            self.format_knowledge_base_markdown(),
                            elem_classes=["side-card"],
                        )

            stage_outputs = [chatbot, pending_message, message]
            response_outputs = [
                chatbot,
                history_state,
                pending_message,
                sources,
                details,
                command_preview,
                status_summary,
                parsed_result,
            ]

            submit_event = message.submit(
                self._stage_user_message,
                inputs=[message, history_state],
                outputs=stage_outputs,
                queue=False,
            )
            submit_event.then(
                self._complete_chat_turn,
                inputs=[pending_message, history_state],
                outputs=response_outputs,
            )

            send_event = send.click(
                self._stage_user_message,
                inputs=[message, history_state],
                outputs=stage_outputs,
                queue=False,
            )
            send_event.then(
                self._complete_chat_turn,
                inputs=[pending_message, history_state],
                outputs=response_outputs,
            )

            clear_button.click(
                self._clear_demo_state,
                inputs=None,
                outputs=[
                    chatbot,
                    history_state,
                    pending_message,
                    sources,
                    details,
                    command_preview,
                    status_summary,
                    parsed_result,
                    message,
                ],
                queue=False,
            )

            details_button.click(
                self._open_details_panel,
                inputs=None,
                outputs=[side_panel, panel_title, details_panel, settings_panel],
                queue=False,
            )
            settings_button.click(
                self._open_settings_panel,
                inputs=None,
                outputs=[side_panel, panel_title, details_panel, settings_panel],
                queue=False,
            )
            close_panel.click(
                self._hide_side_panel,
                inputs=None,
                outputs=[side_panel, panel_title, details_panel, settings_panel],
                queue=False,
            )

            build_kb_button.click(
                self._build_knowledge_base,
                inputs=[upload_files],
                outputs=[runtime_markdown, knowledge_base_markdown, upload_files],
            )

        demo.queue()
        return demo

    def launch_cli(self) -> None:
        """Launch the terminal-based chat loop."""
        print("DAC-3D IIM Assistant")
        print("Type '/quit' to exit or '/clear' to clear the chat history.")
        while True:
            message = input("\nYou: ").strip()
            if not message:
                continue
            if message == "/quit":
                print("Assistant: Session closed.")
                return
            if message == "/clear":
                self.history.clear()
                print("Assistant: History cleared.")
                continue

            response = self.handler(message, self.history)
            rendered = response.render_text() if hasattr(response, "render_text") else str(response)
            self.history.append((message, rendered))
            print("Assistant:")
            if self.streaming:
                self._stream_text(rendered)
            else:
                print(rendered)

    def _chat_turn(
        self,
        message: str,
        history: Sequence[tuple[str, str]] | None,
    ) -> Iterator[tuple[list[dict[str, str]], list[tuple[str, str]], str, str, Any, Any, Any, str]]:
        """Backward-compatible single-step turn handler used by tests."""
        clean_message = (message or "").strip()
        current_history = list(history or [])
        if not clean_message:
            yield (
                self._to_chatbot_messages(current_history),
                current_history,
                "**来源依据**\n\n请输入有效问题，再查看检索证据。",
                "**当前结果**\n\n请输入有效问题。",
                None,
                None,
                None,
                "",
            )
            return

        response = self.handler(clean_message, current_history)
        sources_markdown = self.format_sources_markdown(response)
        details_markdown = self.format_details_markdown(response)
        command_preview = getattr(response, "command_preview", None)
        status_summary = getattr(response, "status_summary", None)
        parsed_result = getattr(response, "parsed_result", None)

        if not self.streaming:
            updated_history = current_history + [(clean_message, response.answer)]
            yield (
                self._to_chatbot_messages(updated_history),
                updated_history,
                sources_markdown,
                details_markdown,
                command_preview,
                status_summary,
                parsed_result,
                "",
            )
            return

        partial_answer = ""
        for chunk in self._stream_chunks(response.answer):
            partial_answer += chunk
            updated_history = current_history + [(clean_message, partial_answer)]
            yield (
                self._to_chatbot_messages(updated_history),
                updated_history,
                sources_markdown,
                details_markdown,
                command_preview,
                status_summary,
                parsed_result,
                "",
            )

    def _stage_user_message(
        self,
        message: str,
        history: Sequence[tuple[str, str]] | None,
    ) -> tuple[list[dict[str, str]], str, str]:
        clean_message = (message or "").strip()
        current_history = list(history or [])
        if not clean_message:
            return self._to_chatbot_messages(current_history), "", ""

        staged_messages = self._to_chatbot_messages(current_history)
        staged_messages.append({"role": "user", "content": clean_message})
        return staged_messages, clean_message, ""

    def _complete_chat_turn(
        self,
        pending_message: str,
        history: Sequence[tuple[str, str]] | None,
    ) -> Iterator[tuple[list[dict[str, str]], list[tuple[str, str]], str, str, str, Any, Any, Any]]:
        clean_message = (pending_message or "").strip()
        current_history = list(history or [])
        if not clean_message:
            yield (
                self._to_chatbot_messages(current_history),
                current_history,
                "",
                self._empty_sources_markdown(),
                self._empty_details_markdown(),
                None,
                None,
                None,
            )
            return

        response = self.handler(clean_message, current_history)
        sources_markdown = self.format_sources_markdown(response)
        details_markdown = self.format_details_markdown(response)
        command_preview = getattr(response, "command_preview", None)
        status_summary = getattr(response, "status_summary", None)
        parsed_result = getattr(response, "parsed_result", None)

        if not self.streaming:
            updated_history = current_history + [(clean_message, response.answer)]
            self.history = updated_history
            yield (
                self._to_chatbot_messages(updated_history),
                updated_history,
                "",
                sources_markdown,
                details_markdown,
                command_preview,
                status_summary,
                parsed_result,
            )
            return

        base_messages = self._to_chatbot_messages(current_history)
        base_messages.append({"role": "user", "content": clean_message})
        partial_answer = ""
        updated_history = current_history
        for chunk in self._stream_chunks(response.answer):
            partial_answer += chunk
            updated_history = current_history + [(clean_message, partial_answer)]
            self.history = updated_history
            yield (
                base_messages + [{"role": "assistant", "content": partial_answer}],
                updated_history,
                "",
                sources_markdown,
                details_markdown,
                command_preview,
                status_summary,
                parsed_result,
            )

    def _build_knowledge_base(
        self,
        uploaded_files: Sequence[Any] | None,
    ) -> tuple[str, str, None]:
        runtime_markdown = self.format_runtime_markdown()
        if self.knowledge_base_builder is None:
            return (
                runtime_markdown,
                self.format_knowledge_base_markdown(
                    {"status_message": "当前会话未启用知识库上传与构建接口。"}
                ),
                None,
            )

        try:
            summary = self.knowledge_base_builder(uploaded_files)
            return (
                self.format_runtime_markdown(),
                self.format_knowledge_base_markdown(summary),
                None,
            )
        except Exception as exc:
            fallback_summary = (
                self.knowledge_base_summary_getter() if self.knowledge_base_summary_getter else {}
            )
            fallback_summary = dict(fallback_summary)
            fallback_summary["status_message"] = f"知识库构建失败: {exc}"
            return (
                runtime_markdown,
                self.format_knowledge_base_markdown(fallback_summary),
                None,
            )

    def _clear_state(self) -> tuple[list[dict[str, str]], list[tuple[str, str]], str, str, None, None, None, str]:
        self.history.clear()
        return (
            [],
            [],
            self._empty_sources_markdown(),
            self._empty_details_markdown(),
            None,
            None,
            None,
            "",
        )

    def _clear_demo_state(
        self,
    ) -> tuple[list[dict[str, str]], list[tuple[str, str]], str, str, str, None, None, None, str]:
        self.history.clear()
        return (
            [],
            [],
            "",
            self._empty_sources_markdown(),
            self._empty_details_markdown(),
            None,
            None,
            None,
            "",
        )

    def _open_details_panel(self) -> tuple[Any, str, Any, Any]:
        import gradio as gr

        return (
            gr.update(visible=True),
            "**详细资料**",
            gr.update(visible=True),
            gr.update(visible=False),
        )

    def _open_settings_panel(self) -> tuple[Any, str, Any, Any]:
        import gradio as gr

        return (
            gr.update(visible=True),
            "**设置**",
            gr.update(visible=False),
            gr.update(visible=True),
        )

    def _hide_side_panel(self) -> tuple[Any, str, Any, Any]:
        import gradio as gr

        return (
            gr.update(visible=False),
            "**详细资料**",
            gr.update(visible=True),
            gr.update(visible=False),
        )

    def _to_chatbot_messages(self, history: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
        """Convert internal tuple history into the message format required by Gradio Chatbot."""
        messages: list[dict[str, str]] = []
        for user_message, assistant_message in history:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_message})
        return messages

    def format_sources_markdown(self, response: Any) -> str:
        """Render retrieval sources for the detail panel."""
        source_items = list(getattr(response, "source_items", []) or [])
        if not source_items:
            return self._empty_sources_markdown()

        lines = [
            "**来源依据**",
            "",
            f"> 命中 `{len(source_items)}` 条证据，已按相关度排序。",
            "",
        ]
        for index, item in enumerate(source_items, start=1):
            source = item.get("source", "unknown")
            section = item.get("section") or "未标注章节"
            document_type = item.get("document_type") or "unknown"
            score = self._format_score(item.get("score"))
            lines.append(f"{index}. `{source}`")
            lines.append(f"章节：{section} | 类型：{document_type} | 相关度：{score}")
        return "\n".join(lines)

    def format_details_markdown(self, response: Any) -> str:
        """Render the current turn summary."""
        intent = getattr(response, "intent", "unknown")
        source_items = list(getattr(response, "source_items", []) or [])
        lines = [
            "**当前结果**",
            "",
            f"- 请求类型: `{self._intent_label(intent)}`",
            f"- 证据条数: `{len(source_items)}`",
        ]
        command_preview = getattr(response, "command_preview", None)
        status_summary = getattr(response, "status_summary", None)
        parsed_result = getattr(response, "parsed_result", None)

        if command_preview:
            lines.append(f"- 扫描区域: `{command_preview.get('scan_area_mm')}`")
            lines.append(f"- 模式: `{command_preview.get('mode')}`")
            lines.append(f"- 选区: `{command_preview.get('region')}`")
            lines.append(f"- 分辨率: `{command_preview.get('resolution')}`")
            missing_fields = command_preview.get("missing_fields") or []
            if missing_fields:
                lines.append(f"- 缺失字段: {', '.join(missing_fields)}")
            warnings = command_preview.get("warnings") or []
            if warnings:
                lines.append(f"- 风险提示: {'；'.join(warnings)}")

        if status_summary:
            lines.append(f"- 状态: `{status_summary.get('state')}`")
            lines.append(f"- 进度: `{status_summary.get('progress')}%`")
            lines.append(f"- 状态消息: {status_summary.get('message')}")

        if parsed_result:
            lines.append(f"- 缺陷类型: `{parsed_result.get('defect_type')}`")
            lines.append(f"- 严重度: `{parsed_result.get('severity')}`")
            lines.append(f"- 位置: `{parsed_result.get('location')}`")
            lines.append(f"- 置信度: `{self._format_score(parsed_result.get('confidence'))}`")
            lines.append(f"- 判定依据: {parsed_result.get('rule_reason')}")

        if not command_preview and not status_summary and not parsed_result:
            lines.append("- 当前轮以自然语言回答为主，命令、状态和判定结果会自动同步到这里。")

        return "\n".join(lines)

    def format_runtime_markdown(self, summary: dict[str, Any] | None = None) -> str:
        """Render runtime information for settings."""
        runtime_summary = summary or (self.runtime_summary_getter() if self.runtime_summary_getter else {})
        dac3d = dict(runtime_summary.get("dac3d", {}))
        dac3d_status = dict(dac3d.get("status", {}))
        lines = ["**运行时概览**", ""]
        if runtime_summary:
            lines.append(f"- Provider: `{runtime_summary.get('provider', 'unknown')}`")
            lines.append(f"- Model: `{runtime_summary.get('model', 'unknown')}`")
            lines.append(f"- 检索 Top-K: `{runtime_summary.get('retrieval_top_k', 'unknown')}`")
            lines.append(f"- 向量库: `{runtime_summary.get('vector_store', 'unknown')}`")
            lines.append(
                f"- 知识库规模: `{runtime_summary.get('document_count', 0)} 文档 / {runtime_summary.get('chunk_count', 0)} 分块`"
            )
            lines.append(f"- 最近构建: `{self._format_timestamp(runtime_summary.get('latest_build_at'))}`")
            lines.append(f"- Mock Mode: `{self._format_boolean(runtime_summary.get('mock_mode'))}`")
            lines.append(f"- DAC-3D 模式: `{dac3d.get('mode', 'unknown')}`")
            lines.append(f"- 设备状态: `{dac3d_status.get('state', 'unknown')}`")
            lines.append(f"- 状态进度: `{dac3d_status.get('progress', 0)}%`")
            lines.append(f"- Endpoint: `{dac3d.get('endpoint', 'unknown')}`")
        else:
            lines.append("- 运行时信息暂不可用。")
        return "\n".join(lines)

    def format_knowledge_base_markdown(self, summary: dict[str, Any] | None = None) -> str:
        """Render the knowledge-base upload/build summary and build history."""
        kb_summary = summary or (
            self.knowledge_base_summary_getter() if self.knowledge_base_summary_getter else {}
        )
        if not kb_summary:
            return "**知识库构建历史**\n\n暂无知识库摘要。"

        lines = ["**知识库构建历史**", ""]
        status_message = kb_summary.get("status_message")
        if status_message:
            lines.append(f"> {status_message}")
            lines.append("")

        lines.append(f"- 当前文档数: `{kb_summary.get('document_count', 0)}`")
        lines.append(f"- 当前分块数: `{kb_summary.get('chunk_count', 0)}`")
        lines.append(f"- 最近构建时间: `{self._format_timestamp(kb_summary.get('latest_build_at'))}`")
        lines.append(f"- 存储后端: `{kb_summary.get('storage_backend', 'unknown')}`")
        lines.append("")

        documents = list(kb_summary.get("documents", []) or [])
        if documents:
            lines.append("**已收录文档**")
            lines.append("")
            for name in documents[:6]:
                lines.append(f"- `{name}`")
            if len(documents) > 6:
                lines.append(f"- 其余 `{len(documents) - 6}` 个文档已省略显示")
            lines.append("")

        history_entries = list(kb_summary.get("history", []) or [])
        if history_entries:
            lines.append("**最近构建记录**")
            lines.append("")
            for entry in history_entries[:5]:
                lines.append(self._format_build_history_entry(entry))
        else:
            lines.append("- 还没有构建记录。")

        return "\n".join(lines)

    def _format_build_history_entry(self, entry: dict[str, Any]) -> str:
        uploaded_files = list(entry.get("uploaded_files", []) or [])
        uploaded_text = f" | 上传: {', '.join(uploaded_files)}" if uploaded_files else ""
        return (
            f"- `{self._format_timestamp(entry.get('generated_at'))}`"
            f" | {entry.get('trigger', 'manual_build')}"
            f" | 文档 `{entry.get('document_count', 0)}`"
            f" | 分块 `{entry.get('chunk_count', 0)}`"
            f"{uploaded_text}"
        )

    def _format_timestamp(self, raw_timestamp: Any) -> str:
        if not raw_timestamp:
            return "未构建"
        try:
            parsed = datetime.fromisoformat(str(raw_timestamp))
        except ValueError:
            return str(raw_timestamp)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone()
        return parsed.strftime("%Y-%m-%d %H:%M:%S")

    def _empty_sources_markdown(self) -> str:
        return (
            "**来源依据**\n\n当前没有证据记录。\n\n"
            "- 对话主界面只保留聊天\n"
            "- 点击“详细资料”后在这里查看来源依据"
        )

    def _empty_details_markdown(self) -> str:
        return (
            "**当前结果**\n\n等待对话结果。\n\n"
            "- 命令预览、状态摘要和缺陷判定会同步显示在这里\n"
            "- 知识库上传和重建已经移到“设置”面板"
        )

    def _intent_label(self, intent: str) -> str:
        mapping = {
            "query": "知识问答",
            "operation": "扫描操作",
            "interpretation": "结果判读",
            "guidance": "操作指导",
            "status": "设备状态",
            "error": "系统错误",
        }
        return mapping.get(intent, intent)

    def _format_score(self, value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if 0.0 <= numeric <= 1.0:
            return f"{numeric * 100:.1f}%"
        return f"{numeric:.3f}"

    def _format_boolean(self, value: Any) -> str:
        return "开启" if bool(value) else "关闭"

    def _topbar_html(self) -> str:
        return """
<section class="topbar-copy-shell">
  <div class="topbar-eyebrow">DAC-3D IIM Assistant</div>
  <h1>面向检测任务的极简对话界面</h1>
  <p>主界面只保留聊天。知识库构建放进设置，详细结果放进侧边面板。</p>
</section>
"""

    def _demo_css(self) -> str:
        return """
:root {
  --page-bg: #f5f7fb;
  --page-bg-2: #eef2f8;
  --panel: rgba(255, 255, 255, 0.88);
  --panel-strong: rgba(255, 255, 255, 0.96);
  --line: rgba(15, 23, 42, 0.08);
  --line-strong: rgba(15, 23, 42, 0.16);
  --ink: #162033;
  --muted: #64748b;
  --primary: #2563eb;
  --primary-2: #0f172a;
  --shadow: 0 24px 60px rgba(15, 23, 42, 0.10);
  --shadow-soft: 0 14px 36px rgba(15, 23, 42, 0.08);
}
body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 24%),
    linear-gradient(180deg, var(--page-bg) 0%, #fbfcfe 48%, var(--page-bg-2) 100%);
  color: var(--ink);
  font-family: "Aptos", "Segoe UI Variable", "Microsoft YaHei UI", sans-serif;
}
.block-container {
  max-width: 1680px !important;
  padding: 18px !important;
}
.app-shell {
  align-items: stretch;
  gap: 18px;
}
.chat-column {
  gap: 14px;
}
.topbar {
  align-items: center;
  gap: 10px;
  margin-bottom: 2px;
}
.topbar-copy {
  flex: 1 1 auto;
}
.topbar-copy-shell {
  padding: 6px 2px;
}
.topbar-eyebrow {
  font-size: 11px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
}
.topbar-copy-shell h1 {
  margin: 6px 0 6px;
  font-size: clamp(24px, 2.2vw, 34px);
  line-height: 1.08;
  color: var(--ink);
  font-family: "Bahnschrift SemiCondensed", "Aptos Display", "Microsoft YaHei UI", sans-serif;
}
.topbar-copy-shell p {
  margin: 0;
  color: var(--muted);
  font-size: 14px;
  line-height: 1.6;
}
.toolbar-button button {
  min-height: 42px;
  padding: 0 16px !important;
  border-radius: 999px !important;
  border: 1px solid var(--line) !important;
  background: rgba(255,255,255,0.78) !important;
  color: var(--ink) !important;
  font-weight: 700;
  box-shadow: none !important;
}
.ghost-toolbar button {
  background: rgba(255,255,255,0.62) !important;
}
.chat-shell {
  border-radius: 30px !important;
  border: 1px solid var(--line) !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,253,0.90)) !important;
  box-shadow: var(--shadow);
  overflow: hidden;
}
#chat-stage .message {
  border-radius: 24px !important;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
}
#chat-stage [data-testid="user"] .message {
  background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
  color: #ffffff !important;
  border: none !important;
}
#chat-stage [data-testid="assistant"] .message,
#chat-stage [data-testid="bot"] .message {
  background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(244,247,252,0.94)) !important;
  border: 1px solid rgba(15, 23, 42, 0.06) !important;
}
.composer-row {
  align-items: end;
  gap: 12px;
  padding: 10px 12px;
  border: 1px solid var(--line);
  border-radius: 28px;
  background: var(--panel-strong);
  box-shadow: var(--shadow-soft);
}
.composer-box {
  flex: 1 1 auto;
}
.composer-box textarea {
  min-height: 34px !important;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  resize: none !important;
  padding: 8px 10px !important;
  font-size: 15px !important;
  line-height: 1.7 !important;
}
.composer-box textarea:focus {
  box-shadow: none !important;
}
.send-button {
  width: 54px;
  min-width: 54px;
}
.send-button button {
  width: 54px;
  min-width: 54px;
  height: 54px;
  min-height: 54px;
  border-radius: 999px !important;
  border: none !important;
  background: linear-gradient(135deg, var(--primary-2), var(--primary)) !important;
  color: #ffffff !important;
  font-size: 22px !important;
  font-weight: 700 !important;
  box-shadow: 0 18px 34px rgba(37, 99, 235, 0.28);
}
.side-panel {
  border-left: 1px solid rgba(15, 23, 42, 0.06);
  padding-left: 4px;
  gap: 12px;
}
.side-header {
  align-items: center;
  gap: 10px;
}
.side-title {
  flex: 1 1 auto;
  border-radius: 22px;
  border: 1px solid var(--line);
  background: rgba(255,255,255,0.78);
  box-shadow: var(--shadow-soft);
  padding: 10px 14px;
}
.side-section {
  gap: 12px;
}
.side-card,
.json-shell,
.settings-card {
  border-radius: 24px;
  border: 1px solid var(--line);
  background: var(--panel);
  box-shadow: var(--shadow-soft);
}
.side-card {
  padding: 16px 18px;
}
.settings-card {
  padding: 14px 16px;
}
.settings-copy {
  margin-bottom: 6px;
}
.settings-build-button button {
  border-radius: 999px !important;
  min-height: 46px;
  border: none !important;
  background: linear-gradient(135deg, var(--primary), #3b82f6) !important;
  box-shadow: 0 16px 30px rgba(37, 99, 235, 0.22);
}
.json-shell {
  overflow: hidden;
}
.json-shell .label-wrap {
  padding-left: 2px;
}
.side-card h1,
.side-card h2,
.side-card h3,
.side-card p,
.side-card li {
  color: var(--ink);
}
.side-card blockquote {
  margin: 0;
  padding: 10px 12px;
  border-left: 3px solid var(--primary);
  background: rgba(37, 99, 235, 0.06);
  border-radius: 12px;
}
.side-card ul,
.side-card ol {
  padding-left: 18px;
  margin-bottom: 0;
}
.side-card code {
  background: rgba(15, 23, 42, 0.06);
  border-radius: 8px;
  padding: 1px 6px;
}
@media (max-width: 1280px) {
  .app-shell {
    flex-direction: column;
  }
  .side-panel {
    padding-left: 0;
    border-left: none;
  }
}
@media (max-width: 768px) {
  .block-container {
    padding: 12px !important;
  }
  .topbar {
    flex-wrap: wrap;
  }
  .chat-shell {
    border-radius: 24px !important;
  }
  .composer-row {
    border-radius: 22px;
    padding: 8px 10px;
  }
  .send-button,
  .send-button button {
    width: 48px;
    min-width: 48px;
    height: 48px;
    min-height: 48px;
  }
}
"""

    def _stream_text(self, text: str) -> None:
        for character in text:
            print(character, end="", flush=True)
            time.sleep(self.stream_delay_seconds)
        print()

    def _stream_chunks(self, text: str) -> Iterator[str]:
        for character in text:
            yield character
