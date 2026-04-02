"""Tests for the LLM provider layer and UI formatting helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from config import AppConfig
from rag.llm_client import LLMClient
from rag.retriever import RetrievalItem
from ui.chat_widget import ChatWidget


def _make_retrieval_item() -> RetrievalItem:
    return RetrievalItem(
        text="分辨率决定扫描点间距，点间距越小，细节越丰富。",
        source="parameter_notes.md",
        title="DAC-3D 参数说明",
        section="分辨率",
        document_type="manual",
        chunk_id=1,
        score=0.92,
        metadata={"source": "parameter_notes.md", "section": "分辨率"},
    )


def test_mock_llm_refuses_without_grounding(tmp_path: Path) -> None:
    """The mock provider should refuse when it has no retrieval evidence."""
    config = AppConfig(base_dir=tmp_path, provider="mock", mock_mode=True)
    llm = LLMClient(config)

    answer = llm.generate(
        "unused",
        task="query",
        question="这个参数是什么意思？",
        retrieval_items=[],
    )

    assert "依据" in answer


def test_anthropic_provider_uses_fake_sdk(tmp_path: Path) -> None:
    """The Anthropic provider should work when the SDK is available."""
    class FakeTextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeStream:
        def __init__(self) -> None:
            self.text_stream = ["流", "式", "输", "出"]

        def __enter__(self) -> "FakeStream":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class FakeMessages:
        def create(self, **kwargs):
            del kwargs
            return types.SimpleNamespace(content=[FakeTextBlock("Anthropic OK")])

        def stream(self, **kwargs):
            del kwargs
            return FakeStream()

    class FakeAnthropic:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.messages = FakeMessages()

    fake_module = types.ModuleType("anthropic")
    fake_module.Anthropic = FakeAnthropic
    original_module = sys.modules.get("anthropic")
    sys.modules["anthropic"] = fake_module
    try:
        config = AppConfig(
            base_dir=tmp_path,
            provider="anthropic",
            api_key="fake-key",
            mock_mode=True,
        )
        llm = LLMClient(config)
        items = [_make_retrieval_item()]

        answer = llm.generate(
            "prompt",
            task="query",
            question="这个参数是什么意思？",
            retrieval_items=items,
        )
        streamed = "".join(
            llm.stream_generate(
                "prompt",
                task="query",
                question="这个参数是什么意思？",
                retrieval_items=items,
            )
        )
    finally:
        if original_module is None:
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = original_module

    assert answer == "Anthropic OK"
    assert streamed == "流式输出"


def test_chat_widget_formats_sources_without_gradio() -> None:
    """The UI helper should render source markdown without importing Gradio."""
    widget = ChatWidget(lambda message, history=None: None)
    response = types.SimpleNamespace(
        source_items=[
            {
                "source": "parameter_notes.md",
                "section": "分辨率",
                "document_type": "manual",
                "score": 0.93,
            }
        ]
    )

    markdown = widget.format_sources_markdown(response)

    assert "parameter_notes.md" in markdown
    assert "分辨率" in markdown


def test_chat_widget_emits_gradio_message_format() -> None:
    """The chat widget should emit message dictionaries for the Gradio Chatbot."""
    widget = ChatWidget(
        lambda message, history=None: types.SimpleNamespace(
            answer=f"reply to {message}",
            source_items=[],
            command_preview=None,
            status_summary=None,
            parsed_result=None,
            intent="query",
        )
    )

    turn = next(widget._chat_turn("你好", []))
    chatbot_messages, state_history = turn[0], turn[1]

    assert chatbot_messages == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "reply to 你好"},
    ]
    assert state_history == [("你好", "reply to 你好")]


def test_chat_widget_stages_user_message_before_answer() -> None:
    """The web UI should stage the user bubble before generating the assistant reply."""
    widget = ChatWidget(
        lambda message, history=None: types.SimpleNamespace(
            answer=f"reply to {message}",
            source_items=[],
            command_preview=None,
            status_summary=None,
            parsed_result=None,
            intent="query",
        )
    )

    staged_messages, pending_message, cleared_input = widget._stage_user_message("你好", [])

    assert staged_messages == [{"role": "user", "content": "你好"}]
    assert pending_message == "你好"
    assert cleared_input == ""

    completed = next(widget._complete_chat_turn(pending_message, []))
    chatbot_messages, state_history = completed[0], completed[1]

    assert chatbot_messages == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "reply to 你好"},
    ]
    assert state_history == [("你好", "reply to 你好")]


def test_chat_widget_formats_knowledge_base_build_history() -> None:
    """The UI helper should render the latest KB build date and upload history."""
    widget = ChatWidget(
        lambda message, history=None: None,
        knowledge_base_summary_getter=lambda: {
            "document_count": 3,
            "chunk_count": 11,
            "latest_build_at": "2026-03-19T10:20:30+00:00",
            "storage_backend": "chroma",
            "documents": ["manual.md", "faq.md"],
            "history": [
                {
                    "generated_at": "2026-03-19T10:20:30+00:00",
                    "trigger": "web_upload",
                    "document_count": 3,
                    "chunk_count": 11,
                    "uploaded_files": ["faq.md"],
                }
            ],
        },
    )

    markdown = widget.format_knowledge_base_markdown()

    assert "最近构建时间" in markdown
    assert "faq.md" in markdown
    assert "web_upload" in markdown
