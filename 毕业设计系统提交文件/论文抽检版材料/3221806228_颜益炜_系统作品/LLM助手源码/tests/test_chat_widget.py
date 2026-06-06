"""Focused tests for chat widget streaming behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ui.chat_widget import ChatWidget


@dataclass
class DummyResponse:
    answer: str
    intent: str = "query"
    source_items: list[dict[str, Any]] = field(default_factory=list)
    command_preview: dict[str, Any] | None = None
    status_summary: dict[str, Any] | None = None
    parsed_result: dict[str, Any] | None = None


def test_stream_chunks_yields_full_text() -> None:
    widget = ChatWidget(lambda _message, _history: DummyResponse(answer="abc"), streaming=True)

    assert list(widget._stream_chunks("abc")) == ["a", "b", "c"]


def test_chat_turn_streams_incremental_answers() -> None:
    widget = ChatWidget(lambda _message, _history: DummyResponse(answer="你好"), streaming=True)

    updates = list(widget._chat_turn("测试", []))

    assert [update[1][-1][1] for update in updates] == ["你", "你好"]
    assert updates[-1][1][-1] == ("测试", "你好")
