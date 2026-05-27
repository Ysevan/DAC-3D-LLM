"""Tests for JSON-backed conversation memory."""

from __future__ import annotations

import json
from pathlib import Path

from config import AppConfig
from memory import ConversationMemoryStore


def test_conversation_memory_store_persists_json_and_searches_layers(tmp_path: Path) -> None:
    config = AppConfig(base_dir=tmp_path, vector_store_type="manifest")
    store = ConversationMemoryStore.from_config(config)
    store.ensure_directories()

    store.append_turn(
        session_id="web-session",
        user="样品表面反光很强怎么办？",
        assistant="建议先降低曝光，再做小范围校准扫描。",
        intent="guidance",
        structured_data={"recommendations": ["降低曝光", "校准扫描"]},
    )

    session_file = config.conversation_memory_dir / "sessions" / "web-session.json"
    index_file = config.conversation_memory_dir / "index.json"
    assert session_file.exists()
    assert index_file.exists()

    session_payload = json.loads(session_file.read_text(encoding="utf-8"))
    assert session_payload["turns"][0]["user"] == "样品表面反光很强怎么办？"
    assert session_payload["summary"]["turn_count"] == 1

    hits = store.search("反光曝光", session_id="web-session", limit=5)

    assert hits
    assert any(hit.layer in {"session_recent_json", "long_term_session_search"} for hit in hits)
    assert any("反光" in hit.snippet for hit in hits)


def test_conversation_memory_context_formats_multiple_layers(tmp_path: Path) -> None:
    config = AppConfig(base_dir=tmp_path, vector_store_type="manifest")
    store = ConversationMemoryStore.from_config(config)
    store.ensure_directories()
    store.append_turn(
        session_id="web-session",
        user="扫描 10mm x 10mm 区域",
        assistant="已生成扫描命令预览，执行前需要确认。",
        intent="operation_preview",
        structured_data={"command_preview": {"action": "start_online_scan"}},
    )

    context, hits = store.format_context(
        "刚才那个扫描可以执行吗？",
        session_id="web-session",
        history=[("我想看状态", "当前为空闲状态。")],
        recent_limit=2,
        search_limit=5,
    )

    assert "短期记忆" in context
    assert "会话记忆" in context
    assert "长期记忆检索" in context
    assert hits
