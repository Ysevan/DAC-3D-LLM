"""Tests for redaction, memory secret guards, and audit hash chains."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import AppConfig
from memory import ConversationMemoryStore
from security.secrets import SecretDetectedError
from tracing.logger import AuditTraceLogger, ZERO_HASH
from tracing.redaction import REDACTION, redact_exception, redact_text, redact_value


def test_api_key_redacted() -> None:
    text = redact_text("DAC3D_LLM_API_KEY=sk-1234567890abcdef")

    assert "sk-1234567890abcdef" not in text
    assert REDACTION in text


def test_password_redacted() -> None:
    payload = redact_value({"username": "admin", "password": "123456789"})

    assert payload["username"] == "admin"
    assert payload["password"] == REDACTION


def test_token_redacted() -> None:
    payload = redact_value({"headers": {"Authorization": "Bearer abcdefghijklmnopqrstuvwxyz"}})

    assert payload["headers"]["Authorization"] == REDACTION


def test_exception_redacted() -> None:
    exc = RuntimeError("provider failed with token=super-secret-token-value")

    summary = redact_exception(exc)

    assert "super-secret-token-value" not in summary
    assert "RuntimeError" in summary


def test_memory_write_with_secret_rejected(tmp_path: Path) -> None:
    config = AppConfig(base_dir=tmp_path, vector_store_type="manifest")
    store = ConversationMemoryStore.from_config(config)
    store.ensure_directories()

    with pytest.raises(SecretDetectedError):
        store.append_turn(
            session_id="memory-secret",
            user="DAC3D_LLM_API_KEY=sk-1234567890abcdef",
            assistant="不会保存密钥。",
            intent="query",
        )

    assert not list(config.conversation_memory_dir.rglob("*.json"))


def test_trace_append_only_hash_chain_valid(tmp_path: Path) -> None:
    logger = AuditTraceLogger(tmp_path / "audit.jsonl")

    first = logger.append_event(
        event_type="tool_call",
        trace_id="trace-1",
        request_id="request-1",
        session_id="session-1",
        actor={"session_id": "session-1", "operator_id": "operator-1"},
        tool_call={"name": "dac3d_status"},
        policy_decision={"allowed": True},
    )
    second = logger.append_event(
        event_type="confirmation",
        trace_id="trace-1",
        request_id="request-2",
        session_id="session-1",
        actor={"session_id": "session-1", "operator_id": "operator-1"},
        confirmation={"preview_id": "p1", "used": True},
    )

    lines = logger.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert first["previous_hash"] == ZERO_HASH
    assert second["previous_hash"] == first["event_hash"]
    assert logger.verify_hash_chain().valid
    assert logger.verify_hash_chain().checked == 2


def test_query_and_redacted_export(tmp_path: Path) -> None:
    logger = AuditTraceLogger(tmp_path / "audit.jsonl")
    logger.append_event(
        event_type="tool_call",
        trace_id="trace-export",
        request_id="request-1",
        session_id="session-1",
        actor={"session_id": "session-1", "operator_token": "secret-token-value"},
        tool_call={"name": "dac3d_tool", "input": {"api_key": "sk-1234567890abcdef"}},
        payload={"exception": "password=hidden-password"},
    )

    queried = logger.query_trace("trace-export")
    exported = logger.export_redacted_trace("trace-export")
    serialized = json.dumps(exported, ensure_ascii=False)

    assert len(queried) == 1
    assert "sk-1234567890abcdef" not in serialized
    assert "hidden-password" not in serialized
    assert REDACTION in serialized


def test_tampered_trace_detected(tmp_path: Path) -> None:
    logger = AuditTraceLogger(tmp_path / "audit.jsonl")
    logger.append_event(event_type="api_request", trace_id="trace-1", request_id="request-1")
    event = json.loads(logger.path.read_text(encoding="utf-8").splitlines()[0])
    event["policy_decision"] = {"allowed": False}
    logger.path.write_text(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    result = logger.verify_hash_chain()

    assert not result.valid
    assert result.error == "EVENT_HASH_MISMATCH"
