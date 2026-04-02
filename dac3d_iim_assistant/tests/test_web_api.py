"""Tests for the FastAPI web surface."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import DAC3DAssistant
from tests.test_assistant import make_config
from ui.web_api import create_api_app


def test_web_api_chat_endpoint_returns_structured_payload(tmp_path) -> None:
    """The API should expose the assistant response schema for the React client."""
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.post(
        "/api/chat",
        json={"message": "这个参数是什么意思？", "history": []},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "query"
    assert payload["answer"]
    assert isinstance(payload["sources"], list)


def test_web_api_runtime_endpoint_returns_runtime_summary(tmp_path) -> None:
    """The API should expose runtime data for the settings drawer."""
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.get("/api/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "mock"
    assert "dac3d" in payload


def test_web_api_chat_endpoint_rejects_empty_message(tmp_path) -> None:
    """The API should reject invalid chat payloads with a client error."""
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.post("/api/chat", json={"message": "   ", "history": []})

    assert response.status_code == 422


def test_web_api_chat_stream_emits_sse_events(tmp_path) -> None:
    """The streaming endpoint should emit metadata, deltas, and the final payload."""
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.post(
        "/api/chat/stream",
        json={"message": "what does this parameter mean?", "history": []},
    )

    assert response.status_code == 200
    body = response.text
    assert "event: meta" in body
    assert "event: delta" in body
    assert "event: done" in body


def test_web_api_chat_stream_emits_multiple_deltas_progressively(tmp_path) -> None:
    """The streaming endpoint should emit multiple incremental delta events for grounded answers."""
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.post(
        "/api/chat/stream",
        json={"message": "样品太反光怎么办？", "history": []},
    )

    assert response.status_code == 200
    body = response.text
    assert body.count("event: delta") >= 2
    assert body.index("event: meta") < body.index("event: delta") < body.index("event: done")
