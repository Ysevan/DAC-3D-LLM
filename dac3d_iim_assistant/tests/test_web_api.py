"""Tests for the FastAPI web surface."""

from __future__ import annotations

from fastapi.testclient import TestClient

from agent_runtime import DAC3DAgentChatAdapter, DAC3DAgentRuntime
from app import DAC3DAssistant
from tests.test_assistant import make_config
from ui.web_api import create_api_app


SESSION_HEADERS = {"X-DAC3D-Session-ID": "test-session"}


def test_web_api_chat_endpoint_returns_structured_payload(tmp_path) -> None:
    """The API should expose the assistant response schema for the React client."""
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.post(
        "/api/chat",
        json={"message": "这个参数是什么意思？", "history": []},
        headers=SESSION_HEADERS,
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

    response = client.get("/api/runtime", headers=SESSION_HEADERS)

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "mock"
    assert "dac3d" in payload


def test_web_api_chat_endpoint_rejects_empty_message(tmp_path) -> None:
    """The API should reject invalid chat payloads with a client error."""
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.post(
        "/api/chat",
        json={"message": "   ", "history": []},
        headers=SESSION_HEADERS,
    )

    assert response.status_code == 422


def test_web_api_chat_stream_emits_sse_events(tmp_path) -> None:
    """The streaming endpoint should emit metadata, deltas, and the final payload."""
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.post(
        "/api/chat/stream",
        json={"message": "what does this parameter mean?", "history": []},
        headers=SESSION_HEADERS,
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
        headers=SESSION_HEADERS,
    )

    assert response.status_code == 200
    body = response.text
    assert body.count("event: delta") >= 2
    assert body.index("event: meta") < body.index("event: delta") < body.index("event: done")


def test_machine_agent_snapshot_endpoint_returns_dashboard_payload(tmp_path) -> None:
    """The industrial Agent dashboard endpoint should expose status and demo data."""
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.get("/api/machine-agent/snapshot", headers=SESSION_HEADERS)

    assert response.status_code == 200
    payload = response.json()
    assert payload["current_status"]["machine_id"] == "IM-Press-01"
    assert payload["alarm_records"]
    assert "现在设备状态怎么样？" in payload["demo_questions"]


def test_machine_agent_chat_endpoint_exposes_tool_calls(tmp_path) -> None:
    """Machine Agent chat should show the tools used to answer a question."""
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.post(
        "/api/machine-agent/chat",
        json={"message": "为什么最近温度报警变多了？"},
        headers=SESSION_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    tool_names = [item["name"] for item in payload["tool_calls"]]
    assert "get_alarm_records" in tool_names
    assert "detect_abnormal_patterns" in tool_names
    assert payload["abnormal_result"]["alarm_count"] > 0


def test_unified_chat_endpoint_enters_agent_runtime_first(tmp_path, monkeypatch) -> None:
    """The single chat endpoint should delegate the message to the Agent runtime."""
    def fake_run_sync(self, message: str, *, session_id: str = "default") -> str:
        del self
        assert session_id == "chrome-session-1"
        assert message == "为什么最近温度报警变多了？"
        return (
            '{"answer":"LLM 已选择 machine_agent_chat 工具处理温度报警问题。",'
            '"structured_data":{'
            '"intent":"machine_alarm_analysis",'
            '"tool_calls":[{"name":"machine_agent_chat","purpose":"分析温度报警"}],'
            '"findings":[{"metric":"temperature_high","value":"最近30天 6 次"}],'
            '"recommendations":["检查冷却水路"]'
            "}}"
        )

    monkeypatch.setattr(DAC3DAgentRuntime, "run_sync", fake_run_sync)
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    response = client.post(
        "/api/chat",
        json={
            "message": "为什么最近温度报警变多了？",
            "history": [],
            "session_id": "chrome-session-1",
        },
        headers={"X-DAC3D-Session-ID": "chrome-session-1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "machine_alarm_analysis"
    assert "machine_agent_chat" in payload["answer"]
    assert payload["parsed_result"]["tool_calls"][0]["name"] == "machine_agent_chat"


def test_unified_chat_stream_uses_client_session_id(tmp_path, monkeypatch) -> None:
    """The streaming chat endpoint should not force every client into the same Agent session."""
    seen_sessions: list[str] = []

    def fake_run_sync(self, message: str, *, session_id: str = "default") -> str:
        del self
        assert message == "当前检测状态是什么？"
        seen_sessions.append(session_id)
        return (
            '{"answer":"当前 DAC-3D 处于空闲状态。",'
            '"structured_data":{'
            '"intent":"status",'
            '"tool_calls":[{"name":"dac3d_status","purpose":"读取状态"}],'
            '"status_summary":{"state":"idle","progress":0}'
            "}}"
        )

    monkeypatch.setattr(DAC3DAgentRuntime, "run_sync", fake_run_sync)
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    response = client.post(
        "/api/chat/stream",
        json={
            "message": "当前检测状态是什么？",
            "history": [],
            "session_id": "ui-session-42",
        },
        headers={"X-DAC3D-Session-ID": "ui-session-42"},
    )

    assert response.status_code == 200
    assert "event: done" in response.text
    assert seen_sessions == ["ui-session-42"]
