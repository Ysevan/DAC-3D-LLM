"""Tests for the industrial machine information-management Agent."""

from __future__ import annotations

from datetime import datetime

from machine_agent import MachineAgentService, MockMachineDataProvider


def make_agent() -> MachineAgentService:
    provider = MockMachineDataProvider(now=datetime(2026, 5, 26, 10, 0, 0))
    return MachineAgentService(provider=provider)


def test_machine_agent_answers_current_status_with_tool_trace() -> None:
    agent = make_agent()

    payload = agent.chat("现在设备状态怎么样？")

    assert payload["intent"] == "machine_agent"
    assert payload["current_status"]["machine_id"] == "IM-Press-01"
    assert payload["tool_calls"][0]["name"] == "get_current_machine_status"
    assert "Agent 工具调用" in payload["answer"] or "工具调用" in payload["answer"]


def test_machine_agent_summarizes_last_month_from_history_and_alarms() -> None:
    agent = make_agent()

    payload = agent.chat("上个月运行情况怎么样？")
    tool_names = [item["name"] for item in payload["tool_calls"]]

    assert "get_machine_history" in tool_names
    assert "get_alarm_records" in tool_names
    assert "summarize_machine_condition" in tool_names
    assert payload["summary_result"]["sample_count"] > 0
    assert payload["summary_result"]["alarm_count"] > 0


def test_machine_agent_detects_recent_temperature_alarm_pattern() -> None:
    agent = make_agent()

    payload = agent.chat("为什么最近温度报警变多了？")
    tool_names = [item["name"] for item in payload["tool_calls"]]

    assert "get_alarm_records" in tool_names
    assert "detect_abnormal_patterns" in tool_names
    assert payload["abnormal_result"]["alarm_count"] > 0
    assert payload["abnormal_result"]["top_alarm_types"][0]["type"] == "temperature_high"


def test_machine_agent_searches_error_code_documents() -> None:
    agent = make_agent()

    payload = agent.chat("E102 错误代码是什么意思？")

    assert payload["tool_calls"][0]["name"] == "search_machine_docs"
    assert payload["doc_results"]
    assert "E102" in payload["answer"]
