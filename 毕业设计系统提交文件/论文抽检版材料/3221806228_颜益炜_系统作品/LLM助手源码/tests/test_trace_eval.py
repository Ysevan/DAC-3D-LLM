"""Tests for append-only traces and deterministic Agent evals."""

from __future__ import annotations

from agent_runtime import DAC3DAgentChatAdapter, DAC3DAgentRuntime, LOCAL_VALIDATION_MODEL_NAME
from app import DAC3DAssistant
from tests.test_agent_runtime import make_agent_config
from trace_eval import CodexHandoffGenerator, TraceLogger


def _make_adapter(tmp_path) -> DAC3DAgentChatAdapter:
    config = make_agent_config(tmp_path)
    config.agent_model_name = LOCAL_VALIDATION_MODEL_NAME
    assistant = DAC3DAssistant.create(config=config)
    return DAC3DAgentChatAdapter(DAC3DAgentRuntime(assistant=assistant, config=config))


def test_trace_logger_records_chat_and_can_read_trace(tmp_path) -> None:
    adapter = _make_adapter(tmp_path)

    response = adapter.handle_message("当前检测状态是什么？", [], session_id="trace-read")

    trace_eval = response.parsed_result["trace_eval"]
    trace = adapter.read_trace(trace_eval["trace_id"])
    assert trace["session_id"] == "trace-read"
    assert trace["intent"] == "status"
    assert trace["user_message"] == "当前检测状态是什么？"
    assert trace["final_response"]


def test_eval_runner_covers_safety_approval_memory_and_status(tmp_path) -> None:
    adapter = _make_adapter(tmp_path)

    result = adapter.run_evals()

    assert result["case_count"] >= 20
    assert result["failed"] == 0
    categories = {item["category"] for item in result["results"]}
    assert {"state_query", "command_preview", "confirmation_required", "prompt_injection", "memory_write"}.issubset(categories)
    injection = next(item for item in result["results"] if item["category"] == "prompt_injection")
    assert injection["passed"] is True
    assert injection["trace_id"]


def test_trace_to_eval_drafts_are_saved_outside_approved_cases(tmp_path) -> None:
    adapter = _make_adapter(tmp_path)
    response = adapter.handle_message("当前检测状态是什么？", [], session_id="draft-session")
    trace_id = response.parsed_result["trace_eval"]["trace_id"]

    result = adapter.generate_eval_drafts(trace_ids=[trace_id])

    assert result["count"] == 1
    assert result["auto_approved"] is False
    draft = result["drafts"][0]["draft"]
    assert draft["draft"] is True
    assert draft["source_trace_id"] == trace_id
    assert draft["input"] == "当前检测状态是什么？"
    assert draft["expected"]["intent"] == "status"
    assert "dac3d_status" in draft["expected"]["must_call_tools"]
    assert (tmp_path / "evals" / "drafts" / f"{draft['id']}.json").exists()
    assert not (tmp_path / "evals" / "cases" / f"{draft['id']}.json").exists()


def test_codex_handoff_generator_writes_failed_eval_and_trace_summary(tmp_path) -> None:
    trace_logger = TraceLogger(tmp_path / "agent_traces.jsonl")
    trace = trace_logger.append(
        {
            "trace_id": "trace-status-mismatch",
            "event_type": "agent_chat",
            "session_id": "handoff-session",
            "user_message": "当前检测状态是什么？",
            "intent": "query",
            "tool_calls": [{"name": "dac3d_answer"}],
            "final_response": "我查询了知识库。",
        }
    )
    eval_result = {
        "backend": "local_deterministic_eval",
        "case_count": 1,
        "passed": 0,
        "failed": 1,
        "pass_rate": 0.0,
        "results": [
            {
                "id": "state_query_status",
                "category": "state_query",
                "input": "当前检测状态是什么？",
                "passed": False,
                "checks": [
                    {
                        "name": "response.intent",
                        "passed": False,
                        "expected": "status",
                        "actual": "query",
                    }
                ],
                "intent": "query",
                "answer_preview": "我查询了知识库。",
                "trace_id": trace["trace_id"],
            }
        ],
    }
    output_path = tmp_path / "docs" / "generated" / "codex_handoff_next.md"

    result = CodexHandoffGenerator(
        trace_logger=trace_logger,
        output_path=output_path,
    ).generate(eval_result=eval_result)

    assert result["failed_count"] == 1
    assert result["trace_count"] >= 1
    assert result["auto_applied"] is False
    assert result["path"] == str(output_path)
    markdown = output_path.read_text(encoding="utf-8")
    assert "state_query_status" in markdown
    assert "trace-status-mismatch" in markdown
    assert "Tune intent routing" in markdown
    assert "python -m pytest tests/test_trace_eval.py tests/test_web_api.py -q" in markdown
