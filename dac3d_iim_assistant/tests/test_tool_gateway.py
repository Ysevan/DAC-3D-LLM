"""Tests for the Agent ToolGateway policy boundary."""

from __future__ import annotations

from agent_core import AGENT_TOOL_NAMES, DEFAULT_AGENT_TOOL_GATEWAY
from agent_core.tool_gateway import TOOL_DIRECT_USE_DISABLED, TOOL_NOT_REGISTERED, ToolGateway


def test_default_gateway_registers_every_agent_tool() -> None:
    registered = set(DEFAULT_AGENT_TOOL_GATEWAY.registered_tool_names())

    assert set(AGENT_TOOL_NAMES) == registered


def test_unknown_tool_fails_closed_without_invoking_handler() -> None:
    called = False

    def handler() -> dict[str, object]:
        nonlocal called
        called = True
        return {"answer": "should not run"}

    payload = ToolGateway({}).invoke("unknown_tool", {}, handler)

    assert called is False
    assert payload["policy_decision"]["allowed"] is False
    assert payload["policy_decision"]["reason"] == TOOL_NOT_REGISTERED


def test_gateway_rejects_unexpected_arguments_without_invoking_handler() -> None:
    called = False

    def handler() -> dict[str, object]:
        nonlocal called
        called = True
        return {"answer": "should not run"}

    payload = DEFAULT_AGENT_TOOL_GATEWAY.invoke(
        "dac3d_status",
        {"unexpected": "value"},
        handler,
    )

    assert called is False
    assert payload["policy_decision"]["allowed"] is False
    assert payload["policy_decision"]["reason"] == "TOOL_ARGUMENTS_INVALID"


def test_gateway_blocks_agent_direct_write_tools() -> None:
    called = False

    def handler() -> dict[str, object]:
        nonlocal called
        called = True
        return {"answer": "should not run"}

    payload = DEFAULT_AGENT_TOOL_GATEWAY.invoke(
        "dac3d_rebuild_knowledge_base",
        {},
        handler,
    )

    assert called is False
    assert payload["policy_decision"]["allowed"] is False
    assert payload["policy_decision"]["reason"] == TOOL_DIRECT_USE_DISABLED
    assert payload["policy_decision"]["side_effect"] == "filesystem_write"
