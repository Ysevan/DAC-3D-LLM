"""Command lifecycle security tests for DAC Tool Gateway."""

from __future__ import annotations

from pathlib import Path

from agent_core import DAC3DAgentSessionStore
from app import DAC3DAssistant
from tests.test_tool_gateway import make_gateway_config
from tool_gateway import DAC3DToolGateway


def make_gateway(tmp_path: Path, *, session_id: str = "lifecycle") -> tuple[DAC3DToolGateway, DAC3DAgentSessionStore]:
    config = make_gateway_config(tmp_path)
    assistant = DAC3DAssistant.create(config=config)
    sessions = DAC3DAgentSessionStore(base_dir=config.base_dir)
    return DAC3DToolGateway(assistant=assistant, sessions=sessions, session_id=session_id), sessions


def test_command_lifecycle_preview_created_but_not_submitted(tmp_path: Path) -> None:
    gateway, sessions = make_gateway(tmp_path, session_id="preview-only")

    payload = gateway.preview_command("扫描 10mm x 10mm 区域")
    pending = sessions.get_pending_command("preview-only")
    history = gateway.read_command_history(limit=5).to_dict()["result"]["events"]

    assert pending is not None
    assert pending.lifecycle_state == "awaiting_confirmation"
    assert pending.preview_id == payload["command_preview"]["gateway"]["preview_id"]
    assert pending.preview_hash == payload["command_preview"]["gateway"]["preview_hash"]
    assert pending.confirmation_token == payload["command_preview"]["gateway"]["confirmation_token"]
    assert pending.expires_at == payload["command_preview"]["gateway"]["confirmation_expires_at"]
    assert [event["state"] for event in pending.lifecycle_events] == [
        "preview_created",
        "validation_passed",
        "awaiting_confirmation",
    ]
    assert history[-1]["event"] == "preview_command"
    assert history[-1]["lifecycle_state"] == "awaiting_confirmation"


def test_command_lifecycle_submit_before_confirmation_rejected(tmp_path: Path) -> None:
    gateway, _sessions = make_gateway(tmp_path, session_id="missing-token")
    preview = gateway.preview_command("扫描 10mm x 10mm 区域")["command_preview"]

    blocked = gateway.submit_command(preview["gateway"]["preview_id"], "")

    tool_gateway = blocked["parsed_result"]["tool_gateway"]
    assert tool_gateway["blocked"] is True
    assert tool_gateway["reason"] == "policy_denied"
    assert "schema_invalid" in tool_gateway["policy"]["blocking_reasons"]


def test_command_lifecycle_expired_preview_rejected(tmp_path: Path) -> None:
    config = make_gateway_config(tmp_path)
    config.command_confirmation_ttl_seconds = 0
    assistant = DAC3DAssistant.create(config=config)
    sessions = DAC3DAgentSessionStore(base_dir=config.base_dir)
    gateway = DAC3DToolGateway(assistant=assistant, sessions=sessions, session_id="expired")

    preview = gateway.preview_command("扫描 10mm x 10mm 区域")["command_preview"]
    blocked = gateway.submit_command(
        preview["gateway"]["preview_id"],
        preview["gateway"]["confirmation_token"],
    )
    history = gateway.read_command_history(limit=10).to_dict()["result"]["events"]

    assert blocked["parsed_result"]["tool_gateway"]["blocked"] is True
    assert blocked["parsed_result"]["tool_gateway"]["reason"] == "confirmation_expired"
    assert sessions.get_pending_command("expired") is None
    assert any(event.get("reason") == "confirmation_expired" for event in history)


def test_command_lifecycle_modified_preview_invalidates_confirmation(tmp_path: Path) -> None:
    gateway, sessions = make_gateway(tmp_path, session_id="modified")
    preview = gateway.preview_command("扫描 10mm x 10mm 区域")["command_preview"]
    pending = sessions.get_pending_command("modified")
    assert pending is not None
    pending.command_preview["scan_area_mm"]["width"] = 20

    blocked = gateway.submit_command(
        preview["gateway"]["preview_id"],
        preview["gateway"]["confirmation_token"],
    )

    tool_gateway = blocked["parsed_result"]["tool_gateway"]
    assert tool_gateway["blocked"] is True
    assert tool_gateway["reason"] in {"preview_id_mismatch", "preview_hash_mismatch"}
    assert sessions.get_pending_command("modified") is not None


def test_command_lifecycle_token_mismatch_and_replay_rejected(tmp_path: Path) -> None:
    gateway, sessions = make_gateway(tmp_path, session_id="replay")
    preview = gateway.preview_command("扫描 10mm x 10mm 区域")["command_preview"]
    preview_id = preview["gateway"]["preview_id"]
    token = preview["gateway"]["confirmation_token"]

    mismatch = gateway.submit_command(preview_id, "confirm-wrong")
    submitted = gateway.submit_command(preview_id, token)
    replay = gateway.submit_command(preview_id, token)

    assert mismatch["parsed_result"]["tool_gateway"]["reason"] == "invalid_confirmation"
    assert submitted["parsed_result"]["tool_gateway"]["tool"] == "submit_command"
    assert submitted["status_summary"]["state"] == "queued"
    assert sessions.get_pending_command("replay") is None
    assert replay["parsed_result"]["tool_gateway"]["reason"] == "confirmation_replay"


def test_command_lifecycle_direct_command_writer_not_exposed(tmp_path: Path) -> None:
    gateway, _sessions = make_gateway(tmp_path, session_id="manifest")

    manifest = gateway.describe()
    tool_names = {tool["name"] for tool in manifest["tools"]}

    assert "submit_command" in tool_names
    assert "write_command_json" not in tool_names
    assert "write_command_file" not in tool_names
    assert "direct_command_writer" not in tool_names
