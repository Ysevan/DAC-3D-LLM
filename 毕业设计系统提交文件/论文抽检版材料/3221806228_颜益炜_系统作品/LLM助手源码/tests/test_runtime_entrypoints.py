"""Tests for runtime entrypoint configuration and temp directory setup."""

from __future__ import annotations

from pathlib import Path

from app import build_argument_parser
from config import AppConfig


def test_app_config_creates_centralized_temp_directories(tmp_path) -> None:
    config = AppConfig(base_dir=tmp_path)

    config.ensure_directories()

    assert config.temp_root_dir.exists()
    assert config.upload_temp_dir.exists()


def test_argument_parser_supports_web_only_flag() -> None:
    parser = build_argument_parser()

    args = parser.parse_args(["--web-only"])

    assert args.web_only is True


def test_argument_parser_supports_agent_flags() -> None:
    parser = build_argument_parser()

    args = parser.parse_args(
        [
            "--agent",
            "--agent-web",
            "--assistant-router",
            "--agent-model",
            "gpt-5.4-mini",
            "--agent-api-base-url",
            "https://llm.example.test/v1",
            "--agent-api-key",
            "third-party-key",
            "--agent-api-type",
            "chat_completions",
        ]
    )

    assert args.agent is True
    assert args.agent_web is True
    assert args.assistant_router is True
    assert args.agent_model == "gpt-5.4-mini"
    assert args.agent_api_base_url == "https://llm.example.test/v1"
    assert args.agent_api_key == "third-party-key"
    assert args.agent_api_type == "chat_completions"


def test_dac3d_host_launches_assistant_in_agent_web_mode() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    host_ui = repo_root / "福特科" / "xxp_ui" / "window" / "ui.py"

    source = host_ui.read_text(encoding="utf-8")

    assert "[sys.executable, str(app_py), '--agent-web']" in source
    assert "DAC3D_ENDPOINT" in source
    assert "DAC3D_COMMAND_PATH" in source
