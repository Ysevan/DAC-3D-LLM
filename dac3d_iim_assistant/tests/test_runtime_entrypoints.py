"""Tests for runtime entrypoint configuration and temp directory setup."""

from __future__ import annotations

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
