"""Tests for production security configuration validation."""

from __future__ import annotations

import pytest

from security.production_config import ProductionSecurityConfig, SecurityConfigError


def _prod_config(**overrides: object) -> ProductionSecurityConfig:
    values = {
        "env": "prod",
        "allow_command_submit": False,
        "require_confirmation": True,
        "allowed_input_dirs": ("/safe/input",),
        "allowed_command_output_dir": "/safe/output",
        "enable_memory_write": True,
        "enable_skill_patch_apply": False,
        "enable_remote_tools": False,
        "enable_open_world_tools": False,
        "trace_redaction": True,
        "audit_trace_enabled": True,
        "audit_trace_signing_key_configured": True,
        "cors_allowed_origins": ("https://dac3d.example",),
        "rate_limit_enabled": True,
        "debug_mode": False,
        "mock_mode": True,
    }
    values.update(overrides)
    return ProductionSecurityConfig(**values)


def test_prod_insecure_config_rejected() -> None:
    config = _prod_config(
        require_confirmation=False,
        trace_redaction=False,
        enable_remote_tools=True,
        debug_mode=True,
    )

    with pytest.raises(SecurityConfigError) as exc_info:
        config.validate_or_raise()

    message = str(exc_info.value)
    assert "REQUIRE_CONFIRMATION" in message
    assert "TRACE_REDACTION" in message
    assert "ENABLE_REMOTE_TOOLS" in message
    assert "Debug mode" in message


def test_dev_can_be_permissive_with_warning() -> None:
    config = ProductionSecurityConfig(
        env="dev",
        require_confirmation=False,
        cors_allowed_origins=("*",),
        enable_remote_tools=True,
        enable_open_world_tools=True,
        debug_mode=True,
    )

    config.validate_or_raise()

    warnings = config.warnings()
    assert any("Confirmation" in warning for warning in warnings)
    assert any("Wildcard CORS" in warning for warning in warnings)


def test_missing_allowlist_rejected() -> None:
    config = _prod_config(allowed_input_dirs=(), allowed_command_output_dir="")

    errors = config.validate()

    assert any("ALLOWED_INPUT_DIRS" in error for error in errors)
    assert any("ALLOWED_COMMAND_OUTPUT_DIR" in error for error in errors)


def test_missing_audit_trace_signing_key_rejected_in_prod() -> None:
    config = _prod_config(audit_trace_signing_key_configured=False)

    errors = config.validate()

    assert any("AUDIT_TRACE_SIGNING_KEY" in error for error in errors)


def test_disabled_audit_trace_rejected_in_prod() -> None:
    config = _prod_config(audit_trace_enabled=False)

    errors = config.validate()

    assert any("AUDIT_TRACE_ENABLED" in error for error in errors)


def test_wildcard_cors_rejected_in_prod() -> None:
    config = _prod_config(cors_allowed_origins=("*",))

    errors = config.validate()

    assert any("CORS wildcard" in error for error in errors)


def test_confirmation_disabled_rejected_in_prod() -> None:
    config = _prod_config(require_confirmation=False)

    errors = config.validate()

    assert any("REQUIRE_CONFIRMATION" in error for error in errors)


def test_mock_writer_rejected_when_prod_submit_enabled() -> None:
    config = _prod_config(allow_command_submit=True, mock_mode=True)

    errors = config.validate()

    assert any("mock DAC-3D writer" in error for error in errors)
