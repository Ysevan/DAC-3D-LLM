"""Path and filesystem policy tests for DAC-Agent Runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

from safety.path_policy import PathPolicy


def test_path_policy_allows_path_inside_read_allowlist(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    policy = PathPolicy(allowed_read_roots=[allowed])

    decision = policy.validate_input_dir(allowed / "batch-a")

    assert decision.allowed is True
    assert decision.matched_root == str(allowed.resolve())
    assert policy.is_allowed_read(allowed / "batch-a") is True


def test_path_policy_rejects_parent_traversal_even_if_it_normalizes_inside(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    policy = PathPolicy(allowed_read_roots=[allowed])

    decision = policy.validate_input_dir(allowed / "nested" / ".." / "batch-a")

    assert decision.allowed is False
    assert "traversal" in decision.reason.lower()


def test_path_policy_rejects_symlink_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    link = allowed / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink not available in this environment: {exc}")
    policy = PathPolicy(allowed_read_roots=[allowed])

    decision = policy.validate_input_dir(link / "batch-a")

    assert decision.allowed is False
    assert decision.path.startswith(str(outside.resolve()))


def test_path_policy_rejects_outside_allowlist(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    policy = PathPolicy(allowed_read_roots=[allowed])

    decision = policy.validate_input_dir(outside / "batch-a")

    assert decision.allowed is False
    assert "outside allowed" in decision.reason


def test_path_policy_rejects_command_output_outside_write_allowlist(tmp_path: Path) -> None:
    output = tmp_path / "command_output"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    policy = PathPolicy(allowed_write_roots=[output])

    allowed = policy.validate_command_output(output / "dac3d_assistant_command.json")
    blocked = policy.validate_command_output(outside / "dac3d_assistant_command.json")

    assert allowed.allowed is True
    assert blocked.allowed is False


def test_path_policy_rejects_secret_file_read_and_config_overwrite(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    output = tmp_path / "output"
    allowed.mkdir()
    output.mkdir()
    policy = PathPolicy(allowed_read_roots=[allowed], allowed_write_roots=[output])

    secret = policy.validate_input_dir(allowed / ".env")
    config = policy.validate_command_output(output / "config.py")

    assert secret.allowed is False
    assert "Secret-bearing" in secret.reason
    assert config.allowed is False
    assert "configuration" in config.reason


def test_path_policy_normalizes_windows_paths_and_blocks_windows_escape() -> None:
    policy = PathPolicy(
        allowed_read_roots=[r"C:\DACData\Offline"],
        allowed_write_roots=[r"C:\DACData\CommandOut"],
    )

    allowed = policy.validate_input_dir(r"C:/DACData/Offline/batch_0527")
    blocked = policy.validate_input_dir(r"C:\DACData\Secrets\api_key.txt")
    output = policy.validate_command_output(r"C:/DACData/CommandOut/dac3d_assistant_command.json")

    assert allowed.allowed is True
    assert allowed.path == r"C:\DACData\Offline\batch_0527"
    assert blocked.allowed is False
    assert output.allowed is True


def test_path_policy_rejects_unc_without_explicit_network_allowlist() -> None:
    policy = PathPolicy(allowed_read_roots=[r"C:\DACData\Offline"])

    decision = policy.validate_input_dir(r"\\server\share\batch")

    assert decision.allowed is False
    assert "UNC" in decision.reason


def test_path_policy_delete_is_always_forbidden(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    policy = PathPolicy(allowed_read_roots=[allowed], allowed_write_roots=[allowed])

    decision = policy.validate_delete(allowed / "old-command.json")

    assert decision.allowed is False
    assert "Deleting files is forbidden" in decision.reason
