"""Tests for centralized DAC-3D filesystem path policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from integration.dac3d_client import DAC3DClient, DAC3DValidationError
from security.path_policy import PathPolicy, PathPolicyError


def _make_offline_images(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for surface in ("surface1", "surface2"):
        for camera in ("焦前", "焦面", "焦后"):
            (path / f"pos1_{surface}_{camera}.jpg").write_bytes(b"fake")


def _validation_command(image_folder: Path) -> dict[str, object]:
    return {
        "action": "validate_offline_folder",
        "payload": {"image_folder": str(image_folder)},
        "safety": {"safe_to_auto_execute": True},
    }


def test_path_policy_allows_canonical_input_directory(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    image_dir = allowed_root / "batch-1"
    _make_offline_images(image_dir)
    policy = PathPolicy(allowed_input_dirs=(allowed_root,))

    assert policy.validate_existing_input_dir(image_dir) == image_dir.resolve()


def test_path_policy_rejects_parent_traversal(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    policy = PathPolicy(allowed_input_dirs=(allowed_root,))

    with pytest.raises(PathPolicyError) as exc_info:
        policy.validate_existing_input_dir(allowed_root / ".." / "outside")

    assert exc_info.value.code == "PATH_TRAVERSAL"


def test_path_policy_rejects_symlink_escape(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed_root.mkdir()
    _make_offline_images(outside)
    link = allowed_root / "escape"
    link.symlink_to(outside, target_is_directory=True)
    policy = PathPolicy(allowed_input_dirs=(allowed_root,))

    with pytest.raises(PathPolicyError) as exc_info:
        policy.validate_existing_input_dir(link)

    assert exc_info.value.code == "PATH_OUTSIDE_ALLOWED_ROOTS"


def test_path_policy_rejects_secret_status_file_read(tmp_path: Path) -> None:
    secret_file = tmp_path / ".env"
    secret_file.write_text("DAC3D_LLM_API_KEY=secret-value", encoding="utf-8")
    policy = PathPolicy()

    with pytest.raises(PathPolicyError) as exc_info:
        policy.validate_status_file_read(secret_file)

    assert exc_info.value.code == "SECRET_PATH_REJECTED"


def test_path_policy_rejects_unsafe_upload_filenames(tmp_path: Path) -> None:
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    policy = PathPolicy()

    assert policy.safe_upload_destination(upload_root, "manual.md") == (upload_root / "manual.md").resolve()
    with pytest.raises(PathPolicyError) as traversal:
        policy.safe_upload_destination(upload_root, "../manual.md")
    with pytest.raises(PathPolicyError) as secret:
        policy.safe_upload_destination(upload_root, ".env")

    assert traversal.value.code == "UPLOAD_FILENAME_PATH"
    assert secret.value.code == "SECRET_PATH_REJECTED"


def test_path_policy_restricts_command_output_dir_and_filename(tmp_path: Path) -> None:
    command_root = tmp_path / "commands"
    command_root.mkdir()
    policy = PathPolicy(allowed_command_output_dir=command_root)

    allowed_path = policy.validate_command_output_path(command_root / "dac3d_assistant_command.json")
    assert allowed_path == (command_root / "dac3d_assistant_command.json").resolve()
    with pytest.raises(PathPolicyError) as outside:
        policy.validate_command_output_path(tmp_path / "dac3d_assistant_command.json")
    with pytest.raises(PathPolicyError) as wrong_name:
        policy.validate_command_output_path(command_root / "other.json")

    assert outside.value.code == "PATH_OUTSIDE_ALLOWED_ROOTS"
    assert wrong_name.value.code == "COMMAND_FILENAME_NOT_ALLOWED"


def test_dac3d_client_blocks_offline_folder_outside_allowlist(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    outside_dir = tmp_path / "outside"
    allowed_root.mkdir()
    _make_offline_images(outside_dir)
    client = DAC3DClient(
        mock_mode=True,
        path_policy=PathPolicy(allowed_input_dirs=(allowed_root,)),
    )

    validation = client.validate_offline_folder(_validation_command(outside_dir))

    assert validation["ready"] is False
    assert "path_policy:PATH_OUTSIDE_ALLOWED_ROOTS" in validation["missing_requirements"]
    assert validation["path_policy"]["allowed"] is False


def test_dac3d_client_rejects_unsafe_offline_start(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    outside_dir = tmp_path / "outside"
    allowed_root.mkdir()
    _make_offline_images(outside_dir)
    client = DAC3DClient(
        mock_mode=True,
        path_policy=PathPolicy(allowed_input_dirs=(allowed_root,)),
    )

    with pytest.raises(DAC3DValidationError) as exc_info:
        client.submit_scan_command(
            {
                "action": "start_offline_detection",
                "payload": {"image_folder": str(outside_dir)},
                "safety": {"needs_confirmation": True},
            }
        )

    assert "PathPolicy" in str(exc_info.value)


def test_dac3d_client_rejects_command_bridge_outside_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_file = tmp_path / "dac3d_runtime_status.json"
    status_file.write_text(json.dumps({"status": {"state": "idle"}}), encoding="utf-8")
    allowed_output = tmp_path / "commands"
    outside_output = tmp_path / "outside"
    command_file = outside_output / "dac3d_assistant_command.json"
    monkeypatch.setenv("DAC3D_COMMAND_PATH", str(command_file))
    client = DAC3DClient(
        endpoint=status_file.as_uri(),
        path_policy=PathPolicy(allowed_command_output_dir=allowed_output),
    )

    with pytest.raises(DAC3DValidationError) as exc_info:
        client.submit_scan_command(
            {
                "action": "scan",
                "scan_area_mm": {"width": 10.0, "height": 10.0},
                "region": "current_selection",
                "mode": "standard",
                "payload": {},
                "safety": {"needs_confirmation": True},
            }
        )

    assert "PathPolicy" in str(exc_info.value)
    assert not command_file.exists()


def test_dac3d_client_rejects_secret_status_file_endpoint(tmp_path: Path) -> None:
    secret_file = tmp_path / ".env"
    secret_file.write_text("DAC3D_LLM_API_KEY=secret-value", encoding="utf-8")
    client = DAC3DClient(endpoint=secret_file.as_uri(), path_policy=PathPolicy())

    status = client.query_current_status()

    assert status["state"] == "error"
    assert status["path_policy"]["code"] == "SECRET_PATH_REJECTED"
    assert "secret-value" not in status["message"]
