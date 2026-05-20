"""Embedded DAC-3D runtime bridge tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app import DAC3DAssistant
from config import AppConfig
from integration.dac3d_client import DAC3DClient


class FakeRuntimeBridge:
    """Small stand-in for the xxp_ui main-window bridge."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.status = {"state": "idle", "progress": 0, "message": "ready"}

    def start_online_scan(self, command: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("start_online_scan", command))
        self.status = {"state": "running", "progress": 1, "message": "online scan queued"}
        return {"accepted": True, "status": self.status}

    def start_offline_detection(self, command: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("start_offline_detection", command))
        self.status = {"state": "running", "progress": 1, "message": "offline detection queued"}
        return {"accepted": True, "status": self.status}

    def stop_detection(self, command: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("stop_detection", command))
        self.status = {"state": "stopped", "progress": 1, "message": "stop requested"}
        return {"accepted": True, "status": self.status}

    def query_current_status(self, command: dict[str, Any] | None = None) -> dict[str, Any]:
        if command is not None:
            self.calls.append(("query_current_status", command))
        return dict(self.status)

    def validate_offline_folder(self, command: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("validate_offline_folder", command))
        payload = dict(command.get("payload") or {})
        return {
            "validation": {
                "image_folder": payload.get("image_folder"),
                "ready": True,
                "missing_requirements": [],
            },
            "status": self.status,
        }

    def get_latest_result_summary(self, command: dict[str, Any] | None = None) -> dict[str, Any]:
        if command is not None:
            self.calls.append(("get_latest_result_summary", command))
        return {
            "result_root": "embedded://latest",
            "files": ["defect_detail.csv", "defect_summary.csv"],
            "parsed_result": {
                "defect_type": "scratch",
                "location": "ROI-1",
                "confidence": 0.91,
                "measurements": {"depth_um": 16.0, "length_mm": 0.55},
            },
        }


def _make_config(tmp_path: Path) -> AppConfig:
    config = AppConfig(
        base_dir=tmp_path,
        mock_mode=False,
        provider="mock",
        vector_store_type="manifest",
        embedding_download_allowed=False,
    )
    config.ensure_directories()
    (config.documents_dir / "manual.md").write_text(
        "# DAC-3D Manual\n\n## Runtime\nDAC-3D supports online scan, offline detection, status query, and result retrieval.",
        encoding="utf-8",
    )
    return config


def test_dac3d_client_delegates_control_commands_to_runtime_bridge() -> None:
    bridge = FakeRuntimeBridge()
    client = DAC3DClient(mock_mode=False, endpoint="embedded://xxp_ui", runtime_bridge=bridge)

    response = client.submit_scan_command(
        {
            "action": "start_online_scan",
            "scan_area_mm": None,
            "resolution": None,
            "region": "current_selection",
            "mode": "standard",
            "payload": {"func": "Scan"},
            "safety": {"needs_confirmation": True},
            "missing_fields": [],
            "warnings": [],
        }
    )

    assert response["mode"] == "embedded"
    assert response["status"]["state"] == "running"
    assert bridge.calls[0][0] == "start_online_scan"


def test_assistant_can_control_embedded_runtime_flow(tmp_path: Path) -> None:
    bridge = FakeRuntimeBridge()
    assistant = DAC3DAssistant.create(
        _make_config(tmp_path),
        rebuild_kb=True,
        runtime_bridge=bridge,
    )

    online = assistant.handle_message("start online scan execute")
    stop = assistant.handle_message("stop detection execute")
    latest = assistant.handle_message("get latest result")

    assert online.intent == "operation"
    assert online.status_summary["state"] == "running"
    assert stop.status_summary["state"] == "stopped"
    assert latest.parsed_result["result_root"] == "embedded://latest"
    assert [name for name, _ in bridge.calls] == [
        "start_online_scan",
        "stop_detection",
        "get_latest_result_summary",
    ]


def test_assistant_validates_offline_folder_through_embedded_bridge(tmp_path: Path) -> None:
    bridge = FakeRuntimeBridge()
    assistant = DAC3DAssistant.create(
        _make_config(tmp_path),
        rebuild_kb=True,
        runtime_bridge=bridge,
    )

    offline_folder = tmp_path / "offline_images"
    offline_folder.mkdir()
    response = assistant.handle_message(f"validate offline folder {offline_folder} execute")

    assert response.intent == "operation"
    assert response.status_summary["state"] == "idle"
    assert response.command_preview["payload"]["image_folder"] == str(offline_folder)
    assert bridge.calls[0][0] == "validate_offline_folder"
