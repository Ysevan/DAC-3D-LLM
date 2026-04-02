"""DAC-3D integration adapter with a deterministic mock mode."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class DAC3DUnavailableError(RuntimeError):
    """Raised when a live DAC-3D adapter is requested but unavailable."""


class DAC3DValidationError(RuntimeError):
    """Raised when a DAC-3D command is invalid for submission."""


class DAC3DClient:
    """Communicate with DAC-3D services or a local mock implementation."""

    def __init__(self, *, mock_mode: bool = True, endpoint: str = "mock://dac3d") -> None:
        self.mock_mode = mock_mode
        self.endpoint = endpoint
        self._last_command: dict[str, Any] | None = None
        self._status = {
            "state": "idle",
            "progress": 0,
            "message": "当前没有正在执行的扫描任务。",
        }
        self._result_profiles = {
            "scratch_high": {
                "defect_type": "scratch",
                "location": "左上 ROI",
                "confidence": 0.94,
                "measurements": {
                    "depth_um": 18.4,
                    "length_mm": 0.62,
                    "width_um": 41.0,
                },
                "sample_id": "SAMPLE-001",
            },
            "pit_medium": {
                "defect_type": "pit",
                "location": "中心 ROI",
                "confidence": 0.88,
                "measurements": {
                    "depth_um": 6.3,
                    "diameter_um": 52.0,
                },
                "sample_id": "SAMPLE-002",
            },
        }
        self._active_result_profile = "scratch_high"

    def submit_scan_command(self, command: dict[str, Any]) -> dict[str, Any]:
        """Queue a structured scan command in mock mode."""
        self._validate_command(command)
        if not self.mock_mode:
            raise DAC3DUnavailableError(
                "Live DAC-3D integration is not implemented in this workspace."
            )

        self._last_command = deepcopy(command)
        self._status = {
            "state": "queued",
            "progress": 0,
            "message": "Mock 扫描任务已进入队列，等待执行。",
        }
        return {
            "accepted": True,
            "mode": "mock",
            "command": deepcopy(command),
            "status": deepcopy(self._status),
        }

    def query_current_status(self) -> dict[str, Any]:
        """Return the latest DAC-3D runtime status."""
        if not self.mock_mode:
            raise DAC3DUnavailableError(
                "Live DAC-3D runtime status is not wired in this workspace."
            )

        status = deepcopy(self._status)
        if self._last_command and status["state"] == "queued":
            status["message"] = "Mock 扫描任务已完成预检查，可从排队状态启动。"
        return status

    def get_recent_inspection_result(self) -> dict[str, Any]:
        """Return the latest inspection result payload."""
        if not self.mock_mode:
            raise DAC3DUnavailableError(
                "Live DAC-3D result retrieval is not wired in this workspace."
            )
        return deepcopy(self._result_profiles[self._active_result_profile])

    def runtime_snapshot(self) -> dict[str, Any]:
        """Return a UI-friendly runtime snapshot."""
        return {
            "mode": "mock" if self.mock_mode else "live",
            "endpoint": self.endpoint,
            "status": deepcopy(self._status),
            "active_result_profile": self._active_result_profile,
            "available_result_profiles": sorted(self._result_profiles),
            "last_command_action": (self._last_command or {}).get("action"),
        }

    def execute(self, command: dict[str, Any]) -> dict[str, Any]:
        """Compatibility wrapper for the scaffold contract."""
        return self.submit_scan_command(command)

    def _validate_command(self, command: dict[str, Any]) -> None:
        if not command.get("action"):
            raise DAC3DValidationError("Command must include an action.")
        if not command.get("scan_area_mm"):
            raise DAC3DValidationError("Command must include scan_area_mm.")
