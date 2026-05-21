"""DAC-3D integration adapter with a deterministic mock mode."""

from __future__ import annotations

import json
import os
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from intent.structured_commands import (
    DEFAULT_REQUIRED_CAMERAS,
    DEFAULT_REQUIRED_SURFACES,
    SUPPORTED_ACTIONS,
    CommandAction,
)

class DAC3DUnavailableError(RuntimeError):
    """Raised when a live DAC-3D adapter is requested but unavailable."""


class DAC3DValidationError(RuntimeError):
    """Raised when a DAC-3D command is invalid for submission."""


class DAC3DClient:
    """Communicate with DAC-3D services or a local mock implementation."""

    def __init__(
        self,
        *,
        mock_mode: bool = True,
        endpoint: str = "mock://dac3d",
        runtime_bridge: Any | None = None,
    ) -> None:
        self.mock_mode = mock_mode
        self.endpoint = endpoint
        self.runtime_bridge = runtime_bridge
        self._last_command: dict[str, Any] | None = None
        self._status = {
            "state": "idle",
            "progress": 0,
            "message": "当前没有正在执行的 DAC-3D 检测任务。",
        }
        self._latest_result_summary = {
            "result_root": "mock://dac3d/results/latest",
            "files": ["defect_detail.csv", "defect_summary.csv"],
            "artifacts": ["mock_detect.jpg", "mock_fused_image.jpg"],
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
                "is_qualified": False,
                "is_ignored": False,
                "sample_quality": False,
                "sample_quality_label": "不合格",
                "region": "ROI",
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
        """Submit or simulate a structured DAC-3D command."""
        self._validate_command(command)
        if self.runtime_bridge is not None:
            return self._submit_via_runtime_bridge(command)
        if self.endpoint.startswith("file://"):
            return self._submit_via_file_bridge(command)

        if not self.mock_mode:
            raise DAC3DUnavailableError(
                "Live DAC-3D integration is not implemented in this workspace."
            )

        action = str(command["action"])
        self._last_command = deepcopy(command)
        if action == CommandAction.QUERY_STATUS:
            return {
                "accepted": True,
                "mode": "mock",
                "command": deepcopy(command),
                "status": self.query_current_status(),
            }
        if action == CommandAction.GET_LATEST_RESULT:
            return {
                "accepted": True,
                "mode": "mock",
                "command": deepcopy(command),
                "status": deepcopy(self._status),
                "result": self.get_latest_result_summary(),
            }
        if action == CommandAction.VALIDATE_OFFLINE_FOLDER:
            return {
                "accepted": True,
                "mode": "mock",
                "command": deepcopy(command),
                "status": deepcopy(self._status),
                "validation": self.validate_offline_folder(command),
            }
        if action == CommandAction.STOP_DETECTION:
            self._status = {
                "state": "stopped",
                "progress": self._status.get("progress", 0),
                "message": "Mock 检测任务已收到停止请求。",
            }
        elif action == CommandAction.START_OFFLINE_DETECTION:
            self._status = {
                "state": "queued",
                "progress": 0,
                "message": "Mock 离线检测任务已排队，将读取三相机图像组并生成结果。",
            }
        elif action == CommandAction.START_ONLINE_SCAN:
            self._status = {
                "state": "queued",
                "progress": 0,
                "message": "Mock 在线 144 点位扫描已排队，等待硬件确认。",
            }
        else:
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
        if self.runtime_bridge is not None:
            method = getattr(self.runtime_bridge, "query_current_status", None)
            if not callable(method):
                raise DAC3DUnavailableError("Runtime bridge does not expose query_current_status().")
            return deepcopy(method())

        file_status = self._read_status_file()
        if file_status is not None:
            return file_status

        if not self.mock_mode:
            raise DAC3DUnavailableError(
                "Live DAC-3D runtime status is not wired in this workspace."
            )

        status = deepcopy(self._status)
        if self._last_command and status["state"] == "queued":
            status["message"] = f"{status['message']} 当前处于 mock 预检阶段。"
        return status

    def get_recent_inspection_result(self) -> dict[str, Any]:
        """Return the latest inspection result payload."""
        if self.runtime_bridge is not None:
            method = getattr(self.runtime_bridge, "get_recent_inspection_result", None)
            if callable(method):
                return deepcopy(method())
            summary_method = getattr(self.runtime_bridge, "get_latest_result_summary", None)
            if callable(summary_method):
                summary = dict(summary_method() or {})
                parsed = summary.get("parsed_result")
                if isinstance(parsed, dict):
                    return deepcopy(parsed)
            raise DAC3DUnavailableError("Runtime bridge does not expose latest inspection result data.")

        file_result = self._read_result_summary_from_status_file()
        if file_result is not None:
            parsed = file_result.get("parsed_result")
            if isinstance(parsed, dict):
                parsed_result = deepcopy(parsed)
                self._enrich_parsed_result_from_summary(parsed_result, file_result)
                return parsed_result
            return deepcopy(file_result)
        if self.endpoint.startswith("file://"):
            raise DAC3DUnavailableError("DAC-3D 主系统尚未发布可解读的最新检测结果。")

        if not self.mock_mode:
            raise DAC3DUnavailableError(
                "Live DAC-3D result retrieval is not wired in this workspace."
            )
        return deepcopy(self._result_profiles[self._active_result_profile])

    def get_latest_result_summary(self) -> dict[str, Any]:
        """Return latest result-file metadata for command responses."""
        if self.runtime_bridge is not None:
            method = getattr(self.runtime_bridge, "get_latest_result_summary", None)
            if not callable(method):
                raise DAC3DUnavailableError("Runtime bridge does not expose get_latest_result_summary().")
            return deepcopy(method())

        file_result = self._read_result_summary_from_status_file()
        if file_result is not None:
            return file_result
        if self.endpoint.startswith("file://"):
            return {
                "result_root": "DAC-3D runtime status",
                "files": [],
                "message": "DAC-3D 主系统尚未发布最新检测结果；请先完成至少一个样品检测。",
                "source": "status_file",
            }

        if not self.mock_mode:
            raise DAC3DUnavailableError(
                "Live DAC-3D result retrieval is not wired in this workspace."
            )
        result = deepcopy(self._latest_result_summary)
        result["parsed_result"] = self.get_recent_inspection_result()
        return result

    def validate_offline_folder(self, command: dict[str, Any]) -> dict[str, Any]:
        """Validate the expected offline image-folder contract."""
        payload = dict(command.get("payload") or {})
        image_folder = payload.get("image_folder")
        required_cameras = list(payload.get("required_cameras") or DEFAULT_REQUIRED_CAMERAS)
        required_surfaces = list(payload.get("required_surfaces") or DEFAULT_REQUIRED_SURFACES)
        result = {
            "image_folder": image_folder,
            "required_cameras": required_cameras,
            "required_surfaces": required_surfaces,
            "exists": False,
            "checked_files": 0,
            "missing_requirements": [],
            "ready": False,
        }
        if not image_folder:
            result["missing_requirements"].append("image_folder")
            return result

        path = Path(str(image_folder))
        result["exists"] = path.exists()
        if not path.exists() or not path.is_dir():
            result["missing_requirements"].append("existing_directory")
            return result

        image_files = [
            file
            for file in path.rglob("*")
            if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ]
        result["checked_files"] = len(image_files)
        names = [file.name for file in image_files]
        for camera in required_cameras:
            if not any(camera in name for name in names):
                result["missing_requirements"].append(f"camera:{camera}")
        for surface in required_surfaces:
            if not any(surface.lower() in name.lower() for name in names):
                result["missing_requirements"].append(f"surface:{surface}")
        result["ready"] = not result["missing_requirements"]
        return result

    def runtime_snapshot(self) -> dict[str, Any]:
        """Return a UI-friendly runtime snapshot."""
        bridge_snapshot = None
        if self.runtime_bridge is not None:
            method = getattr(self.runtime_bridge, "runtime_snapshot", None)
            if callable(method):
                bridge_snapshot = deepcopy(method())
        mode = "embedded" if self.runtime_bridge is not None else ("status_file" if self.endpoint.startswith("file://") else ("mock" if self.mock_mode else "live"))
        return {
            "mode": mode,
            "endpoint": self.endpoint,
            "status": self.query_current_status(),
            "runtime_bridge": bridge_snapshot,
            "active_result_profile": self._active_result_profile,
            "available_result_profiles": sorted(self._result_profiles),
            "last_command_action": (self._last_command or {}).get("action"),
        }

    def execute(self, command: dict[str, Any]) -> dict[str, Any]:
        """Compatibility wrapper for the scaffold contract."""
        return self.submit_scan_command(command)

    def _validate_command(self, command: dict[str, Any]) -> None:
        action = command.get("action")
        if not action:
            raise DAC3DValidationError("Command must include an action.")
        if action not in SUPPORTED_ACTIONS:
            raise DAC3DValidationError(f"Unsupported DAC-3D command action: {action}.")
        if action == CommandAction.SCAN and not command.get("scan_area_mm"):
            raise DAC3DValidationError("Scan command must include scan_area_mm.")
        if action in {CommandAction.START_OFFLINE_DETECTION, CommandAction.VALIDATE_OFFLINE_FOLDER}:
            payload = dict(command.get("payload") or {})
            if not payload.get("image_folder"):
                raise DAC3DValidationError("Offline detection command must include payload.image_folder.")
        if action == CommandAction.START_ONLINE_SCAN:
            payload = dict(command.get("payload") or {})
            if payload.get("func") != "Scan":
                raise DAC3DValidationError("Online scan command must include payload.func='Scan'.")

    def _read_status_file(self) -> dict[str, Any] | None:
        """Read live DAC-3D status from a local file endpoint when configured."""
        if not self.endpoint.startswith("file://"):
            return None

        path = self._file_endpoint_path()
        if path is None:
            return {
                "state": "error",
                "progress": 0,
                "message": f"Invalid DAC-3D status file endpoint: {self.endpoint}",
                "source": "status_file",
            }
        if not path.exists():
            return {
                "state": "unknown",
                "progress": 0,
                "message": f"未读取到 DAC-3D 主系统状态文件: {path}",
                "source": "status_file",
            }

        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            return {
                "state": "error",
                "progress": 0,
                "message": f"DAC-3D 主系统状态文件读取失败: {exc}",
                "source": "status_file",
            }

        status = payload.get("status", payload)
        if not isinstance(status, dict):
            return {
                "state": "error",
                "progress": 0,
                "message": "DAC-3D 主系统状态文件格式无效。",
                "source": "status_file",
            }
        status = deepcopy(status)
        status.setdefault("state", "unknown")
        status.setdefault("progress", 0)
        status.setdefault("message", "已读取 DAC-3D 主系统状态。")
        status.setdefault("source", "status_file")
        return status

    def _read_latest_result_from_status_file(self) -> dict[str, Any] | None:
        """Read the latest DAC-3D sample result published by the host UI."""
        result = self._read_result_summary_from_status_file()
        if result is None:
            return None
        latest = result.get("latest_result")
        if isinstance(latest, dict):
            return deepcopy(latest)
        return result

    def _read_result_summary_from_status_file(self) -> dict[str, Any] | None:
        """Read latest and historical DAC-3D sample results from the host status file."""
        status = self._read_status_file()
        if not isinstance(status, dict):
            return None
        latest = status.get("latest_result")
        history = status.get("result_history")
        if not isinstance(history, list):
            history = []
        normalized_history = [deepcopy(item) for item in history if isinstance(item, dict)]
        if not isinstance(latest, dict) and normalized_history:
            latest = normalized_history[-1]
        if not isinstance(latest, dict) and not normalized_history:
            return None
        result = deepcopy(latest or {})
        if normalized_history:
            result["result_history"] = normalized_history
            result["checked_samples"] = len(normalized_history)
        result.setdefault("result_root", "DAC-3D runtime latest result")
        result.setdefault("files", [])
        result.setdefault("source", "status_file")
        return result

    def _enrich_parsed_result_from_summary(
        self,
        parsed_result: dict[str, Any],
        summary: dict[str, Any],
    ) -> None:
        """Carry sample and defect qualification fields into interpretation payloads."""
        parsed_result.setdefault("sample_quality", summary.get("quality"))
        parsed_result.setdefault("sample_quality_label", summary.get("quality_label"))
        parsed_result.setdefault("defects_num", summary.get("defects_num"))

        defects = summary.get("defects")
        if not isinstance(defects, list) or not defects:
            return

        first_defect = defects[0]
        if not isinstance(first_defect, dict):
            return

        field_map = {
            "is_qualified": "is_qualified",
            "is_ignored": "is_ignored",
            "region": "region",
            "reason": "defect_reason",
        }
        for source_key, target_key in field_map.items():
            if source_key in first_defect and target_key not in parsed_result:
                parsed_result[target_key] = first_defect[source_key]

        if "defect_type" not in parsed_result and "category" in first_defect:
            parsed_result["defect_type"] = first_defect["category"]
        if "confidence" not in parsed_result and "confidence" in first_defect:
            parsed_result["confidence"] = first_defect["confidence"]

    def _file_endpoint_path(self) -> Path | None:
        """Resolve a file:// endpoint to a local Windows path."""
        parsed = urlparse(self.endpoint)
        path_text = unquote(parsed.path or "")
        if parsed.netloc:
            path_text = f"//{parsed.netloc}{path_text}"
        if path_text.startswith("/") and len(path_text) >= 3 and path_text[2] == ":":
            path_text = path_text[1:]
        if not path_text:
            return None
        return Path(path_text)

    def _command_file_path(self) -> Path | None:
        """Return the local command bridge file used by the DAC-3D host UI."""
        configured = os.getenv("DAC3D_COMMAND_PATH", "").strip()
        if configured:
            return Path(configured)
        status_path = self._file_endpoint_path()
        if status_path is None:
            return None
        return status_path.with_name("dac3d_assistant_command.json")

    def _submit_via_file_bridge(self, command: dict[str, Any]) -> dict[str, Any]:
        """Write a command file that the DAC-3D host UI can poll and display."""
        command_path = self._command_file_path()
        if command_path is None:
            raise DAC3DUnavailableError("DAC-3D command bridge path is not configured.")

        action = str(command["action"])
        command_id = uuid.uuid4().hex
        self._last_command = deepcopy(command)
        payload = {
            "id": command_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "dac3d_iim_assistant",
            "status": "pending",
            "command": deepcopy(command),
        }
        command_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = command_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(command_path)

        status = {
            "state": "command_sent",
            "progress": 0,
            "message": f"已向 DAC-3D 主系统下发结构化命令: {action}",
            "step": "assistant_command_bridge",
            "source": "command_file_bridge",
            "command_id": command_id,
            "command_path": str(command_path),
        }
        return {
            "accepted": True,
            "mode": "command_file_bridge",
            "command": deepcopy(command),
            "status": status,
            "message": status["message"],
        }

    def _submit_via_runtime_bridge(self, command: dict[str, Any]) -> dict[str, Any]:
        """Delegate a validated command to the embedded DAC-3D host runtime."""
        action = str(command["action"])
        self._last_command = deepcopy(command)
        method_name_by_action = {
            CommandAction.SCAN: "submit_scan_command",
            CommandAction.START_ONLINE_SCAN: "start_online_scan",
            CommandAction.START_OFFLINE_DETECTION: "start_offline_detection",
            CommandAction.STOP_DETECTION: "stop_detection",
            CommandAction.QUERY_STATUS: "query_current_status",
            CommandAction.GET_LATEST_RESULT: "get_latest_result_summary",
            CommandAction.VALIDATE_OFFLINE_FOLDER: "validate_offline_folder",
        }
        method_name = method_name_by_action.get(action)
        method = getattr(self.runtime_bridge, method_name or "", None)
        if not callable(method):
            raise DAC3DUnavailableError(
                f"Runtime bridge does not expose {method_name}() for action {action}."
            )

        try:
            bridge_result = method(deepcopy(command))
        except TypeError:
            bridge_result = method()
        if not isinstance(bridge_result, dict):
            bridge_result = {"result": bridge_result}

        status = bridge_result.get("status")
        if status is None and action != CommandAction.QUERY_STATUS:
            status = self.query_current_status()
        if action == CommandAction.QUERY_STATUS:
            status = bridge_result if "state" in bridge_result else bridge_result.get("status", bridge_result)

        response = {
            "accepted": bool(bridge_result.get("accepted", True)),
            "mode": "embedded",
            "command": deepcopy(command),
        }
        if status is not None:
            response["status"] = deepcopy(status)
        if action == CommandAction.GET_LATEST_RESULT:
            response["result"] = deepcopy(bridge_result.get("result", bridge_result))
        if action == CommandAction.VALIDATE_OFFLINE_FOLDER:
            response["validation"] = deepcopy(bridge_result.get("validation", bridge_result))
        for key in ("message", "error"):
            if key in bridge_result:
                response[key] = bridge_result[key]
        return response
