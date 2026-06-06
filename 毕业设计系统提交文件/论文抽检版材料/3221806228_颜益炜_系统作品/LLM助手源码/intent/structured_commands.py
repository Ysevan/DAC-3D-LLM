"""Central DAC-3D structured-command contract.

This module is the single source of truth for assistant-facing DAC-3D
commands. Intent parsing can fill partial fields, validation normalizes them,
and integration adapters can rely on the same action names and payload schema.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


class CommandAction:
    """Supported DAC-3D command action names."""

    SCAN = "scan"
    START_OFFLINE_DETECTION = "start_offline_detection"
    START_ONLINE_SCAN = "start_online_scan"
    STOP_DETECTION = "stop_detection"
    QUERY_STATUS = "query_status"
    GET_LATEST_RESULT = "get_latest_result"
    VALIDATE_OFFLINE_FOLDER = "validate_offline_folder"


SUPPORTED_ACTIONS = {
    CommandAction.SCAN,
    CommandAction.START_OFFLINE_DETECTION,
    CommandAction.START_ONLINE_SCAN,
    CommandAction.STOP_DETECTION,
    CommandAction.QUERY_STATUS,
    CommandAction.GET_LATEST_RESULT,
    CommandAction.VALIDATE_OFFLINE_FOLDER,
}
OFFLINE_FOLDER_ACTIONS = {
    CommandAction.START_OFFLINE_DETECTION,
    CommandAction.VALIDATE_OFFLINE_FOLDER,
}
READ_ONLY_ACTIONS = {
    CommandAction.QUERY_STATUS,
    CommandAction.GET_LATEST_RESULT,
    CommandAction.VALIDATE_OFFLINE_FOLDER,
}

DEFAULT_REQUIRED_CAMERAS = ["\u7126\u524d", "\u7126\u9762", "\u7126\u540e"]
DEFAULT_REQUIRED_SURFACES = ["surface1", "surface2"]
DEFAULT_RESULT_FILES = ["defect_detail.csv", "defect_summary.csv"]


@dataclass(slots=True)
class CommandContract:
    """Default payload and safety contract for one command action."""

    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    required_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


COMMAND_CONTRACTS: dict[str, CommandContract] = {
    CommandAction.SCAN: CommandContract(
        action=CommandAction.SCAN,
        safety={
            "needs_confirmation": True,
            "hardware_required": False,
            "safe_to_auto_execute": False,
        },
        required_fields=["scan_area_mm"],
    ),
    CommandAction.START_OFFLINE_DETECTION: CommandContract(
        action=CommandAction.START_OFFLINE_DETECTION,
        payload={
            "image_folder": None,
            "validate_before_run": True,
            "required_cameras": DEFAULT_REQUIRED_CAMERAS,
            "required_surfaces": DEFAULT_REQUIRED_SURFACES,
            "message_type": "offline_detect_folder",
        },
        safety={
            "needs_confirmation": True,
            "hardware_required": False,
            "safe_to_auto_execute": False,
        },
        required_fields=["payload.image_folder"],
        warnings=[
            "离线检测会读取本地图像并生成结果文件，执行前应先通过目录完整性校验。"
        ],
    ),
    CommandAction.START_ONLINE_SCAN: CommandContract(
        action=CommandAction.START_ONLINE_SCAN,
        payload={
            "func": "Scan",
            "plate_layout": "12x12",
            "total_positions": 144,
            "scan_path": "serpentine",
            "require_hardware": True,
        },
        safety={
            "needs_confirmation": True,
            "hardware_required": True,
            "safe_to_auto_execute": False,
        },
        warnings=[
            "在线扫描会驱动相机、光源、运动控制和 144 点位流程，必须人工确认硬件状态。"
        ],
    ),
    CommandAction.STOP_DETECTION: CommandContract(
        action=CommandAction.STOP_DETECTION,
        payload={
            "func": "Stop",
            "reason": "operator_requested",
        },
        safety={
            "needs_confirmation": True,
            "hardware_required": False,
            "safe_to_auto_execute": False,
        },
        warnings=[
            "停止检测可能发生在运动或拍照阶段，实际停止时机取决于 DAC-3D 当前执行阶段。"
        ],
    ),
    CommandAction.QUERY_STATUS: CommandContract(
        action=CommandAction.QUERY_STATUS,
        payload={
            "source": "dac3d_runtime",
            "include_progress": True,
            "include_message": True,
        },
        safety={
            "needs_confirmation": False,
            "hardware_required": False,
            "safe_to_auto_execute": True,
        },
    ),
    CommandAction.GET_LATEST_RESULT: CommandContract(
        action=CommandAction.GET_LATEST_RESULT,
        payload={
            "result_root": None,
            "files": DEFAULT_RESULT_FILES,
            "include_artifacts": True,
        },
        safety={
            "needs_confirmation": False,
            "hardware_required": False,
            "safe_to_auto_execute": True,
        },
    ),
    CommandAction.VALIDATE_OFFLINE_FOLDER: CommandContract(
        action=CommandAction.VALIDATE_OFFLINE_FOLDER,
        payload={
            "image_folder": None,
            "validate_before_run": True,
            "required_cameras": DEFAULT_REQUIRED_CAMERAS,
            "required_surfaces": DEFAULT_REQUIRED_SURFACES,
            "message_type": "offline_detect_folder",
        },
        safety={
            "needs_confirmation": False,
            "hardware_required": False,
            "safe_to_auto_execute": True,
        },
        required_fields=["payload.image_folder"],
    ),
}


def get_command_contract(action: str | None) -> CommandContract | None:
    """Return the contract for an action, if supported."""
    if action is None:
        return None
    return COMMAND_CONTRACTS.get(action)


def default_payload(action: str | None) -> dict[str, Any]:
    """Return a deep copy of the action's default payload."""
    contract = get_command_contract(action)
    return deepcopy(contract.payload) if contract else {}


def default_safety(action: str | None) -> dict[str, Any]:
    """Return a deep copy of the action's default safety policy."""
    contract = get_command_contract(action)
    return deepcopy(contract.safety) if contract else {
        "needs_confirmation": True,
        "hardware_required": False,
        "safe_to_auto_execute": False,
    }


def contract_warnings(action: str | None) -> list[str]:
    """Return default warnings for an action."""
    contract = get_command_contract(action)
    return list(contract.warnings) if contract else []


def missing_required_fields(action: str | None, command: dict[str, Any]) -> list[str]:
    """Return missing required field paths for the command dictionary."""
    contract = get_command_contract(action)
    if contract is None:
        return ["action"]
    missing: list[str] = []
    for field_path in contract.required_fields:
        if _lookup_field(command, field_path) in (None, "", []):
            missing.append(field_path)
    return missing


def _lookup_field(command: dict[str, Any], field_path: str) -> Any:
    current: Any = command
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current
