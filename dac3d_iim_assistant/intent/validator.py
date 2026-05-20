"""Safety validation and normalization for parsed intents."""

from __future__ import annotations

from dataclasses import replace

from intent.schemas import ParsedCommand, ParsedIntent
from intent.structured_commands import (
    OFFLINE_FOLDER_ACTIONS,
    SUPPORTED_ACTIONS,
    CommandAction,
    contract_warnings,
    default_payload,
    default_safety,
)


class IntentValidator:
    """Apply DAC-3D safety defaults and clarification rules."""

    def validate(self, parsed: ParsedIntent) -> ParsedIntent:
        """Return a validated copy of the parsed intent."""
        command = self._copy_command(parsed.command)
        validated = replace(
            parsed,
            hints=dict(parsed.hints),
            command=command,
            missing_fields=list(parsed.missing_fields),
            warnings=list(parsed.warnings),
        )

        if command is None:
            validated.missing_fields = self._dedupe(validated.missing_fields)
            validated.warnings = self._dedupe(validated.warnings)
            return validated

        if not command.action:
            command.action = CommandAction.SCAN
        if command.action not in SUPPORTED_ACTIONS:
            validated.warnings.append(f"Unsupported DAC-3D command action: {command.action}.")

        self._merge_contract_defaults(command)
        self._validate_by_action(command, validated)
        validated.command = command
        validated.missing_fields = self._dedupe(validated.missing_fields)
        validated.warnings = self._dedupe(validated.warnings)
        validated.needs_clarification = bool(validated.missing_fields)
        validated.clarification_question = (
            self._build_clarification_question(command.action, validated.missing_fields)
            if validated.needs_clarification
            else None
        )
        return validated

    def _validate_by_action(self, command: ParsedCommand, parsed: ParsedIntent) -> None:
        if command.action == CommandAction.SCAN:
            self._validate_scan_area(command, parsed)
            self._validate_resolution(command, parsed)
            if not command.region:
                command.region = "current_selection"
                parsed.warnings.append(
                    "未指定扫描区域绑定对象，预览默认绑定到 current_selection，执行前必须确认当前选区。"
                )
            if not command.mode:
                command.mode = "standard"
                parsed.warnings.append("未指定扫描模式，预览默认使用 standard。")
            if command.scan_area_mm is not None:
                area = command.scan_area_mm["width"] * command.scan_area_mm["height"]
                if area > 400.0:
                    parsed.warnings.append("扫描面积较大，可能显著增加扫描时间和数据量。")
            return

        if command.action in OFFLINE_FOLDER_ACTIONS:
            image_folder = command.payload.get("image_folder")
            if not image_folder:
                parsed.missing_fields.append("payload.image_folder")
            command.payload.setdefault("validate_before_run", True)
            command.payload.setdefault("required_cameras", ["焦前", "焦面", "焦后"])
            command.payload.setdefault("required_surfaces", ["surface1", "surface2"])
            command.payload.setdefault("message_type", "offline_detect_folder")
            parsed.warnings.extend(contract_warnings(command.action))
            return

        if command.action == CommandAction.START_ONLINE_SCAN:
            parsed.warnings.extend(contract_warnings(command.action))
            return

        if command.action == CommandAction.STOP_DETECTION:
            parsed.warnings.extend(contract_warnings(command.action))
            return

        if command.action == CommandAction.QUERY_STATUS:
            return

        if command.action == CommandAction.GET_LATEST_RESULT:
            return

    def _merge_contract_defaults(self, command: ParsedCommand) -> None:
        payload = default_payload(command.action)
        payload.update({key: value for key, value in command.payload.items() if value is not None})
        command.payload = payload
        safety = default_safety(command.action)
        safety.update({key: value for key, value in command.safety.items() if value is not None})
        command.safety = safety

    def _validate_scan_area(self, command: ParsedCommand, parsed: ParsedIntent) -> None:
        area = command.scan_area_mm
        if area is None:
            parsed.missing_fields.append("scan_area_mm")
            return

        width = float(area.get("width", 0.0))
        height = float(area.get("height", 0.0))
        if width <= 0.0 or height <= 0.0:
            command.scan_area_mm = None
            parsed.missing_fields.append("scan_area_mm")
            parsed.warnings.append("扫描区域尺寸必须为正数。")
            return

        command.scan_area_mm = {
            "width": round(width, 4),
            "height": round(height, 4),
        }

    def _validate_resolution(self, command: ParsedCommand, parsed: ParsedIntent) -> None:
        resolution = command.resolution
        if resolution is None:
            parsed.warnings.append("未指定分辨率，将由操作员在执行前确认。")
            return

        try:
            value = float(resolution.get("value", 0.0))
        except (TypeError, ValueError):
            value = 0.0

        if value <= 0.0:
            command.resolution = None
            parsed.warnings.append("分辨率必须为正数，当前值已忽略。")
            parsed.warnings.append("未指定分辨率，将由操作员在执行前确认。")
            return

        command.resolution = {
            "value": round(value, 4),
            "unit": "um",
        }

    def _build_clarification_question(self, action: str | None, missing_fields: list[str]) -> str:
        if "payload.image_folder" in missing_fields:
            return "请补充离线检测图片目录，例如 C:\\path\\to\\pre_fusion_images。"
        if "scan_area_mm" in missing_fields:
            return "请补充扫描区域尺寸，例如 10mm x 10mm。"
        missing = "、".join(missing_fields)
        return f"请补充以下信息后再生成命令：{missing}。"

    def _copy_command(self, command: ParsedCommand | None) -> ParsedCommand | None:
        if command is None:
            return None
        return ParsedCommand(
            action=command.action,
            scan_area_mm=None if command.scan_area_mm is None else dict(command.scan_area_mm),
            resolution=None if command.resolution is None else dict(command.resolution),
            region=command.region,
            mode=command.mode,
            payload=dict(command.payload),
            safety=dict(command.safety),
        )

    def _dedupe(self, values: list[str]) -> list[str]:
        unique: list[str] = []
        for value in values:
            if value not in unique:
                unique.append(value)
        return unique
