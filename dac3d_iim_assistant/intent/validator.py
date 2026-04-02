"""Safety validation and normalization for parsed intents."""

from __future__ import annotations

from dataclasses import replace

from intent.schemas import ParsedCommand, ParsedIntent


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

        if validated.intent != "operation":
            validated.missing_fields = self._dedupe(validated.missing_fields)
            validated.warnings = self._dedupe(validated.warnings)
            return validated

        command = validated.command or ParsedCommand(action="scan")
        if not command.action:
            command.action = "scan"

        self._validate_scan_area(command, validated)
        self._validate_resolution(command, validated)

        if not command.region:
            command.region = "current_selection"
            validated.warnings.append(
                "未指定扫描区域绑定对象，预览默认绑定到 current_selection，请在执行前确认当前选区。"
            )

        if not command.mode:
            command.mode = "standard"
            validated.warnings.append("未指定扫描模式，预览默认使用 standard。")

        if command.scan_area_mm is not None:
            area = command.scan_area_mm["width"] * command.scan_area_mm["height"]
            if area > 400.0:
                validated.warnings.append("扫描面积较大，可能显著增加扫描时间和数据量。")

        validated.command = command
        validated.missing_fields = self._dedupe(validated.missing_fields)
        validated.warnings = self._dedupe(validated.warnings)
        validated.needs_clarification = bool(validated.missing_fields)
        if validated.needs_clarification:
            validated.clarification_question = self._build_clarification_question(validated.missing_fields)
        else:
            validated.clarification_question = None
        return validated

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
            parsed.warnings.append("分辨率必须为正数，当前值已被忽略。")
            parsed.warnings.append("未指定分辨率，将由操作员在执行前确认。")
            return

        command.resolution = {
            "value": round(value, 4),
            "unit": "um",
        }

    def _build_clarification_question(self, missing_fields: list[str]) -> str:
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
        )

    def _dedupe(self, values: list[str]) -> list[str]:
        unique: list[str] = []
        for value in values:
            if value not in unique:
                unique.append(value)
        return unique
