"""Structured command generation compatibility layer for DAC-3D requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from intent.parser import IntentParser
from intent.schemas import ParsedCommand


@dataclass(slots=True)
class StructuredCommand:
    """Normalized DAC-3D command preview."""

    action: str
    scan_area_mm: dict[str, float] | None
    resolution: dict[str, float | str] | None
    region: str | None
    mode: str | None
    payload: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    yaml_preview: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the command preview to a serializable mapping."""
        return {
            "action": self.action,
            "scan_area_mm": self.scan_area_mm,
            "resolution": self.resolution,
            "region": self.region,
            "mode": self.mode,
            "payload": self.payload,
            "safety": self.safety,
            "missing_fields": self.missing_fields,
            "warnings": self.warnings,
            "yaml_preview": self.yaml_preview,
        }


class CommandGenerator:
    """Convert natural language into a structured DAC-3D command preview."""

    def __init__(self, parser: IntentParser | None = None) -> None:
        self.parser = parser or IntentParser()

    def generate(self, text: str) -> StructuredCommand:
        """Return a validated structured command representation."""
        parsed = self.parser.parse(text)
        command = parsed.command or ParsedCommand(action="scan")
        structured = StructuredCommand(
            action=command.action or "scan",
            scan_area_mm=command.scan_area_mm,
            resolution=command.resolution,
            region=command.region,
            mode=command.mode,
            payload=dict(command.payload),
            safety=dict(command.safety),
            missing_fields=list(parsed.missing_fields),
            warnings=list(parsed.warnings),
        )
        structured.yaml_preview = self._render_yaml(structured)
        return structured

    def _render_yaml(self, command: StructuredCommand) -> str:
        area = command.scan_area_mm or {}
        resolution = command.resolution or {}
        lines = [
            f"action: {command.action}",
            "scan_area_mm:",
            f"  width: {area.get('width', 'null')}",
            f"  height: {area.get('height', 'null')}",
            "resolution:",
            f"  value: {resolution.get('value', 'null')}",
            f"  unit: {resolution.get('unit', 'null')}",
            f"region: {command.region or 'null'}",
            f"mode: {command.mode or 'null'}",
            "payload:",
        ]
        if command.payload:
            lines.extend(self._render_yaml_mapping(command.payload, indent=2))
        else:
            lines.append("  {}")
        lines.append("safety:")
        if command.safety:
            lines.extend(self._render_yaml_mapping(command.safety, indent=2))
        else:
            lines.append("  {}")
        lines.append("missing_fields:")
        if command.missing_fields:
            lines.extend(f"  - {field_name}" for field_name in command.missing_fields)
        else:
            lines.append("  []")
        lines.append("warnings:")
        if command.warnings:
            lines.extend(f"  - {warning}" for warning in command.warnings)
        else:
            lines.append("  []")
        return "\n".join(lines)

    def _render_yaml_mapping(self, value: dict[str, Any], *, indent: int) -> list[str]:
        spaces = " " * indent
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, dict):
                lines.append(f"{spaces}{key}:")
                lines.extend(self._render_yaml_mapping(item, indent=indent + 2))
            elif isinstance(item, list):
                lines.append(f"{spaces}{key}:")
                if item:
                    lines.extend(f"{spaces}  - {entry}" for entry in item)
                else:
                    lines.append(f"{spaces}  []")
            else:
                rendered = "null" if item is None else item
                lines.append(f"{spaces}{key}: {rendered}")
        return lines
