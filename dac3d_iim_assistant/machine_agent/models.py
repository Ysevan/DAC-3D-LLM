"""Stable schemas for the industrial machine Agent prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


def _iso(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat()


@dataclass(slots=True)
class MachineStatus:
    """Current machine state exposed to the Agent and UI."""

    machine_id: str
    machine_name: str
    timestamp: datetime
    state: str
    temperature_c: float
    pressure_mpa: float
    rpm: int
    current_a: float
    output_count: int
    utilization_pct: float
    active_alarm_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = _iso(self.timestamp)
        return payload


@dataclass(slots=True)
class MachineHistoryPoint:
    """One sampled historical telemetry record."""

    timestamp: datetime
    state: str
    temperature_c: float
    pressure_mpa: float
    rpm: int
    current_a: float
    output_count: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = _iso(self.timestamp)
        return payload


@dataclass(slots=True)
class AlarmRecord:
    """One alarm or abnormal-event record."""

    alarm_id: str
    timestamp: datetime
    code: str
    alarm_type: str
    severity: str
    status: str
    message: str
    related_metric: str
    metric_value: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = _iso(self.timestamp)
        return payload


@dataclass(slots=True)
class DocumentChunk:
    """Searchable machine-document chunk."""

    source: str
    title: str
    section: str
    text: str
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TimeRange:
    """Resolved time range from a user request."""

    start: datetime
    end: datetime
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": _iso(self.start),
            "end": _iso(self.end),
            "label": self.label,
        }


@dataclass(slots=True)
class ToolCallRecord:
    """One Agent tool invocation for transparent UI display."""

    name: str
    arguments: dict[str, Any]
    reason: str
    result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
