"""Data-access layer for the industrial machine Agent prototype."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from machine_agent.models import AlarmRecord, DocumentChunk, MachineHistoryPoint, MachineStatus


IMAGELESS_SUFFIXES = {".md", ".txt"}


class MachineDataProvider(Protocol):
    """Unified data interface for real or mock industrial systems."""

    def get_current_machine_status(self) -> MachineStatus:
        """Return the latest machine status."""

    def get_machine_history(self, start: datetime, end: datetime) -> list[MachineHistoryPoint]:
        """Return historical telemetry for the requested time range."""

    def get_alarm_records(self, start: datetime, end: datetime) -> list[AlarmRecord]:
        """Return alarm and abnormal-event records for the requested time range."""

    def search_machine_docs(self, query: str, limit: int = 4) -> list[DocumentChunk]:
        """Search machine manuals, maintenance documents, and alarm-code notes."""


class MockMachineDataProvider:
    """Deterministic seed data for demoing the industrial machine Agent."""

    def __init__(
        self,
        *,
        now: datetime | None = None,
        documents_dir: Path | None = None,
    ) -> None:
        self.now = (now or datetime.now()).replace(microsecond=0)
        self.machine_id = "IM-Press-01"
        self.machine_name = "智能液压压装单元 A"
        self.documents_dir = documents_dir or Path(__file__).resolve().parent / "documents"
        self._history = self._build_history()
        self._alarms = self._build_alarm_records()
        self._documents = self._load_documents()

    def get_current_machine_status(self) -> MachineStatus:
        latest = self._history[-1]
        active_alarms = [
            alarm.code
            for alarm in self._alarms
            if alarm.status != "closed" and self.now - alarm.timestamp <= timedelta(hours=24)
        ][:3]
        return MachineStatus(
            machine_id=self.machine_id,
            machine_name=self.machine_name,
            timestamp=self.now,
            state="running" if not active_alarms else "warning",
            temperature_c=latest.temperature_c,
            pressure_mpa=latest.pressure_mpa,
            rpm=latest.rpm,
            current_a=latest.current_a,
            output_count=latest.output_count,
            utilization_pct=86.5 if not active_alarms else 78.2,
            active_alarm_codes=active_alarms,
        )

    def get_machine_history(self, start: datetime, end: datetime) -> list[MachineHistoryPoint]:
        return [point for point in self._history if start <= point.timestamp <= end]

    def get_alarm_records(self, start: datetime, end: datetime) -> list[AlarmRecord]:
        return [alarm for alarm in self._alarms if start <= alarm.timestamp <= end]

    def search_machine_docs(self, query: str, limit: int = 4) -> list[DocumentChunk]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        ranked: list[DocumentChunk] = []
        for chunk in self._documents:
            haystack = f"{chunk.title} {chunk.section} {chunk.text}".lower()
            exact_bonus = 1.5 if query.lower() in haystack else 0.0
            code_bonus = 2.5 if _extract_error_code(query) and _extract_error_code(query).lower() in haystack else 0.0
            overlap = sum(1 for token in query_tokens if token in haystack)
            if overlap == 0 and not exact_bonus and not code_bonus:
                continue
            ranked.append(
                DocumentChunk(
                    source=chunk.source,
                    title=chunk.title,
                    section=chunk.section,
                    text=chunk.text,
                    score=round(overlap + exact_bonus + code_bonus, 4),
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[: max(1, limit)]

    def _build_history(self) -> list[MachineHistoryPoint]:
        start = self.now - timedelta(days=120)
        points: list[MachineHistoryPoint] = []
        cumulative_output = 128_000
        for index in range(120 * 4 + 1):
            timestamp = start + timedelta(hours=6 * index)
            day_age = (self.now - timestamp).days
            hour = timestamp.hour
            shift_factor = 1.0 if 8 <= hour <= 20 else 0.72
            weekly_wave = math.sin(index / 4.8)
            recent_heat_bias = 12.0 if day_age <= 21 and 10 <= hour <= 20 else 0.0
            monthly_heat_bias = 4.5 if day_age <= 55 and timestamp.weekday() in {1, 2, 3} else 0.0
            temperature = 71.0 + weekly_wave * 3.2 + recent_heat_bias + monthly_heat_bias
            pressure = 0.62 + math.sin(index / 7.0) * 0.035
            rpm = int(1450 + math.cos(index / 5.0) * 80 * shift_factor)
            current = 32.0 + math.sin(index / 6.5) * 2.2 + (recent_heat_bias * 0.22)
            state = "running" if shift_factor > 0.8 else "standby"
            if temperature >= 84.0 or current >= 37.0:
                state = "warning"
            cumulative_output += int(max(0, rpm * shift_factor / 115))
            points.append(
                MachineHistoryPoint(
                    timestamp=timestamp,
                    state=state,
                    temperature_c=round(temperature, 2),
                    pressure_mpa=round(pressure, 3),
                    rpm=rpm,
                    current_a=round(current, 2),
                    output_count=cumulative_output,
                )
            )
        return points

    def _build_alarm_records(self) -> list[AlarmRecord]:
        alarms: list[AlarmRecord] = []
        specs = [
            ("E102", "temperature_high", "high", "温度超过 85C，冷却效率下降或负载偏高。", "temperature_c"),
            ("E201", "pressure_drop", "medium", "油压低于工艺下限，可能存在泄压或滤芯堵塞。", "pressure_mpa"),
            ("E305", "over_current", "high", "电机电流高于安全阈值，可能由过载或轴承阻力增加引起。", "current_a"),
            ("E410", "vibration_warning", "medium", "振动趋势升高，需要检查固定、轴承和联轴器。", "rpm"),
        ]
        event_days = [92, 84, 76, 63, 51, 45, 32, 28, 24, 20, 17, 14, 12, 10, 8, 6, 4, 2, 1]
        for index, day_age in enumerate(event_days):
            code, alarm_type, severity, message, metric = specs[index % len(specs)]
            if day_age <= 21 and index % 2 == 0:
                code, alarm_type, severity, message, metric = specs[0]
            timestamp = (self.now - timedelta(days=day_age)).replace(hour=14 if code == "E102" else 9)
            matched_point = min(self._history, key=lambda point: abs(point.timestamp - timestamp))
            metric_value = float(getattr(matched_point, metric))
            alarms.append(
                AlarmRecord(
                    alarm_id=f"ALM-{timestamp.strftime('%Y%m%d')}-{index + 1:03d}",
                    timestamp=timestamp,
                    code=code,
                    alarm_type=alarm_type,
                    severity=severity,
                    status="open" if day_age <= 2 else ("acknowledged" if day_age <= 8 else "closed"),
                    message=message,
                    related_metric=metric,
                    metric_value=round(metric_value, 3),
                )
            )
        return sorted(alarms, key=lambda alarm: alarm.timestamp)

    def _load_documents(self) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        if self.documents_dir.exists():
            for path in sorted(self.documents_dir.glob("*")):
                if path.suffix.lower() not in IMAGELESS_SUFFIXES or not path.is_file():
                    continue
                chunks.extend(_split_markdown_document(path))
        if chunks:
            return chunks
        return [
            DocumentChunk(
                source="builtin_fault_codes",
                title="常见故障代码说明",
                section="E102 温度报警",
                text="E102 表示主轴或液压单元温度高。优先检查冷却水流量、散热风扇、滤网堵塞、环境温度和近期负载变化。",
            )
        ]


def _split_markdown_document(path: Path) -> list[DocumentChunk]:
    title = path.stem
    section = title
    buffer: list[str] = []
    chunks: list[DocumentChunk] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            if buffer:
                chunks.append(
                    DocumentChunk(
                        source=path.name,
                        title=title,
                        section=section,
                        text="\n".join(buffer).strip(),
                    )
                )
                buffer = []
            heading = line.lstrip("#").strip()
            if line.startswith("# "):
                title = heading
            section = heading or section
            continue
        if line:
            buffer.append(line)
    if buffer:
        chunks.append(
            DocumentChunk(
                source=path.name,
                title=title,
                section=section,
                text="\n".join(buffer).strip(),
            )
        )
    return chunks


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z]\d{2,4}|[A-Za-z_]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]{2,}", text)
    ]


def _extract_error_code(text: str) -> str:
    match = re.search(r"\b[A-Z]\d{3,4}\b", text.upper())
    return match.group(0) if match else ""


def alarm_type_counts(alarms: list[AlarmRecord]) -> Counter[str]:
    """Return alarm type counts for UI summaries and tests."""
    return Counter(alarm.alarm_type for alarm in alarms)
