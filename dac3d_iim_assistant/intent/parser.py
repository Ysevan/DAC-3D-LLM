"""Unified intent parser with a pluggable backend."""

from __future__ import annotations

import re
from typing import Callable, Protocol

from intent.schemas import ParsedCommand, ParsedIntent
from intent.validator import IntentValidator


AREA_PATTERNS = (
    re.compile(
        r"(?P<width>\d+(?:\.\d+)?)\s*(?:mm|毫米)\s*(?:x|×|\*)\s*(?P<height>\d+(?:\.\d+)?)\s*(?:mm|毫米)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<width>\d+(?:\.\d+)?)\s*(?:mm|毫米)\s*by\s*(?P<height>\d+(?:\.\d+)?)\s*(?:mm|毫米)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<width>\d+(?:\.\d+)?)\s*(?:mm|毫米)\s*(?:乘|乘以)\s*(?P<height>\d+(?:\.\d+)?)\s*(?:mm|毫米)",
        re.IGNORECASE,
    ),
)
RESOLUTION_PATTERNS = (
    re.compile(
        r"(?:resolution|分辨率|点间距)\s*(?:为|是|:)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>um|μm|µm|微米|mm|毫米)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>um|μm|µm|微米|mm|毫米)\s*(?:resolution|分辨率|点间距)",
        re.IGNORECASE,
    ),
)


class ParserBackend(Protocol):
    """Protocol for future LLM-backed structured intent parsers."""

    name: str

    def parse(self, text: str) -> ParsedIntent:
        """Return an unvalidated parsed intent."""


class LLMIntentBackend:
    """Placeholder backend for a future LLM structured parser."""

    name = "llm"

    def __init__(self, parser_callable: Callable[[str], ParsedIntent] | None = None) -> None:
        self._parser_callable = parser_callable

    def parse(self, text: str) -> ParsedIntent:
        if self._parser_callable is None:
            raise RuntimeError("LLM intent backend is not configured.")
        return self._parser_callable(text)


class FallbackIntentBackend:
    """Rule-based fallback backend that preserves current offline behavior."""

    name = "fallback"

    _status_keywords = (
        "current inspection status",
        "inspection status",
        "runtime status",
        "current status",
        "status now",
        "当前检测状态",
        "当前状态",
        "运行状态",
        "检测状态",
        "当前检测进度",
    )
    _guidance_keywords = (
        "what should i do",
        "how should i",
        "what do i do",
        "too reflective",
        "reflective",
        "glare",
        "unstable scan",
        "guidance",
        "suggest",
        "样品太反光",
        "反光",
        "怎么处理",
        "如何处理",
        "应该怎么办",
        "扫描不稳定",
        "预览不稳定",
        "夹具振动",
        "振动",
    )
    _interpret_keywords = (
        "defect serious",
        "is this defect",
        "severity",
        "interpret",
        "explain result",
        "serious defect",
        "缺陷严重",
        "严重吗",
        "结果解读",
        "结果判读",
        "解释结果",
    )
    _operation_keywords = (
        "scan",
        "inspect",
        "start scan",
        "run scan",
        "execute scan",
        "acquire",
        "扫描",
        "开始扫描",
        "执行扫描",
        "采集",
        "检测",
        "重新扫描",
    )
    _submit_keywords = (
        "execute",
        "submit",
        "run now",
        "start now",
        "立即执行",
        "马上扫描",
        "提交",
        "执行",
    )
    _question_markers = (
        "?",
        "？",
        "what",
        "how",
        "why",
        "when",
        "can i",
        "should i",
        "could i",
        "怎么",
        "如何",
        "能不能",
        "是否",
        "吗",
        "什么",
    )
    _guidance_markers = (
        "应该",
        "先做什么",
        "再做什么",
        "还能",
        "还要看什么",
        "能不能",
        "是否",
        "之前",
        "前",
        "怎么办",
        "如何",
        "建议",
    )
    _operation_prefixes = (
        "scan ",
        "scan a",
        "start scan",
        "run scan",
        "execute scan",
        "inspect ",
        "扫描",
        "开始扫描",
        "执行扫描",
        "采集",
        "检测",
        "重新扫描",
    )
    _parameter_keywords = {
        "resolution": ("resolution", "分辨率", "点间距"),
        "scan_area": ("scan area", "扫描区域", "扫描面积"),
        "mode": ("mode", "模式"),
        "region": ("region", "区域", "选区"),
        "status": ("status", "状态"),
        "parameter": ("parameter", "参数"),
    }

    def parse(self, text: str) -> ParsedIntent:
        raw_text = text.strip()
        normalized = raw_text.lower()
        command = self._extract_command(raw_text, normalized)

        hints: dict[str, object] = {
            "parser_backend": self.name,
            "contains_area": command.scan_area_mm is not None,
            "contains_resolution": command.resolution is not None,
            "submit_requested": any(keyword in normalized for keyword in self._submit_keywords),
        }
        parameter_name = self._extract_parameter_name(normalized)
        if parameter_name is not None:
            hints["parameter_name"] = parameter_name

        has_status_keyword = self._contains_any(normalized, self._status_keywords)
        has_guidance_keyword = self._contains_any(normalized, self._guidance_keywords)
        has_interpret_keyword = self._contains_any(normalized, self._interpret_keywords)
        has_operation_keyword = self._contains_any(normalized, self._operation_keywords)
        has_question_marker = self._contains_any(normalized, self._question_markers)
        guidance_question = self._looks_like_guidance_question(
            normalized,
            has_operation_keyword=has_operation_keyword,
            has_question_marker=has_question_marker,
        )
        interpretation_question = self._looks_like_interpretation(normalized, has_interpret_keyword)
        operation_request = self._looks_like_operation_request(
            normalized,
            has_question_marker=has_question_marker,
            has_operation_keyword=has_operation_keyword,
            guidance_question=guidance_question,
            parameter_name=parameter_name,
            command=command,
            submit_requested=bool(hints["submit_requested"]),
        )

        if has_status_keyword:
            hints["status_query"] = True
            return ParsedIntent(intent="status", confidence=0.99, hints=hints)

        if has_guidance_keyword or guidance_question:
            return ParsedIntent(intent="guidance", confidence=0.90, hints=hints)

        if interpretation_question:
            return ParsedIntent(intent="interpretation", confidence=0.92, hints=hints)

        if operation_request:
            return ParsedIntent(intent="operation", confidence=0.90, hints=hints, command=command)

        if parameter_name is not None and has_question_marker:
            return ParsedIntent(intent="query", confidence=0.82, hints=hints)

        if any(token in normalized for token in ("manual", "faq", "parameter", "参数", "手册")):
            return ParsedIntent(intent="query", confidence=0.75, hints=hints)

        if has_question_marker:
            return ParsedIntent(intent="query", confidence=0.72, hints=hints)

        return ParsedIntent(intent="query", confidence=0.60, hints=hints)

    def _extract_command(self, raw_text: str, normalized: str) -> ParsedCommand:
        return ParsedCommand(
            action=self._extract_action(normalized),
            scan_area_mm=self._extract_scan_area(raw_text),
            resolution=self._extract_resolution(raw_text),
            region=self._extract_region(normalized),
            mode=self._extract_mode(normalized),
        )

    def _extract_action(self, normalized: str) -> str | None:
        if self._contains_any(normalized, self._operation_keywords):
            return "scan"
        return None

    def _extract_scan_area(self, text: str) -> dict[str, float] | None:
        for pattern in AREA_PATTERNS:
            match = pattern.search(text)
            if match:
                return {
                    "width": float(match.group("width")),
                    "height": float(match.group("height")),
                }
        return None

    def _extract_resolution(self, text: str) -> dict[str, float | str] | None:
        for pattern in RESOLUTION_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            value = float(match.group("value"))
            unit = match.group("unit").lower()
            if unit in {"mm", "毫米"}:
                value *= 1000.0
            return {"value": round(value, 4), "unit": "um"}
        return None

    def _extract_region(self, normalized: str) -> str | None:
        mapping = {
            "top_left": ("top left", "upper left", "左上"),
            "top_right": ("top right", "upper right", "右上"),
            "bottom_left": ("bottom left", "lower left", "左下"),
            "bottom_right": ("bottom right", "lower right", "右下"),
            "center": ("center", "centre", "中央", "中心"),
            "current_selection": ("current selection", "current region", "当前选区", "当前区域"),
        }
        for region_name, keywords in mapping.items():
            if self._contains_any(normalized, keywords):
                return region_name
        return None

    def _extract_mode(self, normalized: str) -> str | None:
        if self._contains_any(normalized, ("precision", "high detail", "精细", "高精度", "高细节")):
            return "precision"
        if self._contains_any(normalized, ("fast", "preview", "quick", "快速", "预览")):
            return "fast"
        if self._contains_any(normalized, ("standard", "normal", "标准", "常规")):
            return "standard"
        return None

    def _extract_parameter_name(self, normalized: str) -> str | None:
        for parameter_name, keywords in self._parameter_keywords.items():
            if self._contains_any(normalized, keywords):
                return parameter_name
        return None

    def _looks_like_guidance_question(
        self,
        normalized: str,
        *,
        has_operation_keyword: bool,
        has_question_marker: bool,
    ) -> bool:
        if not has_question_marker:
            return False
        if self._contains_any(normalized, self._guidance_keywords):
            return True
        if self._contains_any(normalized, ("what should i do", "how should i", "怎么办", "如何处理")):
            return True
        return has_operation_keyword and self._contains_any(normalized, self._guidance_markers)

    def _looks_like_interpretation(self, normalized: str, has_interpret_keyword: bool) -> bool:
        if has_interpret_keyword:
            return True
        defect_terms = ("pit", "scratch", "defect", "缺陷", "划痕", "点蚀")
        severity_terms = ("serious", "severity", "严重", "阈值")
        return self._contains_any(normalized, defect_terms) and self._contains_any(normalized, severity_terms)

    def _looks_like_operation_request(
        self,
        normalized: str,
        *,
        has_question_marker: bool,
        has_operation_keyword: bool,
        guidance_question: bool,
        parameter_name: str | None,
        command: ParsedCommand,
        submit_requested: bool,
    ) -> bool:
        if guidance_question:
            return False
        if parameter_name is not None and has_question_marker and not has_operation_keyword:
            return False
        if submit_requested:
            return True
        starts_with_operation = any(normalized.startswith(prefix) for prefix in self._operation_prefixes)
        has_command_fields = any(
            value is not None
            for value in (
                command.scan_area_mm,
                command.resolution,
                command.region,
                command.mode,
            )
        )
        if starts_with_operation and not has_question_marker:
            return True
        if has_operation_keyword and has_command_fields and not has_question_marker:
            return True
        if has_operation_keyword and not has_question_marker:
            return True
        if command.scan_area_mm is not None and self._contains_any(normalized, ("扫描", "scan", "区域", "area")):
            return not has_question_marker
        return False

    def _contains_any(self, normalized: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in normalized for keyword in keywords)


class IntentParser:
    """Parse user text into a unified structured intent."""

    def __init__(
        self,
        *,
        backend: ParserBackend | None = None,
        validator: IntentValidator | None = None,
    ) -> None:
        self.backend = backend or FallbackIntentBackend()
        self.validator = validator or IntentValidator()

    def parse(self, text: str) -> ParsedIntent:
        """Return a validated unified parse result."""
        parsed = self.backend.parse(text)
        return self.validator.validate(parsed)
