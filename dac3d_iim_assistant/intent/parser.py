"""Unified intent parser with DAC-3D command extraction."""

from __future__ import annotations

import re
from pathlib import PureWindowsPath
from typing import Callable, Protocol

from intent.schemas import ParsedCommand, ParsedIntent
from intent.structured_commands import CommandAction, default_payload, default_safety
from intent.validator import IntentValidator


AREA_PATTERNS = (
    re.compile(
        r"(?P<width>\d+(?:\.\d+)?)\s*(?:mm|毫米)\s*(?:x|×|\*|by|乘|乘以)\s*"
        r"(?P<height>\d+(?:\.\d+)?)\s*(?:mm|毫米)",
        re.IGNORECASE,
    ),
)
RESOLUTION_PATTERNS = (
    re.compile(
        r"(?:resolution|分辨率|步长|步距|点间距)\s*(?:为|是|=|:|：)?\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>um|μm|微米|mm|毫米)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>um|μm|微米|mm|毫米)\s*"
        r"(?:resolution|分辨率|步长|步距|点间距)",
        re.IGNORECASE,
    ),
)
WINDOWS_PATH_PATTERN = re.compile(
    r"(?P<path>[A-Za-z]:\\[^\r\n\"'<>|]*)"
)
QUOTED_PATH_PATTERN = re.compile(r"[\"'“”](?P<path>[A-Za-z]:\\[^\"'“”]+)[\"'“”]")


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
    """Rule-based fallback backend that preserves offline behavior."""

    name = "fallback"

    _status_keywords = (
        "当前系统",
        "系统状态",
        "系统信息",
        "所有信息",
        "现在运行状态",
        "当前运行状态",
        "当前检测状态",
        "运行状态",
        "检测状态",
        "实时状态",
        "当前状态",
        "当前进度",
        "运行进度",
        "检测进度",
        "当前系统在做什么",
        "系统在做什么",
        "现在在做什么",
        "正在做什么",
        "在做什么",
        "当前任务",
        "正在运行",
        "正在执行",
        "正在检测",
        "运行到哪一步",
        "执行到哪一步",
        "检测到哪一步",
        "做到哪一步",
        "哪一步",
        "当前步骤",
        "执行步骤",
        "运行步骤",
        "status",
        "current inspection",
        "query status",
        "当前状态",
        "检测状态",
        "运行状态",
        "进度",
        "鐘",
        "杩涘害",
    )
    _result_keywords = (
        "latest result",
        "recent result",
        "inspection result",
        "result csv",
        "defect_detail",
        "defect_summary",
        "最新结果",
        "最近结果",
        "检测结果",
        "结果文件",
        "缺陷明细",
        "缂",
    )
    _stop_keywords = (
        "stop",
        "cancel",
        "abort",
        "停止",
        "中止",
        "取消检测",
        "停止检测",
    )
    _validate_folder_keywords = (
        "validate offline folder",
        "check offline folder",
        "validate folder",
        "check image folder",
        "校验离线目录",
        "检查离线目录",
        "校验图片目录",
        "检查图片目录",
        "验证离线目录",
    )
    _offline_keywords = (
        "offline detection",
        "offline detect",
        "offline test",
        "offline_detect_folder",
        "离线",
        "离线检测",
        "离线测试",
        "本地检测",
        "离线图片",
        "离线目录",
        "原图目录",
        "图片目录",
        "pre_fusion_images",
    )
    _online_keywords = (
        "online scan",
        "online detection",
        "start online",
        "144",
        "plate scan",
        "在线检测",
        "在线扫描",
        "整盘检测",
        "整盘扫描",
        "144点",
        "144 点",
    )
    _guidance_keywords = (
        "what should i do",
        "how should i",
        "too reflective",
        "reflective",
        "glare",
        "unstable scan",
        "guidance",
        "suggest",
        "反光",
        "怎么处理",
        "如何处理",
        "应该怎么办",
        "扫描不稳定",
        "预览不稳定",
        "鍙嶅厜",
        "鎬庝箞",
    )
    _interpret_keywords = (
        "defect serious",
        "is this defect",
        "severity",
        "interpret",
        "serious defect",
        "缺陷严重",
        "严重吗",
        "结果解读",
        "缂",
    )
    _operation_keywords = (
        "scan",
        "inspect",
        "start scan",
        "run scan",
        "execute scan",
        "acquire",
        "选择",
        "测试",
        "离线测试",
        "扫描",
        "开始扫描",
        "执行扫描",
        "采集",
        "检测",
        "重新扫描",
        "鎵",
        "妫",
    )
    _submit_keywords = (
        "execute",
        "submit",
        "run now",
        "start now",
        "执行",
        "提交",
        "立即开始",
        "马上扫描",
    )
    _question_markers = (
        "?",
        "？",
        "what",
        "how",
        "why",
        "can i",
        "should i",
        "怎么",
        "如何",
        "能不能",
        "是否",
        "吗",
        "什么",
        "浠",
    )

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
        if command.action:
            hints["command_action"] = command.action

        if command.action == "scan" and self._looks_like_scan_policy_question(normalized, command):
            return ParsedIntent(intent="guidance", confidence=0.88, hints=hints)

        if command.action in {
            CommandAction.START_OFFLINE_DETECTION,
            CommandAction.START_ONLINE_SCAN,
            CommandAction.STOP_DETECTION,
            CommandAction.GET_LATEST_RESULT,
            CommandAction.VALIDATE_OFFLINE_FOLDER,
        }:
            return ParsedIntent(intent="operation", confidence=0.94, hints=hints, command=command)

        if self._contains_any(normalized, self._status_keywords):
            status_command = ParsedCommand(
                action=CommandAction.QUERY_STATUS,
                payload=default_payload(CommandAction.QUERY_STATUS),
                safety=default_safety(CommandAction.QUERY_STATUS),
            )
            hints["command_action"] = CommandAction.QUERY_STATUS
            return ParsedIntent(intent="status", confidence=0.99, hints=hints, command=status_command)

        if self._looks_like_guidance(normalized):
            return ParsedIntent(intent="guidance", confidence=0.90, hints=hints)

        if self._looks_like_interpretation(normalized):
            return ParsedIntent(intent="interpretation", confidence=0.92, hints=hints)

        if self._looks_like_operation_request(normalized, command):
            return ParsedIntent(intent="operation", confidence=0.90, hints=hints, command=command)

        if self._contains_any(normalized, self._question_markers):
            return ParsedIntent(intent="query", confidence=0.72, hints=hints)

        return ParsedIntent(intent="query", confidence=0.60, hints=hints)

    def _extract_command(self, raw_text: str, normalized: str) -> ParsedCommand:
        action = self._extract_action(normalized)
        payload = self._extract_payload(action, raw_text, normalized)
        safety = self._default_safety(action)
        return ParsedCommand(
            action=action,
            scan_area_mm=self._extract_scan_area(raw_text),
            resolution=self._extract_resolution(raw_text),
            region=self._extract_region(normalized),
            mode=self._extract_mode(normalized),
            payload=payload,
            safety=safety,
        )

    def _extract_action(self, normalized: str) -> str | None:
        if self._contains_any(normalized, self._stop_keywords):
            return CommandAction.STOP_DETECTION
        if self._contains_any(normalized, self._validate_folder_keywords):
            return CommandAction.VALIDATE_OFFLINE_FOLDER
        if self._contains_any(normalized, self._result_keywords) and not self._contains_any(
            normalized, self._interpret_keywords
        ):
            return CommandAction.GET_LATEST_RESULT
        if self._contains_any(normalized, self._offline_keywords):
            return CommandAction.START_OFFLINE_DETECTION
        if self._contains_any(normalized, self._online_keywords):
            return CommandAction.START_ONLINE_SCAN
        if self._contains_any(normalized, self._status_keywords):
            return CommandAction.QUERY_STATUS
        if self._contains_any(normalized, self._operation_keywords):
            return CommandAction.SCAN
        return None

    def _extract_payload(
        self,
        action: str | None,
        raw_text: str,
        normalized: str,
    ) -> dict[str, object]:
        payload = default_payload(action)
        if action in {CommandAction.START_OFFLINE_DETECTION, CommandAction.VALIDATE_OFFLINE_FOLDER}:
            image_folder = self._extract_windows_path(raw_text)
            payload["image_folder"] = image_folder
            return payload
        if action == CommandAction.GET_LATEST_RESULT:
            payload["result_root"] = self._extract_windows_path(raw_text)
            sample_position = self._extract_sample_position(raw_text, normalized)
            if sample_position is not None:
                payload["sample_position"] = sample_position
            return payload
        if payload:
            return payload
        if "144" in normalized:
            return {"total_positions": 144}
        return {}

    def _extract_sample_position(self, raw_text: str, normalized: str) -> int | None:
        patterns = (
            r"(?:第\s*)?(?P<num>\d{1,3})\s*(?:个|号)?\s*(?:样品|产品|工件|片)",
            r"(?:样品|产品|工件|片)\s*(?:第\s*)?(?P<num>\d{1,3})\s*(?:个|号)?",
        )
        for pattern in patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                return int(match.group("num"))

        chinese_digits = {
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }
        match = re.search(r"第\s*(?P<num>[一二两三四五六七八九十]{1,3})\s*(?:个|号)?\s*(?:样品|产品|工件|片)", raw_text)
        if not match:
            match = re.search(r"(?:样品|产品|工件|片)\s*第\s*(?P<num>[一二两三四五六七八九十]{1,3})", raw_text)
        if match:
            text = match.group("num")
            if text in chinese_digits:
                return chinese_digits[text]
            if len(text) == 2 and text[0] == "十" and text[1] in chinese_digits:
                return 10 + chinese_digits[text[1]]
            if len(text) == 2 and text[1] == "十" and text[0] in chinese_digits:
                return chinese_digits[text[0]] * 10
            if len(text) == 3 and text[1] == "十" and text[0] in chinese_digits and text[2] in chinese_digits:
                return chinese_digits[text[0]] * 10 + chinese_digits[text[2]]
        return None

    def _default_safety(self, action: str | None) -> dict[str, object]:
        return default_safety(action)

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
            "top_left": ("top left", "upper left", "左上", "宸︿笂"),
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
        if self._contains_any(normalized, ("precision", "high detail", "精细", "高精度", "绮")):
            return "precision"
        if self._contains_any(normalized, ("fast", "preview", "quick", "快速", "预览")):
            return "fast"
        if self._contains_any(normalized, ("standard", "normal", "标准", "常规")):
            return "standard"
        return None

    def _extract_windows_path(self, text: str) -> str | None:
        quoted = QUOTED_PATH_PATTERN.search(text)
        match = quoted or WINDOWS_PATH_PATTERN.search(text)
        if not match:
            return None
        path = match.group("path").rstrip("。；;，,")
        for marker in (
            "下的图片",
            "下图片",
            "中的图片",
            "里面的图片",
            "进行",
            "执行",
            "开始",
            "用于",
            "做",
            "测试",
            "检测",
            "，",
            ",",
            "。",
            "；",
            ";",
        ):
            if marker in path:
                path = path.split(marker, 1)[0].rstrip()
        for trailing_keyword in (" execute", " submit", " run now", " start now"):
            if path.lower().endswith(trailing_keyword):
                path = path[: -len(trailing_keyword)].rstrip()
        try:
            return str(PureWindowsPath(path))
        except Exception:
            return path

    def _looks_like_guidance(self, normalized: str) -> bool:
        if not self._contains_any(normalized, self._question_markers):
            return False
        return self._contains_any(normalized, self._guidance_keywords)

    def _looks_like_scan_policy_question(self, normalized: str, command: ParsedCommand) -> bool:
        if not self._contains_any(normalized, self._question_markers):
            return False
        policy_terms = (
            "能不能",
            "是否",
            "应该",
            "需要",
            "直接",
            "扩大",
            "整片",
            "范围",
            "之前",
            "前",
            "还要",
            "看什么",
            "can i",
            "should i",
        )
        has_specific_command_fields = any(
            value is not None
            for value in (
                command.scan_area_mm,
                command.resolution,
                command.region,
                command.mode,
            )
        )
        return not has_specific_command_fields and self._contains_any(normalized, policy_terms)

    def _looks_like_interpretation(self, normalized: str) -> bool:
        if self._contains_any(normalized, self._interpret_keywords):
            defect_terms = ("pit", "scratch", "defect", "缺陷", "划痕", "点蚀", "缂")
            severity_terms = ("serious", "severity", "严重", "阈值", "闃")
            return self._contains_any(normalized, defect_terms) or self._contains_any(
                normalized, severity_terms
            )
        return False

    def _looks_like_operation_request(self, normalized: str, command: ParsedCommand) -> bool:
        if command.action:
            if self._contains_any(normalized, self._question_markers) and command.action == "scan":
                return False
            return True
        has_command_fields = any(
            value is not None
            for value in (
                command.scan_area_mm,
                command.resolution,
                command.region,
                command.mode,
            )
        )
        return has_command_fields and self._contains_any(normalized, self._operation_keywords)

    def _contains_any(self, normalized: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword.lower() in normalized for keyword in keywords)


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
