"""DAC-3D result parsing and normalization."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


MEASUREMENT_PATTERNS = {
    "depth_um": re.compile(r"(?:depth|深度)\s*(?:=|为|是)?\s*(?P<value>\d+(?:\.\d+)?)\s*um", re.IGNORECASE),
    "length_mm": re.compile(r"(?:length|长度)\s*(?:=|为|是)?\s*(?P<value>\d+(?:\.\d+)?)\s*mm", re.IGNORECASE),
}
GENERIC_UM_PATTERN = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*um", re.IGNORECASE)
GENERIC_MM_PATTERN = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)


@dataclass(slots=True)
class ParsedInspectionResult:
    """Stable assistant-facing inspection result schema."""

    defect_type: str
    severity: str
    confidence: float
    location: str
    measurements: dict[str, float]
    trigger_measurement: str | None
    threshold_reference: str | None
    rule_reason: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        return asdict(self)


def parse_result(payload: dict[str, Any]) -> ParsedInspectionResult:
    """Normalize a DAC-3D response payload."""
    defect_type = str(payload.get("defect_type", "unknown")).lower()
    location = str(payload.get("location", "unknown"))
    confidence = float(payload.get("confidence", 0.0))
    measurements = {
        key: float(value)
        for key, value in dict(payload.get("measurements", {})).items()
        if isinstance(value, (int, float))
    }

    derived = _derive_severity_metadata(defect_type, measurements)
    severity = str(payload.get("severity") or derived["severity"])
    trigger_measurement = str(derived["trigger_measurement"]) if derived["trigger_measurement"] else None
    threshold_reference = (
        str(derived["threshold_reference"]) if derived["threshold_reference"] else None
    )
    rule_reason = str(payload.get("rule_reason") or derived["rule_reason"])
    summary = _build_summary(
        defect_type=defect_type,
        severity=severity,
        location=location,
        measurements=measurements,
        rule_reason=rule_reason,
    )
    return ParsedInspectionResult(
        defect_type=defect_type,
        severity=severity,
        confidence=confidence,
        location=location,
        measurements=measurements,
        trigger_measurement=trigger_measurement,
        threshold_reference=threshold_reference,
        rule_reason=rule_reason,
        summary=summary,
    )


def parse_result_from_text(text: str) -> ParsedInspectionResult | None:
    """Parse a user-supplied defect description into the stable result schema."""
    normalized = text.strip().lower()
    defect_type = _extract_defect_type(normalized)
    if defect_type is None:
        return None

    measurements: dict[str, float] = {}
    for key, pattern in MEASUREMENT_PATTERNS.items():
        match = pattern.search(normalized)
        if match:
            measurements[key] = float(match.group("value"))

    if defect_type == "pit" and "depth_um" not in measurements:
        match = GENERIC_UM_PATTERN.search(normalized)
        if match:
            measurements["depth_um"] = float(match.group("value"))

    if defect_type == "scratch":
        if "depth_um" not in measurements:
            match = GENERIC_UM_PATTERN.search(normalized)
            if match:
                measurements["depth_um"] = float(match.group("value"))
        if "length_mm" not in measurements:
            match = GENERIC_MM_PATTERN.search(normalized)
            if match:
                measurements["length_mm"] = float(match.group("value"))

    if not measurements:
        return None

    return parse_result(
        {
            "defect_type": defect_type,
            "location": "user_supplied",
            "confidence": 1.0,
            "measurements": measurements,
        }
    )


def _extract_defect_type(normalized: str) -> str | None:
    if any(keyword in normalized for keyword in ("scratch", "划痕")):
        return "scratch"
    if any(keyword in normalized for keyword in ("pit", "点蚀")):
        return "pit"
    return None


def _derive_severity_metadata(defect_type: str, measurements: dict[str, float]) -> dict[str, str | None]:
    if defect_type == "scratch":
        depth = measurements.get("depth_um", 0.0)
        length = measurements.get("length_mm", 0.0)
        if depth > 15:
            return {
                "severity": "high",
                "trigger_measurement": "depth_um",
                "threshold_reference": "scratch depth > 15 um => high severity",
                "rule_reason": f"depth_um={depth:.2f} 超过 15 um 阈值。",
            }
        if length > 0.50:
            return {
                "severity": "high",
                "trigger_measurement": "length_mm",
                "threshold_reference": "scratch length > 0.50 mm => high severity",
                "rule_reason": f"length_mm={length:.2f} 超过 0.50 mm 阈值。",
            }
        if depth >= 8:
            return {
                "severity": "medium",
                "trigger_measurement": "depth_um",
                "threshold_reference": "scratch depth >= 8 um => medium severity",
                "rule_reason": f"depth_um={depth:.2f} 落在 8-15 um 区间。",
            }
        if length >= 0.20:
            return {
                "severity": "medium",
                "trigger_measurement": "length_mm",
                "threshold_reference": "scratch length >= 0.20 mm => medium severity",
                "rule_reason": f"length_mm={length:.2f} 落在 0.20-0.50 mm 区间。",
            }
        return {
            "severity": "low",
            "trigger_measurement": None,
            "threshold_reference": "scratch below medium thresholds => low severity",
            "rule_reason": "scratch 未命中中高严重度阈值。",
        }

    if defect_type == "pit":
        depth = measurements.get("depth_um", 0.0)
        if depth > 10:
            return {
                "severity": "high",
                "trigger_measurement": "depth_um",
                "threshold_reference": "pit depth > 10 um => high severity",
                "rule_reason": f"depth_um={depth:.2f} 超过 10 um 阈值。",
            }
        if depth >= 5:
            return {
                "severity": "medium",
                "trigger_measurement": "depth_um",
                "threshold_reference": "pit depth >= 5 um => medium severity",
                "rule_reason": f"depth_um={depth:.2f} 落在 5-10 um 区间。",
            }
        return {
            "severity": "low",
            "trigger_measurement": None,
            "threshold_reference": "pit below medium thresholds => low severity",
            "rule_reason": "pit 未命中中高严重度阈值。",
        }

    return {
        "severity": "medium",
        "trigger_measurement": None,
        "threshold_reference": None,
        "rule_reason": "缺少专用阈值规则，按默认中等严重度输出。",
    }


def _build_summary(
    defect_type: str,
    severity: str,
    location: str,
    measurements: dict[str, float],
    rule_reason: str,
) -> str:
    measurement_text = ", ".join(f"{key}={value}" for key, value in measurements.items())
    return (
        f"检测到的 {defect_type} 位于 {location}，当前判定为 {severity} 严重度。"
        f"测量值: {measurement_text or '无可用测量值'}。判定依据: {rule_reason}"
    )
