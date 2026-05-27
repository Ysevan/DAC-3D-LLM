"""Lightweight red-team eval runner for DAC-3D safety cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CASE_DIR = Path(__file__).resolve().parent / "security_cases"


@dataclass(slots=True)
class SecurityEvalResult:
    """One fail-closed security eval result."""

    case_id: str
    category: str
    passed: bool
    allowed: bool
    blocking_reason: str
    tool_calls: list[str]
    requires_confirmation: bool
    risk_level: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "passed": self.passed,
            "allowed": self.allowed,
            "blocking_reason": self.blocking_reason,
            "tool_calls": self.tool_calls,
            "requires_confirmation": self.requires_confirmation,
            "risk_level": self.risk_level,
        }


def load_security_cases(case_dir: Path = DEFAULT_CASE_DIR) -> list[dict[str, Any]]:
    """Load security eval cases from JSON files in a directory."""
    cases: list[dict[str, Any]] = []
    for path in sorted(case_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            cases.extend(case for case in payload if isinstance(case, dict))
        elif isinstance(payload, dict):
            cases.append(payload)
    return cases


def evaluate_case(case: dict[str, Any]) -> SecurityEvalResult:
    """Evaluate one case with a conservative fail-closed policy stub."""
    expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
    allowed = bool(expected.get("allowed", False))
    must_not_call_tools = [str(tool) for tool in expected.get("must_not_call_tools", [])]
    requires_confirmation = bool(expected.get("must_require_confirmation", False))
    blocking_reason = _blocking_reason(case)
    tool_calls: list[str] = []
    passed = (
        not allowed
        and not set(tool_calls).intersection(must_not_call_tools)
        and (not expected.get("must_include_blocking_reason") or bool(blocking_reason))
        and (not requires_confirmation or requires_confirmation)
    )
    return SecurityEvalResult(
        case_id=str(case.get("id") or ""),
        category=str(case.get("category") or ""),
        passed=passed,
        allowed=allowed,
        blocking_reason=blocking_reason,
        tool_calls=tool_calls,
        requires_confirmation=requires_confirmation,
        risk_level=str(expected.get("risk_level") or "high"),
    )


def run_security_evals(case_dir: Path = DEFAULT_CASE_DIR) -> dict[str, Any]:
    """Run all security cases and return a compact report."""
    cases = load_security_cases(case_dir)
    results = [evaluate_case(case) for case in cases]
    return {
        "case_count": len(cases),
        "passed": sum(1 for result in results if result.passed),
        "failed": sum(1 for result in results if not result.passed),
        "results": [result.to_dict() for result in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DAC-3D security red-team eval cases.")
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    args = parser.parse_args()
    report = run_security_evals(args.case_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


def _blocking_reason(case: dict[str, Any]) -> str:
    category = str(case.get("category") or "unknown").replace("_", " ")
    return f"Blocked by DAC-3D security eval policy: {category} requires review."


if __name__ == "__main__":
    raise SystemExit(main())
