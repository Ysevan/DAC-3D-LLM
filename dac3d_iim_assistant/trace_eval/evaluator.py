"""Deterministic local eval runner for DAC-Agent Runtime."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EvalCase:
    """One local eval case loaded from JSON."""

    id: str
    category: str
    input: str
    expected: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalCase":
        return cls(
            id=str(payload.get("id") or ""),
            category=str(payload.get("category") or "uncategorized"),
            input=str(payload.get("input") or ""),
            expected=dict(payload.get("expected") or {}),
        )


class EvalRunner:
    """Run deterministic assertions against the local Agent chat surface."""

    def __init__(
        self,
        *,
        cases_dir: str | Path,
        chat_handler: Callable[[str, str], Any],
        approve_handler: Callable[[str, str | None, str | None], Any] | None = None,
    ) -> None:
        self.cases_dir = Path(cases_dir)
        self.chat_handler = chat_handler
        self.approve_handler = approve_handler

    def load_cases(self) -> list[EvalCase]:
        """Load eval cases from JSON files."""
        cases_dir = self.cases_dir
        if not cases_dir.exists():
            bundled_dir = Path(__file__).resolve().parents[1] / "evals" / "cases"
            cases_dir = bundled_dir
        if not cases_dir.exists():
            return []
        cases: list[EvalCase] = []
        for path in sorted(cases_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                cases.extend(EvalCase.from_dict(item) for item in payload if isinstance(item, dict))
            elif isinstance(payload, dict):
                cases.append(EvalCase.from_dict(payload))
        return [case for case in cases if case.id and case.input]

    def run(self, *, categories: list[str] | None = None) -> dict[str, Any]:
        """Run all matching eval cases."""
        selected_categories = {str(item) for item in categories or [] if str(item).strip()}
        cases = [
            case
            for case in self.load_cases()
            if not selected_categories or case.category in selected_categories
        ]
        results = [self._run_case(case) for case in cases]
        passed = sum(1 for result in results if result["passed"])
        return {
            "backend": "local_deterministic_eval",
            "cases_dir": str(self.cases_dir),
            "case_count": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(passed / len(results), 4) if results else 0.0,
            "results": results,
        }

    def _run_case(self, case: EvalCase) -> dict[str, Any]:
        session_id = f"eval-{case.id}"
        response = self.chat_handler(case.input, session_id)
        response_payload = self._response_to_payload(response)
        checks = self._grade_payload(case.expected, response_payload, prefix="response")

        approval_payload: dict[str, Any] | None = None
        if case.expected.get("approve_after_preview"):
            approval_payload = self._approve_preview(session_id, response_payload)
            checks.extend(self._grade_payload(dict(case.expected.get("after_approval") or {}), approval_payload, prefix="approval"))

        passed = all(check["passed"] for check in checks)
        return {
            "id": case.id,
            "category": case.category,
            "input": case.input,
            "passed": passed,
            "checks": checks,
            "intent": response_payload.get("intent"),
            "answer_preview": str(response_payload.get("answer") or "")[:180],
            "trace_id": self._trace_id(response_payload),
            "approval_trace_id": self._trace_id(approval_payload or {}),
        }

    def _approve_preview(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.approve_handler is None:
            return {
                "intent": "error",
                "answer": "Approval handler is not configured.",
                "parsed_result": {"eval_error": "missing_approve_handler"},
            }
        command_preview = self._record(payload.get("command_preview"))
        gateway = self._record(command_preview.get("gateway"))
        return self._response_to_payload(
            self.approve_handler(
                session_id,
                self._string_or_none(gateway.get("preview_id")),
                self._string_or_none(gateway.get("confirmation_token")),
            )
        )

    def _grade_payload(self, expected: dict[str, Any], payload: dict[str, Any], *, prefix: str) -> list[dict[str, Any]]:
        checks: list[dict[str, Any]] = []

        def add(name: str, passed: bool, expected_value: Any, actual_value: Any) -> None:
            checks.append(
                {
                    "name": f"{prefix}.{name}",
                    "passed": bool(passed),
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )

        if "intent" in expected:
            add("intent", payload.get("intent") == expected["intent"], expected["intent"], payload.get("intent"))
        if expected.get("answer_contains"):
            answer = str(payload.get("answer") or "")
            for phrase in expected["answer_contains"]:
                add(f"answer_contains:{phrase}", str(phrase) in answer, phrase, answer[:240])
        if expected.get("must_call_tools"):
            called = self._tool_names(payload)
            for tool_name in expected["must_call_tools"]:
                add(f"must_call:{tool_name}", tool_name in called, tool_name, called)
        if expected.get("must_not_call_tools"):
            called = self._tool_names(payload)
            for tool_name in expected["must_not_call_tools"]:
                add(f"must_not_call:{tool_name}", tool_name not in called, tool_name, called)
        if "command_action" in expected:
            action = self._record(payload.get("command_preview")).get("action")
            add("command_action", action == expected["command_action"], expected["command_action"], action)
        if "requires_confirmation" in expected:
            actual = self._requires_confirmation(payload)
            add("requires_confirmation", actual is expected["requires_confirmation"], expected["requires_confirmation"], actual)
        if "can_submit" in expected:
            actual = self._can_submit(payload)
            add("can_submit", actual is expected["can_submit"], expected["can_submit"], actual)
        if "risk_level" in expected:
            actual = self._risk_level(payload)
            add("risk_level", actual == expected["risk_level"], expected["risk_level"], actual)
        if "memory_patch_proposed" in expected:
            actual = self._memory_patch_count(payload) > 0
            add("memory_patch_proposed", actual is expected["memory_patch_proposed"], expected["memory_patch_proposed"], actual)
        if expected.get("blocked_by_contains"):
            blocked_by = self._blocked_by(payload)
            for item in expected["blocked_by_contains"]:
                add(f"blocked_by_contains:{item}", item in blocked_by, item, blocked_by)
        return checks

    def _response_to_payload(self, response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            return dict(response)
        to_ui_payload = getattr(response, "to_ui_payload", None)
        if callable(to_ui_payload):
            payload = to_ui_payload()
            return dict(payload) if isinstance(payload, dict) else {}
        return {
            "intent": str(getattr(response, "intent", "")),
            "answer": str(getattr(response, "answer", "")),
            "command_preview": getattr(response, "command_preview", None),
            "status_summary": getattr(response, "status_summary", None),
            "parsed_result": getattr(response, "parsed_result", None),
        }

    def _tool_names(self, payload: dict[str, Any]) -> list[str]:
        parsed = self._record(payload.get("parsed_result"))
        calls = parsed.get("tool_calls")
        if not isinstance(calls, list):
            return []
        return [str(call.get("name") or "") for call in calls if isinstance(call, dict)]

    def _requires_confirmation(self, payload: dict[str, Any]) -> bool:
        preview = self._record(payload.get("command_preview"))
        gateway = self._record(preview.get("gateway"))
        safety = self._record(preview.get("safety"))
        if "confirmation_required" in gateway:
            return gateway.get("confirmation_required") is True
        return safety.get("needs_confirmation") is True

    def _can_submit(self, payload: dict[str, Any]) -> bool | None:
        tool_gateway = self._tool_gateway(payload)
        validation = self._record(tool_gateway.get("validation"))
        if "can_submit" in validation:
            return validation.get("can_submit") is True
        preview = self._record(payload.get("command_preview"))
        gateway = self._record(preview.get("gateway"))
        return True if gateway.get("preview_id") and self._requires_confirmation(payload) else None

    def _risk_level(self, payload: dict[str, Any]) -> str:
        tool_gateway = self._tool_gateway(payload)
        risk = self._record(tool_gateway.get("risk"))
        return str(risk.get("risk_level") or "")

    def _blocked_by(self, payload: dict[str, Any]) -> list[str]:
        tool_gateway = self._tool_gateway(payload)
        risk = self._record(tool_gateway.get("risk"))
        values = risk.get("blocked_by")
        return [str(item) for item in values] if isinstance(values, list) else []

    def _tool_gateway(self, payload: dict[str, Any]) -> dict[str, Any]:
        parsed = self._record(payload.get("parsed_result"))
        return self._record(parsed.get("tool_gateway"))

    def _memory_patch_count(self, payload: dict[str, Any]) -> int:
        parsed = self._record(payload.get("parsed_result"))
        memory_os = self._record(parsed.get("memory_os"))
        return int(memory_os.get("proposed_patch_count") or 0)

    def _trace_id(self, payload: dict[str, Any]) -> str | None:
        parsed = self._record(payload.get("parsed_result"))
        memory_os = self._record(parsed.get("memory_os"))
        trace_eval = self._record(parsed.get("trace_eval"))
        return self._string_or_none(trace_eval.get("trace_id") or memory_os.get("trace_id") or parsed.get("trace_id"))

    def _record(self, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _string_or_none(self, value: Any) -> str | None:
        return str(value) if value not in (None, "") else None
