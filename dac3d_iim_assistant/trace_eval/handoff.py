"""Generate a Codex-ready handoff from eval results and Agent traces."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trace_eval.tracing import TraceLogger


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clip(value: Any, limit: int = 360) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


class CodexHandoffGenerator:
    """Turn failing evals and nearby traces into a reviewable Codex handoff."""

    def __init__(self, *, trace_logger: TraceLogger, output_path: str | Path) -> None:
        self.trace_logger = trace_logger
        self.output_path = Path(output_path)

    def generate(
        self,
        *,
        eval_result: dict[str, Any],
        recent_trace_limit: int = 8,
    ) -> dict[str, Any]:
        """Write the next Codex handoff markdown and return a compact summary."""
        generated_at = _utc_now_iso()
        failed_evals = self._failed_eval_summaries(eval_result)
        traces = self._collect_trace_summaries(failed_evals, recent_trace_limit=recent_trace_limit)
        recommendations = self._recommendations(failed_evals)
        content = self._render_markdown(
            generated_at=generated_at,
            eval_result=eval_result,
            failed_evals=failed_evals,
            traces=traces,
            recommendations=recommendations,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(content, encoding="utf-8")
        return {
            "enabled": True,
            "backend": "codex_handoff_generator",
            "path": str(self.output_path),
            "generated_at": generated_at,
            "failed_count": len(failed_evals),
            "trace_count": len(traces),
            "recommendations": recommendations,
            "failed_evals": failed_evals,
            "eval_summary": {
                "backend": eval_result.get("backend"),
                "case_count": int(eval_result.get("case_count") or 0),
                "passed": int(eval_result.get("passed") or 0),
                "failed": int(eval_result.get("failed") or 0),
                "pass_rate": float(eval_result.get("pass_rate") or 0.0),
            },
            "workflow": "failing evals + traces -> Codex handoff -> next implementation pass",
            "auto_applied": False,
        }

    def _failed_eval_summaries(self, eval_result: dict[str, Any]) -> list[dict[str, Any]]:
        failed: list[dict[str, Any]] = []
        for result in eval_result.get("results", []):
            if not isinstance(result, dict) or result.get("passed") is True:
                continue
            checks = [
                {
                    "name": str(check.get("name") or ""),
                    "expected": check.get("expected"),
                    "actual": check.get("actual"),
                }
                for check in result.get("checks", [])
                if isinstance(check, dict) and check.get("passed") is not True
            ]
            failed.append(
                {
                    "id": str(result.get("id") or ""),
                    "category": str(result.get("category") or ""),
                    "input": str(result.get("input") or ""),
                    "intent": str(result.get("intent") or ""),
                    "trace_id": result.get("trace_id"),
                    "approval_trace_id": result.get("approval_trace_id"),
                    "answer_preview": str(result.get("answer_preview") or ""),
                    "failed_checks": checks,
                }
            )
        return failed

    def _collect_trace_summaries(
        self,
        failed_evals: list[dict[str, Any]],
        *,
        recent_trace_limit: int,
    ) -> list[dict[str, Any]]:
        trace_ids: list[str] = []
        for result in failed_evals:
            for key in ("trace_id", "approval_trace_id"):
                trace_id = str(result.get(key) or "").strip()
                if trace_id and trace_id not in trace_ids:
                    trace_ids.append(trace_id)

        collected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for trace_id in trace_ids:
            trace = self.trace_logger.get(trace_id)
            if trace is not None:
                summary = self._trace_summary(trace)
                collected.append(summary)
                seen.add(str(summary.get("trace_id") or ""))

        safe_limit = max(1, min(50, int(recent_trace_limit or 8)))
        for trace in self.trace_logger.iter_recent(limit=safe_limit):
            summary = self._trace_summary(trace)
            trace_id = str(summary.get("trace_id") or "")
            if trace_id and trace_id not in seen:
                collected.append(summary)
                seen.add(trace_id)
        return collected

    def _trace_summary(self, trace: dict[str, Any]) -> dict[str, Any]:
        tool_calls = [
            str(item.get("name") or "")
            for item in trace.get("tool_calls", [])
            if isinstance(item, dict) and item.get("name")
        ]
        return {
            "trace_id": str(trace.get("trace_id") or ""),
            "timestamp": str(trace.get("timestamp") or ""),
            "event_type": str(trace.get("event_type") or ""),
            "session_id": str(trace.get("session_id") or ""),
            "intent": str(trace.get("intent") or ""),
            "skill": trace.get("skill"),
            "tools": list(dict.fromkeys(tool_calls)),
            "user_message": _clip(trace.get("user_message"), 240),
            "final_response": _clip(trace.get("final_response"), 240),
        }

    def _recommendations(self, failed_evals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        recommendations: list[dict[str, Any]] = []

        def add(title: str, reason: str, target_files: list[str]) -> None:
            if any(item["title"] == title for item in recommendations):
                return
            recommendations.append(
                {
                    "title": title,
                    "reason": reason,
                    "target_files": target_files,
                }
            )

        for result in failed_evals:
            for check in result.get("failed_checks", []):
                name = str(check.get("name") or "")
                if "intent" in name:
                    add(
                        "Tune intent routing and coordinator output parsing",
                        "An eval expected a different intent than the Agent returned.",
                        [
                            "dac3d_iim_assistant/intent/classifier.py",
                            "dac3d_iim_assistant/agent_runtime.py",
                            "dac3d_iim_assistant/evals/cases/",
                        ],
                    )
                elif "must_call" in name:
                    add(
                        "Align specialist handoff and tool selection",
                        "An eval expected a tool call that was missing or unexpectedly present.",
                        [
                            "dac3d_iim_assistant/agent_core/",
                            "dac3d_iim_assistant/agent_runtime.py",
                            "dac3d_iim_assistant/tool_gateway/",
                        ],
                    )
                elif any(marker in name for marker in ("requires_confirmation", "can_submit", "risk_level", "blocked_by")):
                    add(
                        "Repair command preview and execution state handling",
                        "A command-control eval disagreed with the preview, confirmation, or risk metadata.",
                        [
                            "dac3d_iim_assistant/agent_runtime.py",
                            "dac3d_iim_assistant/tool_gateway/gateway.py",
                            "dac3d_iim_assistant/intent/command_generator.py",
                        ],
                    )
                elif "memory_patch" in name:
                    add(
                        "Improve Memory OS write proposal extraction",
                        "A memory eval expected a pending patch but the Agent did not expose one.",
                        [
                            "dac3d_iim_assistant/memory/",
                            "dac3d_iim_assistant/context_engineering/",
                            "dac3d_iim_assistant/agent_runtime.py",
                        ],
                    )
                elif "answer_contains" in name:
                    add(
                        "Update grounded response wording or retrieval coverage",
                        "A response assertion failed against the final answer text.",
                        [
                            "dac3d_iim_assistant/rag/prompts.py",
                            "dac3d_iim_assistant/rag/llm_client.py",
                            "dac3d_iim_assistant/knowledge_base/documents/",
                        ],
                    )
                else:
                    add(
                        "Inspect failed check and add a targeted regression fix",
                        f"Unhandled check failure: {name or 'unknown_check'}",
                        [
                            "dac3d_iim_assistant/agent_runtime.py",
                            "dac3d_iim_assistant/evals/cases/",
                            "dac3d_iim_assistant/tests/",
                        ],
                    )

        if not recommendations:
            recommendations.append(
                {
                    "title": "No failing evals; expand regression coverage from recent traces",
                    "reason": "The current deterministic run passed, so the next useful step is adding evals for uncovered traces or user-reported gaps.",
                    "target_files": [
                        "dac3d_iim_assistant/evals/cases/",
                        "dac3d_iim_assistant/trace_eval/",
                        "dac3d_iim_assistant/tests/test_trace_eval.py",
                    ],
                }
            )
        return recommendations

    def _render_markdown(
        self,
        *,
        generated_at: str,
        eval_result: dict[str, Any],
        failed_evals: list[dict[str, Any]],
        traces: list[dict[str, Any]],
        recommendations: list[dict[str, Any]],
    ) -> str:
        lines = [
            "# Codex Handoff: DAC-Agent Next Pass",
            "",
            f"- Generated at: `{generated_at}`",
            "- Workflow: `failing evals + traces -> Codex handoff -> next implementation pass`",
            "- Auto-applied changes: `false`",
            "",
            "## Eval Summary",
            "",
            f"- Backend: `{eval_result.get('backend') or 'unknown'}`",
            f"- Cases: `{eval_result.get('case_count') or 0}`",
            f"- Passed: `{eval_result.get('passed') or 0}`",
            f"- Failed: `{eval_result.get('failed') or 0}`",
            f"- Pass rate: `{eval_result.get('pass_rate') or 0}`",
            "",
            "## Recommended Code Changes",
            "",
        ]
        for index, item in enumerate(recommendations, start=1):
            targets = ", ".join(f"`{path}`" for path in item.get("target_files", []))
            lines.extend(
                [
                    f"{index}. {item.get('title')}",
                    f"   - Reason: {item.get('reason')}",
                    f"   - Target files: {targets}",
                ]
            )

        lines.extend(["", "## Failing Evals", ""])
        if failed_evals:
            for result in failed_evals:
                lines.extend(
                    [
                        f"### `{result.get('id')}`",
                        "",
                        f"- Category: `{result.get('category')}`",
                        f"- Input: {result.get('input')}",
                        f"- Trace: `{result.get('trace_id') or 'none'}`",
                        f"- Approval trace: `{result.get('approval_trace_id') or 'none'}`",
                        f"- Answer preview: {result.get('answer_preview') or 'none'}",
                        "- Failed checks:",
                    ]
                )
                for check in result.get("failed_checks", []):
                    lines.append(
                        f"  - `{check.get('name')}` expected `{_clip(check.get('expected'), 120)}` actual `{_clip(check.get('actual'), 120)}`"
                    )
                lines.append("")
        else:
            lines.append("No failing evals were found in this run. Use the recent traces below to choose the next coverage gap.")
            lines.append("")

        lines.extend(["## Relevant Traces", ""])
        if traces:
            for trace in traces:
                tools = ", ".join(f"`{tool}`" for tool in trace.get("tools", [])) or "`none`"
                lines.extend(
                    [
                        f"### `{trace.get('trace_id')}`",
                        "",
                        f"- Time: `{trace.get('timestamp')}`",
                        f"- Event: `{trace.get('event_type')}` / intent `{trace.get('intent')}`",
                        f"- Session: `{trace.get('session_id')}`",
                        f"- Tools: {tools}",
                        f"- User: {trace.get('user_message')}",
                        f"- Assistant: {trace.get('final_response')}",
                        "",
                    ]
                )
        else:
            lines.append("No trace records were available.")
            lines.append("")

        lines.extend(
            [
                "## Suggested Verification",
                "",
                "```bash",
                "cd dac3d_iim_assistant",
                "python -m pytest tests/test_trace_eval.py tests/test_web_api.py -q",
                "python -m pytest tests -q",
                "python -m compileall .",
                "```",
                "",
                "## Next Codex Prompt",
                "",
                "Continue upgrading DAC-3D-LLM on the dev branch. Use this handoff as the starting context, fix the failed evals or add coverage for the listed traces, avoid DAC detection algorithm changes, run the suggested verification, then commit and push to dev.",
                "",
            ]
        )
        return "\n".join(lines)
