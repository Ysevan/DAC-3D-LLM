"""Generate reviewable eval drafts from append-only Agent traces."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trace_eval.tracing import TraceLogger


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()
    return slug[:80] or "trace"


class EvalDraftGenerator:
    """Convert selected trace records into human-reviewed eval case drafts."""

    def __init__(self, *, trace_logger: TraceLogger, drafts_dir: str | Path) -> None:
        self.trace_logger = trace_logger
        self.drafts_dir = Path(drafts_dir)

    def generate(
        self,
        *,
        trace_ids: list[str] | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Generate draft eval cases from selected or recent traces."""
        traces = self._select_traces(trace_ids=trace_ids, limit=limit)
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        drafts = []
        for trace in traces:
            draft = self._draft_from_trace(trace)
            if not draft:
                continue
            path = self.drafts_dir / f"{draft['id']}.json"
            path.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            drafts.append({"path": str(path), "draft": draft})
        return {
            "enabled": True,
            "backend": "trace_to_eval_draft",
            "drafts_dir": str(self.drafts_dir),
            "count": len(drafts),
            "drafts": drafts,
            "workflow": "trace -> eval draft -> human review -> evals/cases",
            "auto_approved": False,
        }

    def list_drafts(self) -> dict[str, Any]:
        """List generated eval drafts without treating them as approved cases."""
        if not self.drafts_dir.exists():
            return {
                "enabled": True,
                "backend": "trace_to_eval_draft",
                "drafts_dir": str(self.drafts_dir),
                "count": 0,
                "drafts": [],
                "auto_approved": False,
            }
        drafts = []
        for path in sorted(self.drafts_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                drafts.append({"path": str(path), "draft": payload})
        return {
            "enabled": True,
            "backend": "trace_to_eval_draft",
            "drafts_dir": str(self.drafts_dir),
            "count": len(drafts),
            "drafts": drafts,
            "auto_approved": False,
        }

    def _select_traces(
        self,
        *,
        trace_ids: list[str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if trace_ids:
            traces = []
            for trace_id in trace_ids:
                trace = self.trace_logger.get(trace_id)
                if trace is not None:
                    traces.append(trace)
            return traces
        safe_limit = max(1, min(50, int(limit or 5)))
        recent = self.trace_logger.iter_recent(limit=safe_limit * 3)
        usable = [trace for trace in recent if self._is_usable_trace(trace)]
        return usable[-safe_limit:]

    def _is_usable_trace(self, trace: dict[str, Any]) -> bool:
        message = str(trace.get("user_message") or "").strip()
        if not message:
            return False
        if message == "批准执行":
            return False
        return bool(trace.get("intent") or trace.get("tool_calls") or trace.get("final_response"))

    def _draft_from_trace(self, trace: dict[str, Any]) -> dict[str, Any] | None:
        if not self._is_usable_trace(trace):
            return None
        trace_id = str(trace.get("trace_id") or "")
        intent = str(trace.get("intent") or "agent")
        expected: dict[str, Any] = {"intent": intent}

        tool_names = [
            str(item.get("name") or "")
            for item in trace.get("tool_calls", [])
            if isinstance(item, dict) and item.get("name")
        ]
        if tool_names:
            expected["must_call_tools"] = list(dict.fromkeys(tool_names))

        confirmation = trace.get("confirmation") if isinstance(trace.get("confirmation"), dict) else {}
        if "required" in confirmation and confirmation.get("required") is not None:
            expected["requires_confirmation"] = confirmation.get("required") is True

        risk = trace.get("risk_decision") if isinstance(trace.get("risk_decision"), dict) else {}
        if risk.get("risk_level"):
            expected["risk_level"] = str(risk.get("risk_level"))
        blocked_by = [str(item) for item in risk.get("blocked_by", []) if str(item)]
        if blocked_by:
            expected["blocked_by_contains"] = blocked_by

        if trace.get("memory_patch_ids"):
            expected["memory_patch_proposed"] = True

        generated_at = _utc_now_iso()
        draft_id = f"draft_{generated_at[:10].replace('-', '')}_{_safe_slug(trace_id)}"
        return {
            "id": draft_id,
            "category": f"trace_{_safe_slug(intent)}",
            "input": str(trace.get("user_message") or ""),
            "expected": expected,
            "draft": True,
            "source_trace_id": trace_id,
            "source_event_type": trace.get("event_type"),
            "generated_at": generated_at,
            "review": {
                "status": "draft",
                "instructions": "Review this draft before moving it into evals/cases.",
            },
            "notes": {
                "final_response_preview": str(trace.get("final_response") or "")[:240],
                "context_ids": trace.get("context_ids", []),
            },
        }
