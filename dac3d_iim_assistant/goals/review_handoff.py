"""Local review handoff queue for DAC-Agent workspace.

Review handoffs capture work packets that should be inspected by a human or a
specialist reviewer agent. They are separate from command safety approvals:
this store is about product/code workflow review, comments, and follow-up
status, not execution authorization.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REVIEW_STATUSES = ("pending", "in_review", "approved", "needs_changes", "rejected", "archived")
REVIEW_PRIORITIES = ("low", "normal", "high")


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(default)
    return payload if isinstance(payload, dict) else dict(default)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = _utc_now_iso()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _clip(value: Any, limit: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _normalize_status(value: str, *, default: str = "pending") -> str:
    status = str(value or default).strip().lower()
    if status not in REVIEW_STATUSES:
        raise ValueError(f"Review status must be one of: {', '.join(REVIEW_STATUSES)}.")
    return status


def _normalize_priority(value: str) -> str:
    priority = str(value or "normal").strip().lower()
    if priority not in REVIEW_PRIORITIES:
        raise ValueError(f"Review priority must be one of: {', '.join(REVIEW_PRIORITIES)}.")
    return priority


def _strings(values: list[Any] | tuple[Any, ...] | None, *, limit: int = 30) -> list[str]:
    if not values:
        return []
    result: list[str] = []
    for value in values[:limit]:
        text = _clip(value, 240)
        if text:
            result.append(text)
    return result


def _checklist(values: list[Any] | tuple[Any, ...] | None) -> list[dict[str, Any]]:
    if not values:
        return []
    items: list[dict[str, Any]] = []
    for value in values[:30]:
        if isinstance(value, dict):
            label = _clip(value.get("label") or value.get("title") or value.get("text"), 240)
            if label:
                items.append({"label": label, "checked": bool(value.get("checked"))})
        else:
            label = _clip(value, 240)
            if label:
                items.append({"label": label, "checked": False})
    return items


class ReviewHandoffStore:
    """JSON-backed review packets for local Agent collaboration."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_root(cls, root_dir: Path) -> "ReviewHandoffStore":
        return cls(root_dir / "agent_review_handoffs.json")

    def create_review(
        self,
        title: str,
        *,
        summary: str = "",
        session_id: str = "web",
        status: str = "pending",
        priority: str = "normal",
        task_id: str = "",
        workflow_id: str = "",
        trace_id: str = "",
        files: list[Any] | None = None,
        verification_run_ids: list[Any] | None = None,
        checklist: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_title = _clip(title, 180)
        if not clean_title:
            raise ValueError("Review title is required.")
        normalized_status = _normalize_status(status)
        normalized_priority = _normalize_priority(priority)
        now = _utc_now_iso()
        review = {
            "id": f"review-{uuid.uuid4().hex[:12]}",
            "session_id": str(session_id or "web"),
            "task_id": str(task_id or ""),
            "workflow_id": str(workflow_id or ""),
            "trace_id": str(trace_id or ""),
            "title": clean_title,
            "summary": _clip(summary, 1600),
            "status": normalized_status,
            "priority": normalized_priority,
            "files": _strings(files),
            "verification_run_ids": _strings(verification_run_ids),
            "checklist": _checklist(checklist),
            "comments": [],
            "created_at": now,
            "updated_at": now,
            "completed_at": now if normalized_status in {"approved", "rejected", "archived"} else "",
            "history": [
                {
                    "id": f"history-{uuid.uuid4().hex[:10]}",
                    "created_at": now,
                    "type": "created",
                    "status": normalized_status,
                    "note": "Review handoff created.",
                }
            ],
            "metadata": dict(metadata or {}),
        }
        payload = self._load()
        reviews = self._reviews(payload)
        reviews.insert(0, review)
        payload["reviews"] = reviews
        _write_json(self.path, payload)
        return {"review": review, "created": True}

    def list_reviews(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        reviews = self._reviews(payload)
        if session_id:
            reviews = [review for review in reviews if str(review.get("session_id") or "") == session_id]
        if status:
            normalized_status = _normalize_status(status)
            reviews = [review for review in reviews if str(review.get("status") or "") == normalized_status]
        if priority:
            normalized_priority = _normalize_priority(priority)
            reviews = [review for review in reviews if str(review.get("priority") or "") == normalized_priority]
        safe_limit = max(1, min(200, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_review_handoff_queue",
            "path": str(self.path),
            "reviews": reviews[:safe_limit],
            "count": len(reviews[:safe_limit]),
            "total_count": len(self._reviews(payload)),
            "statuses": list(REVIEW_STATUSES),
            "priorities": list(REVIEW_PRIORITIES),
            "workflow": "create_review_packet -> reviewer_comments -> status_decision",
        }

    def read_review(self, review_id: str) -> dict[str, Any]:
        review = self._find_review(self._reviews(self._load()), review_id)
        if review is None:
            raise ValueError(f"Unknown review handoff: {review_id}")
        return {"enabled": True, "review": review}

    def add_comment(
        self,
        review_id: str,
        body: str,
        *,
        reviewer: str = "human",
    ) -> dict[str, Any]:
        clean_body = _clip(body, 1200)
        if not clean_body:
            raise ValueError("Review comment body is required.")
        payload = self._load()
        reviews = self._reviews(payload)
        review = self._find_review(reviews, review_id)
        if review is None:
            raise ValueError(f"Unknown review handoff: {review_id}")
        now = _utc_now_iso()
        comment = {
            "id": f"comment-{uuid.uuid4().hex[:10]}",
            "created_at": now,
            "reviewer": _clip(reviewer or "human", 120),
            "body": clean_body,
        }
        comments = review.setdefault("comments", [])
        if not isinstance(comments, list):
            comments = []
            review["comments"] = comments
        comments.append(comment)
        review["updated_at"] = now
        self._append_history(
            review,
            "comment_added",
            status=str(review.get("status") or "pending"),
            note=clean_body,
            extra={"comment_id": comment["id"], "reviewer": comment["reviewer"]},
        )
        payload["reviews"] = reviews
        _write_json(self.path, payload)
        return {"review": review, "comment": comment}

    def update_status(
        self,
        review_id: str,
        status: str,
        *,
        note: str = "",
        reviewer: str = "human",
    ) -> dict[str, Any]:
        normalized_status = _normalize_status(status)
        payload = self._load()
        reviews = self._reviews(payload)
        review = self._find_review(reviews, review_id)
        if review is None:
            raise ValueError(f"Unknown review handoff: {review_id}")
        now = _utc_now_iso()
        previous_status = str(review.get("status") or "")
        review["status"] = normalized_status
        review["updated_at"] = now
        if normalized_status in {"approved", "rejected", "archived"}:
            review["completed_at"] = now
        elif previous_status in {"approved", "rejected", "archived"}:
            review["completed_at"] = ""
        history = self._append_history(
            review,
            "status_changed",
            status=normalized_status,
            note=note or f"{previous_status} -> {normalized_status}",
            extra={
                "from": previous_status,
                "to": normalized_status,
                "reviewer": _clip(reviewer or "human", 120),
            },
        )
        payload["reviews"] = reviews
        _write_json(self.path, payload)
        return {"review": review, "history": history}

    def describe(self) -> dict[str, Any]:
        reviews = self._reviews(self._load())
        by_status = {status: 0 for status in REVIEW_STATUSES}
        for review in reviews:
            status = str(review.get("status") or "pending")
            by_status[status] = by_status.get(status, 0) + 1
        latest = reviews[0] if reviews else {}
        return {
            "enabled": True,
            "backend": "local_review_handoff_queue",
            "path": str(self.path),
            "review_count": len(reviews),
            "by_status": by_status,
            "latest_review": latest,
            "workflow": "agent_work_packet -> review_handoff -> comments -> decision",
        }

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.path, {"version": 1, "updated_at": "", "reviews": []})
        if not isinstance(payload.get("reviews"), list):
            payload["reviews"] = []
        return payload

    def _reviews(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [review for review in payload.get("reviews", []) if isinstance(review, dict)]

    def _find_review(
        self,
        reviews: list[dict[str, Any]],
        review_id: str,
    ) -> dict[str, Any] | None:
        for review in reviews:
            if str(review.get("id") or "") == str(review_id or ""):
                return review
        return None

    def _append_history(
        self,
        review: dict[str, Any],
        event_type: str,
        *,
        status: str,
        note: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        history_items = review.setdefault("history", [])
        if not isinstance(history_items, list):
            history_items = []
            review["history"] = history_items
        history = {
            "id": f"history-{uuid.uuid4().hex[:10]}",
            "created_at": _utc_now_iso(),
            "type": event_type,
            "status": status,
            "note": _clip(note, 800),
        }
        if extra:
            history.update(extra)
        history_items.append(history)
        return history
