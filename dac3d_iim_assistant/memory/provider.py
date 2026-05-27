"""Auditable Memory OS provider for DAC-3D Agent runtime."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from memory.conversation_store import ConversationMemoryStore
from memory.policies import MemorySecurityPolicy, MemoryStatus
from safety import PolicyEngine


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clip(value: Any, limit: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(default)
    return payload if isinstance(payload, dict) else dict(default)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


@dataclass(slots=True)
class MemoryBundle:
    """Prompt-ready memory bundle selected for one Agent turn."""

    task: str
    session_id: str
    context_text: str = ""
    hits: list[dict[str, Any]] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)
    knowledge_notes: dict[str, Any] = field(default_factory=dict)
    selected_layers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["backend"] = "json+markdown"
        return payload


@dataclass(slots=True)
class MemoryPatch:
    """A proposed long-term memory write that must be approved before applying."""

    id: str
    created_at: str
    target: str
    content: str
    mode: str = "append"
    reason: str = ""
    source_trace_id: str = ""
    status: str = "pending"
    topic: str = ""
    applied_at: str = ""
    rejected_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryProvider(Protocol):
    """DAC Memory OS interface used by Agent runtime."""

    def prefetch(
        self,
        task: str,
        user: str = "",
        context: dict[str, Any] | None = None,
    ) -> MemoryBundle:
        """Select the memory bundle needed by the current task."""

    def record_trace(self, trace: dict[str, Any]) -> dict[str, Any]:
        """Persist an auditable Agent trace."""

    def propose_writes(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        """Create pending memory patches from one trace."""

    def approve_write(self, patch_id: str) -> dict[str, Any]:
        """Apply a pending memory patch."""

    def reject_write(self, patch_id: str, reason: str = "") -> dict[str, Any]:
        """Reject a pending memory patch."""

    def correct(self, memory_id: str, feedback: str) -> dict[str, Any]:
        """Create a correction patch for an existing memory item."""

    def forget(self, scope: str, memory_id: str | None = None) -> dict[str, Any]:
        """Forget pending or curated memory by scope."""


class LocalMemoryProvider:
    """Local file-backed Memory OS using Markdown, JSON, and JSONL traces."""

    def __init__(self, store: ConversationMemoryStore) -> None:
        self.store = store
        self.trace_path = store.root_dir / "traces.jsonl"
        self.patches_path = store.root_dir / "memory_patches.json"
        self.policy_engine = PolicyEngine()
        self.memory_policy = MemorySecurityPolicy(self.policy_engine)

    def prefetch(
        self,
        task: str,
        user: str = "",
        context: dict[str, Any] | None = None,
    ) -> MemoryBundle:
        """Select prompt context from curated memory, session memory, and notes."""
        context = dict(context or {})
        session_id = str(context.get("session_id") or "default")
        history = context.get("history") if isinstance(context.get("history"), Sequence) else None
        recent_limit = int(context.get("recent_limit") or 4)
        search_limit = int(context.get("search_limit") or 5)
        context_text, hits = self.store.format_context(
            task,
            session_id=session_id,
            history=history,
            recent_limit=recent_limit,
            search_limit=search_limit,
        )
        selected_layers = list(
            dict.fromkeys(str(hit.get("layer") or "") for hit in hits if hit.get("layer"))
        )
        if context_text and "核心记忆" in context_text:
            selected_layers.insert(0, "core_markdown_memory")
        if history:
            selected_layers.insert(0, "short_term_history")
        selected_layers = [layer for layer in dict.fromkeys(selected_layers) if layer]
        return MemoryBundle(
            task=task,
            session_id=session_id,
            context_text=context_text,
            hits=hits,
            profile=self.store.load_curated_memory(),
            knowledge_notes=self.store.list_knowledge_notes(),
            selected_layers=selected_layers,
        )

    def record_trace(self, trace: dict[str, Any]) -> dict[str, Any]:
        """Append one auditable trace record to JSONL."""
        self.store.ensure_directories()
        record = dict(trace)
        record.setdefault("trace_id", f"trace-{uuid.uuid4().hex[:16]}")
        record.setdefault("created_at", _utc_now_iso())
        record.setdefault("runtime", "dac-agent-runtime")
        record["user_message"] = _clip(record.get("user_message"), 2000)
        record["assistant_answer"] = _clip(record.get("assistant_answer"), 2000)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def propose_writes(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        """Create pending memory patches from explicit user preference/correction signals."""
        candidates = self._extract_patch_candidates(trace)
        if not candidates:
            return []

        payload = self._load_patch_payload()
        patches = [entry for entry in payload.get("patches", []) if isinstance(entry, dict)]
        existing_pending_by_key = {
            (
                str(entry.get("target") or ""),
                str(entry.get("topic") or ""),
                str(entry.get("content") or ""),
                "pending",
            ): entry
            for entry in patches
            if str(entry.get("status") or "") == "pending"
        }

        created: list[dict[str, Any]] = []
        for candidate in candidates:
            metadata = dict(candidate.get("metadata") or {})
            candidate_target = str(candidate.get("target") or "memory")
            candidate_content = str(candidate.get("content") or "")
            source_trace_id = str(candidate.get("source_trace_id") or trace.get("trace_id") or "")
            policy_result = self.memory_policy.evaluate_candidate(
                target=candidate_target,
                content=candidate_content,
                source=str(metadata.get("source") or "user"),
                evidence_trace_id=source_trace_id,
                session_id=str(trace.get("session_id") or ""),
                privileged=bool(metadata.get("privileged")),
                existing_profile=self.store.load_curated_memory(),
            )
            policy = self.policy_engine.evaluate_memory_write(
                target=str(candidate.get("target") or "memory"),
                content=str(candidate.get("content") or ""),
                approved=False,
                commit=False,
                privileged=bool(dict(candidate.get("metadata") or {}).get("privileged")),
                source=str(dict(candidate.get("metadata") or {}).get("source") or "user"),
                evidence_trace_id=str(candidate.get("source_trace_id") or trace.get("trace_id") or ""),
            )
            if not policy.allowed:
                rejected_patch = MemoryPatch(
                    id=f"mempatch-{uuid.uuid4().hex[:12]}",
                    created_at=_utc_now_iso(),
                    target=str(candidate.get("target") or "memory"),
                    content=_clip(candidate.get("content"), 1200),
                    mode=str(candidate.get("mode") or "append"),
                    reason=str(candidate.get("reason") or "Memory write rejected by policy."),
                    source_trace_id=str(trace.get("trace_id") or ""),
                    status="rejected",
                    rejected_at=_utc_now_iso(),
                    topic=str(candidate.get("topic") or ""),
                    metadata={
                        **metadata,
                        "memory_status": MemoryStatus.REJECTED,
                        "trust_level": "untrusted",
                        "provenance": policy_result.provenance,
                        "conflict": policy_result.conflict,
                        "policy": policy.to_dict(),
                    },
                ).to_dict()
                patches.append(rejected_patch)
                created.append(rejected_patch)
                continue
            key = (
                str(candidate.get("target") or ""),
                str(candidate.get("topic") or ""),
                str(candidate.get("content") or ""),
                "pending",
            )
            if key in existing_pending_by_key:
                created.append(existing_pending_by_key[key])
                continue
            patch = MemoryPatch(
                id=f"mempatch-{uuid.uuid4().hex[:12]}",
                created_at=_utc_now_iso(),
                target=str(candidate.get("target") or "memory"),
                content=_clip(candidate.get("content"), 1200),
                mode=str(candidate.get("mode") or "append"),
                reason=str(candidate.get("reason") or "Agent trace produced a durable memory candidate."),
                source_trace_id=str(trace.get("trace_id") or ""),
                topic=str(candidate.get("topic") or ""),
                metadata={
                    **metadata,
                    "memory_status": MemoryStatus.PROPOSED,
                    "trust_level": "untrusted",
                    "provenance": policy_result.provenance,
                    "conflict": policy_result.conflict,
                    "policy": policy.to_dict(),
                },
            ).to_dict()
            patches.append(patch)
            created.append(patch)
            existing_pending_by_key[key] = patch

        if created:
            payload["version"] = 1
            payload["updated_at"] = _utc_now_iso()
            payload["patches"] = patches
            _write_json(self.patches_path, payload)
        return created

    def approve_write(self, patch_id: str) -> dict[str, Any]:
        """Apply a pending memory patch and mark it approved."""
        payload = self._load_patch_payload()
        patches = [entry for entry in payload.get("patches", []) if isinstance(entry, dict)]
        patch = self._find_patch(patches, patch_id)
        if patch is None:
            raise ValueError(f"Unknown memory patch: {patch_id}")
        if str(patch.get("status") or "") != "pending":
            return {"patch": patch, "applied": False, "message": "Patch is not pending."}

        target = str(patch.get("target") or "memory")
        metadata = dict(patch.get("metadata") or {})
        policy = self.memory_policy.approval_policy(
            target=target,
            content=str(patch.get("content") or ""),
            privileged=bool(metadata.get("privileged")),
            source=str(metadata.get("source") or "approved_patch"),
            evidence_trace_id=str(patch.get("source_trace_id") or ""),
        )
        if not policy.allowed:
            patch["status"] = "rejected"
            patch["rejected_at"] = _utc_now_iso()
            patch["reject_reason"] = "policy_denied_on_approval"
            patch["policy"] = policy.to_dict()
            payload["updated_at"] = _utc_now_iso()
            payload["patches"] = patches
            _write_json(self.patches_path, payload)
            return {"patch": patch, "applied": False, "policy": policy.to_dict()}

        if target == "knowledge":
            result = self.store.upsert_knowledge_note(
                topic=str(patch.get("topic") or "general"),
                content=str(patch.get("content") or ""),
                mode=str(patch.get("mode") or "append"),
            )
        elif target in {"memory", "user"}:
            result = self.store.update_curated_memory(
                target=target,
                content=str(patch.get("content") or ""),
                mode=str(patch.get("mode") or "append"),
            )
        else:
            raise ValueError("Memory patch target must be memory, user, or knowledge.")

        patch["status"] = "approved"
        patch["applied_at"] = _utc_now_iso()
        patch["metadata"] = {
            **metadata,
            **self.memory_policy.active_metadata(
                patch_id=str(patch.get("id") or ""),
                target=target,
                source_trace_id=str(patch.get("source_trace_id") or ""),
                session_id=str(metadata.get("provenance", {}).get("session_id") or ""),
            ),
        }
        patch["apply_result"] = result
        payload["updated_at"] = _utc_now_iso()
        payload["patches"] = patches
        _write_json(self.patches_path, payload)
        return {"patch": patch, "applied": True, "result": result}

    def reject_write(self, patch_id: str, reason: str = "") -> dict[str, Any]:
        """Reject a pending memory patch without mutating long-term memory."""
        payload = self._load_patch_payload()
        patches = [entry for entry in payload.get("patches", []) if isinstance(entry, dict)]
        patch = self._find_patch(patches, patch_id)
        if patch is None:
            raise ValueError(f"Unknown memory patch: {patch_id}")
        patch["status"] = "rejected"
        patch["rejected_at"] = _utc_now_iso()
        patch["reject_reason"] = str(reason or "")
        payload["updated_at"] = _utc_now_iso()
        payload["patches"] = patches
        _write_json(self.patches_path, payload)
        return {"patch": patch, "rejected": True}

    def correct(self, memory_id: str, feedback: str) -> dict[str, Any]:
        """Create a correction patch instead of silently editing memory."""
        trace = {
            "trace_id": f"manual-correction-{uuid.uuid4().hex[:12]}",
            "user_message": f"纠正记忆 {memory_id}: {feedback}",
            "parsed_result": {
                "memory_write_candidates": [
                    {
                        "target": "memory",
                        "content": f"记忆纠错：{memory_id} -> {feedback}",
                        "reason": "manual_memory_correction",
                    }
                ]
            },
        }
        return {"patches": self.propose_writes(trace)}

    def forget(self, scope: str, memory_id: str | None = None) -> dict[str, Any]:
        """Forget pending patches or curated entries by exact/substring id."""
        normalized_scope = str(scope or "").strip().lower()
        if normalized_scope in {"pending", "pending_patch", "patch"}:
            if not memory_id:
                raise ValueError("memory_id is required for pending patch forget.")
            return self.reject_write(memory_id, reason="forgotten_by_request")
        if normalized_scope in {"memory", "user"}:
            return self._forget_curated(normalized_scope, str(memory_id or ""))
        raise ValueError("Unsupported forget scope. Use pending, memory, or user.")

    def list_patches(self, *, status: str | None = None) -> dict[str, Any]:
        """List memory patches, optionally filtered by status."""
        payload = self._load_patch_payload()
        patches = [entry for entry in payload.get("patches", []) if isinstance(entry, dict)]
        if status:
            patches = [entry for entry in patches if str(entry.get("status") or "") == status]
        return {
            "backend": "json+markdown",
            "patches_path": str(self.patches_path),
            "patches": patches,
            "count": len(patches),
        }

    def describe(self) -> dict[str, Any]:
        """Return Memory OS diagnostics."""
        patches = self.list_patches()["patches"]
        trace_count = 0
        if self.trace_path.exists():
            trace_count = sum(1 for _line in self.trace_path.open(encoding="utf-8"))
        return {
            "provider": "local_memory_os",
            "trace_path": str(self.trace_path),
            "patches_path": str(self.patches_path),
            "trace_count": trace_count,
            "pending_patch_count": sum(1 for patch in patches if patch.get("status") == "pending"),
            "patch_count": len(patches),
            "workflow": "trace -> memory_patch -> approval -> long_term_memory",
        }

    def _extract_patch_candidates(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        parsed = trace.get("parsed_result")
        candidates: list[dict[str, Any]] = []
        if isinstance(parsed, dict) and isinstance(parsed.get("memory_write_candidates"), list):
            candidates.extend(
                candidate
                for candidate in parsed.get("memory_write_candidates", [])
                if isinstance(candidate, dict)
            )

        user_message = str(trace.get("user_message") or "")
        if self._looks_like_explicit_memory_request(user_message):
            candidates.append(
                {
                    "target": "user",
                    "content": f"用户长期偏好/纠正：{_clip(user_message, 500)}",
                    "reason": "explicit_user_memory_request",
                    "metadata": {"source": "user_message"},
                }
            )
        return candidates

    def _looks_like_explicit_memory_request(self, text: str) -> bool:
        return bool(
            re.search(
                r"(记住|以后|下次|不要默认|默认用|偏好|我更喜欢|修正|纠正)",
                text,
            )
        )

    def _load_patch_payload(self) -> dict[str, Any]:
        payload = _read_json(self.patches_path, {"version": 1, "updated_at": "", "patches": []})
        if not isinstance(payload.get("patches"), list):
            payload["patches"] = []
        return payload

    def _find_patch(
        self,
        patches: list[dict[str, Any]],
        patch_id: str,
    ) -> dict[str, Any] | None:
        for patch in patches:
            if str(patch.get("id") or "") == str(patch_id or ""):
                return patch
        return None

    def _forget_curated(self, target: str, memory_id: str) -> dict[str, Any]:
        if not memory_id:
            raise ValueError("memory_id is required for curated memory forget.")
        profile = self.store.load_curated_memory()
        original = str(profile.get(target) or "")
        parts = [part.strip() for part in original.split("§") if part.strip()]
        remaining = [part for part in parts if memory_id not in part]
        removed = len(parts) - len(remaining)
        path = self.store.core_memory_path if target == "memory" else self.store.user_memory_path
        path.write_text("\n§\n".join(remaining), encoding="utf-8")
        return {
            "target": target,
            "removed": removed,
            "memory_id": memory_id,
            "profile": self.store.load_curated_memory(),
        }
