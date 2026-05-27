"""Reviewable skill patch proposals for DAC-Agent Runtime."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safety import PolicyEngine
from skill_system.registry import SkillRegistry


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clip(value: Any, limit: int = 4000) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _safe_patch_id(skill_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(skill_name or "skill")).strip("_").lower()
    return f"skillpatch-{slug or 'skill'}-{uuid.uuid4().hex[:10]}"


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
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


@dataclass(slots=True)
class SkillPatch:
    """One proposed skill change that requires human review."""

    id: str
    created_at: str
    target_skill: str
    reason: str
    diff: str = ""
    replacement_section: str = ""
    evidence_trace_ids: list[str] = field(default_factory=list)
    risk_level: str = "medium"
    status: str = "pending"
    proposed_by: str = "agent"
    approved_at: str = ""
    rejected_at: str = ""
    reject_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SkillPatchStore:
    """File-backed queue for Memento-style skill improvement proposals."""

    def __init__(self, *, patches_path: str | Path, registry: SkillRegistry) -> None:
        self.patches_path = Path(patches_path)
        self.registry = registry
        self.policy_engine = PolicyEngine()

    @classmethod
    def from_root(cls, root_dir: str | Path, registry: SkillRegistry) -> "SkillPatchStore":
        return cls(patches_path=Path(root_dir) / "skill_patches.json", registry=registry)

    def propose_patch(
        self,
        *,
        target_skill: str,
        reason: str,
        diff: str = "",
        replacement_section: str = "",
        evidence_trace_ids: list[str] | None = None,
        risk_level: str = "medium",
        proposed_by: str = "agent",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a pending skill patch without modifying the target skill."""
        clean_skill = self._resolve_skill_name(target_skill)
        clean_diff = _clip(diff, 6000)
        clean_replacement = _clip(replacement_section, 6000)
        clean_reason = _clip(reason, 1200)
        evidence = [str(item) for item in evidence_trace_ids or [] if str(item).strip()]
        patch_metadata = dict(metadata or {})
        content_for_policy = "\n".join([clean_reason, clean_diff, clean_replacement])
        policy = self.policy_engine.evaluate_skill_patch(
            skill_name=clean_skill,
            content=content_for_policy,
            apply=False,
            approved=False,
        )
        status = "pending" if policy.allowed and (clean_diff or clean_replacement) else "rejected"
        reject_reason = "" if status == "pending" else "missing_patch_content"
        if not policy.allowed:
            reject_reason = ",".join(policy.blocking_reasons)

        payload = self._load_payload()
        patches = [entry for entry in payload.get("patches", []) if isinstance(entry, dict)]
        key = (
            clean_skill,
            clean_reason,
            clean_diff,
            clean_replacement,
            "pending",
        )
        for entry in patches:
            existing_key = (
                str(entry.get("target_skill") or ""),
                str(entry.get("reason") or ""),
                str(entry.get("diff") or ""),
                str(entry.get("replacement_section") or ""),
                str(entry.get("status") or ""),
            )
            if existing_key == key:
                return {"patch": entry, "created": False, "duplicate": True}

        now = _utc_now_iso()
        patch = SkillPatch(
            id=_safe_patch_id(clean_skill),
            created_at=now,
            target_skill=clean_skill,
            reason=clean_reason,
            diff=clean_diff,
            replacement_section=clean_replacement,
            evidence_trace_ids=evidence,
            risk_level=str(risk_level or "medium"),
            status=status,
            proposed_by=str(proposed_by or "agent"),
            rejected_at=now if status == "rejected" else "",
            reject_reason=reject_reason,
            metadata={
                **patch_metadata,
                "policy": policy.to_dict(),
                "workflow": "skill_patch -> human_review -> manual_apply",
                "auto_applied": False,
            },
        ).to_dict()
        patches.append(patch)
        payload["version"] = 1
        payload["updated_at"] = now
        payload["patches"] = patches
        _write_json(self.patches_path, payload)
        return {"patch": patch, "created": True, "duplicate": False}

    def list_patches(self, *, status: str | None = "pending") -> dict[str, Any]:
        """List skill patch proposals, optionally filtered by status."""
        payload = self._load_payload()
        patches = [entry for entry in payload.get("patches", []) if isinstance(entry, dict)]
        if status:
            patches = [entry for entry in patches if str(entry.get("status") or "") == status]
        return {
            "backend": "json_skill_patch_queue",
            "patches_path": str(self.patches_path),
            "patches": patches,
            "count": len(patches),
            "workflow": "skill_patch -> human_review -> manual_apply",
            "auto_applied": False,
        }

    def approve_patch(self, patch_id: str) -> dict[str, Any]:
        """Mark a pending skill patch approved without editing production skill files."""
        payload, patches, patch = self._patch_payload_and_entry(patch_id)
        if str(patch.get("status") or "") != "pending":
            return {"patch": patch, "approved": False, "applied": False, "message": "Patch is not pending."}
        patch["status"] = "approved"
        patch["approved_at"] = _utc_now_iso()
        patch["metadata"] = {
            **dict(patch.get("metadata") or {}),
            "review": "approved",
            "auto_applied": False,
            "apply_note": "Approved proposals are review records; SKILL.md is not modified automatically.",
        }
        payload["updated_at"] = _utc_now_iso()
        payload["patches"] = patches
        _write_json(self.patches_path, payload)
        return {"patch": patch, "approved": True, "applied": False}

    def reject_patch(self, patch_id: str, reason: str = "") -> dict[str, Any]:
        """Reject a skill patch proposal."""
        payload, patches, patch = self._patch_payload_and_entry(patch_id)
        patch["status"] = "rejected"
        patch["rejected_at"] = _utc_now_iso()
        patch["reject_reason"] = str(reason or "")
        payload["updated_at"] = _utc_now_iso()
        payload["patches"] = patches
        _write_json(self.patches_path, payload)
        return {"patch": patch, "rejected": True}

    def describe(self) -> dict[str, Any]:
        patches = self.list_patches(status=None)["patches"]
        return {
            "enabled": True,
            "backend": "json_skill_patch_queue",
            "patches_path": str(self.patches_path),
            "patch_count": len(patches),
            "pending_patch_count": sum(1 for patch in patches if patch.get("status") == "pending"),
            "workflow": "skill_patch -> human_review -> manual_apply",
            "auto_applied": False,
        }

    def _resolve_skill_name(self, target_skill: str) -> str:
        clean = str(target_skill or "").strip()
        if not clean:
            raise ValueError("target_skill is required.")
        for skill in self.registry.discover():
            if skill.name == clean or Path(skill.path).name == clean:
                return skill.name
        raise ValueError(f"Unknown DAC skill: {target_skill}")

    def _load_payload(self) -> dict[str, Any]:
        payload = _read_json(self.patches_path, {"version": 1, "updated_at": "", "patches": []})
        if not isinstance(payload.get("patches"), list):
            payload["patches"] = []
        return payload

    def _patch_payload_and_entry(self, patch_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        payload = self._load_payload()
        patches = [entry for entry in payload.get("patches", []) if isinstance(entry, dict)]
        for patch in patches:
            if str(patch.get("id") or "") == str(patch_id or ""):
                return payload, patches, patch
        raise ValueError(f"Unknown skill patch: {patch_id}")
