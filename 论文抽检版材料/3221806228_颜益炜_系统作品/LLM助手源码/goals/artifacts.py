"""Local artifact store for DAC-Agent shared workspace.

Artifacts capture reusable Agent outputs such as markdown notes, JSON payloads,
diffs, and logs. Metadata stays in one JSON index while full content is written
to local files so the workspace can list, search, and reopen outputs across
sessions.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACT_TYPES = ("markdown", "json", "text", "diff", "log", "link")
ARTIFACT_EXTENSIONS = {
    "markdown": ".md",
    "json": ".json",
    "text": ".txt",
    "diff": ".diff",
    "log": ".log",
    "link": ".url",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def _strings(values: list[Any] | tuple[Any, ...] | None, *, limit: int = 20) -> list[str]:
    if not values:
        return []
    result: list[str] = []
    for value in values[:limit]:
        text = _clip(value, 80)
        if text and text not in result:
            result.append(text)
    return result


def _normalize_type(value: str) -> str:
    artifact_type = str(value or "markdown").strip().lower()
    if artifact_type not in ARTIFACT_TYPES:
        raise ValueError(f"Artifact type must be one of: {', '.join(ARTIFACT_TYPES)}.")
    return artifact_type


def _serialize_content(content: Any, artifact_type: str) -> str:
    if artifact_type == "json":
        if isinstance(content, str):
            stripped = content.strip()
            if not stripped:
                raise ValueError("JSON artifact content is required.")
            try:
                json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError("JSON artifact content must be valid JSON.") from exc
            return json.dumps(json.loads(stripped), ensure_ascii=False, indent=2)
        return json.dumps(content, ensure_ascii=False, indent=2)
    text = str(content or "").strip()
    if not text:
        raise ValueError("Artifact content is required.")
    return text


class ArtifactStore:
    """JSON index plus file-backed content for Agent workspace artifacts."""

    def __init__(self, index_path: Path, content_dir: Path) -> None:
        self.index_path = index_path
        self.content_dir = content_dir

    @classmethod
    def from_root(cls, root_dir: Path) -> "ArtifactStore":
        return cls(root_dir / "agent_artifacts.json", root_dir / "artifacts")

    def create_artifact(
        self,
        title: str,
        content: Any,
        *,
        artifact_type: str = "markdown",
        session_id: str = "web",
        task_id: str = "",
        workflow_id: str = "",
        trace_id: str = "",
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_title = _clip(title, 180)
        if not clean_title:
            raise ValueError("Artifact title is required.")
        normalized_type = _normalize_type(artifact_type)
        body = _serialize_content(content, normalized_type)
        now = _utc_now_iso()
        artifact_id = f"artifact-{uuid.uuid4().hex[:12]}"
        relative_content_path = f"artifacts/{artifact_id}{ARTIFACT_EXTENSIONS[normalized_type]}"
        content_path = self.index_path.parent / relative_content_path
        content_path.parent.mkdir(parents=True, exist_ok=True)
        content_path.write_text(body, encoding="utf-8")
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        artifact = {
            "id": artifact_id,
            "session_id": str(session_id or "web"),
            "task_id": str(task_id or ""),
            "workflow_id": str(workflow_id or ""),
            "trace_id": str(trace_id or ""),
            "title": clean_title,
            "artifact_type": normalized_type,
            "content_path": relative_content_path,
            "content_preview": _clip(body, 500),
            "tags": _strings(tags),
            "created_at": now,
            "updated_at": now,
            "size_bytes": len(body.encode("utf-8")),
            "sha256": digest,
            "metadata": dict(metadata or {}),
        }
        payload = self._load()
        artifacts = self._artifacts(payload)
        artifacts.insert(0, artifact)
        payload["artifacts"] = artifacts
        _write_json(self.index_path, payload)
        return {"artifact": artifact, "created": True}

    def list_artifacts(
        self,
        *,
        session_id: str | None = None,
        artifact_type: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        artifacts = self._artifacts(payload)
        if session_id:
            artifacts = [
                artifact
                for artifact in artifacts
                if str(artifact.get("session_id") or "") == session_id
            ]
        if artifact_type:
            normalized_type = _normalize_type(artifact_type)
            artifacts = [
                artifact
                for artifact in artifacts
                if str(artifact.get("artifact_type") or "") == normalized_type
            ]
        if tag:
            normalized_tag = str(tag or "").strip()
            artifacts = [
                artifact
                for artifact in artifacts
                if normalized_tag in [str(item) for item in artifact.get("tags", [])]
            ]
        if query:
            artifacts = [artifact for artifact in artifacts if self._matches_query(artifact, query)]
        safe_limit = max(1, min(200, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_agent_artifact_store",
            "path": str(self.index_path),
            "content_dir": str(self.content_dir),
            "artifacts": artifacts[:safe_limit],
            "count": len(artifacts[:safe_limit]),
            "total_count": len(self._artifacts(payload)),
            "artifact_types": list(ARTIFACT_TYPES),
            "workflow": "agent_output -> artifact_file -> searchable_workspace",
        }

    def read_artifact(self, artifact_id: str) -> dict[str, Any]:
        artifact = self._find_artifact(self._artifacts(self._load()), artifact_id)
        if artifact is None:
            raise ValueError(f"Unknown artifact: {artifact_id}")
        return {
            "enabled": True,
            "backend": "local_agent_artifact_store",
            "artifact": artifact,
            "content": self._read_content(artifact),
        }

    def describe(self) -> dict[str, Any]:
        artifacts = self._artifacts(self._load())
        by_type: dict[str, int] = {artifact_type: 0 for artifact_type in ARTIFACT_TYPES}
        for artifact in artifacts:
            artifact_type = str(artifact.get("artifact_type") or "text")
            by_type[artifact_type] = by_type.get(artifact_type, 0) + 1
        return {
            "enabled": True,
            "backend": "local_agent_artifact_store",
            "path": str(self.index_path),
            "content_dir": str(self.content_dir),
            "artifact_count": len(artifacts),
            "artifact_types": list(ARTIFACT_TYPES),
            "by_type": by_type,
            "workflow": "create_artifact -> list/search -> reopen_from_workspace",
        }

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.index_path, {"version": 1, "updated_at": "", "artifacts": []})
        if not isinstance(payload.get("artifacts"), list):
            payload["artifacts"] = []
        return payload

    def _artifacts(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            artifact
            for artifact in payload.get("artifacts", [])
            if isinstance(artifact, dict)
        ]

    def _find_artifact(
        self,
        artifacts: list[dict[str, Any]],
        artifact_id: str,
    ) -> dict[str, Any] | None:
        for artifact in artifacts:
            if str(artifact.get("id") or "") == str(artifact_id or ""):
                return artifact
        return None

    def _read_content(self, artifact: dict[str, Any]) -> str:
        relative_path = str(artifact.get("content_path") or "")
        if not relative_path:
            return ""
        path = self.index_path.parent / relative_path
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _matches_query(self, artifact: dict[str, Any], query: str) -> bool:
        needle = str(query or "").strip().lower()
        if not needle:
            return True
        haystack = "\n".join(
            [
                str(artifact.get("title") or ""),
                str(artifact.get("artifact_type") or ""),
                str(artifact.get("content_preview") or ""),
                " ".join(str(item) for item in artifact.get("tags", [])),
                json.dumps(artifact.get("metadata") or {}, ensure_ascii=False),
                self._read_content(artifact),
            ]
        ).lower()
        return needle in haystack
