"""Local workflow template store for DAC-Agent Runtime.

Workflow templates persist the DAG-like shape of an Agent workflow preview so a
future UI can render, compare, and reuse known flows without rebuilding them
from scratch every time.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


WORKFLOW_STATUSES = ("draft", "active", "archived")


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


def _normalize_status(value: str, *, default: str = "draft") -> str:
    status = str(value or default).strip().lower()
    if status not in WORKFLOW_STATUSES:
        raise ValueError(f"Workflow status must be one of: {', '.join(WORKFLOW_STATUSES)}.")
    return status


def _normalize_nodes(nodes: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, node in enumerate(nodes):
        payload = dict(node) if isinstance(node, dict) else {"label": str(node)}
        node_id = _clip(payload.get("id") or f"node-{index + 1}", 80)
        if node_id in seen:
            node_id = f"{node_id}-{index + 1}"
        seen.add(node_id)
        normalized.append(
            {
                "id": node_id,
                "label": _clip(payload.get("label") or node_id, 160),
                "kind": _clip(payload.get("kind") or "agent_step", 80),
                "status": _clip(payload.get("status") or "planned", 80),
                "metadata": dict(payload.get("metadata") or {}),
            }
        )
    return normalized


def _normalize_edges(edges: list[Any], node_ids: set[str]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for edge in edges:
        payload = dict(edge) if isinstance(edge, dict) else {}
        source = _clip(payload.get("source") or payload.get("from"), 80)
        target = _clip(payload.get("target") or payload.get("to"), 80)
        if not source or not target:
            continue
        if source not in node_ids or target not in node_ids:
            raise ValueError(f"Workflow edge references unknown node: {source} -> {target}")
        normalized.append({"source": source, "target": target, "label": _clip(payload.get("label"), 160)})
    return normalized


def _edges_from_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    for index in range(len(nodes) - 1):
        edges.append(
            {
                "source": str(nodes[index]["id"]),
                "target": str(nodes[index + 1]["id"]),
                "label": "next",
            }
        )
    return edges


class WorkflowTemplateStore:
    """JSON-backed reusable Agent workflow templates."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_root(cls, root_dir: Path) -> "WorkflowTemplateStore":
        return cls(root_dir / "agent_workflows.json")

    def create_workflow(
        self,
        name: str,
        *,
        session_id: str = "web",
        description: str = "",
        nodes: list[Any] | None = None,
        edges: list[Any] | None = None,
        status: str = "draft",
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_name = _clip(name, 180)
        if not clean_name:
            raise ValueError("Workflow name is required.")
        normalized_nodes = _normalize_nodes(list(nodes or []))
        if not normalized_nodes:
            raise ValueError("Workflow must include at least one node.")
        node_ids = {str(node["id"]) for node in normalized_nodes}
        normalized_edges = _normalize_edges(list(edges or []), node_ids)
        normalized_status = _normalize_status(status)
        now = _utc_now_iso()
        workflow = {
            "id": f"workflow-{uuid.uuid4().hex[:12]}",
            "session_id": str(session_id or "web"),
            "name": clean_name,
            "description": _clip(description, 900),
            "status": normalized_status,
            "nodes": normalized_nodes,
            "edges": normalized_edges,
            "tags": [_clip(tag, 80) for tag in list(tags or []) if _clip(tag, 80)],
            "created_at": now,
            "updated_at": now,
            "metadata": dict(metadata or {}),
        }
        payload = self._load()
        workflows = self._workflows(payload)
        workflows.insert(0, workflow)
        payload["workflows"] = workflows
        _write_json(self.path, payload)
        return {"workflow": workflow, "created": True}

    def create_from_preview(
        self,
        preview: dict[str, Any],
        *,
        name: str = "",
        session_id: str = "web",
        status: str = "draft",
        tags: list[Any] | None = None,
    ) -> dict[str, Any]:
        task = _clip(preview.get("task"), 180)
        workflow_name = _clip(name or task or "Agent workflow", 180)
        nodes = _normalize_nodes(list(preview.get("nodes") or []))
        edges = _edges_from_nodes(nodes)
        return self.create_workflow(
            workflow_name,
            session_id=session_id or str(preview.get("session_id") or "web"),
            description=f"Workflow template generated from task: {task}",
            nodes=nodes,
            edges=edges,
            status=status,
            tags=tags,
            metadata={
                "source": "workflow_preview",
                "preview_backend": preview.get("backend"),
                "agent_path": list(preview.get("agent_path") or []),
                "tool_candidates": list(preview.get("tool_candidates") or []),
                "context_match_count": len(preview.get("context_tree_matches") or []),
            },
        )

    def list_workflows(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        workflows = self._workflows(payload)
        if session_id:
            workflows = [
                workflow
                for workflow in workflows
                if str(workflow.get("session_id") or "") == session_id
            ]
        if status:
            normalized_status = _normalize_status(status)
            workflows = [
                workflow
                for workflow in workflows
                if str(workflow.get("status") or "") == normalized_status
            ]
        safe_limit = max(1, min(200, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_workflow_templates",
            "path": str(self.path),
            "workflows": workflows[:safe_limit],
            "count": len(workflows[:safe_limit]),
            "total_count": len(self._workflows(payload)),
            "workflow": "preview_or_manual_graph -> reusable_template -> workspace",
        }

    def read_workflow(self, workflow_id: str) -> dict[str, Any]:
        workflow = self._find_workflow(self._workflows(self._load()), workflow_id)
        if workflow is None:
            raise ValueError(f"Unknown workflow: {workflow_id}")
        return {"enabled": True, "workflow": workflow}

    def update_status(self, workflow_id: str, status: str) -> dict[str, Any]:
        normalized_status = _normalize_status(status)
        payload = self._load()
        workflows = self._workflows(payload)
        workflow = self._find_workflow(workflows, workflow_id)
        if workflow is None:
            raise ValueError(f"Unknown workflow: {workflow_id}")
        workflow["status"] = normalized_status
        workflow["updated_at"] = _utc_now_iso()
        payload["workflows"] = workflows
        _write_json(self.path, payload)
        return {"workflow": workflow}

    def describe(self) -> dict[str, Any]:
        workflows = self._workflows(self._load())
        by_status: dict[str, int] = {status: 0 for status in WORKFLOW_STATUSES}
        for workflow in workflows:
            status = str(workflow.get("status") or "draft")
            by_status[status] = by_status.get(status, 0) + 1
        return {
            "enabled": True,
            "backend": "local_workflow_templates",
            "path": str(self.path),
            "workflow_count": len(workflows),
            "by_status": by_status,
            "workflow": "create_template -> review_graph -> activate -> reuse",
        }

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.path, {"version": 1, "updated_at": "", "workflows": []})
        if not isinstance(payload.get("workflows"), list):
            payload["workflows"] = []
        return payload

    def _workflows(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            workflow
            for workflow in payload.get("workflows", [])
            if isinstance(workflow, dict)
        ]

    def _find_workflow(
        self,
        workflows: list[dict[str, Any]],
        workflow_id: str,
    ) -> dict[str, Any] | None:
        for workflow in workflows:
            if str(workflow.get("id") or "") == str(workflow_id or ""):
                return workflow
        return None
