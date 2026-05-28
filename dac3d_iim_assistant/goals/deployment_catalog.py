"""Local Agent deployment catalog for low-code DAC-Agent workflows."""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEPLOYMENT_STATUSES = ("draft", "ready", "deployed", "paused", "archived")


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


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "").strip().lower()).strip("-")
    return normalized[:120] or f"deployment-{uuid.uuid4().hex[:8]}"


def _strings(values: list[Any] | tuple[Any, ...] | None, *, limit: int = 80) -> list[str]:
    if not values:
        return []
    result: list[str] = []
    for value in values[:limit]:
        text = _clip(value, 220)
        if text and text not in result:
            result.append(text)
    return result


def _json_dict(value: dict[str, Any] | None, *, field_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(json.dumps(dict(value or {}), ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable.") from exc
    return payload if isinstance(payload, dict) else {}


def _normalize_status(value: str, *, default: str = "draft") -> str:
    status = str(value or default).strip().lower()
    if status not in DEPLOYMENT_STATUSES:
        raise ValueError(f"Deployment status must be one of: {', '.join(DEPLOYMENT_STATUSES)}.")
    return status


def _search_text(deployment: dict[str, Any]) -> str:
    release_text = " ".join(
        " ".join(
            [
                str(release.get("id") or ""),
                str(release.get("version") or ""),
                str(release.get("summary") or ""),
                " ".join(str(item) for item in release.get("artifact_ids") or []),
                " ".join(str(item) for item in release.get("verification_run_ids") or []),
            ]
        )
        for release in deployment.get("releases") or []
    )
    parts = [
        str(deployment.get("id") or ""),
        str(deployment.get("slug") or ""),
        str(deployment.get("name") or ""),
        str(deployment.get("app_type") or ""),
        str(deployment.get("entrypoint") or ""),
        str(deployment.get("environment") or ""),
        str(deployment.get("route_path") or ""),
        str(deployment.get("version") or ""),
        " ".join(str(item) for item in deployment.get("workflow_ids") or []),
        " ".join(str(item) for item in deployment.get("tool_pack_slugs") or []),
        " ".join(str(item) for item in deployment.get("agent_roles") or []),
        " ".join(str(item) for item in deployment.get("config_refs") or []),
        " ".join(str(item) for item in deployment.get("tags") or []),
        release_text,
    ]
    return " ".join(parts).lower()


class AgentDeploymentStore:
    """JSON-backed catalog of deployable Agent apps and release records."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_root(cls, root_dir: Path) -> "AgentDeploymentStore":
        return cls(root_dir / "agent_deployments.json")

    def create_deployment(
        self,
        name: str,
        *,
        entrypoint: str,
        slug: str = "",
        app_type: str = "agent_app",
        version: str = "0.1.0",
        environment: str = "local",
        route_path: str = "",
        status: str = "draft",
        workflow_ids: list[Any] | None = None,
        tool_pack_slugs: list[Any] | None = None,
        agent_roles: list[Any] | None = None,
        config_refs: list[Any] | None = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        created_by: str = "agent",
    ) -> dict[str, Any]:
        clean_name = _clip(name, 180)
        clean_entrypoint = _clip(entrypoint, 1000)
        if not clean_name:
            raise ValueError("Deployment name is required.")
        if not clean_entrypoint:
            raise ValueError("Deployment entrypoint is required.")
        clean_slug = _slugify(slug or clean_name)
        clean_environment = _clip(environment or "local", 120) or "local"
        payload = self._load()
        deployments = self._deployments(payload)
        deployment = self._find_by_identity(
            deployments,
            slug=clean_slug,
            environment=clean_environment,
        )
        now = _utc_now_iso()
        if deployment is None:
            deployment = {
                "id": f"deployment-{uuid.uuid4().hex[:12]}",
                "slug": clean_slug,
                "name": clean_name,
                "app_type": _clip(app_type or "agent_app", 120) or "agent_app",
                "entrypoint": clean_entrypoint,
                "version": _clip(version or "0.1.0", 80) or "0.1.0",
                "environment": clean_environment,
                "route_path": _clip(route_path, 500),
                "status": _normalize_status(status),
                "workflow_ids": _strings(workflow_ids),
                "tool_pack_slugs": _strings(tool_pack_slugs),
                "agent_roles": _strings(agent_roles),
                "config_refs": _strings(config_refs),
                "tags": _strings(tags),
                "metadata": _json_dict(metadata, field_name="Deployment metadata"),
                "created_by": _clip(created_by, 120) or "agent",
                "created_at": now,
                "updated_at": now,
                "deployed_at": now if _normalize_status(status) == "deployed" else "",
                "latest_release_id": "",
                "releases": [],
                "history": [],
            }
            deployments.insert(0, deployment)
            created = True
            event_type = "created"
        else:
            deployment.update(
                {
                    "name": clean_name,
                    "app_type": _clip(app_type or "agent_app", 120) or "agent_app",
                    "entrypoint": clean_entrypoint,
                    "version": _clip(version or "0.1.0", 80) or "0.1.0",
                    "route_path": _clip(route_path, 500),
                    "status": _normalize_status(status),
                    "workflow_ids": _strings(workflow_ids),
                    "tool_pack_slugs": _strings(tool_pack_slugs),
                    "agent_roles": _strings(agent_roles),
                    "config_refs": _strings(config_refs),
                    "tags": _strings(tags),
                    "metadata": _json_dict(metadata, field_name="Deployment metadata"),
                    "updated_at": now,
                }
            )
            if deployment["status"] == "deployed" and not deployment.get("deployed_at"):
                deployment["deployed_at"] = now
            created = False
            event_type = "updated"
        history = self._append_history(deployment, event_type, actor=created_by or "agent")
        payload["deployments"] = deployments[:200]
        _write_json(self.path, payload)
        return {"deployment": deployment, "history": history, "created": created}

    def list_deployments(
        self,
        *,
        status: str | None = None,
        environment: str | None = None,
        app_type: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        deployments = self._deployments(payload)
        if status:
            normalized_status = _normalize_status(status)
            deployments = [
                deployment
                for deployment in deployments
                if str(deployment.get("status") or "") == normalized_status
            ]
        if environment:
            clean_environment = _clip(environment, 120)
            deployments = [
                deployment
                for deployment in deployments
                if str(deployment.get("environment") or "") == clean_environment
            ]
        if app_type:
            clean_app_type = _clip(app_type, 120)
            deployments = [
                deployment
                for deployment in deployments
                if str(deployment.get("app_type") or "") == clean_app_type
            ]
        if tag:
            clean_tag = _clip(tag, 220)
            deployments = [
                deployment
                for deployment in deployments
                if clean_tag in [str(value) for value in deployment.get("tags") or []]
            ]
        if query:
            needle = str(query or "").strip().lower()
            deployments = [deployment for deployment in deployments if needle and needle in _search_text(deployment)]
        safe_limit = max(1, min(100, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_agent_deployment_catalog",
            "path": str(self.path),
            "deployments": [self._deployment_summary(deployment) for deployment in deployments[:safe_limit]],
            "count": len(deployments[:safe_limit]),
            "total_count": len(self._deployments(payload)),
            "statuses": list(DEPLOYMENT_STATUSES),
            "workflow": "workflow_builder -> deployment_catalog -> release_record",
        }

    def read_deployment(self, deployment_id_or_slug: str) -> dict[str, Any]:
        deployments = self._deployments(self._load())
        deployment = self._find_by_id(deployments, deployment_id_or_slug) or self._find_by_slug(
            deployments,
            deployment_id_or_slug,
        )
        if deployment is None:
            raise ValueError(f"Unknown Agent deployment: {deployment_id_or_slug}")
        return {"enabled": True, "backend": "local_agent_deployment_catalog", "deployment": deployment}

    def update_status(
        self,
        deployment_id_or_slug: str,
        status: str,
        *,
        note: str = "",
        actor: str = "agent",
    ) -> dict[str, Any]:
        payload = self._load()
        deployments = self._deployments(payload)
        deployment = self._find_by_id(deployments, deployment_id_or_slug) or self._find_by_slug(
            deployments,
            deployment_id_or_slug,
        )
        if deployment is None:
            raise ValueError(f"Unknown Agent deployment: {deployment_id_or_slug}")
        old_status = str(deployment.get("status") or "draft")
        new_status = _normalize_status(status)
        now = _utc_now_iso()
        deployment["status"] = new_status
        deployment["updated_at"] = now
        if new_status == "deployed":
            deployment["deployed_at"] = now
        history = self._append_history(
            deployment,
            "status_changed",
            actor=actor,
            note=note or f"{old_status} -> {new_status}",
            extra={"from": old_status, "to": new_status},
        )
        _write_json(self.path, payload)
        return {"deployment": deployment, "history": history}

    def record_release(
        self,
        deployment_id_or_slug: str,
        *,
        version: str,
        summary: str = "",
        artifact_ids: list[Any] | None = None,
        verification_run_ids: list[Any] | None = None,
        released_by: str = "agent",
    ) -> dict[str, Any]:
        clean_version = _clip(version, 80)
        if not clean_version:
            raise ValueError("Release version is required.")
        payload = self._load()
        deployments = self._deployments(payload)
        deployment = self._find_by_id(deployments, deployment_id_or_slug) or self._find_by_slug(
            deployments,
            deployment_id_or_slug,
        )
        if deployment is None:
            raise ValueError(f"Unknown Agent deployment: {deployment_id_or_slug}")
        now = _utc_now_iso()
        release = {
            "id": f"release-{uuid.uuid4().hex[:12]}",
            "version": clean_version,
            "summary": _clip(summary, 1200),
            "artifact_ids": _strings(artifact_ids),
            "verification_run_ids": _strings(verification_run_ids),
            "released_by": _clip(released_by, 120) or "agent",
            "created_at": now,
        }
        releases = list(deployment.get("releases") or [])
        releases.insert(0, release)
        deployment["releases"] = releases[:100]
        deployment["latest_release_id"] = release["id"]
        deployment["version"] = clean_version
        deployment["updated_at"] = now
        history = self._append_history(
            deployment,
            "release_recorded",
            actor=released_by or "agent",
            note=summary or f"Release {clean_version} recorded.",
            extra={"release_id": release["id"], "version": clean_version},
        )
        _write_json(self.path, payload)
        return {"deployment": deployment, "release": release, "history": history}

    def describe(self) -> dict[str, Any]:
        deployments = self._deployments(self._load())
        by_status = {status: 0 for status in DEPLOYMENT_STATUSES}
        by_environment: dict[str, int] = {}
        by_app_type: dict[str, int] = {}
        release_count = 0
        for deployment in deployments:
            status = str(deployment.get("status") or "draft")
            by_status[status] = by_status.get(status, 0) + 1
            environment = str(deployment.get("environment") or "local")
            by_environment[environment] = by_environment.get(environment, 0) + 1
            app_type = str(deployment.get("app_type") or "agent_app")
            by_app_type[app_type] = by_app_type.get(app_type, 0) + 1
            release_count += len(deployment.get("releases") or [])
        return {
            "enabled": True,
            "backend": "local_agent_deployment_catalog",
            "path": str(self.path),
            "deployment_count": len(deployments),
            "release_count": release_count,
            "by_status": by_status,
            "by_environment": by_environment,
            "by_app_type": by_app_type,
            "latest_deployment": self._deployment_summary(deployments[0]) if deployments else {},
            "workflow": "low_code_agent_app -> deployment_catalog -> release_history",
        }

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.path, {"version": 1, "updated_at": "", "deployments": []})
        if not isinstance(payload.get("deployments"), list):
            payload["deployments"] = []
        return payload

    def _deployments(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [deployment for deployment in payload.get("deployments", []) if isinstance(deployment, dict)]

    def _find_by_id(self, deployments: list[dict[str, Any]], deployment_id: str) -> dict[str, Any] | None:
        for deployment in deployments:
            if str(deployment.get("id") or "") == str(deployment_id or ""):
                return deployment
        return None

    def _find_by_slug(self, deployments: list[dict[str, Any]], slug: str) -> dict[str, Any] | None:
        clean_slug = _slugify(slug)
        for deployment in deployments:
            if str(deployment.get("slug") or "") == clean_slug:
                return deployment
        return None

    def _find_by_identity(
        self,
        deployments: list[dict[str, Any]],
        *,
        slug: str,
        environment: str,
    ) -> dict[str, Any] | None:
        for deployment in deployments:
            if (
                str(deployment.get("slug") or "") == slug
                and str(deployment.get("environment") or "") == environment
            ):
                return deployment
        return None

    def _append_history(
        self,
        deployment: dict[str, Any],
        event_type: str,
        *,
        actor: str,
        note: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        history = {
            "at": _utc_now_iso(),
            "type": event_type,
            "actor": _clip(actor, 120) or "agent",
            "note": _clip(note, 800),
            **(extra or {}),
        }
        events = list(deployment.get("history") or [])
        events.append(history)
        deployment["history"] = events[-80:]
        return history

    def _deployment_summary(self, deployment: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": deployment.get("id"),
            "slug": deployment.get("slug"),
            "name": deployment.get("name"),
            "app_type": deployment.get("app_type"),
            "entrypoint": deployment.get("entrypoint"),
            "version": deployment.get("version"),
            "environment": deployment.get("environment"),
            "route_path": deployment.get("route_path"),
            "status": deployment.get("status"),
            "workflow_ids": list(deployment.get("workflow_ids") or []),
            "tool_pack_slugs": list(deployment.get("tool_pack_slugs") or []),
            "agent_roles": list(deployment.get("agent_roles") or []),
            "tags": list(deployment.get("tags") or []),
            "release_count": len(deployment.get("releases") or []),
            "latest_release_id": deployment.get("latest_release_id"),
            "deployed_at": deployment.get("deployed_at"),
            "updated_at": deployment.get("updated_at"),
        }
