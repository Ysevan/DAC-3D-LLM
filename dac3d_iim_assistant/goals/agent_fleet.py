"""Local Agent fleet control-plane records for DAC-Agent workspaces."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


FLEET_STATUSES = ("ready", "busy", "paused", "offline", "archived")
ASSIGNMENT_STATUSES = ("queued", "running", "completed", "failed", "cancelled")


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


def _normalize_status(value: str, *, default: str = "ready") -> str:
    status = str(value or default).strip().lower()
    if status not in FLEET_STATUSES:
        raise ValueError(f"Fleet status must be one of: {', '.join(FLEET_STATUSES)}.")
    return status


def _normalize_assignment_status(value: str, *, default: str = "queued") -> str:
    status = str(value or default).strip().lower()
    if status not in ASSIGNMENT_STATUSES:
        raise ValueError(f"Assignment status must be one of: {', '.join(ASSIGNMENT_STATUSES)}.")
    return status


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    return max(minimum, min(maximum, number))


def _search_text(instance: dict[str, Any]) -> str:
    assignment_text = " ".join(
        " ".join(
            [
                str(assignment.get("id") or ""),
                str(assignment.get("task_id") or ""),
                str(assignment.get("summary") or ""),
                str(assignment.get("thread_id") or ""),
                str(assignment.get("workflow_id") or ""),
            ]
        )
        for assignment in instance.get("assignments") or []
    )
    parts = [
        str(instance.get("id") or ""),
        str(instance.get("name") or ""),
        str(instance.get("agent_role") or ""),
        str(instance.get("environment") or ""),
        str(instance.get("endpoint") or ""),
        " ".join(str(item) for item in instance.get("capabilities") or []),
        " ".join(str(item) for item in instance.get("tags") or []),
        assignment_text,
    ]
    return " ".join(parts).lower()


class AgentFleetStore:
    """JSON-backed Agent fleet control plane for local multi-agent runtimes."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_root(cls, root_dir: Path) -> "AgentFleetStore":
        return cls(root_dir / "agent_fleet.json")

    def register_instance(
        self,
        name: str,
        *,
        agent_role: str,
        environment: str = "local",
        endpoint: str = "",
        status: str = "ready",
        capabilities: list[Any] | None = None,
        max_concurrency: int = 1,
        current_load: int = 0,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        owner_agent: str = "agent",
    ) -> dict[str, Any]:
        clean_name = _clip(name, 180)
        clean_role = _clip(agent_role, 120)
        if not clean_name:
            raise ValueError("Fleet instance name is required.")
        if not clean_role:
            raise ValueError("Fleet instance agent_role is required.")

        payload = self._load()
        instances = self._instances(payload)
        instance = self._find_by_identity(
            instances,
            name=clean_name,
            agent_role=clean_role,
            environment=environment,
        )
        now = _utc_now_iso()
        if instance is None:
            instance = {
                "id": f"fleet-{uuid.uuid4().hex[:12]}",
                "name": clean_name,
                "agent_role": clean_role,
                "environment": _clip(environment or "local", 120) or "local",
                "endpoint": _clip(endpoint, 1000),
                "status": _normalize_status(status),
                "capabilities": _strings(capabilities),
                "max_concurrency": _bounded_int(
                    max_concurrency,
                    default=1,
                    minimum=1,
                    maximum=100,
                    field_name="max_concurrency",
                ),
                "current_load": _bounded_int(
                    current_load,
                    default=0,
                    minimum=0,
                    maximum=100,
                    field_name="current_load",
                ),
                "tags": _strings(tags),
                "metadata": _json_dict(metadata, field_name="Fleet metadata"),
                "owner_agent": _clip(owner_agent, 120) or "agent",
                "last_heartbeat_at": "",
                "created_at": now,
                "updated_at": now,
                "assignments": [],
                "history": [],
            }
            instances.insert(0, instance)
            created = True
            event_type = "registered"
        else:
            instance.update(
                {
                    "endpoint": _clip(endpoint, 1000),
                    "status": _normalize_status(status),
                    "capabilities": _strings(capabilities),
                    "max_concurrency": _bounded_int(
                        max_concurrency,
                        default=1,
                        minimum=1,
                        maximum=100,
                        field_name="max_concurrency",
                    ),
                    "current_load": _bounded_int(
                        current_load,
                        default=0,
                        minimum=0,
                        maximum=100,
                        field_name="current_load",
                    ),
                    "tags": _strings(tags),
                    "metadata": _json_dict(metadata, field_name="Fleet metadata"),
                    "owner_agent": _clip(owner_agent, 120) or "agent",
                    "updated_at": now,
                }
            )
            created = False
            event_type = "updated"
        history = self._append_history(instance, event_type, actor=owner_agent or "agent")
        payload["instances"] = instances[:200]
        _write_json(self.path, payload)
        return {"instance": instance, "history": history, "created": created}

    def list_instances(
        self,
        *,
        status: str | None = None,
        agent_role: str | None = None,
        environment: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        instances = self._instances(payload)
        if status:
            normalized_status = _normalize_status(status)
            instances = [
                instance for instance in instances if str(instance.get("status") or "") == normalized_status
            ]
        if agent_role:
            clean_role = _clip(agent_role, 120)
            instances = [instance for instance in instances if str(instance.get("agent_role") or "") == clean_role]
        if environment:
            clean_environment = _clip(environment, 120)
            instances = [
                instance for instance in instances if str(instance.get("environment") or "") == clean_environment
            ]
        if tag:
            clean_tag = _clip(tag, 220)
            instances = [
                instance for instance in instances if clean_tag in [str(value) for value in instance.get("tags") or []]
            ]
        if query:
            needle = str(query or "").strip().lower()
            instances = [instance for instance in instances if needle and needle in _search_text(instance)]
        safe_limit = max(1, min(100, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_agent_fleet",
            "path": str(self.path),
            "instances": [self._instance_summary(instance) for instance in instances[:safe_limit]],
            "count": len(instances[:safe_limit]),
            "total_count": len(self._instances(payload)),
            "statuses": list(FLEET_STATUSES),
            "workflow": "register_instance -> heartbeat -> assign_task -> update_assignment",
        }

    def read_instance(self, instance_id_or_name: str) -> dict[str, Any]:
        instance = self._find_instance(instance_id_or_name)
        if instance is None:
            raise ValueError(f"Unknown fleet instance: {instance_id_or_name}")
        return {"enabled": True, "backend": "local_agent_fleet", "instance": instance}

    def heartbeat(
        self,
        instance_id_or_name: str,
        *,
        status: str = "ready",
        current_load: int | None = None,
        metrics: dict[str, Any] | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        payload = self._load()
        instance = self._find_in_instances(self._instances(payload), instance_id_or_name)
        if instance is None:
            raise ValueError(f"Unknown fleet instance: {instance_id_or_name}")
        instance["status"] = _normalize_status(status)
        if current_load is not None:
            instance["current_load"] = _bounded_int(
                current_load,
                default=0,
                minimum=0,
                maximum=100,
                field_name="current_load",
            )
        instance["metrics"] = _json_dict(metrics, field_name="Fleet metrics")
        instance["last_heartbeat_at"] = _utc_now_iso()
        instance["updated_at"] = instance["last_heartbeat_at"]
        history = self._append_history(
            instance,
            "heartbeat",
            actor=str(instance.get("name") or "fleet"),
            note=note or "Fleet heartbeat recorded.",
        )
        _write_json(self.path, payload)
        return {"instance": instance, "history": history}

    def assign_task(
        self,
        instance_id_or_name: str,
        *,
        task_id: str,
        summary: str = "",
        thread_id: str = "",
        workflow_id: str = "",
        priority: str = "normal",
        metadata: dict[str, Any] | None = None,
        assigned_by: str = "coordinator",
    ) -> dict[str, Any]:
        clean_task_id = _clip(task_id, 180)
        if not clean_task_id:
            raise ValueError("task_id is required.")
        payload = self._load()
        instance = self._find_in_instances(self._instances(payload), instance_id_or_name)
        if instance is None:
            raise ValueError(f"Unknown fleet instance: {instance_id_or_name}")
        now = _utc_now_iso()
        assignment = {
            "id": f"assignment-{uuid.uuid4().hex[:12]}",
            "task_id": clean_task_id,
            "summary": _clip(summary, 1000),
            "thread_id": str(thread_id or ""),
            "workflow_id": str(workflow_id or ""),
            "priority": _clip(priority or "normal", 80) or "normal",
            "status": "queued",
            "metadata": _json_dict(metadata, field_name="Assignment metadata"),
            "assigned_by": _clip(assigned_by, 120) or "coordinator",
            "created_at": now,
            "updated_at": now,
        }
        assignments = list(instance.get("assignments") or [])
        assignments.insert(0, assignment)
        instance["assignments"] = assignments[:200]
        instance["current_load"] = min(
            int(instance.get("max_concurrency") or 1),
            int(instance.get("current_load") or 0) + 1,
        )
        if instance["current_load"] > 0 and instance["status"] not in {"paused", "offline"}:
            instance["status"] = "busy"
        instance["updated_at"] = now
        history = self._append_history(
            instance,
            "task_assigned",
            actor=assigned_by or "coordinator",
            note=f"Assigned task {clean_task_id}.",
        )
        _write_json(self.path, payload)
        return {"instance": instance, "assignment": assignment, "history": history}

    def update_assignment_status(
        self,
        instance_id_or_name: str,
        assignment_id: str,
        status: str,
        *,
        note: str = "",
        actor: str = "agent",
    ) -> dict[str, Any]:
        payload = self._load()
        instance = self._find_in_instances(self._instances(payload), instance_id_or_name)
        if instance is None:
            raise ValueError(f"Unknown fleet instance: {instance_id_or_name}")
        assignment = self._find_assignment(instance, assignment_id)
        if assignment is None:
            raise ValueError(f"Unknown assignment: {assignment_id}")
        old_status = str(assignment.get("status") or "queued")
        new_status = _normalize_assignment_status(status)
        assignment["status"] = new_status
        assignment["updated_at"] = _utc_now_iso()
        active_assignments = [
            item
            for item in instance.get("assignments") or []
            if str(item.get("status") or "") in {"queued", "running"}
        ]
        instance["current_load"] = min(
            int(instance.get("max_concurrency") or 1),
            len(active_assignments),
        )
        if instance["status"] != "paused" and instance["status"] != "offline":
            instance["status"] = "busy" if instance["current_load"] else "ready"
        instance["updated_at"] = assignment["updated_at"]
        history = self._append_history(
            instance,
            "assignment_status_changed",
            actor=actor,
            note=note or f"{old_status} -> {new_status}",
            extra={"assignment_id": assignment_id, "from": old_status, "to": new_status},
        )
        _write_json(self.path, payload)
        return {"instance": instance, "assignment": assignment, "history": history}

    def describe(self) -> dict[str, Any]:
        instances = self._instances(self._load())
        by_status = {status: 0 for status in FLEET_STATUSES}
        by_environment: dict[str, int] = {}
        active_assignments = 0
        total_capacity = 0
        current_load = 0
        for instance in instances:
            status = str(instance.get("status") or "ready")
            by_status[status] = by_status.get(status, 0) + 1
            environment = str(instance.get("environment") or "local")
            by_environment[environment] = by_environment.get(environment, 0) + 1
            total_capacity += int(instance.get("max_concurrency") or 0)
            current_load += int(instance.get("current_load") or 0)
            active_assignments += len(
                [
                    item
                    for item in instance.get("assignments") or []
                    if str(item.get("status") or "") in {"queued", "running"}
                ]
            )
        return {
            "enabled": True,
            "backend": "local_agent_fleet",
            "path": str(self.path),
            "instance_count": len(instances),
            "active_assignment_count": active_assignments,
            "total_capacity": total_capacity,
            "current_load": current_load,
            "by_status": by_status,
            "by_environment": by_environment,
            "latest_instance": self._instance_summary(instances[0]) if instances else {},
            "workflow": "agent_fleet -> workload_assignment -> heartbeat_observation",
        }

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.path, {"version": 1, "updated_at": "", "instances": []})
        if not isinstance(payload.get("instances"), list):
            payload["instances"] = []
        return payload

    def _instances(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [instance for instance in payload.get("instances", []) if isinstance(instance, dict)]

    def _find_instance(self, instance_id_or_name: str) -> dict[str, Any] | None:
        return self._find_in_instances(self._instances(self._load()), instance_id_or_name)

    def _find_in_instances(
        self,
        instances: list[dict[str, Any]],
        instance_id_or_name: str,
    ) -> dict[str, Any] | None:
        for instance in instances:
            if str(instance.get("id") or "") == str(instance_id_or_name or ""):
                return instance
            if str(instance.get("name") or "") == str(instance_id_or_name or ""):
                return instance
        return None

    def _find_by_identity(
        self,
        instances: list[dict[str, Any]],
        *,
        name: str,
        agent_role: str,
        environment: str,
    ) -> dict[str, Any] | None:
        for instance in instances:
            if (
                str(instance.get("name") or "") == name
                and str(instance.get("agent_role") or "") == agent_role
                and str(instance.get("environment") or "") == str(environment or "local")
            ):
                return instance
        return None

    def _find_assignment(self, instance: dict[str, Any], assignment_id: str) -> dict[str, Any] | None:
        for assignment in instance.get("assignments") or []:
            if str(assignment.get("id") or "") == str(assignment_id or ""):
                return assignment
        return None

    def _append_history(
        self,
        instance: dict[str, Any],
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
        events = list(instance.get("history") or [])
        events.append(history)
        instance["history"] = events[-80:]
        return history

    def _instance_summary(self, instance: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": instance.get("id"),
            "name": instance.get("name"),
            "agent_role": instance.get("agent_role"),
            "environment": instance.get("environment"),
            "endpoint": instance.get("endpoint"),
            "status": instance.get("status"),
            "capabilities": list(instance.get("capabilities") or []),
            "max_concurrency": int(instance.get("max_concurrency") or 0),
            "current_load": int(instance.get("current_load") or 0),
            "tags": list(instance.get("tags") or []),
            "assignment_count": len(instance.get("assignments") or []),
            "last_heartbeat_at": instance.get("last_heartbeat_at"),
            "updated_at": instance.get("updated_at"),
        }
