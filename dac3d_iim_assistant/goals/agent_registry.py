"""Local Agent registry for DAC-Agent workspace discovery."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


AGENT_STATUSES = ("active", "paused", "archived")


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


def _strings(values: list[Any] | tuple[Any, ...] | None, *, limit: int = 50) -> list[str]:
    if not values:
        return []
    result: list[str] = []
    for value in values[:limit]:
        text = _clip(value, 180)
        if text and text not in result:
            result.append(text)
    return result


def _json_dict(value: dict[str, Any] | None, *, field_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(json.dumps(dict(value or {}), ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable.") from exc
    return payload if isinstance(payload, dict) else {}


def _normalize_status(value: str, *, default: str = "active") -> str:
    status = str(value or default).strip().lower()
    if status not in AGENT_STATUSES:
        raise ValueError(f"Agent status must be one of: {', '.join(AGENT_STATUSES)}.")
    return status


def _search_text(agent: dict[str, Any]) -> str:
    parts = [
        str(agent.get("id") or ""),
        str(agent.get("name") or ""),
        str(agent.get("role") or ""),
        str(agent.get("description") or ""),
        str(agent.get("handoff_name") or ""),
        " ".join(str(item) for item in agent.get("capabilities") or []),
        " ".join(str(item) for item in agent.get("tools") or []),
        " ".join(str(item) for item in agent.get("triggers") or []),
        " ".join(str(item) for item in agent.get("tags") or []),
    ]
    return " ".join(parts).lower()


def _tokenize(value: str) -> list[str]:
    text = str(value or "").strip().lower()
    if not text:
        return []
    separators = " \t\r\n,.;:，。；：、/\\|()[]{}<>!?！？\"'"
    tokens: list[str] = []
    current: list[str] = []
    for char in text:
        if char in separators:
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
    if current:
        tokens.append("".join(current))
    return [token for token in tokens if token]


class AgentRegistryStore:
    """JSON-backed Agent registry for multi-agent discovery and routing hints."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def from_root(cls, root_dir: Path) -> "AgentRegistryStore":
        return cls(root_dir / "agent_registry.json")

    def seed_defaults(self, agents: list[dict[str, Any]]) -> dict[str, Any]:
        """Register built-in agents when the local registry is empty or missing roles."""
        created = 0
        updated = 0
        for agent in agents:
            result = self.register_agent(
                name=str(agent.get("name") or agent.get("role") or ""),
                role=str(agent.get("role") or ""),
                description=str(agent.get("description") or ""),
                status=str(agent.get("status") or "active"),
                handoff_name=str(agent.get("handoff_name") or ""),
                agent_type=str(agent.get("agent_type") or "specialist"),
                capabilities=agent.get("capabilities") if isinstance(agent.get("capabilities"), list) else None,
                tools=agent.get("tools") if isinstance(agent.get("tools"), list) else None,
                triggers=agent.get("triggers") if isinstance(agent.get("triggers"), list) else None,
                tags=agent.get("tags") if isinstance(agent.get("tags"), list) else None,
                metadata=agent.get("metadata") if isinstance(agent.get("metadata"), dict) else None,
                owner_agent=str(agent.get("owner_agent") or "system"),
                seed=True,
            )
            if result["created"]:
                created += 1
            else:
                updated += 1
        return {"enabled": True, "backend": "local_agent_registry", "created": created, "updated": updated}

    def register_agent(
        self,
        name: str,
        *,
        role: str,
        description: str = "",
        status: str = "active",
        handoff_name: str = "",
        agent_type: str = "specialist",
        capabilities: list[Any] | None = None,
        tools: list[Any] | None = None,
        triggers: list[Any] | None = None,
        tags: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        owner_agent: str = "system",
        seed: bool = False,
    ) -> dict[str, Any]:
        clean_role = _clip(role, 120)
        clean_name = _clip(name, 180)
        if not clean_role:
            raise ValueError("Agent role is required.")
        if not clean_name:
            raise ValueError("Agent name is required.")

        payload = self._load()
        agents = self._agents(payload)
        agent = self._find_by_role(agents, clean_role)
        now = _utc_now_iso()
        event_type = "seeded" if seed else "registered"
        if agent is None:
            agent = {
                "id": f"agent-{uuid.uuid4().hex[:12]}",
                "name": clean_name,
                "role": clean_role,
                "description": _clip(description, 1000),
                "status": _normalize_status(status),
                "agent_type": _clip(agent_type or "specialist", 80) or "specialist",
                "handoff_name": _clip(handoff_name, 160),
                "capabilities": _strings(capabilities),
                "tools": _strings(tools),
                "triggers": _strings(triggers),
                "tags": _strings(tags),
                "metadata": _json_dict(metadata, field_name="Agent metadata"),
                "owner_agent": _clip(owner_agent, 120) or "system",
                "created_at": now,
                "updated_at": now,
                "history": [],
            }
            agents.insert(0, agent)
            created = True
        else:
            agent.update(
                {
                    "name": clean_name,
                    "description": _clip(description, 1000),
                    "status": _normalize_status(status),
                    "agent_type": _clip(agent_type or "specialist", 80) or "specialist",
                    "handoff_name": _clip(handoff_name, 160),
                    "capabilities": _strings(capabilities),
                    "tools": _strings(tools),
                    "triggers": _strings(triggers),
                    "tags": _strings(tags),
                    "metadata": _json_dict(metadata, field_name="Agent metadata"),
                    "owner_agent": _clip(owner_agent, 120) or "system",
                    "updated_at": now,
                }
            )
            created = False
            event_type = "seed_refreshed" if seed else "updated"

        history = self._append_history(agent, event_type, actor=owner_agent or "system")
        payload["agents"] = agents[:100]
        _write_json(self.path, payload)
        return {"agent": agent, "history": history, "created": created}

    def list_agents(
        self,
        *,
        status: str | None = None,
        role: str | None = None,
        capability: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        agents = self._agents(payload)
        if status:
            normalized_status = _normalize_status(status)
            agents = [agent for agent in agents if str(agent.get("status") or "") == normalized_status]
        if role:
            clean_role = _clip(role, 120)
            agents = [agent for agent in agents if str(agent.get("role") or "") == clean_role]
        if capability:
            clean_capability = _clip(capability, 180).lower()
            agents = [
                agent
                for agent in agents
                if clean_capability in [str(item).lower() for item in agent.get("capabilities") or []]
            ]
        if query:
            needle = str(query or "").strip().lower()
            agents = [agent for agent in agents if needle and needle in _search_text(agent)]
        safe_limit = max(1, min(100, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_agent_registry",
            "path": str(self.path),
            "agents": agents[:safe_limit],
            "count": len(agents[:safe_limit]),
            "total_count": len(self._agents(payload)),
            "statuses": list(AGENT_STATUSES),
            "workflow": "agent_register -> capability_discovery -> route_candidates",
        }

    def read_agent(self, agent_id_or_role: str) -> dict[str, Any]:
        agents = self._agents(self._load())
        agent = self._find_by_id(agents, agent_id_or_role) or self._find_by_role(agents, agent_id_or_role)
        if agent is None:
            raise ValueError(f"Unknown agent: {agent_id_or_role}")
        return {"enabled": True, "backend": "local_agent_registry", "agent": agent}

    def route_candidates(self, task: str, *, limit: int = 5) -> dict[str, Any]:
        clean_task = _clip(task, 1000)
        tokens = _tokenize(clean_task)
        agents = [
            agent
            for agent in self._agents(self._load())
            if str(agent.get("status") or "") == "active"
        ]
        scored: list[dict[str, Any]] = []
        for agent in agents:
            score, matched = self._score_agent(agent, clean_task.lower(), tokens)
            if score > 0:
                scored.append(
                    {
                        "agent": agent,
                        "score": score,
                        "matched": matched,
                        "reason": f"Matched {len(matched)} registry hints for role {agent.get('role')}.",
                    }
                )
        scored.sort(key=lambda item: (-int(item["score"]), str(item["agent"].get("role") or "")))
        safe_limit = max(1, min(20, int(limit or 5)))
        return {
            "enabled": True,
            "backend": "local_agent_registry",
            "task": clean_task,
            "candidates": scored[:safe_limit],
            "count": len(scored[:safe_limit]),
            "total_count": len(scored),
            "workflow": "task_text -> registry_hint_match -> specialist_route_candidates",
        }

    def describe(self) -> dict[str, Any]:
        agents = self._agents(self._load())
        by_status = {status: 0 for status in AGENT_STATUSES}
        by_type: dict[str, int] = {}
        capabilities: set[str] = set()
        for agent in agents:
            status = str(agent.get("status") or "active")
            by_status[status] = by_status.get(status, 0) + 1
            agent_type = str(agent.get("agent_type") or "specialist")
            by_type[agent_type] = by_type.get(agent_type, 0) + 1
            for capability in agent.get("capabilities") or []:
                capabilities.add(str(capability))
        return {
            "enabled": True,
            "backend": "local_agent_registry",
            "path": str(self.path),
            "agent_count": len(agents),
            "by_status": by_status,
            "by_type": by_type,
            "capability_count": len(capabilities),
            "latest_agent": agents[0] if agents else {},
            "workflow": "agent_registry -> discover_specialists -> route_or_handoff",
        }

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.path, {"version": 1, "updated_at": "", "agents": []})
        if not isinstance(payload.get("agents"), list):
            payload["agents"] = []
        return payload

    def _agents(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [agent for agent in payload.get("agents", []) if isinstance(agent, dict)]

    def _find_by_id(self, agents: list[dict[str, Any]], agent_id: str) -> dict[str, Any] | None:
        for agent in agents:
            if str(agent.get("id") or "") == str(agent_id or ""):
                return agent
        return None

    def _find_by_role(self, agents: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
        for agent in agents:
            if str(agent.get("role") or "") == str(role or ""):
                return agent
        return None

    def _append_history(self, agent: dict[str, Any], event_type: str, *, actor: str) -> dict[str, Any]:
        history = {
            "at": _utc_now_iso(),
            "type": event_type,
            "actor": _clip(actor or "system", 120),
        }
        events = list(agent.get("history") or [])
        events.append(history)
        agent["history"] = events[-20:]
        return history

    def _score_agent(self, agent: dict[str, Any], task_text: str, task_tokens: list[str]) -> tuple[int, list[str]]:
        score = 0
        matched: list[str] = []
        weighted_fields = [
            ("triggers", 5),
            ("capabilities", 3),
            ("tools", 2),
            ("tags", 2),
        ]
        for field, weight in weighted_fields:
            for hint in agent.get(field) or []:
                text = str(hint or "").strip().lower()
                if text and text in task_text:
                    score += weight
                    matched.append(f"{field}:{hint}")
        searchable = _search_text(agent)
        for token in task_tokens:
            if len(token) >= 2 and token in searchable:
                score += 1
                matched.append(f"token:{token}")
        return score, matched[:20]
