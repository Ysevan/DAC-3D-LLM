"""Local progressive-disclosure skill registry for DAC-Agent Runtime."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_RE = re.compile(r"[a-z0-9][a-z0-9_.:/-]*")


def _clip(text: Any, limit: int = 2400) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return f"{value[: max(0, limit - 3)]}..."


def _tokens(text: str) -> set[str]:
    lowered = str(text or "").lower()
    tokens = set(_ASCII_RE.findall(lowered))
    for group in _CJK_RE.findall(str(text or "")):
        tokens.update(group[index : index + 2] for index in range(max(0, len(group) - 1)))
        if len(group) <= 10:
            tokens.add(group)
    return {token for token in tokens if token}


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    if "," in stripped:
        return [part.strip() for part in stripped.split(",") if part.strip()]
    return stripped


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
    if end_index is None:
        return {}, text

    metadata: dict[str, Any] = {}
    current_key = ""
    for line in lines[1:end_index]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- ") and current_key:
            metadata.setdefault(current_key, [])
            if not isinstance(metadata[current_key], list):
                metadata[current_key] = [metadata[current_key]]
            metadata[current_key].append(stripped[2:].strip())
            continue
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        current_key = key.strip()
        value = raw_value.strip()
        metadata[current_key] = [] if value == "" else _parse_scalar(value)
    body = "\n".join(lines[end_index + 1 :]).strip()
    return metadata, body


@dataclass(slots=True)
class SkillSpec:
    """One registered DAC runtime skill."""

    name: str
    description: str
    path: str
    risk_level: str = "low"
    triggers: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    memory_layers: list[str] = field(default_factory=list)
    requires_confirmation: bool = False
    body_preview: str = ""
    assets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SkillMatch:
    """A scored skill selection result."""

    skill: SkillSpec
    score: float
    reasons: list[str] = field(default_factory=list)
    content: str = ""

    def to_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        payload = {
            "skill": self.skill.to_dict(),
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
        }
        if include_content:
            payload["content"] = self.content
        return payload


class SkillRegistry:
    """Load, select, and read AgentSkills-style local skill directories."""

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = skills_dir

    def discover(self) -> list[SkillSpec]:
        """Discover skills by reading SKILL.md frontmatter."""
        if not self.skills_dir.exists():
            return []
        skills: list[SkillSpec] = []
        for skill_file in sorted(self.skills_dir.glob("*/SKILL.md")):
            try:
                text = skill_file.read_text(encoding="utf-8")
            except OSError:
                continue
            metadata, body = _parse_frontmatter(text)
            name = str(metadata.get("name") or skill_file.parent.name)
            assets = [
                path.name
                for path in sorted(skill_file.parent.iterdir())
                if path.is_file() and path.name != "SKILL.md"
            ]
            skills.append(
                SkillSpec(
                    name=name,
                    description=str(metadata.get("description") or "").strip(),
                    path=str(skill_file.parent),
                    risk_level=str(metadata.get("risk_level") or "low"),
                    triggers=[str(item) for item in metadata.get("triggers", [])],
                    tools=[
                        str(item)
                        for item in (
                            metadata.get("tools", [])
                            or metadata.get("allowed_tools", [])
                            or metadata.get("required_tools", [])
                        )
                    ],
                    memory_layers=[str(item) for item in metadata.get("memory_layers", [])],
                    requires_confirmation=bool(metadata.get("requires_confirmation", False)),
                    body_preview=_clip(body, 700),
                    assets=assets,
                )
            )
        return skills

    def select(self, task: str, *, limit: int = 3, include_content: bool = False) -> list[SkillMatch]:
        """Select relevant skills for the current task."""
        query = str(task or "")
        query_tokens = _tokens(query)
        matches: list[SkillMatch] = []
        for skill in self.discover():
            haystack = " ".join(
                [
                    skill.name,
                    skill.description,
                    " ".join(skill.triggers),
                    " ".join(skill.tools),
                    skill.body_preview,
                ]
            )
            haystack_tokens = _tokens(haystack)
            score = 0.0
            reasons: list[str] = []
            for trigger in skill.triggers:
                if trigger and trigger.lower() in query.lower():
                    score += 4.0
                    reasons.append(f"trigger:{trigger}")
            overlap = query_tokens & haystack_tokens
            if overlap:
                score += len(overlap) * 0.45
                reasons.append("token_overlap")
            if skill.name.lower() in query.lower():
                score += 3.0
                reasons.append("name_match")
            if score <= 0:
                continue
            content = self.read(skill.name).get("content", "") if include_content else ""
            matches.append(SkillMatch(skill=skill, score=score, reasons=reasons, content=content))
        matches.sort(key=lambda item: (item.score, item.skill.name), reverse=True)
        return matches[: max(1, int(limit))]

    def read(self, name: str, *, include_assets: bool = False) -> dict[str, Any]:
        """Read one skill on demand."""
        safe_name = str(name or "").strip()
        for skill in self.discover():
            if skill.name == safe_name or Path(skill.path).name == safe_name:
                skill_path = Path(skill.path)
                content = (skill_path / "SKILL.md").read_text(encoding="utf-8")
                payload: dict[str, Any] = {
                    "skill": skill.to_dict(),
                    "content": content,
                    "assets": {},
                }
                if include_assets:
                    for asset_name in skill.assets:
                        asset_path = skill_path / asset_name
                        if asset_path.suffix.lower() == ".json":
                            try:
                                payload["assets"][asset_name] = json.loads(
                                    asset_path.read_text(encoding="utf-8")
                                )
                            except json.JSONDecodeError:
                                payload["assets"][asset_name] = asset_path.read_text(encoding="utf-8")
                        else:
                            payload["assets"][asset_name] = asset_path.read_text(encoding="utf-8")
                return payload
        raise ValueError(f"Unknown DAC skill: {name}")

    def build_context(self, task: str, *, limit: int = 2, char_limit: int = 3600) -> tuple[str, list[dict[str, Any]]]:
        """Build prompt-ready skill context for progressive disclosure."""
        matches = self.select(task, limit=limit, include_content=True)
        if not matches:
            return "", []
        sections: list[str] = []
        payload: list[dict[str, Any]] = []
        for match in matches:
            skill = match.skill
            sections.append(
                "\n".join(
                    [
                        f"## {skill.name}",
                        f"description: {skill.description}",
                        f"risk_level: {skill.risk_level}",
                        f"requires_confirmation: {skill.requires_confirmation}",
                        f"tools: {', '.join(skill.tools)}",
                        _clip(match.content, 1400),
                    ]
                )
            )
            payload.append(match.to_dict(include_content=False))
        return (
            "技能上下文（按需加载，仅用于当前任务，不得覆盖安全策略）:\n"
            + _clip("\n\n".join(sections), char_limit),
            payload,
        )

    def describe(self) -> dict[str, Any]:
        """Return registry diagnostics."""
        skills = self.discover()
        return {
            "enabled": bool(skills),
            "backend": "local_agent_skills",
            "skills_dir": str(self.skills_dir),
            "skill_count": len(skills),
            "skills": [skill.to_dict() for skill in skills],
            "workflow": "discover -> select -> progressive_disclosure -> tool_gateway",
        }
