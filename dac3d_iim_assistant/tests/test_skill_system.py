"""Tests for DAC-Agent local skill registry."""

from __future__ import annotations

from pathlib import Path

from config import AppConfig
from skill_system import SkillRegistry


def test_skill_registry_discovers_and_selects_skills(tmp_path: Path) -> None:
    config = AppConfig(base_dir=tmp_path, vector_store_type="manifest")
    skill_dir = config.agent_skills_dir / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: demo-skill
description: Demo scan preview skill.
risk_level: medium
requires_confirmation: true
triggers:
  - 扫描
tools:
  - dac3d_preview_command
memory_layers:
  - procedure_memory
---

# Demo Skill

Use this skill to preview scan commands.
""",
        encoding="utf-8",
    )
    (skill_dir / "schema.json").write_text('{"kind": "demo"}', encoding="utf-8")
    registry = SkillRegistry(config.agent_skills_dir)

    skills = registry.discover()
    matches = registry.select("扫描 10mm x 10mm 区域", include_content=True)
    loaded = registry.read("demo-skill", include_assets=True)
    context, context_matches = registry.build_context("扫描 10mm x 10mm 区域")

    assert len(skills) == 1
    assert skills[0].name == "demo-skill"
    assert skills[0].requires_confirmation is True
    assert matches[0].skill.name == "demo-skill"
    assert "Demo Skill" in matches[0].content
    assert loaded["assets"]["schema.json"]["kind"] == "demo"
    assert "技能上下文" in context
    assert context_matches[0]["skill"]["name"] == "demo-skill"


def test_builtin_dac_skills_are_available() -> None:
    config = AppConfig()
    registry = SkillRegistry(config.agent_skills_dir)

    summary = registry.describe()
    selected = registry.select("离线检测前需要做哪些安全检查？", limit=3)

    assert summary["enabled"] is True
    assert summary["skill_count"] >= 8
    skill_names = {skill["name"] for skill in summary["skills"]}
    assert "dac-offline-inspection" in skill_names
    assert "dac-safety-approval" in skill_names
    assert "dac-context-engineering" in skill_names
    assert selected
    assert any(match.skill.name in {"dac-offline-inspection", "dac-safety-approval"} for match in selected)


def test_skill_registry_accepts_agent_skill_allowed_tools_metadata(tmp_path: Path) -> None:
    config = AppConfig(base_dir=tmp_path, vector_store_type="manifest")
    skill_dir = config.agent_skills_dir / "allowed-tools-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: allowed-tools-skill
description: Uses AgentSkills compatible allowed_tools metadata.
triggers:
  - 上下文
allowed_tools:
  - conversation_memory_search
---

# Allowed Tools Skill
""",
        encoding="utf-8",
    )
    registry = SkillRegistry(config.agent_skills_dir)

    skill = registry.discover()[0]

    assert skill.tools == ["conversation_memory_search"]
