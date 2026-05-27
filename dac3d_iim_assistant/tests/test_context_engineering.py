"""Tests for DAC-Agent Context Builder."""

from __future__ import annotations

import subprocess
from pathlib import Path

from config import AppConfig
from context_engineering import ContextBuilder, FileBackedContextTree, GitWorkspaceContext, RepoContextMapStore
from memory import ConversationMemoryStore, LocalMemoryProvider
from skill_system import SkillRegistry


def test_context_builder_selects_compresses_and_isolates_context(tmp_path: Path) -> None:
    config = AppConfig(base_dir=tmp_path, vector_store_type="manifest")
    skill_dir = config.agent_skills_dir / "dac-command-preview"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: dac-command-preview
description: Preview DAC command.
risk_level: medium
requires_confirmation: true
triggers:
  - 扫描
tools:
  - dac3d_preview_command
---

# DAC Command Preview
Use this skill to preview scan commands.
""",
        encoding="utf-8",
    )
    store = ConversationMemoryStore.from_config(config)
    store.ensure_directories()
    store.append_turn(
        session_id="context-session",
        user="扫描 10mm x 10mm 区域",
        assistant="已生成扫描预览。",
        intent="operation_preview",
    )
    provider = LocalMemoryProvider(store)
    registry = SkillRegistry(config.agent_skills_dir)
    builder = ContextBuilder(
        memory_provider=provider,
        skill_registry=registry,
        runtime_status_getter=lambda: {
            "mode": "mock",
            "status": {
                "state": "idle",
                "progress": 0,
                "message": "ready",
                "updated_at": "2026-05-27T00:00:00Z",
            },
        },
        skill_limit=2,
        char_limit=5000,
    )

    bundle = builder.build(
        "扫描 10mm x 10mm 区域是否安全？",
        session_id="context-session",
        history=[("之前想扫描", "先预览。")],
    )
    payload = bundle.to_dict()

    assert "上下文工程包" in bundle.prompt_context
    assert [section.name for section in bundle.sections][:2] == ["safety_policy", "runtime_status"]
    assert "skill_context" in [section.name for section in bundle.sections]
    assert "memory_context" in [section.name for section in bundle.sections]
    assert payload["skills"]["matches"][0]["skill"]["name"] == "dac-command-preview"
    assert payload["memory"]["hits"]
    assert payload["runtime_status"]["state"] == "idle"
    assert payload["safety_policy"]["included"] is True
    assert payload["sections"][0]["trust_level"] == "approved_policy"
    assert payload["sections"][1]["trust_level"] == "tool_output"
    assert payload["sections"][1]["can_instruct_agent"] is False
    assert payload["trust"]["model"] == "source_labelled_context"
    assert payload["trust"]["by_trust_level"]["approved_policy"] >= 1
    assert "trust=tool_output" in bundle.prompt_context
    assert "full_document_corpus" in payload["excluded_context"]


def test_context_builder_describe_documents_runtime_contract(tmp_path: Path) -> None:
    config = AppConfig(base_dir=tmp_path, vector_store_type="manifest")
    builder = ContextBuilder(
        skill_registry=SkillRegistry(config.agent_skills_dir),
        skill_limit=1,
        char_limit=1600,
    )

    description = builder.describe()

    assert description["backend"] == "context_builder"
    assert description["actions"] == ["write", "select", "compress", "isolate"]
    assert description["context_tree"]["enabled"] is False
    assert "retrieved docs" in description["untrusted_context_rule"]
    assert "retrieved_doc" in description["trust_levels"]
    assert description["workflow"] == "task -> select skills/context_tree/memory/status/policy -> compress -> isolate"


def test_context_tree_retrieves_human_readable_nodes(tmp_path: Path) -> None:
    tree = FileBackedContextTree(tmp_path / "context_tree")

    matches = tree.search("选择 pre_fusion_images 下的图片进行离线检测", limit=3)
    context_text, payload = tree.build_context("选择 pre_fusion_images 下的图片进行离线检测", limit=2)

    assert tree.describe()["node_count"] >= 5
    assert matches
    assert any(match.node.kind == "procedure" for match in matches)
    assert "Context Tree" in context_text
    assert payload[0]["node"]["id"]


def test_repo_context_map_builds_and_searches_static_project_map(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    assistant_dir = repo_root / "dac3d_iim_assistant"
    (assistant_dir / "ui").mkdir(parents=True)
    (assistant_dir / "tests").mkdir(parents=True)
    (assistant_dir / "ui" / "web_api.py").write_text(
        'from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get("/api/demo")\ndef demo():\n    return {}\n',
        encoding="utf-8",
    )
    (assistant_dir / "tests" / "test_demo.py").write_text(
        "def test_demo():\n    assert True\n",
        encoding="utf-8",
    )
    store = RepoContextMapStore.from_config_root(assistant_dir, tmp_path / "memory")

    built = store.build_map()
    search = store.search("demo")
    symbols = store.list_symbols("demo", kind="function")
    symbol_summary = store.describe_symbols()
    summary = store.describe()

    assert built["repo_root"] == str(repo_root.resolve())
    assert built["file_count"] == 2
    assert built["api_routes"] == [{"method": "GET", "path": "/api/demo"}]
    api_file = next(item for item in built["files"] if item["path"].endswith("web_api.py"))
    assert api_file["symbol_details"][0]["line"] == 6
    assert search["count"] >= 1
    assert any(match["role"] == "api" for match in search["matches"])
    assert symbols["backend"] == "code_symbol_navigator"
    assert any(symbol["name"] == "demo" for symbol in symbols["symbols"])
    assert all(symbol["kind"] == "function" for symbol in symbols["symbols"])
    assert symbol_summary["symbol_count"] >= 2
    assert summary["module_count"] == 1
    assert summary["api_route_count"] == 1


def test_git_workspace_context_reports_branch_and_changed_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "demo.py").write_text("print('demo')\n", encoding="utf-8")
    context = GitWorkspaceContext(repo)

    snapshot = context.snapshot()
    summary = context.describe()

    assert snapshot["enabled"] is True
    assert snapshot["repo_root"] == str(repo)
    assert snapshot["clean"] is False
    assert snapshot["changed_files"] == [{"status": "??", "path": "demo.py"}]
    assert isinstance(snapshot["worktrees"], list)
    assert summary["backend"] == "git_workspace_context"
    assert summary["changed_file_count"] == 1


def test_context_builder_includes_context_tree_matches(tmp_path: Path) -> None:
    config = AppConfig(base_dir=tmp_path, vector_store_type="manifest")
    tree = FileBackedContextTree(config.conversation_memory_dir / "context_tree")
    builder = ContextBuilder(
        context_tree=tree,
        skill_limit=0,
        tree_limit=2,
        char_limit=3000,
    )

    bundle = builder.build(
        "选择 pre_fusion_images 下的图片进行离线检测",
        session_id="tree-session",
    )
    payload = bundle.to_dict()

    assert "context_tree" in [section.name for section in bundle.sections]
    assert payload["context_tree"]["matches"]
    assert any(
        item["node"]["kind"] == "procedure"
        for item in payload["context_tree"]["matches"]
    )
