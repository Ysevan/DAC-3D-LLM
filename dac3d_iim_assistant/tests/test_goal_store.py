"""Tests for DAC-Agent goal tracking."""

from __future__ import annotations

from pathlib import Path

from goals import GoalStore, TaskBoardStore


def test_goal_store_tracks_progress_and_completion(tmp_path: Path) -> None:
    store = GoalStore.from_root(tmp_path)

    created = store.create_goal("目标：升级 DAC-Agent 工作流", session_id="goal-session")
    goal_id = created["goal"]["id"]
    progress = store.append_progress(goal_id, "已完成 Context Tree 接入。")
    completed = store.complete_goal(goal_id, note="目标已验收。")
    listed = store.list_goals(status="completed")

    assert created["created"] is True
    assert progress["goal"]["progress"][0]["note"] == "已完成 Context Tree 接入。"
    assert completed["goal"]["status"] == "completed"
    assert listed["count"] == 1
    assert store.describe()["by_status"]["completed"] == 1


def test_goal_store_deduplicates_active_session_goals(tmp_path: Path) -> None:
    store = GoalStore.from_root(tmp_path)

    first = store.create_goal("目标：完善 Agent 目标功能", session_id="same-session")
    second = store.create_goal("目标：完善 Agent 目标功能", session_id="same-session")

    assert first["created"] is True
    assert second["created"] is False
    assert second["duplicate"] is True
    assert store.list_goals(status="active")["count"] == 1


def test_task_board_store_tracks_workflow_cards(tmp_path: Path) -> None:
    store = TaskBoardStore.from_root(tmp_path)

    created = store.create_task(
        "选择 pre_fusion_images 做离线检测",
        session_id="task-session",
        status="ready",
        priority="high",
        agent_path=["coordinator", "dac3d_control_agent"],
        tool_candidates=["dac3d_preview_command", "dac_tool_validate_command"],
    )
    task_id = created["task"]["id"]
    moved = store.update_status(task_id, "in_progress", note="正在生成命令预览。")
    done = store.update_status(task_id, "done", note="工作流已完成。")
    listed = store.list_tasks(session_id="task-session", status="done")

    assert created["task"]["priority"] == "high"
    assert moved["event"]["from"] == "ready"
    assert moved["event"]["to"] == "in_progress"
    assert done["task"]["completed_at"]
    assert listed["count"] == 1
    assert store.describe()["by_status"]["done"] == 1


def test_task_board_store_creates_card_from_workflow_preview(tmp_path: Path) -> None:
    store = TaskBoardStore.from_root(tmp_path)

    created = store.create_from_workflow_preview(
        {
            "backend": "agent_workflow_preview",
            "task": "当前检测状态是什么？",
            "session_id": "workflow-session",
            "agent_path": ["coordinator", "dac3d_control_agent"],
            "tool_candidates": ["dac3d_status"],
            "context_sections": [{"name": "runtime_status"}],
            "nodes": [{"id": "coordinator"}],
        },
        session_id="workflow-session",
        status="ready",
    )

    assert created["task"]["status"] == "ready"
    assert created["task"]["metadata"]["source"] == "workflow_preview"
    assert created["task"]["tool_candidates"] == ["dac3d_status"]
