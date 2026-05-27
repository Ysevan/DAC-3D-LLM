"""Tests for DAC-Agent goal tracking."""

from __future__ import annotations

from pathlib import Path

from goals import GoalStore


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
