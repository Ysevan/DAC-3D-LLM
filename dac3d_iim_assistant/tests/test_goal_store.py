"""Tests for DAC-Agent goal tracking."""

from __future__ import annotations

from pathlib import Path

from goals import (
    ArtifactStore,
    AgentDeploymentStore,
    AgentFleetStore,
    AgentLabelingStore,
    AgentRegistryStore,
    AutomationPlannerStore,
    BrowserContextStore,
    CheckpointStore,
    ConversationThreadStore,
    EventQueueStore,
    GoalStore,
    AgentGroundingStore,
    ObservabilityReporter,
    AgentPerformanceStore,
    ReviewHandoffStore,
    SharedStateStore,
    TaskBoardStore,
    ToolMarketplaceStore,
    VerificationRunnerStore,
    WorkflowTemplateStore,
)
from trace_eval import TraceLogger


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


def test_automation_planner_store_tracks_schedule_and_runs(tmp_path: Path) -> None:
    store = AutomationPlannerStore.from_root(tmp_path)

    created = store.create_automation(
        "每日 DAC 状态摘要",
        "每天生成一次 DAC-3D 运行状态摘要。",
        session_id="automation-session",
        schedule={"type": "daily", "time": "09:30"},
    )
    automation_id = created["automation"]["id"]
    paused = store.update_status(automation_id, "paused", note="演示期间暂停。")
    resumed = store.update_status(automation_id, "active")
    run = store.record_run(
        automation_id,
        result="已生成状态摘要。",
        trace_id="trace-123",
    )
    listed = store.list_automations(session_id="automation-session", status="active")

    assert created["automation"]["schedule_summary"] == "daily at 09:30"
    assert created["automation"]["next_run_at"]
    assert paused["automation"]["status"] == "paused"
    assert paused["automation"]["next_run_at"] == ""
    assert resumed["automation"]["next_run_at"]
    assert run["automation"]["run_count"] == 1
    assert run["run"]["trace_id"] == "trace-123"
    assert listed["count"] == 1
    assert store.describe()["by_status"]["active"] == 1


def test_automation_planner_store_validates_schedule(tmp_path: Path) -> None:
    store = AutomationPlannerStore.from_root(tmp_path)

    try:
        store.create_automation(
            "过短间隔",
            "检查状态。",
            schedule={"type": "interval", "interval_minutes": 1},
        )
    except ValueError as exc:
        assert "interval_minutes" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected invalid interval to be rejected.")


def test_workflow_template_store_creates_and_reads_dag(tmp_path: Path) -> None:
    store = WorkflowTemplateStore.from_root(tmp_path)

    created = store.create_workflow(
        "离线检测流程模板",
        session_id="workflow-template-session",
        nodes=[
            {"id": "coordinator", "label": "Coordinator", "kind": "agent"},
            {"id": "control", "label": "Control Agent", "kind": "agent"},
            {"id": "preview", "label": "Preview Command", "kind": "tool"},
        ],
        edges=[
            {"source": "coordinator", "target": "control"},
            {"source": "control", "target": "preview"},
        ],
        status="draft",
        tags=["offline"],
    )
    workflow_id = created["workflow"]["id"]
    activated = store.update_status(workflow_id, "active")
    read = store.read_workflow(workflow_id)
    listed = store.list_workflows(session_id="workflow-template-session", status="active")

    assert created["workflow"]["edges"][0]["source"] == "coordinator"
    assert activated["workflow"]["status"] == "active"
    assert read["workflow"]["nodes"][2]["kind"] == "tool"
    assert listed["count"] == 1
    assert store.describe()["by_status"]["active"] == 1


def test_workflow_template_store_creates_from_preview(tmp_path: Path) -> None:
    store = WorkflowTemplateStore.from_root(tmp_path)

    created = store.create_from_preview(
        {
            "backend": "agent_workflow_preview",
            "task": "当前检测状态是什么？",
            "session_id": "workflow-preview-session",
            "agent_path": ["coordinator", "dac3d_control_agent"],
            "tool_candidates": ["dac3d_status"],
            "context_tree_matches": [{}],
            "nodes": [
                {"id": "route", "label": "Route", "kind": "agent", "status": "selected"},
                {"id": "tool", "label": "dac3d_status", "kind": "tool", "status": "selected"},
            ],
        },
        session_id="workflow-preview-session",
        status="active",
    )

    assert created["workflow"]["metadata"]["source"] == "workflow_preview"
    assert created["workflow"]["edges"] == [{"source": "route", "target": "tool", "label": "next"}]
    assert created["workflow"]["metadata"]["tool_candidates"] == ["dac3d_status"]


def test_artifact_store_creates_searches_and_reads_files(tmp_path: Path) -> None:
    store = ArtifactStore.from_root(tmp_path)

    created = store.create_artifact(
        "离线检测摘要",
        "# 离线检测摘要\n\n命令预览已生成，等待执行。",
        artifact_type="markdown",
        session_id="artifact-session",
        task_id="task-1",
        tags=["offline", "summary"],
        metadata={"source": "test"},
    )
    artifact_id = created["artifact"]["id"]
    listed = store.list_artifacts(session_id="artifact-session", tag="offline", query="命令预览")
    read = store.read_artifact(artifact_id)
    summary = store.describe()

    assert created["created"] is True
    assert created["artifact"]["artifact_type"] == "markdown"
    assert created["artifact"]["task_id"] == "task-1"
    assert listed["count"] == 1
    assert listed["artifacts"][0]["id"] == artifact_id
    assert "命令预览" in read["content"]
    assert (tmp_path / created["artifact"]["content_path"]).exists()
    assert summary["artifact_count"] == 1
    assert summary["by_type"]["markdown"] == 1


def test_event_queue_store_claims_and_completes_due_events(tmp_path: Path) -> None:
    store = EventQueueStore.from_root(tmp_path)

    delayed = store.enqueue_event(
        "automation.run",
        session_id="event-session",
        priority="high",
        scheduled_for="2999-01-01T00:00:00Z",
    )
    queued = store.enqueue_event(
        "workflow.resume",
        payload={"workflow_id": "workflow-1"},
        session_id="event-session",
        priority="normal",
        workflow_id="workflow-1",
    )
    event_id = queued["event"]["id"]
    due = store.due_events()
    claimed = store.claim_next(worker_id="worker-1", session_id="event-session")
    completed = store.update_status(
        event_id,
        "completed",
        note="Worker finished event.",
        result={"ok": True},
    )
    listed = store.list_events(session_id="event-session", status="completed")
    summary = store.describe()

    assert delayed["event"]["scheduled_for"] == "2999-01-01T00:00:00Z"
    assert due["count"] == 1
    assert due["events"][0]["id"] == event_id
    assert claimed["claimed"] is True
    assert claimed["event"]["worker_id"] == "worker-1"
    assert completed["event"]["status"] == "completed"
    assert completed["event"]["result"] == {"ok": True}
    assert listed["count"] == 1
    assert summary["event_count"] == 2
    assert summary["by_status"]["completed"] == 1


def test_verification_runner_store_runs_and_records_compileall(tmp_path: Path) -> None:
    base_dir = tmp_path / "workspace"
    base_dir.mkdir()
    (base_dir / "demo.py").write_text("value = 1\n", encoding="utf-8")
    store = VerificationRunnerStore.from_root(tmp_path, base_dir)

    presets = store.presets()
    result = store.run_preset("python_compile", timeout_seconds=30)
    listed = store.list_runs(preset_id="python_compile")
    summary = store.describe()

    assert presets["count"] >= 3
    assert result["run"]["status"] == "passed"
    assert result["run"]["preset_id"] == "python_compile"
    assert listed["count"] == 1
    assert listed["runs"][0]["id"] == result["run"]["id"]
    assert summary["by_status"]["passed"] == 1


def test_review_handoff_store_tracks_comments_and_decisions(tmp_path: Path) -> None:
    store = ReviewHandoffStore.from_root(tmp_path)

    created = store.create_review(
        "Review Verification Feedback Runner",
        summary="确认 preset、API 和 workspace 输出。",
        session_id="review-session",
        priority="high",
        files=["dac3d_iim_assistant/goals/verification.py"],
        verification_run_ids=["verification-1"],
        checklist=["pytest passed", {"label": "docs updated", "checked": True}],
    )
    review_id = created["review"]["id"]
    commented = store.add_comment(review_id, "结构清晰，可以继续。", reviewer="reviewer-agent")
    approved = store.update_status(review_id, "approved", note="LGTM", reviewer="human")
    listed = store.list_reviews(session_id="review-session", status="approved")
    read = store.read_review(review_id)
    summary = store.describe()

    assert created["review"]["priority"] == "high"
    assert created["review"]["checklist"][0]["checked"] is False
    assert commented["comment"]["reviewer"] == "reviewer-agent"
    assert approved["review"]["status"] == "approved"
    assert approved["review"]["completed_at"]
    assert listed["count"] == 1
    assert read["review"]["comments"][0]["body"] == "结构清晰，可以继续。"
    assert summary["by_status"]["approved"] == 1


def test_checkpoint_store_captures_and_restores_resume_state(tmp_path: Path) -> None:
    store = CheckpointStore.from_root(tmp_path)

    created = store.create_checkpoint(
        "离线检测流程检查点",
        state={"step": "preview_ready", "tool": "dac3d_preview_command"},
        session_id="checkpoint-session",
        task_id="task-1",
        tags=["offline", "preview"],
        metadata={"source": "test"},
    )
    checkpoint_id = created["checkpoint"]["id"]
    restored = store.restore_checkpoint(
        checkpoint_id,
        note="继续从命令预览后恢复。",
        actor="agent-worker",
    )
    listed = store.list_checkpoints(session_id="checkpoint-session", status="restored", tag="offline")
    read = store.read_checkpoint(checkpoint_id)
    summary = store.describe()

    assert created["checkpoint"]["state"]["step"] == "preview_ready"
    assert restored["restored"] is True
    assert restored["checkpoint"]["status"] == "restored"
    assert restored["checkpoint"]["restored_at"]
    assert listed["count"] == 1
    assert read["checkpoint"]["history"][-1]["actor"] == "agent-worker"
    assert summary["last_restored_checkpoint_id"] == checkpoint_id
    assert summary["by_status"]["restored"] == 1


def test_observability_reporter_summarizes_workspace_signals(tmp_path: Path) -> None:
    trace_logger = TraceLogger(tmp_path / "agent_traces.jsonl")
    event_store = EventQueueStore.from_root(tmp_path)
    verification_store = VerificationRunnerStore.from_root(tmp_path, tmp_path)
    review_store = ReviewHandoffStore.from_root(tmp_path)
    checkpoint_store = CheckpointStore.from_root(tmp_path)
    trace_logger.append(
        {
            "event_type": "agent_chat",
            "session_id": "observe-session",
            "intent": "query",
            "tool_calls": [{"name": "dac3d_answer"}],
            "final_response": "ok",
        }
    )
    event_store.enqueue_event("workflow.resume", session_id="observe-session")
    review_store.create_review("待评审输出", session_id="observe-session")
    checkpoint_store.create_checkpoint("恢复点", state={"step": "observe"}, session_id="observe-session")
    reporter = ObservabilityReporter(
        trace_logger=trace_logger,
        event_queue_store=event_store,
        verification_store=verification_store,
        review_handoff_store=review_store,
        checkpoint_store=checkpoint_store,
    )

    snapshot = reporter.snapshot()
    summary = reporter.describe()

    assert snapshot["backend"] == "local_agent_observability"
    assert snapshot["traces"]["by_intent"]["query"] == 1
    assert snapshot["traces"]["tool_calls"]["dac3d_answer"] == 1
    assert snapshot["attention"]["queued_events"] == 1
    assert snapshot["attention"]["pending_reviews"] == 1
    assert snapshot["attention"]["active_checkpoints"] == 1
    assert summary["recent_trace_count"] == 1


def test_shared_state_store_upserts_scoped_json_state(tmp_path: Path) -> None:
    store = SharedStateStore.from_root(tmp_path)

    created = store.set_state(
        "current_batch",
        {"folder": "pre_fusion_images", "count": 3},
        session_id="shared-session",
        scope="workflow",
        namespace="offline-inspection",
        owner_agent="control-agent",
        workflow_id="workflow-1",
        tags=["offline"],
    )
    state_id = created["state"]["id"]
    updated = store.set_state(
        "current_batch",
        {"folder": "pre_fusion_images", "count": 4},
        session_id="shared-session",
        scope="workflow",
        namespace="offline-inspection",
        owner_agent="result-agent",
        workflow_id="workflow-1",
        tags=["offline", "updated"],
        note="结果 Agent 更新样品数量。",
    )
    listed = store.list_states(
        session_id="shared-session",
        scope="workflow",
        namespace="offline-inspection",
        tag="updated",
    )
    read = store.read_state(state_id)
    summary = store.describe()

    assert created["created"] is True
    assert updated["created"] is False
    assert updated["state"]["version"] == 2
    assert updated["state"]["value"]["count"] == 4
    assert updated["history"]["actor"] == "result-agent"
    assert listed["count"] == 1
    assert read["state"]["key"] == "current_batch"
    assert summary["by_scope"]["workflow"] == 1


def test_agent_registry_store_registers_and_routes_agents(tmp_path: Path) -> None:
    store = AgentRegistryStore.from_root(tmp_path)

    created = store.register_agent(
        "Offline Control Agent",
        role="offline_control",
        description="生成离线检测命令预览。",
        capabilities=["command_preview", "offline_inspection"],
        tools=["dac3d_preview_command"],
        triggers=["离线检测", "扫描"],
        tags=["control"],
    )
    updated = store.register_agent(
        "Offline Control Agent",
        role="offline_control",
        description="生成离线检测命令预览，并读取任务状态。",
        capabilities=["command_preview", "runtime_status"],
        tools=["dac3d_preview_command", "dac3d_status"],
        triggers=["离线检测", "扫描", "状态"],
        tags=["control", "offline"],
        owner_agent="registry-test",
    )
    listed = store.list_agents(capability="runtime_status")
    routed = store.route_candidates("选择图片做离线检测并查看状态", limit=3)
    read = store.read_agent("offline_control")
    summary = store.describe()

    assert created["created"] is True
    assert updated["created"] is False
    assert updated["agent"]["tools"] == ["dac3d_preview_command", "dac3d_status"]
    assert updated["history"]["actor"] == "registry-test"
    assert listed["count"] == 1
    assert routed["candidates"][0]["agent"]["role"] == "offline_control"
    assert "triggers:离线检测" in routed["candidates"][0]["matched"]
    assert read["agent"]["id"] == created["agent"]["id"]
    assert summary["agent_count"] == 1
    assert summary["by_status"]["active"] == 1


def test_agent_fleet_store_tracks_instances_heartbeats_and_assignments(tmp_path: Path) -> None:
    store = AgentFleetStore.from_root(tmp_path)

    created = store.register_instance(
        "local-control-1",
        agent_role="dac3d_control",
        environment="local",
        capabilities=["command_preview", "status"],
        max_concurrency=2,
        tags=["control"],
    )
    instance_id = created["instance"]["id"]
    heartbeat = store.heartbeat(
        instance_id,
        status="ready",
        current_load=0,
        metrics={"latency_ms": 12},
    )
    assigned = store.assign_task(
        instance_id,
        task_id="task-offline-1",
        summary="生成离线检测命令预览。",
        thread_id="thread-1",
        assigned_by="coordinator",
    )
    completed = store.update_assignment_status(
        instance_id,
        assigned["assignment"]["id"],
        "completed",
        actor="local-control-1",
    )
    listed = store.list_instances(agent_role="dac3d_control", tag="control", query="离线检测")
    read = store.read_instance("local-control-1")
    summary = store.describe()

    assert created["created"] is True
    assert heartbeat["instance"]["last_heartbeat_at"]
    assert assigned["instance"]["status"] == "busy"
    assert assigned["assignment"]["task_id"] == "task-offline-1"
    assert completed["assignment"]["status"] == "completed"
    assert completed["instance"]["current_load"] == 0
    assert listed["count"] == 1
    assert read["instance"]["id"] == instance_id
    assert summary["instance_count"] == 1
    assert summary["total_capacity"] == 2


def test_conversation_thread_store_tracks_messages_and_context_refs(tmp_path: Path) -> None:
    store = ConversationThreadStore.from_root(tmp_path)

    created = store.create_thread(
        "离线检测协作",
        session_id="thread-session",
        participants=["coordinator", "dac3d_control"],
        shared_state_ids=["state-1"],
        tags=["offline"],
    )
    thread_id = created["thread"]["id"]
    appended = store.append_message(
        thread_id,
        role="agent",
        agent_role="dac3d_control",
        content="已生成离线检测命令预览。",
        tool_calls=[{"name": "dac3d_preview_command"}],
        artifact_ids=["artifact-1"],
        shared_state_ids=["state-2"],
    )
    resolved = store.update_status(thread_id, "resolved", note="协作完成。")
    listed = store.list_threads(session_id="thread-session", participant="dac3d_control")
    read = store.read_thread(thread_id)
    summary = store.describe()

    assert created["created"] is True
    assert appended["message"]["tool_calls"][0]["name"] == "dac3d_preview_command"
    assert appended["thread"]["message_count"] == 1
    assert "artifact-1" in appended["thread"]["artifact_ids"]
    assert "state-2" in appended["thread"]["shared_state_ids"]
    assert resolved["history"]["to"] == "resolved"
    assert listed["count"] == 1
    assert read["thread"]["messages"][0]["content"] == "已生成离线检测命令预览。"
    assert summary["thread_count"] == 1
    assert summary["message_count"] == 1


def test_browser_context_store_records_shared_observations(tmp_path: Path) -> None:
    store = BrowserContextStore.from_root(tmp_path)

    created = store.create_context(
        "参考页面",
        url="http://localhost:7890",
        session_id="browser-session",
        thread_id="thread-1",
        owner_agent="researcher",
        tags=["ui"],
    )
    context_id = created["context"]["id"]
    observed = store.append_observation(
        context_id,
        title="DAC Agent Workspace",
        text="页面包含 Agent workspace 和状态面板。",
        agent_role="researcher",
        screenshot_path="/tmp/workspace.png",
        artifact_ids=["artifact-browser"],
    )
    captured = store.update_status(context_id, "captured", actor="researcher")
    listed = store.list_contexts(session_id="browser-session", tag="ui", query="workspace")
    read = store.read_context(context_id)
    summary = store.describe()

    assert created["created"] is True
    assert observed["context"]["observation_count"] == 1
    assert observed["observation"]["screenshot_path"] == "/tmp/workspace.png"
    assert "artifact-browser" in observed["context"]["artifact_ids"]
    assert captured["history"]["to"] == "captured"
    assert listed["count"] == 1
    assert read["context"]["observations"][0]["title"] == "DAC Agent Workspace"
    assert summary["context_count"] == 1
    assert summary["observation_count"] == 1


def test_tool_marketplace_store_registers_and_filters_tool_packs(tmp_path: Path) -> None:
    store = ToolMarketplaceStore.from_root(tmp_path)

    created = store.register_entry(
        "Offline Inspection Tool Pack",
        slug="offline-inspection",
        description="离线检测命令预览工作流工具包。",
        category="dac3d_operations",
        status="available",
        tools=["dac3d_preview_command", "dac3d_status"],
        required_context=["runtime_status", "command_schema"],
        prompt_examples=["选择图片做离线检测"],
        tags=["offline", "workflow"],
    )
    updated = store.update_status(
        "offline-inspection",
        "installed",
        actor="tool-marketplace-test",
    )
    listed = store.list_entries(category="dac3d_operations", tag="offline", query="预览")
    read = store.read_entry(created["entry"]["id"])
    summary = store.describe()

    assert created["created"] is True
    assert created["entry"]["slug"] == "offline-inspection"
    assert updated["entry"]["status"] == "installed"
    assert updated["history"]["actor"] == "tool-marketplace-test"
    assert listed["count"] == 1
    assert read["entry"]["tools"] == ["dac3d_preview_command", "dac3d_status"]
    assert summary["entry_count"] == 1
    assert summary["tool_count"] == 2


def test_agent_deployment_store_tracks_apps_status_and_releases(tmp_path: Path) -> None:
    store = AgentDeploymentStore.from_root(tmp_path)

    created = store.create_deployment(
        "Offline Workflow App",
        slug="offline-workflow",
        app_type="workflow_app",
        entrypoint="workflow:offline-inspection",
        version="0.1.0",
        environment="local",
        route_path="/agent/offline",
        status="ready",
        workflow_ids=["workflow-offline"],
        tool_pack_slugs=["dac3d-control-pack"],
        agent_roles=["coordinator", "dac3d_control"],
        config_refs=["local.env"],
        tags=["offline", "demo"],
    )
    release = store.record_release(
        "offline-workflow",
        version="0.2.0",
        summary="接入离线检测 workflow 模板。",
        artifact_ids=["artifact-release"],
        verification_run_ids=["verification-1"],
        released_by="deployment-test",
    )
    deployed = store.update_status("offline-workflow", "deployed", actor="deployment-test")
    listed = store.list_deployments(status="deployed", environment="local", query="离线检测")
    read = store.read_deployment(created["deployment"]["id"])
    summary = store.describe()

    assert created["created"] is True
    assert created["deployment"]["slug"] == "offline-workflow"
    assert release["release"]["version"] == "0.2.0"
    assert release["deployment"]["latest_release_id"] == release["release"]["id"]
    assert deployed["deployment"]["status"] == "deployed"
    assert deployed["deployment"]["deployed_at"]
    assert listed["count"] == 1
    assert read["deployment"]["releases"][0]["artifact_ids"] == ["artifact-release"]
    assert summary["deployment_count"] == 1
    assert summary["release_count"] == 1


def test_agent_labeling_store_labels_trace_samples_and_exports_jsonl(tmp_path: Path) -> None:
    store = AgentLabelingStore.from_root(tmp_path)

    created = store.create_from_trace(
        {
            "trace_id": "trace-label-1",
            "session_id": "label-session",
            "intent": "status",
            "user_message": "当前检测状态是什么？",
            "final_response": "当前处于空闲状态。",
            "tool_calls": [{"name": "dac3d_status"}],
            "timestamp": "2026-05-29T00:00:00Z",
        },
        tags=["status"],
        created_by="labeling-test",
    )
    item_id = created["item"]["id"]
    labeled = store.label_item(
        item_id,
        labels={"answer_quality": "good", "intent_correct": True},
        outcome="accepted",
        score=5,
        comment="回答和意图都正确。",
        labeler="qa",
    )
    listed = store.list_items(status="labeled", source_type="trace", query="空闲")
    exported = store.export_items(status="labeled", mark_exported=True)
    read = store.read_item(item_id)
    summary = store.describe()

    assert created["created"] is True
    assert created["item"]["source_id"] == "trace-label-1"
    assert created["item"]["tool_names"] == ["dac3d_status"]
    assert labeled["item"]["status"] == "labeled"
    assert labeled["annotation"]["score"] == 5
    assert listed["count"] == 1
    assert exported["count"] == 1
    assert "trace-label-1" in exported["jsonl"]
    assert read["item"]["status"] == "exported"
    assert summary["item_count"] == 1
    assert summary["annotation_count"] == 1


def test_agent_performance_store_records_trace_metrics_and_summarizes(tmp_path: Path) -> None:
    store = AgentPerformanceStore.from_root(tmp_path)

    manual = store.record_metric(
        "agent_quality_score",
        4.5,
        unit="score",
        category="quality",
        target="status",
        source_type="manual",
        tags=["llmops"],
    )
    trace_metrics = store.record_from_trace(
        {
            "trace_id": "trace-perf-1",
            "session_id": "perf-session",
            "intent": "status",
            "duration_ms": 850,
            "token_usage": {"total_tokens": 321},
            "tool_calls": [{"name": "dac3d_status"}],
            "quality_score": 5,
        },
        tags=["status"],
        recorded_by="performance-test",
    )
    listed = store.list_metrics(category="latency", source_type="trace", query="trace-perf-1")
    summary = store.summarize(category="quality")
    archived = store.update_status(manual["metric"]["id"], "archived", actor="performance-test")
    description = store.describe()

    assert manual["created"] is True
    assert trace_metrics["count"] == 4
    assert listed["count"] == 1
    assert listed["metrics"][0]["metric_name"] == "agent_latency_ms"
    assert summary["by_metric"]["agent_quality_score"]["count"] == 2
    assert archived["metric"]["status"] == "archived"
    assert description["metric_count"] == 5


def test_agent_grounding_store_records_sources_and_builds_context_bundle(tmp_path: Path) -> None:
    store = AgentGroundingStore.from_root(tmp_path)

    created = store.record_source(
        "Gemini CLI grounding docs",
        query="Gemini CLI search grounding",
        source_type="web",
        url="https://example.com/gemini-cli",
        snippet="Gemini CLI supports web fetching and search grounding.",
        summary="Search grounding can attach cited web evidence to agent answers.",
        citations=["Gemini CLI README"],
        confidence=0.86,
        trace_id="trace-grounding-1",
        tags=["gemini", "grounding"],
    )
    source_id = created["source"]["id"]
    listed = store.list_sources(source_type="web", tag="grounding", query="cited", min_confidence=0.8)
    bundle = store.context_bundle(query="grounding", min_confidence=0.8)
    stale = store.update_status(source_id, "stale", actor="grounding-test")
    read = store.read_source(source_id)
    summary = store.describe()

    assert created["created"] is True
    assert listed["count"] == 1
    assert bundle["count"] == 1
    assert "[1] Gemini CLI grounding docs" in bundle["context"]
    assert stale["source"]["status"] == "stale"
    assert read["source"]["citations"] == ["Gemini CLI README"]
    assert summary["source_count"] == 1
    assert summary["by_source_type"]["web"] == 1
