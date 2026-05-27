"""Tests for the FastAPI web surface."""

from __future__ import annotations

import subprocess

from fastapi.testclient import TestClient

from agent_runtime import DAC3DAgentChatAdapter, DAC3DAgentRuntime, LOCAL_VALIDATION_MODEL_NAME
from app import DAC3DAssistant
from tests.test_assistant import make_config
from ui.web_api import create_api_app


def test_web_api_chat_endpoint_returns_structured_payload(tmp_path) -> None:
    """The API should expose the assistant response schema for the React client."""
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.post(
        "/api/chat",
        json={"message": "这个参数是什么意思？", "history": []},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "query"
    assert payload["answer"]
    assert isinstance(payload["sources"], list)


def test_web_api_runtime_endpoint_returns_runtime_summary(tmp_path) -> None:
    """The API should expose runtime data for the settings drawer."""
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.get("/api/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "mock"
    assert "dac3d" in payload


def test_web_api_mcp_manifest_endpoint_exposes_discovery_payload(tmp_path) -> None:
    """The API should expose MCP-compatible tools, resources, and prompts for discovery."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    response = client.get("/api/mcp/manifest", params={"session_id": "mcp-web"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["protocol"]["style"] == "mcp-compatible"
    assert payload["capabilities"]["tools"]["count"] >= 8
    assert {resource["uri"] for resource in payload["resources"]} >= {
        "dac3d://runtime/status",
        "dac3d://memory/profile",
    }
    assert "future_agents_sdk_orchestration" in payload["deployment_modes"]


def test_web_api_chat_endpoint_rejects_empty_message(tmp_path) -> None:
    """The API should reject invalid chat payloads with a client error."""
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.post("/api/chat", json={"message": "   ", "history": []})

    assert response.status_code == 422


def test_web_api_chat_stream_emits_sse_events(tmp_path) -> None:
    """The streaming endpoint should emit metadata, deltas, and the final payload."""
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.post(
        "/api/chat/stream",
        json={"message": "what does this parameter mean?", "history": []},
    )

    assert response.status_code == 200
    body = response.text
    assert "event: meta" in body
    assert "event: delta" in body
    assert "event: done" in body


def test_web_api_chat_stream_emits_multiple_deltas_progressively(tmp_path) -> None:
    """The streaming endpoint should emit multiple incremental delta events for grounded answers."""
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.post(
        "/api/chat/stream",
        json={"message": "样品太反光怎么办？", "history": []},
    )

    assert response.status_code == 200
    body = response.text
    assert body.count("event: delta") >= 2
    assert body.index("event: meta") < body.index("event: delta") < body.index("event: done")


def test_machine_agent_snapshot_endpoint_returns_dashboard_payload(tmp_path) -> None:
    """The industrial Agent dashboard endpoint should expose status and demo data."""
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.get("/api/machine-agent/snapshot")

    assert response.status_code == 200
    payload = response.json()
    assert payload["current_status"]["machine_id"] == "IM-Press-01"
    assert payload["alarm_records"]
    assert "现在设备状态怎么样？" in payload["demo_questions"]


def test_machine_agent_chat_endpoint_exposes_tool_calls(tmp_path) -> None:
    """Machine Agent chat should show the tools used to answer a question."""
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.post(
        "/api/machine-agent/chat",
        json={"message": "为什么最近温度报警变多了？"},
    )

    assert response.status_code == 200
    payload = response.json()
    tool_names = [item["name"] for item in payload["tool_calls"]]
    assert "get_alarm_records" in tool_names
    assert "detect_abnormal_patterns" in tool_names
    assert payload["abnormal_result"]["alarm_count"] > 0


def test_unified_chat_endpoint_enters_agent_runtime_first(tmp_path, monkeypatch) -> None:
    """The single chat endpoint should delegate the message to the Agent runtime."""
    def fake_run_sync(self, message: str, *, session_id: str = "default") -> str:
        del self
        assert session_id == "chrome-session-1"
        assert "当前用户问题:\n为什么最近温度报警变多了？" in message
        assert "上下文工程包" in message
        return (
            '{"answer":"LLM 已选择 machine_agent_chat 工具处理温度报警问题。",'
            '"structured_data":{'
            '"intent":"machine_alarm_analysis",'
            '"tool_calls":[{"name":"machine_agent_chat","purpose":"分析温度报警"}],'
            '"findings":[{"metric":"temperature_high","value":"最近30天 6 次"}],'
            '"recommendations":["检查冷却水路"]'
            "}}"
        )

    monkeypatch.setattr(DAC3DAgentRuntime, "run_sync", fake_run_sync)
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    response = client.post(
        "/api/chat",
        json={
            "message": "为什么最近温度报警变多了？",
            "history": [],
            "session_id": "chrome-session-1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "machine_alarm_analysis"
    assert "machine_agent_chat" in payload["answer"]
    assert payload["parsed_result"]["tool_calls"][0]["name"] == "machine_agent_chat"


def test_unified_chat_stream_uses_client_session_id(tmp_path, monkeypatch) -> None:
    """The streaming chat endpoint should not force every client into the same Agent session."""
    seen_sessions: list[str] = []

    def fake_run_sync(self, message: str, *, session_id: str = "default") -> str:
        del self
        assert "当前用户问题:\n当前检测状态是什么？" in message
        assert "上下文工程包" in message
        seen_sessions.append(session_id)
        return (
            '{"answer":"当前 DAC-3D 处于空闲状态。",'
            '"structured_data":{'
            '"intent":"status",'
            '"tool_calls":[{"name":"dac3d_status","purpose":"读取状态"}],'
            '"status_summary":{"state":"idle","progress":0}'
            "}}"
        )

    monkeypatch.setattr(DAC3DAgentRuntime, "run_sync", fake_run_sync)
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    response = client.post(
        "/api/chat/stream",
        json={
            "message": "当前检测状态是什么？",
            "history": [],
            "session_id": "ui-session-42",
        },
    )

    assert response.status_code == 200
    assert "event: done" in response.text
    assert seen_sessions == ["ui-session-42"]


def test_web_api_approval_endpoint_submits_pending_gateway_command(tmp_path) -> None:
    """The web UI approval button should continue the pending Tool Gateway command."""
    config = make_config(tmp_path)
    config.agent_model_name = LOCAL_VALIDATION_MODEL_NAME
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    runtime = DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    agent_runtime = DAC3DAgentChatAdapter(runtime)
    client = TestClient(create_api_app(agent_runtime))

    preview_response = client.post(
        "/api/chat",
        json={
            "message": "扫描 10mm × 10mm 区域",
            "history": [],
            "session_id": "approval-ui-session",
        },
    )

    assert preview_response.status_code == 200
    preview_payload = preview_response.json()
    gateway = preview_payload["command_preview"]["gateway"]
    assert gateway["confirmation_required"] is True
    assert runtime.sessions is not None
    assert runtime.sessions.get_pending_command("approval-ui-session") is not None

    approve_response = client.post(
        "/api/commands/approve",
        json={
            "session_id": "approval-ui-session",
            "preview_id": gateway["preview_id"],
            "confirmation_token": gateway["confirmation_token"],
        },
    )

    assert approve_response.status_code == 200
    approved_payload = approve_response.json()
    assert "提交" in approved_payload["answer"] or "已排队" in approved_payload["answer"]
    assert approved_payload["parsed_result"]["tool_gateway"]["tool"] == "submit_command"
    assert runtime.sessions.get_pending_command("approval-ui-session") is None


def test_web_api_eval_endpoint_runs_local_regression_cases(tmp_path) -> None:
    """The API should expose the local trace/eval improvement loop."""
    config = make_config(tmp_path)
    config.agent_model_name = LOCAL_VALIDATION_MODEL_NAME
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    response = client.post("/api/evals/run", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["case_count"] >= 5
    assert payload["failed"] == 0
    assert payload["trace_logger"]["trace_count"] >= payload["case_count"]


def test_web_api_eval_draft_endpoint_generates_reviewable_trace_cases(tmp_path) -> None:
    """Trace-to-eval conversion should save drafts without approving them as regression cases."""
    config = make_config(tmp_path)
    config.agent_model_name = LOCAL_VALIDATION_MODEL_NAME
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))
    chat_response = client.post(
        "/api/chat",
        json={"message": "当前检测状态是什么？", "history": [], "session_id": "draft-ui-session"},
    )
    trace_id = chat_response.json()["parsed_result"]["trace_eval"]["trace_id"]

    response = client.post("/api/evals/drafts", json={"trace_ids": [trace_id]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["auto_approved"] is False
    draft = payload["drafts"][0]["draft"]
    assert draft["source_trace_id"] == trace_id
    assert draft["expected"]["intent"] == "status"
    assert "dac3d_status" in draft["expected"]["must_call_tools"]
    assert (tmp_path / "evals" / "drafts" / f"{draft['id']}.json").exists()
    assert not (tmp_path / "evals" / "cases" / f"{draft['id']}.json").exists()

    list_response = client.get("/api/evals/drafts")
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1


def test_web_api_codex_handoff_endpoint_generates_reviewable_markdown(tmp_path) -> None:
    """The API should generate a Codex handoff from eval results and traces."""
    config = make_config(tmp_path)
    config.agent_model_name = LOCAL_VALIDATION_MODEL_NAME
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    response = client.post("/api/evals/codex-handoff", json={"recent_trace_limit": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "codex_handoff_generator"
    assert payload["auto_applied"] is False
    assert payload["eval_summary"]["case_count"] >= 5
    assert payload["trace_count"] >= 1
    handoff_path = tmp_path / "docs" / "generated" / "codex_handoff_next.md"
    assert payload["path"] == str(handoff_path)
    assert handoff_path.exists()
    assert "Codex Handoff" in handoff_path.read_text(encoding="utf-8")


def test_web_api_memory_patch_review_endpoints(tmp_path) -> None:
    """The web UI should be able to review, approve, and reject Memory OS patches."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    assert agent_runtime.memory_provider is not None
    trace = agent_runtime.memory_provider.record_trace(
        {
            "session_id": "memory-review-session",
            "user_message": "测试记忆补丁审核。",
            "assistant_answer": "已生成待审核记忆。",
            "intent": "query",
            "parsed_result": {
                "memory_write_candidates": [
                    {
                        "target": "user",
                        "content": "用户偏好：执行前先查看命令预览。",
                        "reason": "test_memory_patch_review",
                    },
                    {
                        "target": "memory",
                        "content": "流程经验：批准前应校验命令预览。",
                        "reason": "test_memory_patch_review",
                    },
                ]
            },
        }
    )
    patches = agent_runtime.memory_provider.propose_writes(trace)
    client = TestClient(create_api_app(agent_runtime))

    list_response = client.get("/api/memory/patches")

    assert list_response.status_code == 200
    listed = list_response.json()
    assert listed["enabled"] is True
    assert listed["count"] == 2

    approve_response = client.post(f"/api/memory/patches/{patches[0]['id']}/approve")
    reject_response = client.post(
        f"/api/memory/patches/{patches[1]['id']}/reject",
        json={"reason": "test_rejected"},
    )

    assert approve_response.status_code == 200
    assert approve_response.json()["applied"] is True
    assert reject_response.status_code == 200
    assert reject_response.json()["rejected"] is True
    pending_response = client.get("/api/memory/patches")
    assert pending_response.json()["count"] == 0


def test_web_api_procedure_memory_uses_patch_approval_before_markdown_write(tmp_path) -> None:
    """Reviewed procedure memory should be proposed first, then written as Markdown after approval."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))
    procedure_path = config.conversation_memory_dir / "procedures" / "offline-inspection-flow.md"

    propose_response = client.post(
        "/api/memory/procedures",
        json={
            "name": "offline-inspection-flow",
            "content": "离线检测目录选择后，先生成命令预览，再等待用户审核。",
            "reason": "reviewed_flow",
        },
    )

    assert propose_response.status_code == 200
    payload = propose_response.json()
    patch = payload["patches"][0]
    assert patch["target"] == "procedure_memory"
    assert patch["status"] == "pending"
    assert not procedure_path.exists()

    approve_response = client.post(f"/api/memory/patches/{patch['id']}/approve")
    list_response = client.get("/api/memory/procedures")
    read_response = client.get("/api/memory/procedures/offline-inspection-flow")

    assert approve_response.status_code == 200
    assert approve_response.json()["applied"] is True
    assert procedure_path.exists()
    assert list_response.json()["count"] == 1
    procedure = read_response.json()
    assert procedure["frontmatter"]["target"] == "procedure_memory"
    assert procedure["frontmatter"]["provenance"]["trace_id"] == patch["source_trace_id"]
    assert "命令预览" in procedure["body"]


def test_web_api_skill_patch_review_endpoints(tmp_path) -> None:
    """The web UI should review skill patch proposals without editing SKILL.md."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))
    skill_path = config.agent_skills_dir / "dac-command-preview" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        """---
name: dac-command-preview
description: Preview DAC commands.
triggers:
  - 扫描
tools:
  - dac3d_preview_command
---

# DAC Command Preview

Original preview flow.
""",
        encoding="utf-8",
    )
    original_skill = skill_path.read_text(encoding="utf-8")

    propose_response = client.post(
        "/api/skills/patches",
        json={
            "target_skill": "dac-command-preview",
            "reason": "trace shows preview validation should be called out",
            "diff": "+ Require validate_command in the preview checklist.",
            "evidence_trace_ids": ["trace-skill-1"],
            "risk_level": "medium",
        },
    )

    assert propose_response.status_code == 200
    patch = propose_response.json()["patch"]
    assert patch["status"] == "pending"
    assert patch["evidence_trace_ids"] == ["trace-skill-1"]

    list_response = client.get("/api/skills/patches")
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1

    approve_response = client.post(f"/api/skills/patches/{patch['id']}/approve")
    assert approve_response.status_code == 200
    assert approve_response.json()["approved"] is True
    assert approve_response.json()["applied"] is False
    assert skill_path.read_text(encoding="utf-8") == original_skill

    approved_response = client.get("/api/skills/patches?status=approved")
    assert approved_response.json()["count"] == 1


def test_web_api_agent_workspace_and_workflow_preview(tmp_path) -> None:
    """The React client should be able to inspect the multi-agent workspace."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    workspace_response = client.get("/api/agent/workspace")
    preview_response = client.post(
        "/api/agent/workflow/preview",
        json={
            "task": "选择 pre_fusion_images 下的图片进行离线检测",
            "session_id": "workspace-ui-session",
        },
    )

    assert workspace_response.status_code == 200
    workspace = workspace_response.json()
    assert workspace["backend"] == "dac_agent_workspace"
    assert workspace["context_tree"]["node_count"] >= 5
    assert "coordinator_route" in workspace["workflow"]

    assert preview_response.status_code == 200
    preview = preview_response.json()
    assert preview["backend"] == "agent_workflow_preview"
    assert preview["agent_path"][0] == "coordinator"
    assert "dac3d_preview_command" in preview["tool_candidates"]
    assert preview["context_tree_matches"]
    assert any(
        item["node"]["kind"] == "procedure"
        for item in preview["context_tree_matches"]
    )


def test_web_api_repo_context_map_endpoints(tmp_path) -> None:
    """The web UI should build, read, and search the static repo context map."""
    config = make_config(tmp_path)
    (config.base_dir / "ui" / "web_api.py").parent.mkdir(parents=True, exist_ok=True)
    (config.base_dir / "ui" / "web_api.py").write_text(
        'from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get("/api/repo-demo")\ndef demo():\n    return {}\n',
        encoding="utf-8",
    )
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    build_response = client.post("/api/agent/repo-map/build", json={"max_files": 200})
    read_response = client.get("/api/agent/repo-map")
    search_response = client.get("/api/agent/repo-map/search?q=repo-demo")
    symbols_response = client.get("/api/agent/symbols?q=demo&kind=function")
    workspace_response = client.get("/api/agent/workspace")

    assert build_response.status_code == 200
    assert {"method": "GET", "path": "/api/repo-demo"} in build_response.json()["api_routes"]
    assert read_response.status_code == 200
    assert read_response.json()["file_count"] >= 2
    assert search_response.status_code == 200
    assert search_response.json()["count"] >= 1
    assert symbols_response.status_code == 200
    assert symbols_response.json()["symbols"][0]["name"] == "demo"
    assert symbols_response.json()["symbols"][0]["line"] == 6
    assert workspace_response.json()["repo_context_map"]["file_count"] >= 2
    assert workspace_response.json()["code_symbols"]["symbol_count"] >= 1


def test_web_api_git_workspace_status_endpoint(tmp_path) -> None:
    """The web UI should inspect read-only Git branch and changed-file context."""
    config = make_config(tmp_path)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "api-change.txt").write_text("changed\n", encoding="utf-8")
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    status_response = client.get("/api/agent/git/status")
    workspace_response = client.get("/api/agent/workspace")

    assert status_response.status_code == 200
    payload = status_response.json()
    assert payload["enabled"] is True
    assert payload["clean"] is False
    assert any(item["path"] == "api-change.txt" for item in payload["changed_files"])
    assert workspace_response.json()["git_workspace"]["changed_file_count"] >= 1


def test_web_api_agent_workflow_template_endpoints(tmp_path) -> None:
    """The web UI should persist workflow previews as reusable DAG templates."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    create_response = client.post(
        "/api/agent/workflows/from-preview",
        json={
            "task": "选择 pre_fusion_images 下的图片进行离线检测",
            "name": "离线检测 DAG",
            "session_id": "workflow-ui-session",
            "status": "active",
        },
    )
    workflow_id = create_response.json()["workflow"]["id"]
    read_response = client.get(f"/api/agent/workflows/{workflow_id}")
    list_response = client.get("/api/agent/workflows?session_id=workflow-ui-session&status=active")
    archive_response = client.post(
        f"/api/agent/workflows/{workflow_id}/status",
        json={"status": "archived"},
    )
    workspace_response = client.get("/api/agent/workspace")

    assert create_response.status_code == 200
    assert create_response.json()["workflow"]["metadata"]["source"] == "workflow_preview"
    assert create_response.json()["workflow"]["edges"]
    assert read_response.status_code == 200
    assert read_response.json()["workflow"]["name"] == "离线检测 DAG"
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1
    assert archive_response.status_code == 200
    assert archive_response.json()["workflow"]["status"] == "archived"
    assert workspace_response.json()["workflow_templates"]["workflow_count"] == 1


def test_web_api_agent_artifact_store_endpoints(tmp_path) -> None:
    """The web UI should create, search, and read shared Agent artifacts."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    create_response = client.post(
        "/api/agent/artifacts",
        json={
            "title": "离线检测执行摘要",
            "content": {"result": "queued", "step": "preview"},
            "artifact_type": "json",
            "session_id": "artifact-ui-session",
            "tags": ["offline", "summary"],
            "metadata": {"task": "offline_inspection"},
        },
    )
    artifact_id = create_response.json()["artifact"]["id"]
    list_response = client.get(
        "/api/agent/artifacts?session_id=artifact-ui-session&artifact_type=json&q=queued"
    )
    read_response = client.get(f"/api/agent/artifacts/{artifact_id}")
    workspace_response = client.get("/api/agent/workspace")

    assert create_response.status_code == 200
    assert create_response.json()["artifact"]["artifact_type"] == "json"
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1
    assert read_response.status_code == 200
    assert '"result": "queued"' in read_response.json()["content"]
    assert workspace_response.json()["artifacts"]["artifact_count"] == 1


def test_web_api_agent_event_queue_endpoints(tmp_path) -> None:
    """The web UI should enqueue, claim, complete, and list Agent events."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    create_response = client.post(
        "/api/agent/events",
        json={
            "event_type": "workflow.resume",
            "payload": {"workflow_id": "workflow-web"},
            "session_id": "event-ui-session",
            "priority": "high",
            "workflow_id": "workflow-web",
        },
    )
    event_id = create_response.json()["event"]["id"]
    due_response = client.get("/api/agent/events/due")
    claim_response = client.post(
        "/api/agent/events/claim",
        json={"worker_id": "web-worker", "session_id": "event-ui-session"},
    )
    complete_response = client.post(
        f"/api/agent/events/{event_id}/status",
        json={"status": "completed", "note": "UI worker completed.", "result": {"ok": True}},
    )
    list_response = client.get("/api/agent/events?session_id=event-ui-session&status=completed")
    workspace_response = client.get("/api/agent/workspace")

    assert create_response.status_code == 200
    assert create_response.json()["event"]["priority"] == "high"
    assert due_response.status_code == 200
    assert due_response.json()["count"] == 1
    assert claim_response.status_code == 200
    assert claim_response.json()["event"]["worker_id"] == "web-worker"
    assert complete_response.status_code == 200
    assert complete_response.json()["event"]["status"] == "completed"
    assert complete_response.json()["event"]["result"] == {"ok": True}
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1
    assert workspace_response.json()["event_queue"]["event_count"] == 1


def test_web_api_agent_verification_feedback_endpoints(tmp_path) -> None:
    """The web UI should list presets, run one verification, and read feedback records."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    presets_response = client.get("/api/agent/verifications/presets")
    run_response = client.post(
        "/api/agent/verifications/run",
        json={"preset_id": "python_compile", "timeout_seconds": 30},
    )
    list_response = client.get("/api/agent/verifications?preset_id=python_compile")
    workspace_response = client.get("/api/agent/workspace")

    assert presets_response.status_code == 200
    assert presets_response.json()["backend"] == "local_verification_runner"
    assert run_response.status_code == 200
    assert run_response.json()["run"]["status"] == "passed"
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1
    assert workspace_response.json()["verification_feedback"]["run_count"] == 1


def test_web_api_agent_review_handoff_endpoints(tmp_path) -> None:
    """The web UI should create, comment on, decide, and list review handoffs."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    create_response = client.post(
        "/api/agent/reviews",
        json={
            "title": "Review Agent workspace upgrade",
            "summary": "检查 review handoff 是否进入 workspace。",
            "session_id": "review-ui-session",
            "priority": "high",
            "files": ["dac3d_iim_assistant/ui/web_api.py"],
            "verification_run_ids": ["verification-web"],
            "checklist": ["API returns 200"],
        },
    )
    review_id = create_response.json()["review"]["id"]
    comment_response = client.post(
        f"/api/agent/reviews/{review_id}/comments",
        json={"body": "API roundtrip ok.", "reviewer": "reviewer-agent"},
    )
    approve_response = client.post(
        f"/api/agent/reviews/{review_id}/status",
        json={"status": "approved", "note": "LGTM", "reviewer": "human"},
    )
    read_response = client.get(f"/api/agent/reviews/{review_id}")
    list_response = client.get("/api/agent/reviews?session_id=review-ui-session&status=approved")
    workspace_response = client.get("/api/agent/workspace")

    assert create_response.status_code == 200
    assert create_response.json()["review"]["priority"] == "high"
    assert comment_response.status_code == 200
    assert comment_response.json()["comment"]["reviewer"] == "reviewer-agent"
    assert approve_response.status_code == 200
    assert approve_response.json()["review"]["status"] == "approved"
    assert read_response.status_code == 200
    assert read_response.json()["review"]["verification_run_ids"] == ["verification-web"]
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1
    assert workspace_response.json()["review_handoffs"]["review_count"] == 1


def test_web_api_agent_checkpoint_endpoints(tmp_path) -> None:
    """The web UI should create, restore, read, and list Agent checkpoints."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    create_response = client.post(
        "/api/agent/checkpoints",
        json={
            "title": "离线检测恢复点",
            "state": {"step": "review_handoff_done"},
            "session_id": "checkpoint-ui-session",
            "tags": ["offline", "review"],
        },
    )
    checkpoint_id = create_response.json()["checkpoint"]["id"]
    restore_response = client.post(
        f"/api/agent/checkpoints/{checkpoint_id}/restore",
        json={"note": "继续下一步。", "actor": "web-worker"},
    )
    read_response = client.get(f"/api/agent/checkpoints/{checkpoint_id}")
    list_response = client.get(
        "/api/agent/checkpoints?session_id=checkpoint-ui-session&status=restored&tag=offline"
    )
    workspace_response = client.get("/api/agent/workspace")

    assert create_response.status_code == 200
    assert create_response.json()["checkpoint"]["state"]["step"] == "review_handoff_done"
    assert restore_response.status_code == 200
    assert restore_response.json()["checkpoint"]["status"] == "restored"
    assert read_response.status_code == 200
    assert read_response.json()["checkpoint"]["history"][-1]["actor"] == "web-worker"
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1
    assert workspace_response.json()["checkpoints"]["checkpoint_count"] == 1


def test_web_api_agent_observability_endpoint(tmp_path) -> None:
    """The web UI should read traces and workspace signal counts in one snapshot."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    assert agent_runtime.trace_logger is not None
    agent_runtime.trace_logger.append(
        {
            "event_type": "agent_chat",
            "session_id": "observability-ui-session",
            "intent": "result_query",
            "tool_calls": [{"name": "dac3d_latest_result"}],
            "final_response": "ok",
        }
    )
    client = TestClient(create_api_app(agent_runtime))

    client.post("/api/agent/events", json={"event_type": "workflow.resume"})
    client.post("/api/agent/reviews", json={"title": "待评审输出"})
    client.post(
        "/api/agent/checkpoints",
        json={"title": "恢复点", "state": {"step": "observe"}},
    )
    response = client.get("/api/agent/observability?recent_trace_limit=5")
    workspace_response = client.get("/api/agent/workspace")

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "local_agent_observability"
    assert payload["traces"]["by_intent"]["result_query"] == 1
    assert payload["traces"]["tool_calls"]["dac3d_latest_result"] == 1
    assert payload["attention"]["queued_events"] == 1
    assert payload["attention"]["pending_reviews"] == 1
    assert payload["attention"]["active_checkpoints"] == 1
    assert workspace_response.json()["observability"]["recent_trace_count"] == 1


def test_web_api_agent_task_board_endpoints(tmp_path) -> None:
    """The web UI should create, move, and list Agent task-board cards."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    create_response = client.post(
        "/api/agent/tasks/from-workflow",
        json={
            "task": "选择 pre_fusion_images 下的图片进行离线检测",
            "session_id": "task-ui-session",
            "priority": "high",
        },
    )
    task_id = create_response.json()["task"]["id"]
    move_response = client.post(
        f"/api/agent/tasks/{task_id}/status",
        json={"status": "in_progress", "note": "UI 已开始处理。"},
    )
    list_response = client.get("/api/agent/tasks?session_id=task-ui-session&status=in_progress")
    workspace_response = client.get("/api/agent/workspace")

    assert create_response.status_code == 200
    assert create_response.json()["task"]["metadata"]["source"] == "workflow_preview"
    assert move_response.status_code == 200
    assert move_response.json()["task"]["status"] == "in_progress"
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1
    assert workspace_response.json()["task_board"]["task_count"] == 1


def test_web_api_agent_automation_planner_endpoints(tmp_path) -> None:
    """The web UI should create, pause, record, and list automation plans."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    create_response = client.post(
        "/api/agent/automations",
        json={
            "name": "每日 DAC 状态摘要",
            "prompt": "每天生成一次 DAC-3D 状态摘要。",
            "session_id": "automation-ui-session",
            "schedule": {"type": "daily", "time": "08:30"},
        },
    )
    automation_id = create_response.json()["automation"]["id"]
    run_response = client.post(
        f"/api/agent/automations/{automation_id}/runs",
        json={"result": "已完成摘要。", "trace_id": "trace-web-auto"},
    )
    pause_response = client.post(
        f"/api/agent/automations/{automation_id}/status",
        json={"status": "paused", "note": "演示暂停。"},
    )
    list_response = client.get("/api/agent/automations?session_id=automation-ui-session")
    due_response = client.get("/api/agent/automations/due")
    workspace_response = client.get("/api/agent/workspace")

    assert create_response.status_code == 200
    assert create_response.json()["automation"]["schedule_summary"] == "daily at 08:30"
    assert run_response.status_code == 200
    assert run_response.json()["run"]["trace_id"] == "trace-web-auto"
    assert pause_response.status_code == 200
    assert pause_response.json()["automation"]["status"] == "paused"
    assert list_response.status_code == 200
    assert list_response.json()["count"] == 1
    assert due_response.status_code == 200
    assert due_response.json()["count"] == 0
    assert workspace_response.json()["automations"]["automation_count"] == 1


def test_web_api_goal_tracker_endpoints(tmp_path) -> None:
    """The web UI should create, update, complete, and list Agent goals."""
    config = make_config(tmp_path)
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    agent_runtime = DAC3DAgentChatAdapter(
        DAC3DAgentRuntime(assistant=assistant, config=assistant.config)
    )
    client = TestClient(create_api_app(agent_runtime))

    create_response = client.post(
        "/api/goals",
        json={"objective": "目标：新增 Agent goal 目标功能", "session_id": "goal-ui-session"},
    )
    goal_id = create_response.json()["goal"]["id"]
    progress_response = client.post(
        f"/api/goals/{goal_id}/progress",
        json={"note": "已接入 API。"},
    )
    complete_response = client.post(
        f"/api/goals/{goal_id}/complete",
        json={"note": "已完成 UI 联调。"},
    )
    active_response = client.get("/api/goals?status=active")
    completed_response = client.get("/api/goals?status=completed")
    workspace_response = client.get("/api/agent/workspace")

    assert create_response.status_code == 200
    assert create_response.json()["created"] is True
    assert progress_response.status_code == 200
    assert progress_response.json()["goal"]["progress"][0]["note"] == "已接入 API。"
    assert complete_response.status_code == 200
    assert complete_response.json()["goal"]["status"] == "completed"
    assert active_response.json()["count"] == 0
    assert completed_response.json()["count"] == 1
    assert workspace_response.json()["goals"]["goal_count"] == 1
