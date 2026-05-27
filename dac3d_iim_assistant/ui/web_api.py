"""FastAPI application for the DAC-3D assistant web client."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator, Sequence
from inspect import signature
from pathlib import Path
from typing import Any


def create_api_app(assistant: Any, frontend_dist_dir: Path | None = None) -> Any:
    """Create the FastAPI app that powers the React web client."""
    try:
        from fastapi import Body, FastAPI, File, HTTPException, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "FastAPI dependencies are not installed. Install project dependencies first."
        ) from exc

    app = FastAPI(title="DAC-3D IIM Assistant API")
    try:
        from machine_agent import MachineAgentService

        machine_agent = MachineAgentService()
    except Exception:
        machine_agent = None
    frontend_dev_url = assistant.config.frontend_dev_url
    allowed_origins = [
        origin
        for origin in {
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            frontend_dev_url,
        }
        if origin
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runtime")
    def runtime_summary() -> dict[str, Any]:
        return assistant.runtime_summary()

    @app.get("/api/mcp/manifest")
    def mcp_manifest(session_id: str | None = None) -> dict[str, Any]:
        manifest = getattr(assistant, "mcp_capability_manifest", None)
        if not callable(manifest):
            raise HTTPException(status_code=503, detail="MCP capability manifest is unavailable.")
        return manifest(session_id=(session_id or "web"))

    @app.get("/api/agent/workspace")
    def agent_workspace() -> dict[str, Any]:
        workspace = getattr(assistant, "agent_workspace", None)
        if not callable(workspace):
            raise HTTPException(status_code=503, detail="Agent workspace is unavailable.")
        return workspace()

    @app.get("/api/agent/verifications")
    def list_agent_verification_runs(
        status: str | None = None,
        preset_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        list_runs = getattr(assistant, "list_agent_verification_runs", None)
        if not callable(list_runs):
            raise HTTPException(status_code=503, detail="Verification feedback runner is unavailable.")
        try:
            return list_runs(status=status, preset_id=preset_id, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/agent/verifications/presets")
    def list_agent_verification_presets() -> dict[str, Any]:
        list_presets = getattr(assistant, "list_agent_verification_presets", None)
        if not callable(list_presets):
            raise HTTPException(status_code=503, detail="Verification feedback runner is unavailable.")
        return list_presets()

    @app.post("/api/agent/verifications/run")
    def run_agent_verification(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        run_verification = getattr(assistant, "run_agent_verification", None)
        if not callable(run_verification):
            raise HTTPException(status_code=503, detail="Verification feedback runner is unavailable.")
        preset_id = str(request.get("preset_id") or request.get("preset") or "").strip()
        if not preset_id:
            raise HTTPException(status_code=422, detail="The `preset_id` field is required.")
        try:
            timeout_seconds = int(request.get("timeout_seconds") or 120)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="The `timeout_seconds` field must be an integer.") from exc
        try:
            return run_verification(preset_id, timeout_seconds=timeout_seconds)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/agent/reviews")
    def list_agent_review_handoffs(
        session_id: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        list_reviews = getattr(assistant, "list_agent_review_handoffs", None)
        if not callable(list_reviews):
            raise HTTPException(status_code=503, detail="Review handoff queue is unavailable.")
        try:
            return list_reviews(session_id=session_id, status=status, priority=priority, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/agent/reviews/{review_id}")
    def read_agent_review_handoff(review_id: str) -> dict[str, Any]:
        read_review = getattr(assistant, "read_agent_review_handoff", None)
        if not callable(read_review):
            raise HTTPException(status_code=503, detail="Review handoff queue is unavailable.")
        try:
            return read_review(review_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/agent/reviews")
    def create_agent_review_handoff(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        create_review = getattr(assistant, "create_agent_review_handoff", None)
        if not callable(create_review):
            raise HTTPException(status_code=503, detail="Review handoff queue is unavailable.")
        title = str(request.get("title") or "").strip()
        if not title:
            raise HTTPException(status_code=422, detail="The `title` field is required.")
        try:
            return create_review(
                title,
                summary=str(request.get("summary") or ""),
                session_id=str(request.get("session_id") or "web").strip() or "web",
                status=str(request.get("status") or "pending"),
                priority=str(request.get("priority") or "normal"),
                task_id=str(request.get("task_id") or ""),
                workflow_id=str(request.get("workflow_id") or ""),
                trace_id=str(request.get("trace_id") or ""),
                files=request.get("files") if isinstance(request.get("files"), list) else None,
                verification_run_ids=request.get("verification_run_ids")
                if isinstance(request.get("verification_run_ids"), list)
                else None,
                checklist=request.get("checklist") if isinstance(request.get("checklist"), list) else None,
                metadata=request.get("metadata") if isinstance(request.get("metadata"), dict) else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/agent/reviews/{review_id}/comments")
    def add_agent_review_comment(review_id: str, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        add_comment = getattr(assistant, "add_agent_review_comment", None)
        if not callable(add_comment):
            raise HTTPException(status_code=503, detail="Review handoff queue is unavailable.")
        body = str(request.get("body") or "").strip()
        if not body:
            raise HTTPException(status_code=422, detail="The `body` field is required.")
        try:
            return add_comment(
                review_id,
                body,
                reviewer=str(request.get("reviewer") or "human"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/agent/reviews/{review_id}/status")
    def update_agent_review_status(review_id: str, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        update_review = getattr(assistant, "update_agent_review_status", None)
        if not callable(update_review):
            raise HTTPException(status_code=503, detail="Review handoff queue is unavailable.")
        status = str(request.get("status") or "").strip()
        if not status:
            raise HTTPException(status_code=422, detail="The `status` field is required.")
        try:
            return update_review(
                review_id,
                status,
                note=str(request.get("note") or ""),
                reviewer=str(request.get("reviewer") or "human"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/agent/repo-map")
    def read_repo_context_map() -> dict[str, Any]:
        read_map = getattr(assistant, "read_repo_context_map", None)
        if not callable(read_map):
            raise HTTPException(status_code=503, detail="Repo context map is unavailable.")
        try:
            return read_map()
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/agent/repo-map/build")
    def build_repo_context_map(request: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        build_map = getattr(assistant, "build_repo_context_map", None)
        if not callable(build_map):
            raise HTTPException(status_code=503, detail="Repo context map is unavailable.")
        payload = dict(request or {})
        try:
            max_files = int(payload.get("max_files") or 1200)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="The `max_files` field must be an integer.") from exc
        try:
            return build_map(max_files=max_files)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/agent/repo-map/search")
    def search_repo_context_map(q: str, limit: int = 20) -> dict[str, Any]:
        search_map = getattr(assistant, "search_repo_context_map", None)
        if not callable(search_map):
            raise HTTPException(status_code=503, detail="Repo context map is unavailable.")
        try:
            return search_map(q, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/agent/git/status")
    def git_workspace_status(
        recent_limit: int = 5,
        include_diff_stat: bool = True,
        include_worktrees: bool = True,
    ) -> dict[str, Any]:
        git_status = getattr(assistant, "git_workspace_status", None)
        if not callable(git_status):
            raise HTTPException(status_code=503, detail="Git workspace context is unavailable.")
        try:
            return git_status(
                recent_limit=recent_limit,
                include_diff_stat=include_diff_stat,
                include_worktrees=include_worktrees,
            )
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/agent/artifacts")
    def list_agent_artifacts(
        session_id: str | None = None,
        artifact_type: str | None = None,
        tag: str | None = None,
        q: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        list_artifacts = getattr(assistant, "list_agent_artifacts", None)
        if not callable(list_artifacts):
            raise HTTPException(status_code=503, detail="Agent artifact store is unavailable.")
        return list_artifacts(
            session_id=session_id,
            artifact_type=artifact_type,
            tag=tag,
            query=q,
            limit=limit,
        )

    @app.get("/api/agent/artifacts/{artifact_id}")
    def read_agent_artifact(artifact_id: str) -> dict[str, Any]:
        read_artifact = getattr(assistant, "read_agent_artifact", None)
        if not callable(read_artifact):
            raise HTTPException(status_code=503, detail="Agent artifact store is unavailable.")
        try:
            return read_artifact(artifact_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/agent/artifacts")
    def create_agent_artifact(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        create_artifact = getattr(assistant, "create_agent_artifact", None)
        if not callable(create_artifact):
            raise HTTPException(status_code=503, detail="Agent artifact store is unavailable.")
        title = str(request.get("title") or "").strip()
        if not title:
            raise HTTPException(status_code=422, detail="The `title` field is required.")
        if "content" not in request:
            raise HTTPException(status_code=422, detail="The `content` field is required.")
        try:
            return create_artifact(
                title,
                request.get("content"),
                artifact_type=str(request.get("artifact_type") or "markdown"),
                session_id=str(request.get("session_id") or "web").strip() or "web",
                task_id=str(request.get("task_id") or ""),
                workflow_id=str(request.get("workflow_id") or ""),
                trace_id=str(request.get("trace_id") or ""),
                tags=request.get("tags") if isinstance(request.get("tags"), list) else None,
                metadata=request.get("metadata") if isinstance(request.get("metadata"), dict) else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/agent/events")
    def list_agent_events(
        session_id: str | None = None,
        status: str | None = None,
        event_type: str | None = None,
        priority: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        list_events = getattr(assistant, "list_agent_events", None)
        if not callable(list_events):
            raise HTTPException(status_code=503, detail="Agent event queue is unavailable.")
        return list_events(
            session_id=session_id,
            status=status,
            event_type=event_type,
            priority=priority,
            limit=limit,
        )

    @app.get("/api/agent/events/due")
    def list_due_agent_events(limit: int = 20) -> dict[str, Any]:
        list_due = getattr(assistant, "list_due_agent_events", None)
        if not callable(list_due):
            raise HTTPException(status_code=503, detail="Agent event queue is unavailable.")
        return list_due(limit=limit)

    @app.post("/api/agent/events")
    def enqueue_agent_event(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        enqueue = getattr(assistant, "enqueue_agent_event", None)
        if not callable(enqueue):
            raise HTTPException(status_code=503, detail="Agent event queue is unavailable.")
        event_type = str(request.get("event_type") or "").strip()
        if not event_type:
            raise HTTPException(status_code=422, detail="The `event_type` field is required.")
        try:
            return enqueue(
                event_type,
                payload=request.get("payload") if isinstance(request.get("payload"), dict) else None,
                session_id=str(request.get("session_id") or "web").strip() or "web",
                priority=str(request.get("priority") or "normal"),
                scheduled_for=str(request.get("scheduled_for") or ""),
                task_id=str(request.get("task_id") or ""),
                workflow_id=str(request.get("workflow_id") or ""),
                automation_id=str(request.get("automation_id") or ""),
                goal_id=str(request.get("goal_id") or ""),
                metadata=request.get("metadata") if isinstance(request.get("metadata"), dict) else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/agent/events/claim")
    def claim_next_agent_event(request: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        claim = getattr(assistant, "claim_next_agent_event", None)
        if not callable(claim):
            raise HTTPException(status_code=503, detail="Agent event queue is unavailable.")
        payload = dict(request or {})
        try:
            return claim(
                worker_id=str(payload.get("worker_id") or "agent-worker"),
                session_id=str(payload.get("session_id") or "").strip() or None,
                event_type=str(payload.get("event_type") or "").strip() or None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/agent/events/{event_id}/status")
    def update_agent_event_status(event_id: str, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        update_status = getattr(assistant, "update_agent_event_status", None)
        if not callable(update_status):
            raise HTTPException(status_code=503, detail="Agent event queue is unavailable.")
        status = str(request.get("status") or "").strip()
        if not status:
            raise HTTPException(status_code=422, detail="The `status` field is required.")
        try:
            return update_status(
                event_id,
                status,
                note=str(request.get("note") or ""),
                result=request.get("result") if isinstance(request.get("result"), dict) else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/agent/workflow/preview")
    def preview_agent_workflow(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        preview = getattr(assistant, "preview_agent_workflow", None)
        if not callable(preview):
            raise HTTPException(status_code=503, detail="Agent workflow preview is unavailable.")
        task = str(request.get("task") or "").strip()
        session_id = str(request.get("session_id") or "web-preview").strip() or "web-preview"
        if not task:
            raise HTTPException(status_code=422, detail="The `task` field is required.")
        try:
            return preview(task, session_id=session_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/agent/workflows")
    def list_agent_workflows(
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        list_workflows = getattr(assistant, "list_agent_workflows", None)
        if not callable(list_workflows):
            raise HTTPException(status_code=503, detail="Agent workflow templates are unavailable.")
        return list_workflows(session_id=session_id, status=status, limit=limit)

    @app.get("/api/agent/workflows/{workflow_id}")
    def read_agent_workflow(workflow_id: str) -> dict[str, Any]:
        read_workflow = getattr(assistant, "read_agent_workflow", None)
        if not callable(read_workflow):
            raise HTTPException(status_code=503, detail="Agent workflow templates are unavailable.")
        try:
            return read_workflow(workflow_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/agent/workflows")
    def create_agent_workflow(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        create_workflow = getattr(assistant, "create_agent_workflow", None)
        if not callable(create_workflow):
            raise HTTPException(status_code=503, detail="Agent workflow templates are unavailable.")
        name = str(request.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=422, detail="The `name` field is required.")
        try:
            return create_workflow(
                name,
                session_id=str(request.get("session_id") or "web").strip() or "web",
                description=str(request.get("description") or ""),
                nodes=request.get("nodes") if isinstance(request.get("nodes"), list) else None,
                edges=request.get("edges") if isinstance(request.get("edges"), list) else None,
                status=str(request.get("status") or "draft"),
                tags=request.get("tags") if isinstance(request.get("tags"), list) else None,
                metadata=request.get("metadata") if isinstance(request.get("metadata"), dict) else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/agent/workflows/from-preview")
    def create_agent_workflow_from_preview(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        create_from_preview = getattr(assistant, "create_agent_workflow_from_preview", None)
        if not callable(create_from_preview):
            raise HTTPException(status_code=503, detail="Agent workflow templates are unavailable.")
        task = str(request.get("task") or "").strip()
        if not task:
            raise HTTPException(status_code=422, detail="The `task` field is required.")
        try:
            return create_from_preview(
                task,
                name=str(request.get("name") or ""),
                session_id=str(request.get("session_id") or "web").strip() or "web",
                status=str(request.get("status") or "draft"),
                tags=request.get("tags") if isinstance(request.get("tags"), list) else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/agent/workflows/{workflow_id}/status")
    def update_agent_workflow_status(workflow_id: str, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        update_status = getattr(assistant, "update_agent_workflow_status", None)
        if not callable(update_status):
            raise HTTPException(status_code=503, detail="Agent workflow templates are unavailable.")
        status = str(request.get("status") or "").strip()
        if not status:
            raise HTTPException(status_code=422, detail="The `status` field is required.")
        try:
            return update_status(workflow_id, status)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/knowledge-base/summary")
    def knowledge_base_summary() -> dict[str, Any]:
        return assistant.knowledge_base_summary()

    @app.get("/api/machine-agent/snapshot")
    def machine_agent_snapshot() -> dict[str, Any]:
        if machine_agent is None:
            raise HTTPException(status_code=503, detail="Machine Agent is unavailable.")
        return machine_agent.snapshot()

    @app.post("/api/machine-agent/chat")
    def machine_agent_chat(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        if machine_agent is None:
            raise HTTPException(status_code=503, detail="Machine Agent is unavailable.")
        message = str(request.get("message", "")).strip()
        if not message:
            raise HTTPException(status_code=422, detail="The `message` field is required.")
        return machine_agent.chat(message)

    @app.get("/api/machine-agent/status")
    def machine_agent_status() -> dict[str, Any]:
        if machine_agent is None:
            raise HTTPException(status_code=503, detail="Machine Agent is unavailable.")
        return machine_agent.get_current_machine_status()

    @app.post("/api/chat")
    def chat(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            message, history, session_id = _parse_chat_request(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        response = _call_assistant_message(assistant, message, history, session_id=session_id)
        return response.to_ui_payload()

    @app.post("/api/chat/stream")
    def chat_stream(request: dict[str, Any] = Body(...)) -> StreamingResponse:
        try:
            message, history, session_id = _parse_chat_request(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return StreamingResponse(
            _stream_chat_events(assistant, message, history, session_id=session_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/commands/approve")
    def approve_command(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        session_id = str(request.get("session_id") or "web").strip() or "web"
        preview_id = str(request.get("preview_id") or "").strip() or None
        confirmation_token = str(request.get("confirmation_token") or "").strip() or None
        response = _approve_pending_command(
            assistant,
            session_id=session_id,
            preview_id=preview_id,
            confirmation_token=confirmation_token,
        )
        return response.to_ui_payload()

    @app.get("/api/goals")
    def list_goals(
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        list_goal_items = getattr(assistant, "list_goals", None)
        if not callable(list_goal_items):
            raise HTTPException(status_code=503, detail="Goal tracker is unavailable.")
        return list_goal_items(session_id=session_id, status=status, limit=limit)

    @app.post("/api/goals")
    def create_goal(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        create_goal_item = getattr(assistant, "create_goal", None)
        if not callable(create_goal_item):
            raise HTTPException(status_code=503, detail="Goal tracker is unavailable.")
        objective = str(request.get("objective") or "").strip()
        session_id = str(request.get("session_id") or "web").strip() or "web"
        if not objective:
            raise HTTPException(status_code=422, detail="The `objective` field is required.")
        try:
            return create_goal_item(objective, session_id=session_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/goals/{goal_id}/progress")
    def append_goal_progress(goal_id: str, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        append_progress = getattr(assistant, "append_goal_progress", None)
        if not callable(append_progress):
            raise HTTPException(status_code=503, detail="Goal tracker is unavailable.")
        note = str(request.get("note") or "").strip()
        status = str(request.get("status") or "").strip() or None
        if not note:
            raise HTTPException(status_code=422, detail="The `note` field is required.")
        try:
            return append_progress(goal_id, note, status=status)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/goals/{goal_id}/complete")
    def complete_goal(goal_id: str, request: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        complete_goal_item = getattr(assistant, "complete_goal", None)
        if not callable(complete_goal_item):
            raise HTTPException(status_code=503, detail="Goal tracker is unavailable.")
        note = str(dict(request or {}).get("note") or "")
        try:
            return complete_goal_item(goal_id, note=note)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/agent/tasks")
    def list_agent_tasks(
        session_id: str | None = None,
        status: str | None = None,
        goal_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        list_tasks = getattr(assistant, "list_agent_tasks", None)
        if not callable(list_tasks):
            raise HTTPException(status_code=503, detail="Agent task board is unavailable.")
        return list_tasks(session_id=session_id, status=status, goal_id=goal_id, limit=limit)

    @app.post("/api/agent/tasks")
    def create_agent_task(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        create_task = getattr(assistant, "create_agent_task", None)
        if not callable(create_task):
            raise HTTPException(status_code=503, detail="Agent task board is unavailable.")
        title = str(request.get("title") or "").strip()
        if not title:
            raise HTTPException(status_code=422, detail="The `title` field is required.")
        try:
            return create_task(
                title,
                session_id=str(request.get("session_id") or "web").strip() or "web",
                description=str(request.get("description") or ""),
                status=str(request.get("status") or "backlog"),
                priority=str(request.get("priority") or "normal"),
                goal_id=str(request.get("goal_id") or ""),
                agent_path=request.get("agent_path") if isinstance(request.get("agent_path"), list) else None,
                tool_candidates=request.get("tool_candidates")
                if isinstance(request.get("tool_candidates"), list)
                else None,
                dependencies=request.get("dependencies") if isinstance(request.get("dependencies"), list) else None,
                metadata=request.get("metadata") if isinstance(request.get("metadata"), dict) else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/agent/tasks/from-workflow")
    def create_agent_task_from_workflow(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        create_from_workflow = getattr(assistant, "create_agent_task_from_workflow", None)
        if not callable(create_from_workflow):
            raise HTTPException(status_code=503, detail="Agent task board is unavailable.")
        task = str(request.get("task") or "").strip()
        if not task:
            raise HTTPException(status_code=422, detail="The `task` field is required.")
        try:
            return create_from_workflow(
                task,
                session_id=str(request.get("session_id") or "web").strip() or "web",
                goal_id=str(request.get("goal_id") or ""),
                status=str(request.get("status") or "ready"),
                priority=str(request.get("priority") or "normal"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/agent/tasks/{task_id}/status")
    def update_agent_task_status(task_id: str, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        update_status = getattr(assistant, "update_agent_task_status", None)
        if not callable(update_status):
            raise HTTPException(status_code=503, detail="Agent task board is unavailable.")
        status = str(request.get("status") or "").strip()
        if not status:
            raise HTTPException(status_code=422, detail="The `status` field is required.")
        try:
            return update_status(task_id, status, note=str(request.get("note") or ""))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/agent/automations")
    def list_agent_automations(
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        list_automations = getattr(assistant, "list_agent_automations", None)
        if not callable(list_automations):
            raise HTTPException(status_code=503, detail="Agent automation planner is unavailable.")
        return list_automations(session_id=session_id, status=status, limit=limit)

    @app.get("/api/agent/automations/due")
    def list_due_agent_automations(limit: int = 20) -> dict[str, Any]:
        list_due = getattr(assistant, "list_due_agent_automations", None)
        if not callable(list_due):
            raise HTTPException(status_code=503, detail="Agent automation planner is unavailable.")
        return list_due(limit=limit)

    @app.post("/api/agent/automations")
    def create_agent_automation(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        create_automation = getattr(assistant, "create_agent_automation", None)
        if not callable(create_automation):
            raise HTTPException(status_code=503, detail="Agent automation planner is unavailable.")
        name = str(request.get("name") or "").strip()
        prompt = str(request.get("prompt") or "").strip()
        if not name:
            raise HTTPException(status_code=422, detail="The `name` field is required.")
        if not prompt:
            raise HTTPException(status_code=422, detail="The `prompt` field is required.")
        try:
            return create_automation(
                name,
                prompt,
                session_id=str(request.get("session_id") or "web").strip() or "web",
                schedule=request.get("schedule") if isinstance(request.get("schedule"), dict) else None,
                status=str(request.get("status") or "active"),
                task_id=str(request.get("task_id") or ""),
                goal_id=str(request.get("goal_id") or ""),
                metadata=request.get("metadata") if isinstance(request.get("metadata"), dict) else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/agent/automations/{automation_id}/status")
    def update_agent_automation_status(
        automation_id: str,
        request: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        update_status = getattr(assistant, "update_agent_automation_status", None)
        if not callable(update_status):
            raise HTTPException(status_code=503, detail="Agent automation planner is unavailable.")
        status = str(request.get("status") or "").strip()
        if not status:
            raise HTTPException(status_code=422, detail="The `status` field is required.")
        try:
            return update_status(automation_id, status, note=str(request.get("note") or ""))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/agent/automations/{automation_id}/runs")
    def record_agent_automation_run(
        automation_id: str,
        request: dict[str, Any] | None = Body(default=None),
    ) -> dict[str, Any]:
        record_run = getattr(assistant, "record_agent_automation_run", None)
        if not callable(record_run):
            raise HTTPException(status_code=503, detail="Agent automation planner is unavailable.")
        payload = dict(request or {})
        try:
            return record_run(
                automation_id,
                result=str(payload.get("result") or ""),
                status=str(payload.get("status") or "completed"),
                trace_id=str(payload.get("trace_id") or ""),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/evals/run")
    def run_evals(request: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        request = dict(request or {})
        categories = request.get("categories")
        if categories is not None and not isinstance(categories, list):
            raise HTTPException(status_code=422, detail="The `categories` field must be a list.")
        run = getattr(assistant, "run_evals", None)
        if not callable(run):
            raise HTTPException(status_code=503, detail="Eval runner is unavailable.")
        return run(categories=[str(item) for item in categories] if categories else None)

    @app.get("/api/evals/drafts")
    def list_eval_drafts() -> dict[str, Any]:
        list_drafts = getattr(assistant, "list_eval_drafts", None)
        if not callable(list_drafts):
            raise HTTPException(status_code=503, detail="Eval draft generator is unavailable.")
        try:
            return list_drafts()
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/evals/drafts")
    def generate_eval_drafts(request: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        request = dict(request or {})
        trace_ids = request.get("trace_ids")
        limit = request.get("limit", 5)
        if trace_ids is not None and not isinstance(trace_ids, list):
            raise HTTPException(status_code=422, detail="The `trace_ids` field must be a list.")
        try:
            safe_limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="The `limit` field must be an integer.") from exc
        generate = getattr(assistant, "generate_eval_drafts", None)
        if not callable(generate):
            raise HTTPException(status_code=503, detail="Eval draft generator is unavailable.")
        try:
            return generate(
                trace_ids=[str(item) for item in trace_ids] if trace_ids else None,
                limit=safe_limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/evals/codex-handoff")
    def generate_codex_handoff(request: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        request = dict(request or {})
        categories = request.get("categories")
        recent_trace_limit = request.get("recent_trace_limit", 8)
        if categories is not None and not isinstance(categories, list):
            raise HTTPException(status_code=422, detail="The `categories` field must be a list.")
        try:
            safe_trace_limit = int(recent_trace_limit)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="The `recent_trace_limit` field must be an integer.") from exc
        generate = getattr(assistant, "generate_codex_handoff", None)
        if not callable(generate):
            raise HTTPException(status_code=503, detail="Codex handoff generator is unavailable.")
        try:
            return generate(
                categories=[str(item) for item in categories] if categories else None,
                recent_trace_limit=safe_trace_limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/traces/{trace_id}")
    def read_trace(trace_id: str) -> dict[str, Any]:
        read = getattr(assistant, "read_trace", None)
        if not callable(read):
            raise HTTPException(status_code=503, detail="Trace logger is unavailable.")
        try:
            return read(trace_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/skills/patches")
    def list_skill_patches(status: str = "pending") -> dict[str, Any]:
        list_patches = getattr(assistant, "list_skill_patches", None)
        if not callable(list_patches):
            raise HTTPException(status_code=503, detail="Skill patch review is unavailable.")
        return list_patches(status=status)

    @app.post("/api/skills/patches")
    def propose_skill_patch(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        propose = getattr(assistant, "propose_skill_patch", None)
        if not callable(propose):
            raise HTTPException(status_code=503, detail="Skill patch review is unavailable.")
        target_skill = str(request.get("target_skill") or "").strip()
        reason = str(request.get("reason") or "").strip()
        diff = str(request.get("diff") or "")
        replacement_section = str(request.get("replacement_section") or "")
        risk_level = str(request.get("risk_level") or "medium").strip() or "medium"
        trace_ids = request.get("evidence_trace_ids") or []
        if not isinstance(trace_ids, list):
            raise HTTPException(status_code=422, detail="The `evidence_trace_ids` field must be a list.")
        if not target_skill:
            raise HTTPException(status_code=422, detail="The `target_skill` field is required.")
        if not reason:
            raise HTTPException(status_code=422, detail="The `reason` field is required.")
        if not diff and not replacement_section:
            raise HTTPException(status_code=422, detail="Provide `diff` or `replacement_section`.")
        try:
            return propose(
                target_skill=target_skill,
                reason=reason,
                diff=diff,
                replacement_section=replacement_section,
                evidence_trace_ids=[str(item) for item in trace_ids],
                risk_level=risk_level,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/skills/patches/{patch_id}/approve")
    def approve_skill_patch(patch_id: str) -> dict[str, Any]:
        approve = getattr(assistant, "approve_skill_patch", None)
        if not callable(approve):
            raise HTTPException(status_code=503, detail="Skill patch review is unavailable.")
        try:
            return approve(patch_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/skills/patches/{patch_id}/reject")
    def reject_skill_patch(
        patch_id: str,
        request: dict[str, Any] | None = Body(default=None),
    ) -> dict[str, Any]:
        reject = getattr(assistant, "reject_skill_patch", None)
        if not callable(reject):
            raise HTTPException(status_code=503, detail="Skill patch review is unavailable.")
        reason = str(dict(request or {}).get("reason") or "ui_rejected")
        try:
            return reject(patch_id, reason=reason)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/memory/procedures")
    def list_procedure_memories() -> dict[str, Any]:
        list_procedures = getattr(assistant, "list_procedure_memories", None)
        if not callable(list_procedures):
            raise HTTPException(status_code=503, detail="Procedure memory is unavailable.")
        return list_procedures()

    @app.get("/api/memory/procedures/{name}")
    def read_procedure_memory(name: str) -> dict[str, Any]:
        read_procedure = getattr(assistant, "read_procedure_memory", None)
        if not callable(read_procedure):
            raise HTTPException(status_code=503, detail="Procedure memory is unavailable.")
        return read_procedure(name)

    @app.post("/api/memory/procedures")
    def propose_procedure_memory(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        propose = getattr(assistant, "propose_procedure_memory", None)
        if not callable(propose):
            raise HTTPException(status_code=503, detail="Procedure memory is unavailable.")
        name = str(request.get("name") or "").strip()
        content = str(request.get("content") or "").strip()
        reason = str(request.get("reason") or "procedure_memory_candidate").strip()
        mode = str(request.get("mode") or "append").strip() or "append"
        if not name:
            raise HTTPException(status_code=422, detail="The `name` field is required.")
        if not content:
            raise HTTPException(status_code=422, detail="The `content` field is required.")
        try:
            return propose(name=name, content=content, reason=reason, mode=mode)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/memory/patches")
    def list_memory_patches(status: str = "pending") -> dict[str, Any]:
        list_patches = getattr(assistant, "list_memory_patches", None)
        if not callable(list_patches):
            raise HTTPException(status_code=503, detail="Memory patch review is unavailable.")
        return list_patches(status=status)

    @app.post("/api/memory/patches/{patch_id}/approve")
    def approve_memory_patch(patch_id: str) -> dict[str, Any]:
        approve = getattr(assistant, "approve_memory_patch", None)
        if not callable(approve):
            raise HTTPException(status_code=503, detail="Memory patch review is unavailable.")
        try:
            return approve(patch_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/memory/patches/{patch_id}/reject")
    def reject_memory_patch(
        patch_id: str,
        request: dict[str, Any] | None = Body(default=None),
    ) -> dict[str, Any]:
        reject = getattr(assistant, "reject_memory_patch", None)
        if not callable(reject):
            raise HTTPException(status_code=503, detail="Memory patch review is unavailable.")
        reason = str(dict(request or {}).get("reason") or "ui_rejected")
        try:
            return reject(patch_id, reason=reason)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/knowledge-base/build")
    async def build_knowledge_base(files: list[UploadFile] | None = File(default=None)) -> dict[str, Any]:
        uploaded_paths: list[Path] = []
        temp_dir: Path | None = None
        if files:
            temp_dir = Path(tempfile.mkdtemp(prefix="dac3d-kb-", dir=assistant.config.upload_temp_dir))
            for file in files:
                if not file.filename:
                    continue
                destination = temp_dir / Path(file.filename).name
                destination.write_bytes(await file.read())
                uploaded_paths.append(destination)

        try:
            summary = assistant.build_knowledge_base_from_uploads(uploaded_paths)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            if temp_dir is not None:
                for path in temp_dir.glob("*"):
                    path.unlink(missing_ok=True)
                temp_dir.rmdir()

        return {
            "runtime": assistant.runtime_summary(),
            "knowledge_base": summary,
        }

    selected_frontend_dist = frontend_dist_dir or assistant.config.frontend_dist_dir

    if selected_frontend_dist.exists():
        assets_dir = selected_frontend_dist / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

        @app.get("/{static_path:path}")
        async def frontend_static(static_path: str) -> HTMLResponse | FileResponse:
            candidate = selected_frontend_dist / static_path
            if static_path and candidate.exists() and candidate.is_file():
                return FileResponse(candidate)
            return HTMLResponse(selected_frontend_dist.joinpath("index.html").read_text(encoding="utf-8"))

        @app.get("/", response_class=HTMLResponse)
        async def frontend_index() -> HTMLResponse:
            return HTMLResponse(selected_frontend_dist.joinpath("index.html").read_text(encoding="utf-8"))

    else:
        @app.get("/", response_class=HTMLResponse)
        async def frontend_missing() -> HTMLResponse:
            return HTMLResponse(_frontend_hint_html())

    return app


def _stream_chat_events(
    assistant: Any,
    message: str,
    history: Sequence[tuple[str, str]],
    *,
    session_id: str,
) -> Iterator[str]:
    """Yield SSE events for a single chat request."""
    try:
        for event_name, payload in _call_assistant_stream(
            assistant,
            message,
            history,
            session_id=session_id,
        ):
            yield _sse_event(event_name, payload)
    except Exception as exc:  # pragma: no cover - defensive streaming guard
        yield _sse_event("error", {"message": str(exc)})


def _parse_chat_request(payload: dict[str, Any]) -> tuple[str, list[tuple[str, str]], str]:
    """Normalize the chat request body into the assistant's internal schema."""
    message = str(payload.get("message", "")).strip()
    if not message:
        raise ValueError("The `message` field is required.")
    session_id = str(payload.get("session_id") or "web").strip() or "web"

    normalized_history: list[tuple[str, str]] = []
    raw_history = payload.get("history", [])
    if raw_history is None:
        raw_history = []
    if not isinstance(raw_history, list):
        raise ValueError("The `history` field must be a list.")

    for index, turn in enumerate(raw_history):
        if not isinstance(turn, dict):
            raise ValueError(f"History turn {index} must be an object.")
        user_message = str(turn.get("user", ""))
        assistant_message = str(turn.get("assistant", ""))
        normalized_history.append((user_message, assistant_message))

    return message, normalized_history, session_id


def _call_assistant_message(
    assistant: Any,
    message: str,
    history: Sequence[tuple[str, str]],
    *,
    session_id: str,
) -> Any:
    """Call chat backends with session ids when supported."""
    if _supports_session_id(assistant.handle_message):
        return assistant.handle_message(message, history, session_id=session_id)
    return assistant.handle_message(message, history)


def _call_assistant_stream(
    assistant: Any,
    message: str,
    history: Sequence[tuple[str, str]],
    *,
    session_id: str,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Call streaming backends with session ids when supported."""
    if _supports_session_id(assistant.stream_message):
        yield from assistant.stream_message(message, history, session_id=session_id)
        return
    yield from assistant.stream_message(message, history)


def _approve_pending_command(
    assistant: Any,
    *,
    session_id: str,
    preview_id: str | None,
    confirmation_token: str | None,
) -> Any:
    """Approve a command preview through the Agent gateway when available."""
    approve = getattr(assistant, "approve_pending_command", None)
    if callable(approve):
        return approve(
            session_id=session_id,
            preview_id=preview_id,
            confirmation_token=confirmation_token,
        )
    if _supports_session_id(assistant.handle_message):
        return assistant.handle_message("确认执行", [], session_id=session_id)
    return assistant.handle_message("确认执行", [])


def _supports_session_id(callable_obj: Any) -> bool:
    try:
        return "session_id" in signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


def _sse_event(event: str, data: Any) -> str:
    serialized = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {serialized}\n\n"


def _frontend_hint_html() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>DAC-3D IIM Assistant</title>
    <style>
      body {
        margin: 0;
        font-family: "Segoe UI Variable", "Microsoft YaHei UI", sans-serif;
        background: linear-gradient(180deg, #f5f7fb 0%, #eef2f8 100%);
        color: #162033;
      }
      main {
        max-width: 760px;
        margin: 60px auto;
        padding: 28px 32px;
        border-radius: 28px;
        background: rgba(255, 255, 255, 0.92);
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.1);
      }
      code {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.08);
      }
      pre {
        overflow: auto;
        padding: 16px 18px;
        border-radius: 18px;
        background: #0f172a;
        color: #e2e8f0;
      }
    </style>
  </head>
  <body>
    <main>
      <h1>前端尚未构建</h1>
      <p>FastAPI 后端已经启动，但 <code>frontend/dist</code> 还不存在。</p>
      <p>在仓库根目录执行：</p>
      <pre>cd frontend
npm install
npm run build</pre>
      <p>然后重新运行 <code>python app.py</code>。</p>
    </main>
  </body>
</html>
"""
