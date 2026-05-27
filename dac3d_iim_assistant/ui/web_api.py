"""FastAPI application for the DAC-3D assistant web client."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator, Sequence
from html.parser import HTMLParser
from inspect import signature
from pathlib import Path
from typing import Any

from security.production_config import ProductionSecurityConfig, SecurityConfigError
from tracing.logger import AuditTraceLogger
from tracing.redaction import redact_exception
from ui.auth import ApiSecurityError, PRIVILEGED_ROLES, require_api_actor
from ui.security_middleware import install_security_middleware
from ui.session import ConfirmationTokenStore

try:  # FastAPI resolves postponed route annotations from module globals.
    from fastapi import Request, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
except Exception:  # pragma: no cover - optional dependency guard
    Request = Any  # type: ignore[misc,assignment]
    UploadFile = Any  # type: ignore[misc,assignment]
    FileResponse = Any  # type: ignore[misc,assignment]
    HTMLResponse = Any  # type: ignore[misc,assignment]
    StreamingResponse = Any  # type: ignore[misc,assignment]


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
    security_config = ProductionSecurityConfig.from_env(
        base_dir=assistant.config.base_dir,
        mock_mode=assistant.config.mock_mode,
        debug_mode=getattr(assistant.config, "debug_logging_enabled", False),
    )
    try:
        security_config.validate_or_raise()
    except SecurityConfigError as exc:
        raise RuntimeError(f"Unsafe production security configuration: {exc}") from exc
    app.state.security_config = security_config
    app.state.command_confirmations = ConfirmationTokenStore()
    app.state.audit_logger = (
        AuditTraceLogger(assistant.config.audit_trace_path)
        if getattr(assistant.config, "audit_trace_enabled", True)
        else None
    )
    install_security_middleware(
        app,
        rate_limit_enabled=security_config.rate_limit_enabled,
        audit_logger=app.state.audit_logger,
    )
    try:
        from machine_agent import MachineAgentService

        machine_agent = MachineAgentService()
    except Exception:
        machine_agent = None
    frontend_dev_url = assistant.config.frontend_dev_url
    is_production = security_config.is_prod
    allowed_origins = [
        origin
        for origin in (
            security_config.cors_allowed_origins
            or (
                "http://127.0.0.1:5173",
                "http://localhost:5173",
                frontend_dev_url,
            )
        )
        if origin and (not is_production or origin != "*")
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runtime")
    def runtime_summary(request: Request) -> dict[str, Any]:
        _require_actor(request)
        return _attach_trace(request, assistant.runtime_summary())

    @app.get("/api/mcp/manifest")
    def mcp_manifest(session_id: str | None = None) -> dict[str, Any]:
        manifest = getattr(assistant, "mcp_capability_manifest", None)
        if not callable(manifest):
            raise HTTPException(status_code=503, detail="MCP capability manifest is unavailable.")
        return manifest(session_id=(session_id or "web"))

    @app.get("/api/agent/workspace")
    def agent_workspace(request: Request) -> dict[str, Any]:
        _require_actor(request)
        workspace = getattr(assistant, "agent_workspace", None)
        if not callable(workspace):
            raise HTTPException(status_code=503, detail="Agent workspace is unavailable.")
        return _attach_trace(request, workspace())

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

    @app.post("/api/agent/workflow/preview")
    def preview_agent_workflow(http_request: Request, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        actor = _require_actor(http_request, payload=request)
        preview = getattr(assistant, "preview_agent_workflow", None)
        if not callable(preview):
            raise HTTPException(status_code=503, detail="Agent workflow preview is unavailable.")
        task = str(request.get("task") or "").strip()
        session_id = str(request.get("session_id") or actor.session_id).strip() or actor.session_id
        if not task:
            raise HTTPException(status_code=422, detail="The `task` field is required.")
        try:
            return _attach_trace(http_request, preview(task, session_id=session_id))
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
    def knowledge_base_summary(request: Request) -> dict[str, Any]:
        _require_actor(request)
        return _attach_trace(request, assistant.knowledge_base_summary())

    @app.get("/api/machine-agent/snapshot")
    def machine_agent_snapshot(request: Request) -> dict[str, Any]:
        _require_actor(request)
        if machine_agent is None:
            raise HTTPException(status_code=503, detail="Machine Agent is unavailable.")
        return _attach_trace(request, machine_agent.snapshot())

    @app.post("/api/machine-agent/chat")
    def machine_agent_chat(http_request: Request, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        _require_actor(http_request, payload=request)
        if machine_agent is None:
            raise HTTPException(status_code=503, detail="Machine Agent is unavailable.")
        message = str(request.get("message", "")).strip()
        if not message:
            raise HTTPException(status_code=422, detail="The `message` field is required.")
        return _attach_trace(http_request, machine_agent.chat(message))

    @app.get("/api/machine-agent/status")
    def machine_agent_status(request: Request) -> dict[str, Any]:
        _require_actor(request)
        if machine_agent is None:
            raise HTTPException(status_code=503, detail="Machine Agent is unavailable.")
        return _attach_trace(request, machine_agent.get_current_machine_status())

    @app.post("/api/chat")
    def chat(http_request: Request, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        actor = _require_actor(http_request, payload=request)
        try:
            message, history, session_id = _parse_chat_request(request, default_session_id=actor.session_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if _message_requires_confirmation(message):
            raise ApiSecurityError(
                code="COMMAND_CONFIRMATION_REQUIRED",
                message="Command execution requires preview and confirmation token validation.",
                status_code=403,
            )
        response = _call_assistant_message(assistant, message, history, session_id=session_id)
        return _attach_trace(http_request, response.to_ui_payload())

    @app.post("/api/chat/stream")
    def chat_stream(http_request: Request, request: dict[str, Any] = Body(...)) -> StreamingResponse:
        actor = _require_actor(http_request, payload=request)
        try:
            message, history, session_id = _parse_chat_request(request, default_session_id=actor.session_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if _message_requires_confirmation(message):
            raise ApiSecurityError(
                code="COMMAND_CONFIRMATION_REQUIRED",
                message="Command execution requires preview and confirmation token validation.",
                status_code=403,
            )
        return StreamingResponse(
            _stream_chat_events(
                assistant,
                message,
                history,
                session_id=session_id,
                request_id=getattr(http_request.state, "request_id", None),
                trace_id=getattr(http_request.state, "trace_id", None),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/commands/preview")
    def command_preview(http_request: Request, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        actor = _require_actor(http_request, payload=request, require_operator=True)
        message = str(request.get("message", "")).strip()
        if not message:
            raise HTTPException(status_code=422, detail="The `message` field is required.")
        history = _parse_history(request.get("history", []))
        response = _call_assistant_preview(assistant, message, history, session_id=actor.session_id)
        payload = _response_payload(response)
        command_preview = payload.get("command_preview")
        if isinstance(command_preview, dict) and not command_preview.get("missing_fields"):
            safety = dict(command_preview.get("safety") or {})
            if safety.get("needs_confirmation"):
                token_store: ConfirmationTokenStore = http_request.app.state.command_confirmations
                stored = token_store.create(
                    session_id=actor.session_id,
                    operator_id=str(actor.operator_id),
                    command_preview=command_preview,
                )
                payload["confirmation"] = {
                    "required": True,
                    "preview_id": stored.preview_id,
                    "preview_hash": stored.preview_hash,
                    "confirmation_token": stored.confirmation_token,
                    "expires_at": stored.expires_at,
                }
            else:
                payload["confirmation"] = {"required": False}
        _audit_event(
            http_request,
            actor=actor,
            event_type="command_preview",
            tool_call={
                "action": command_preview.get("action") if isinstance(command_preview, dict) else None,
                "message_length": len(message),
            },
            policy_decision={
                "missing_fields": command_preview.get("missing_fields") if isinstance(command_preview, dict) else [],
                "needs_confirmation": bool(payload.get("confirmation", {}).get("required")),
            },
            confirmation={
                key: value
                for key, value in dict(payload.get("confirmation") or {}).items()
                if key != "confirmation_token"
            },
        )
        return _attach_trace(http_request, payload)

    @app.post("/api/commands/confirm")
    def command_confirm(http_request: Request, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        actor = _require_actor(http_request, payload=request, write=True, require_operator=True)
        security_config: ProductionSecurityConfig = http_request.app.state.security_config
        if not security_config.allow_command_submit:
            raise ApiSecurityError(
                code="COMMAND_SUBMIT_DISABLED",
                message="Command submission is disabled by security configuration.",
                status_code=403,
            )
        token_store: ConfirmationTokenStore = http_request.app.state.command_confirmations
        stored = token_store.consume(
            preview_id=str(request.get("preview_id") or ""),
            confirmation_token=str(request.get("confirmation_token") or ""),
            session_id=actor.session_id,
            operator_id=str(actor.operator_id),
            preview_hash=str(request.get("preview_hash") or ""),
        )
        response = _call_assistant_execute_prepared(
            assistant,
            stored.command_preview,
            session_id=actor.session_id,
        )
        payload = _response_payload(response)
        payload["confirmation"] = {
            "preview_id": stored.preview_id,
            "preview_hash": stored.preview_hash,
            "used": True,
        }
        _audit_event(
            http_request,
            actor=actor,
            event_type="command_confirm",
            tool_call={"action": stored.command_preview.get("action")},
            policy_decision={"status": "submitted"},
            confirmation=payload["confirmation"],
        )
        return _attach_trace(http_request, payload)

    @app.post("/api/commands/approve")
    def approve_command(http_request: Request, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        actor = _require_actor(http_request, payload=request, write=True, require_operator=True)
        session_id = str(request.get("session_id") or actor.session_id).strip() or actor.session_id
        preview_id = str(request.get("preview_id") or "").strip() or None
        confirmation_token = str(request.get("confirmation_token") or "").strip() or None
        response = _approve_pending_command(
            assistant,
            session_id=session_id,
            preview_id=preview_id,
            confirmation_token=confirmation_token,
        )
        return _attach_trace(http_request, _response_payload(response))

    @app.get("/api/goals")
    def list_goals(
        request: Request,
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        _require_actor(request)
        list_goal_items = getattr(assistant, "list_goals", None)
        if not callable(list_goal_items):
            raise HTTPException(status_code=503, detail="Goal tracker is unavailable.")
        return _attach_trace(request, list_goal_items(session_id=session_id, status=status, limit=limit))

    @app.post("/api/goals")
    def create_goal(http_request: Request, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        actor = _require_actor(http_request, payload=request, write=True, require_operator=True)
        create_goal_item = getattr(assistant, "create_goal", None)
        if not callable(create_goal_item):
            raise HTTPException(status_code=503, detail="Goal tracker is unavailable.")
        objective = str(request.get("objective") or "").strip()
        session_id = str(request.get("session_id") or actor.session_id).strip() or actor.session_id
        if not objective:
            raise HTTPException(status_code=422, detail="The `objective` field is required.")
        try:
            return _attach_trace(http_request, create_goal_item(objective, session_id=session_id))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/goals/{goal_id}/progress")
    def append_goal_progress(
        http_request: Request,
        goal_id: str,
        request: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        _require_actor(http_request, payload=request, write=True, require_operator=True)
        append_progress = getattr(assistant, "append_goal_progress", None)
        if not callable(append_progress):
            raise HTTPException(status_code=503, detail="Goal tracker is unavailable.")
        note = str(request.get("note") or "").strip()
        status = str(request.get("status") or "").strip() or None
        if not note:
            raise HTTPException(status_code=422, detail="The `note` field is required.")
        try:
            return _attach_trace(http_request, append_progress(goal_id, note, status=status))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/goals/{goal_id}/complete")
    def complete_goal(
        http_request: Request,
        goal_id: str,
        request: dict[str, Any] | None = Body(default=None),
    ) -> dict[str, Any]:
        payload = dict(request or {})
        _require_actor(http_request, payload=payload, write=True, require_operator=True)
        complete_goal_item = getattr(assistant, "complete_goal", None)
        if not callable(complete_goal_item):
            raise HTTPException(status_code=503, detail="Goal tracker is unavailable.")
        note = str(payload.get("note") or "")
        try:
            return _attach_trace(http_request, complete_goal_item(goal_id, note=note))
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
    def run_evals(http_request: Request, request: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        _require_actor(http_request)
        request = dict(request or {})
        categories = request.get("categories")
        if categories is not None and not isinstance(categories, list):
            raise HTTPException(status_code=422, detail="The `categories` field must be a list.")
        run = getattr(assistant, "run_evals", None)
        if not callable(run):
            raise HTTPException(status_code=503, detail="Eval runner is unavailable.")
        return _attach_trace(http_request, run(categories=[str(item) for item in categories] if categories else None))

    @app.get("/api/evals/drafts")
    def list_eval_drafts(request: Request) -> dict[str, Any]:
        _require_actor(request)
        list_drafts = getattr(assistant, "list_eval_drafts", None)
        if not callable(list_drafts):
            raise HTTPException(status_code=503, detail="Eval draft generator is unavailable.")
        try:
            return _attach_trace(request, list_drafts())
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/evals/drafts")
    def generate_eval_drafts(
        http_request: Request,
        request: dict[str, Any] | None = Body(default=None),
    ) -> dict[str, Any]:
        payload = dict(request or {})
        _require_actor(http_request, payload=payload, write=True, require_operator=True)
        trace_ids = payload.get("trace_ids")
        limit = payload.get("limit", 5)
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
            return _attach_trace(
                http_request,
                generate(
                    trace_ids=[str(item) for item in trace_ids] if trace_ids else None,
                    limit=safe_limit,
                ),
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
    def read_trace(request: Request, trace_id: str) -> dict[str, Any]:
        _require_actor(request)
        read = getattr(assistant, "read_trace", None)
        if not callable(read):
            raise HTTPException(status_code=503, detail="Trace logger is unavailable.")
        try:
            return _attach_trace(request, read(trace_id))
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
    def list_memory_patches(request: Request, status: str = "pending") -> dict[str, Any]:
        _require_actor(request)
        list_patches = getattr(assistant, "list_memory_patches", None)
        if not callable(list_patches):
            raise HTTPException(status_code=503, detail="Memory patch review is unavailable.")
        return _attach_trace(request, list_patches(status=status))

    @app.post("/api/memory/patches/{patch_id}/approve")
    def approve_memory_patch(request: Request, patch_id: str) -> dict[str, Any]:
        _require_actor(request, write=True, require_operator=True, required_roles=PRIVILEGED_ROLES)
        approve = getattr(assistant, "approve_memory_patch", None)
        if not callable(approve):
            raise HTTPException(status_code=503, detail="Memory patch review is unavailable.")
        try:
            return _attach_trace(request, approve(patch_id))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/memory/patches/{patch_id}/reject")
    def reject_memory_patch(
        http_request: Request,
        patch_id: str,
        request: dict[str, Any] | None = Body(default=None),
    ) -> dict[str, Any]:
        payload = dict(request or {})
        _require_actor(http_request, payload=payload, write=True, require_operator=True, required_roles=PRIVILEGED_ROLES)
        reject = getattr(assistant, "reject_memory_patch", None)
        if not callable(reject):
            raise HTTPException(status_code=503, detail="Memory patch review is unavailable.")
        reason = str(payload.get("reason") or "ui_rejected")
        try:
            return _attach_trace(http_request, reject(patch_id, reason=reason))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/memory/approve")
    def memory_approve(http_request: Request, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        actor = _require_actor(
            http_request,
            payload=request,
            write=True,
            require_operator=True,
            required_roles=PRIVILEGED_ROLES,
        )
        del actor
        raise HTTPException(status_code=501, detail="Memory approval API is not implemented yet.")

    @app.post("/api/memory/reject")
    def memory_reject(http_request: Request, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        actor = _require_actor(
            http_request,
            payload=request,
            write=True,
            require_operator=True,
            required_roles=PRIVILEGED_ROLES,
        )
        del actor
        raise HTTPException(status_code=501, detail="Memory rejection API is not implemented yet.")

    @app.post("/api/skill-patches/apply")
    def skill_patch_apply(http_request: Request, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        actor = _require_actor(
            http_request,
            payload=request,
            write=True,
            require_operator=True,
            required_roles=PRIVILEGED_ROLES,
        )
        del actor
        raise HTTPException(status_code=501, detail="Skill patch apply API is not implemented yet.")

    @app.post("/api/knowledge-base/build")
    async def build_knowledge_base(
        request: Request,
        files: list[UploadFile] | None = File(default=None),
    ) -> dict[str, Any]:
        _require_actor(request, write=True, require_operator=True)
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

        return _attach_trace(request, {
            "runtime": assistant.runtime_summary(),
            "knowledge_base": summary,
        })

    selected_frontend_dist = frontend_dist_dir or assistant.config.frontend_dist_dir

    if _frontend_dist_is_usable(selected_frontend_dist):
        assets_dir = selected_frontend_dist / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

        @app.get("/{static_path:path}", include_in_schema=False, response_model=None)
        async def frontend_static(static_path: str) -> HTMLResponse | FileResponse:
            candidate = _static_file_candidate(selected_frontend_dist, static_path)
            if candidate is not None:
                return FileResponse(candidate)
            return HTMLResponse(selected_frontend_dist.joinpath("index.html").read_text(encoding="utf-8"))

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def frontend_index() -> HTMLResponse:
            return HTMLResponse(selected_frontend_dist.joinpath("index.html").read_text(encoding="utf-8"))

    else:
        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def frontend_missing() -> HTMLResponse:
            return HTMLResponse(_frontend_hint_html())

    return app


class _FrontendAssetParser(HTMLParser):
    """Collect Vite asset references from a built frontend index."""

    def __init__(self) -> None:
        super().__init__()
        self.assets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name not in {"href", "src"} or not value:
                continue
            clean_value = value.split("?", 1)[0].split("#", 1)[0]
            if clean_value.startswith("/assets/"):
                self.assets.append(clean_value.removeprefix("/"))
            elif clean_value.startswith("assets/"):
                self.assets.append(clean_value)


def _frontend_dist_is_usable(dist_dir: Path) -> bool:
    """Return whether the built frontend has an index and referenced assets."""
    index_path = dist_dir / "index.html"
    if not index_path.is_file():
        return False
    try:
        html = index_path.read_text(encoding="utf-8")
    except OSError:
        return False

    parser = _FrontendAssetParser()
    parser.feed(html)
    dist_root = dist_dir.resolve()
    for asset in parser.assets:
        asset_path = (dist_dir / asset).resolve()
        try:
            asset_path.relative_to(dist_root)
        except ValueError:
            return False
        if not asset_path.is_file():
            return False
    return True


def _static_file_candidate(dist_dir: Path, static_path: str) -> Path | None:
    """Return a static file only when it stays inside the frontend dist directory."""
    if not static_path:
        return None
    dist_root = dist_dir.resolve()
    candidate = (dist_dir / static_path).resolve()
    try:
        candidate.relative_to(dist_root)
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    return None


def _stream_chat_events(
    assistant: Any,
    message: str,
    history: Sequence[tuple[str, str]],
    *,
    session_id: str,
    request_id: str | None = None,
    trace_id: str | None = None,
) -> Iterator[str]:
    """Yield SSE events for a single chat request."""
    try:
        for event_name, payload in _call_assistant_stream(
            assistant,
            message,
            history,
            session_id=session_id,
        ):
            if event_name in {"meta", "done", "error"} and isinstance(payload, dict):
                payload = dict(payload)
                payload.setdefault("request_id", request_id)
                payload.setdefault("trace_id", trace_id)
            yield _sse_event(event_name, payload)
    except Exception as exc:  # pragma: no cover - defensive streaming guard
        yield _sse_event("error", {"message": redact_exception(exc), "request_id": request_id, "trace_id": trace_id})


def _parse_chat_request(
    payload: dict[str, Any],
    *,
    default_session_id: str = "web",
) -> tuple[str, list[tuple[str, str]], str]:
    """Normalize the chat request body into the assistant's internal schema."""
    message = str(payload.get("message", "")).strip()
    if not message:
        raise ValueError("The `message` field is required.")
    session_id = str(payload.get("session_id") or default_session_id).strip() or default_session_id

    normalized_history = _parse_history(payload.get("history", []))
    return message, normalized_history, session_id


def _parse_history(raw_history: Any) -> list[tuple[str, str]]:
    """Normalize chat history values into internal tuple pairs."""
    normalized_history: list[tuple[str, str]] = []
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
    return normalized_history


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


def _call_assistant_preview(
    assistant: Any,
    message: str,
    history: Sequence[tuple[str, str]],
    *,
    session_id: str,
) -> Any:
    """Call the safest available command-preview backend under either chat wrapper."""
    runtime = getattr(assistant, "runtime", None)
    if runtime is not None and hasattr(runtime, "preview_command"):
        return runtime.preview_command(message, session_id=session_id)
    if hasattr(assistant, "preview_command"):
        return assistant.preview_command(message, session_id=session_id)
    core = _core_assistant(assistant)
    if hasattr(core, "preview_operation_command"):
        return core.preview_operation_command(message, history)
    raise ApiSecurityError(
        code="COMMAND_PREVIEW_UNAVAILABLE",
        message="Command preview is unavailable for this runtime.",
        status_code=503,
    )


def _call_assistant_execute_prepared(
    assistant: Any,
    preview: dict[str, Any],
    *,
    session_id: str,
) -> Any:
    """Execute a stored preview through Tool Gateway when the runtime exposes it."""
    runtime = getattr(assistant, "runtime", None)
    gateway = preview.get("gateway") if isinstance(preview.get("gateway"), dict) else {}
    if runtime is not None and hasattr(runtime, "approve_pending_command") and gateway:
        return runtime.approve_pending_command(
            session_id=session_id,
            preview_id=str(gateway.get("preview_id") or ""),
            confirmation_token=str(gateway.get("confirmation_token") or ""),
        )
    approve = getattr(assistant, "approve_pending_command", None)
    if callable(approve) and gateway:
        return approve(
            session_id=session_id,
            preview_id=str(gateway.get("preview_id") or ""),
            confirmation_token=str(gateway.get("confirmation_token") or ""),
        )
    core = _core_assistant(assistant)
    if hasattr(core, "execute_prepared_command"):
        return core.execute_prepared_command(preview, confirmed_by_user=True)
    raise ApiSecurityError(
        code="COMMAND_CONFIRM_UNAVAILABLE",
        message="Command confirmation is unavailable for this runtime.",
        status_code=503,
    )


def _core_assistant(assistant: Any) -> Any:
    """Return the underlying deterministic assistant from optional Agent adapters."""
    runtime = getattr(assistant, "runtime", None)
    if runtime is not None and hasattr(runtime, "assistant"):
        return runtime.assistant
    return assistant


def _response_payload(response: Any) -> dict[str, Any]:
    """Normalize assistant responses and dict payloads."""
    if isinstance(response, dict):
        return dict(response)
    if hasattr(response, "to_ui_payload"):
        return dict(response.to_ui_payload())
    raise ApiSecurityError(
        code="INVALID_ASSISTANT_RESPONSE",
        message="Assistant returned an invalid response.",
        status_code=500,
    )


def _require_actor(
    request: Any,
    *,
    payload: dict[str, Any] | None = None,
    write: bool = False,
    require_operator: bool = False,
    required_roles: set[str] | None = None,
) -> Any:
    """Validate API actor context from headers, query, and optional JSON body."""
    return require_api_actor(
        request.headers,
        payload=payload,
        query_params=request.query_params,
        write=write,
        require_operator=require_operator,
        required_roles=required_roles,
    )


def _attach_trace(request: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Attach request and trace ids to successful API responses."""
    result = dict(payload)
    result.setdefault("request_id", getattr(request.state, "request_id", None))
    result.setdefault("trace_id", getattr(request.state, "trace_id", None))
    return result


def _audit_event(
    request: Any,
    *,
    actor: Any,
    event_type: str,
    tool_call: dict[str, Any],
    policy_decision: dict[str, Any] | None = None,
    confirmation: dict[str, Any] | None = None,
) -> None:
    logger = getattr(request.app.state, "audit_logger", None)
    if logger is None:
        return
    try:
        logger.append_event(
            event_type=event_type,
            trace_id=getattr(request.state, "trace_id", None),
            request_id=getattr(request.state, "request_id", None),
            session_id=getattr(actor, "session_id", None),
            actor={
                "session_id": getattr(actor, "session_id", None),
                "operator_id": getattr(actor, "operator_id", None),
                "roles": list(getattr(actor, "roles", ()) or ()),
            },
            tool_call=tool_call,
            policy_decision=policy_decision or {},
            confirmation=confirmation or {},
        )
    except Exception:
        return


def _message_requires_confirmation(message: str) -> bool:
    """Return whether generic chat is trying to submit a side-effect command."""
    lowered = message.strip().lower()
    preview_only_markers = (
        "不要执行",
        "不执行",
        "不要下发",
        "不下发",
        "只预览",
        "仅预览",
        "preview only",
        "do not execute",
        "don't execute",
    )
    if any(marker in lowered for marker in preview_only_markers):
        return False
    submit_markers = (
        "execute",
        "submit",
        "run now",
        "start now",
        "stop detection",
        "abort detection",
        "开始扫描",
        "执行扫描",
        "确认执行",
        "立即开始",
        "马上扫描",
        "开始在线扫描",
        "执行在线扫描",
        "离线检测",
        "离线测试",
        "停止检测",
        "中止检测",
        "提交命令",
    )
    return any(marker in lowered for marker in submit_markers)


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
