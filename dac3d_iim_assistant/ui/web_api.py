"""FastAPI application for the DAC-3D assistant web client."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterator, Sequence
from inspect import signature
from pathlib import Path
from typing import Any

from memory import ConversationMemoryStore, MemoryApprovalError
from security.path_policy import PathPolicy, PathPolicyError
from security.production_config import ProductionSecurityConfig, SecurityConfigError
from tracing.logger import AuditTraceLogger, DEFAULT_TRACE_QUERY_LIMIT, MAX_TRACE_QUERY_LIMIT
from tracing.redaction import redact_exception
from ui.auth import ApiSecurityError, PRIVILEGED_ROLES, require_api_actor
from ui.security_middleware import install_security_middleware
from ui.session import ConfirmationTokenStore


_OPENAPI_SECURITY_SCHEMES: dict[str, dict[str, str]] = {
    "DAC3DSessionId": {
        "type": "apiKey",
        "in": "header",
        "name": "X-DAC3D-Session-ID",
        "description": "DAC-3D session id. It is bound to body/query session_id when both are supplied.",
    },
    "DAC3DOperatorId": {
        "type": "apiKey",
        "in": "header",
        "name": "X-DAC3D-Operator-ID",
        "description": "Human operator id required for command, memory, upload, and other operator-scoped actions.",
    },
    "DAC3DRoles": {
        "type": "apiKey",
        "in": "header",
        "name": "X-DAC3D-Roles",
        "description": "Comma-separated actor roles. Use operator for writes; admin or security_admin for privileged memory APIs.",
    },
}

_OPENAPI_SECURITY_POLICIES: dict[tuple[str, str], dict[str, Any]] = {
    ("GET", "/api/runtime"): {"schemes": ("DAC3DSessionId",), "level": "read"},
    ("GET", "/api/knowledge-base/summary"): {"schemes": ("DAC3DSessionId",), "level": "read"},
    ("GET", "/api/machine-agent/snapshot"): {"schemes": ("DAC3DSessionId",), "level": "read"},
    ("POST", "/api/machine-agent/chat"): {"schemes": ("DAC3DSessionId",), "level": "read"},
    ("GET", "/api/machine-agent/status"): {"schemes": ("DAC3DSessionId",), "level": "read"},
    ("POST", "/api/chat"): {"schemes": ("DAC3DSessionId",), "level": "read"},
    ("POST", "/api/chat/stream"): {"schemes": ("DAC3DSessionId",), "level": "read"},
    ("POST", "/api/commands/preview"): {
        "schemes": ("DAC3DSessionId", "DAC3DOperatorId"),
        "level": "operator_preview",
    },
    ("POST", "/api/commands/confirm"): {
        "schemes": ("DAC3DSessionId", "DAC3DOperatorId", "DAC3DRoles"),
        "level": "write",
        "required_roles": ("operator", "admin", "security_admin"),
    },
    ("GET", "/api/memory/pending"): {
        "schemes": ("DAC3DSessionId", "DAC3DOperatorId", "DAC3DRoles"),
        "level": "privileged",
        "required_roles": ("admin", "security_admin"),
    },
    ("POST", "/api/memory/approve"): {
        "schemes": ("DAC3DSessionId", "DAC3DOperatorId", "DAC3DRoles"),
        "level": "privileged_write",
        "required_roles": ("admin", "security_admin"),
    },
    ("POST", "/api/memory/reject"): {
        "schemes": ("DAC3DSessionId", "DAC3DOperatorId", "DAC3DRoles"),
        "level": "privileged_write",
        "required_roles": ("admin", "security_admin"),
    },
    ("POST", "/api/memory/delete"): {
        "schemes": ("DAC3DSessionId", "DAC3DOperatorId", "DAC3DRoles"),
        "level": "privileged_write",
        "required_roles": ("admin", "security_admin"),
    },
    ("POST", "/api/skill-patches/apply"): {
        "schemes": ("DAC3DSessionId", "DAC3DOperatorId", "DAC3DRoles"),
        "level": "privileged_write",
        "required_roles": ("admin", "security_admin"),
    },
    ("POST", "/api/knowledge-base/build"): {
        "schemes": ("DAC3DSessionId", "DAC3DOperatorId", "DAC3DRoles"),
        "level": "write",
        "required_roles": ("operator", "admin", "security_admin"),
    },
    ("GET", "/api/audit/traces"): {
        "schemes": ("DAC3DSessionId", "DAC3DOperatorId", "DAC3DRoles"),
        "level": "privileged",
        "required_roles": ("admin", "security_admin"),
    },
    ("GET", "/api/audit/traces/{trace_id}"): {
        "schemes": ("DAC3DSessionId", "DAC3DOperatorId", "DAC3DRoles"),
        "level": "privileged",
        "required_roles": ("admin", "security_admin"),
    },
    ("GET", "/api/audit/export"): {
        "schemes": ("DAC3DSessionId", "DAC3DOperatorId", "DAC3DRoles"),
        "level": "privileged_export",
        "required_roles": ("admin", "security_admin"),
    },
}
_TRACE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")

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
        from fastapi import Body, FastAPI, File, HTTPException
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
    app.state.path_policy = PathPolicy.from_runtime_config(
        allowed_input_dirs=security_config.allowed_input_dirs,
        allowed_command_output_dir=security_config.allowed_command_output_dir,
    )
    if hasattr(getattr(assistant, "dac3d_client", None), "path_policy"):
        assistant.dac3d_client.path_policy = app.state.path_policy
    app.state.memory_store = _ensure_memory_store(assistant)
    app.state.command_confirmations = ConfirmationTokenStore()
    audit_trace_signing_key = getattr(assistant.config, "audit_trace_signing_key", "") or os.getenv(
        "DAC3D_AUDIT_TRACE_SIGNING_KEY",
        "",
    ).strip()
    audit_trace_key_id = (
        getattr(assistant.config, "audit_trace_key_id", "")
        or os.getenv("DAC3D_AUDIT_TRACE_KEY_ID", "local-audit-key").strip()
        or "local-audit-key"
    )
    app.state.audit_logger = (
        AuditTraceLogger(
            assistant.config.audit_trace_path,
            signing_key=audit_trace_signing_key,
            signing_key_id=audit_trace_key_id,
        )
        if getattr(assistant.config, "audit_trace_enabled", True)
        else None
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
    install_security_middleware(
        app,
        rate_limit_enabled=security_config.rate_limit_enabled,
        audit_logger=app.state.audit_logger,
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runtime")
    def runtime_summary(request: Request) -> dict[str, Any]:
        _require_actor(request)
        return _attach_trace(request, assistant.runtime_summary())

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

    @app.get("/api/memory/pending")
    def memory_pending(http_request: Request) -> dict[str, Any]:
        actor = _require_actor(
            http_request,
            require_operator=True,
            required_roles=PRIVILEGED_ROLES,
        )
        memory_store = _require_memory_store(http_request)
        patches = memory_store.list_pending_patches(
            session_id=http_request.query_params.get("session_id"),
            status=http_request.query_params.get("status") or "pending",
        )
        return _attach_trace(
            http_request,
            {
                "memory_patches": patches,
                "memory": memory_store.describe(),
                "actor": {"session_id": actor.session_id, "operator_id": actor.operator_id},
            },
        )

    @app.post("/api/memory/approve")
    def memory_approve(http_request: Request, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        actor = _require_actor(
            http_request,
            payload=request,
            write=True,
            require_operator=True,
            required_roles=PRIVILEGED_ROLES,
        )
        security_config: ProductionSecurityConfig = http_request.app.state.security_config
        if not security_config.enable_memory_write:
            raise ApiSecurityError(
                code="MEMORY_WRITE_DISABLED",
                message="Memory writes are disabled by security configuration.",
                status_code=403,
            )
        memory_store = _require_memory_store(http_request)
        try:
            patch = memory_store.approve_pending_patch(
                patch_id=str(request.get("patch_id") or request.get("memory_patch_id") or ""),
                operator_id=str(actor.operator_id or ""),
                session_id=request.get("session_id"),
            )
        except MemoryApprovalError as exc:
            raise ApiSecurityError(code=exc.code, message=str(exc), status_code=400) from exc
        _audit_event(
            http_request,
            actor=actor,
            event_type="memory_approve",
            tool_call={"name": "memory_approve", "patch_id": patch.get("id")},
            policy_decision={"status": patch.get("status")},
        )
        return _attach_trace(
            http_request,
            {
                "memory_patch": patch,
                "memory": memory_store.describe(),
            },
        )

    @app.post("/api/memory/reject")
    def memory_reject(http_request: Request, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        actor = _require_actor(
            http_request,
            payload=request,
            write=True,
            require_operator=True,
            required_roles=PRIVILEGED_ROLES,
        )
        memory_store = _require_memory_store(http_request)
        try:
            patch = memory_store.reject_pending_patch(
                patch_id=str(request.get("patch_id") or request.get("memory_patch_id") or ""),
                operator_id=str(actor.operator_id or ""),
                reason=str(request.get("reason") or ""),
                session_id=request.get("session_id"),
            )
        except MemoryApprovalError as exc:
            raise ApiSecurityError(code=exc.code, message=str(exc), status_code=400) from exc
        _audit_event(
            http_request,
            actor=actor,
            event_type="memory_reject",
            tool_call={"name": "memory_reject", "patch_id": patch.get("id")},
            policy_decision={"status": patch.get("status")},
        )
        return _attach_trace(
            http_request,
            {
                "memory_patch": patch,
                "memory": memory_store.describe(),
            },
        )

    @app.post("/api/memory/delete")
    def memory_delete(http_request: Request, request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        actor = _require_actor(
            http_request,
            payload=request,
            write=True,
            require_operator=True,
            required_roles=PRIVILEGED_ROLES,
        )
        memory_store = _require_memory_store(http_request)
        try:
            deleted = memory_store.delete_turn(
                session_id=str(request.get("session_id") or actor.session_id),
                turn_id=str(request.get("turn_id") or ""),
                operator_id=str(actor.operator_id or ""),
                reason=str(request.get("reason") or ""),
            )
        except MemoryApprovalError as exc:
            raise ApiSecurityError(code=exc.code, message=str(exc), status_code=400) from exc
        _audit_event(
            http_request,
            actor=actor,
            event_type="memory_delete",
            tool_call={"name": "memory_delete", "turn_id": deleted.get("turn_id")},
            policy_decision={"status": "deleted"},
        )
        return _attach_trace(
            http_request,
            {
                "deleted_memory": deleted,
                "memory": memory_store.describe(),
            },
        )

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
        try:
            if files:
                temp_dir = Path(tempfile.mkdtemp(prefix="dac3d-kb-", dir=assistant.config.upload_temp_dir))
                path_policy: PathPolicy = request.app.state.path_policy
                for file in files:
                    if not file.filename:
                        continue
                    try:
                        destination = path_policy.safe_upload_destination(temp_dir, file.filename)
                    except PathPolicyError as exc:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unsafe upload filename rejected by PathPolicy: {exc.code}.",
                        ) from exc
                    destination.write_bytes(await file.read())
                    uploaded_paths.append(destination)
            summary = assistant.build_knowledge_base_from_uploads(uploaded_paths)
        except HTTPException:
            raise
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

    @app.get("/api/audit/traces")
    def audit_traces(http_request: Request) -> dict[str, Any]:
        actor = _require_actor(
            http_request,
            require_operator=True,
            required_roles=PRIVILEGED_ROLES,
        )
        trace_id = _parse_optional_trace_id(http_request.query_params.get("trace_id"))
        return _audit_trace_query_response(http_request, actor=actor, trace_id=trace_id)

    @app.get("/api/audit/traces/{trace_id}")
    def audit_trace_by_id(http_request: Request, trace_id: str) -> dict[str, Any]:
        actor = _require_actor(
            http_request,
            require_operator=True,
            required_roles=PRIVILEGED_ROLES,
        )
        return _audit_trace_query_response(
            http_request,
            actor=actor,
            trace_id=_parse_required_trace_id(trace_id),
        )

    @app.get("/api/audit/export")
    def audit_trace_export(http_request: Request) -> dict[str, Any]:
        actor = _require_actor(
            http_request,
            require_operator=True,
            required_roles=PRIVILEGED_ROLES,
        )
        trace_id = _parse_optional_trace_id(http_request.query_params.get("trace_id"))
        return _audit_trace_query_response(
            http_request,
            actor=actor,
            trace_id=trace_id,
            export=True,
        )

    selected_frontend_dist = frontend_dist_dir or assistant.config.frontend_dist_dir

    if selected_frontend_dist.exists():
        assets_dir = selected_frontend_dist / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

        @app.get("/{static_path:path}", include_in_schema=False, response_model=None)
        async def frontend_static(static_path: str) -> HTMLResponse | FileResponse:
            candidate = selected_frontend_dist / static_path
            if static_path and candidate.exists() and candidate.is_file():
                return FileResponse(candidate)
            return HTMLResponse(selected_frontend_dist.joinpath("index.html").read_text(encoding="utf-8"))

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def frontend_index() -> HTMLResponse:
            return HTMLResponse(selected_frontend_dist.joinpath("index.html").read_text(encoding="utf-8"))

    else:
        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def frontend_missing() -> HTMLResponse:
            return HTMLResponse(_frontend_hint_html())

    _install_openapi_security_schema(app)
    return app


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


def _call_assistant_preview(
    assistant: Any,
    message: str,
    history: Sequence[tuple[str, str]],
    *,
    session_id: str,
) -> Any:
    """Call the deterministic command-preview backend under either chat wrapper."""
    core = _core_assistant(assistant)
    if hasattr(core, "preview_operation_command"):
        return core.preview_operation_command(message, history)
    runtime = getattr(assistant, "runtime", None)
    if runtime is not None and hasattr(runtime, "preview_command"):
        return runtime.preview_command(message, session_id=session_id)
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
    """Execute a stored preview through the backend safety path."""
    del session_id
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


def _ensure_memory_store(assistant: Any) -> ConversationMemoryStore | None:
    """Return the app memory store, creating one when memory is enabled."""
    memory_store = getattr(assistant, "memory_store", None)
    config = getattr(assistant, "config", None)
    if memory_store is None and getattr(config, "memory_enabled", False):
        memory_store = ConversationMemoryStore.from_config(config)
        memory_store.ensure_directories()
        if hasattr(assistant, "memory_store"):
            assistant.memory_store = memory_store
    return memory_store


def _require_memory_store(request: Any) -> ConversationMemoryStore:
    """Return the configured memory store or fail closed."""
    memory_store = getattr(request.app.state, "memory_store", None)
    if memory_store is None:
        raise ApiSecurityError(
            code="MEMORY_DISABLED",
            message="Conversation memory is disabled.",
            status_code=403,
        )
    return memory_store


def _require_audit_logger(request: Any) -> AuditTraceLogger:
    """Return the configured audit logger or fail closed."""
    audit_logger = getattr(request.app.state, "audit_logger", None)
    if audit_logger is None:
        raise ApiSecurityError(
            code="AUDIT_TRACE_DISABLED",
            message="Audit trace logging is disabled.",
            status_code=403,
        )
    return audit_logger


def _audit_trace_query_response(
    request: Any,
    *,
    actor: Any,
    trace_id: str | None,
    export: bool = False,
) -> dict[str, Any]:
    audit_logger = _require_audit_logger(request)
    limit, offset = _parse_trace_pagination(request.query_params)
    query_result = audit_logger.query_redacted_events(
        trace_id=trace_id,
        offset=offset,
        limit=limit,
    )
    verification = audit_logger.verify_hash_chain().to_dict()
    payload = query_result.to_dict()
    payload["verification"] = verification
    payload["export_format"] = "redacted-json" if export else "redacted-page"
    _audit_event(
        request,
        actor=actor,
        event_type="audit_trace_export" if export else "audit_trace_query",
        tool_call={
            "name": "audit_trace_export" if export else "audit_trace_query",
            "trace_id": trace_id,
            "offset": offset,
            "limit": limit,
        },
        policy_decision={
            "returned": payload["returned"],
            "total": payload["total"],
            "verification_valid": verification.get("valid"),
        },
    )
    return _attach_trace(
        request,
        {
            "audit_trace": payload,
            "actor": {
                "session_id": actor.session_id,
                "operator_id": actor.operator_id,
                "roles": list(actor.roles),
            },
        },
    )


def _parse_trace_pagination(query_params: Any) -> tuple[int, int]:
    limit = _parse_bounded_int(
        query_params.get("limit"),
        field_name="limit",
        default=DEFAULT_TRACE_QUERY_LIMIT,
        minimum=1,
        maximum=MAX_TRACE_QUERY_LIMIT,
    )
    offset = _parse_bounded_int(
        query_params.get("offset"),
        field_name="offset",
        default=0,
        minimum=0,
        maximum=1_000_000,
    )
    return limit, offset


def _parse_bounded_int(
    value: Any,
    *,
    field_name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = int(str(value).strip())
    except ValueError as exc:
        raise ApiSecurityError(
            code=f"INVALID_TRACE_{field_name.upper()}",
            message=f"trace {field_name} must be an integer.",
            status_code=422,
        ) from exc
    if parsed < minimum or parsed > maximum:
        raise ApiSecurityError(
            code=f"INVALID_TRACE_{field_name.upper()}",
            message=f"trace {field_name} must be between {minimum} and {maximum}.",
            status_code=422,
        )
    return parsed


def _parse_optional_trace_id(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return _parse_required_trace_id(value)


def _parse_required_trace_id(value: Any) -> str:
    text = str(value or "").strip()
    if not _TRACE_ID_RE.fullmatch(text):
        raise ApiSecurityError(
            code="INVALID_TRACE_ID",
            message="trace_id must contain only letters, digits, underscore, dot, colon, or hyphen.",
            status_code=422,
        )
    return text


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


def _install_openapi_security_schema(app: Any) -> None:
    """Expose DAC-3D API actor headers in Swagger/OpenAPI."""
    try:
        from fastapi.openapi.utils import get_openapi
    except Exception:  # pragma: no cover - optional FastAPI dependency guard
        return

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        components = schema.setdefault("components", {})
        security_schemes = components.setdefault("securitySchemes", {})
        security_schemes.update(_OPENAPI_SECURITY_SCHEMES)
        _apply_openapi_security_policies(schema)
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi


def _apply_openapi_security_policies(schema: dict[str, Any]) -> None:
    paths = schema.get("paths")
    if not isinstance(paths, dict):
        return
    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if not isinstance(operation, dict):
                continue
            policy = _OPENAPI_SECURITY_POLICIES.get((method.upper(), path))
            if not policy:
                continue
            schemes = tuple(str(scheme) for scheme in policy.get("schemes", ()))
            operation["security"] = [{scheme: [] for scheme in schemes}]
            operation["x-dac3d-security"] = {
                "level": policy.get("level"),
                "required_headers": [
                    _OPENAPI_SECURITY_SCHEMES[scheme]["name"]
                    for scheme in schemes
                    if scheme in _OPENAPI_SECURITY_SCHEMES
                ],
                "required_roles": list(policy.get("required_roles", ())),
            }


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
