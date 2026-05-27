"""Security middleware and structured API errors for FastAPI."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from ui.auth import ApiSecurityError, REQUEST_ID_HEADER, SESSION_HEADER


class InMemoryRateLimiter:
    """Small per-process rate-limit skeleton keyed by session or client host."""

    def __init__(self, *, limit: int = 120, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        """Return whether a request is within the current window."""
        now = time.time()
        events = self._events[key]
        while events and events[0] <= now - self.window_seconds:
            events.popleft()
        if len(events) >= self.limit:
            return False
        events.append(now)
        return True


def install_security_middleware(app: Any, *, rate_limit: int = 120) -> None:
    """Install request id, trace id, rate-limit, and structured error handlers."""
    try:
        from fastapi import HTTPException, Request
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse
    except Exception as exc:  # pragma: no cover - FastAPI optional dependency
        raise RuntimeError("FastAPI dependencies are required for API security.") from exc

    limiter = InMemoryRateLimiter(limit=rate_limit)

    @app.middleware("http")
    async def security_context_middleware(request: Request, call_next: Callable[..., Any]) -> Any:
        request_id = _safe_header(request.headers.get(REQUEST_ID_HEADER)) or uuid.uuid4().hex
        trace_id = uuid.uuid4().hex
        request.state.request_id = request_id
        request.state.trace_id = trace_id

        rate_key = (
            _safe_header(request.headers.get(SESSION_HEADER))
            or (request.client.host if request.client else "unknown")
        )
        if request.url.path.startswith("/api/") and not limiter.allow(rate_key):
            return structured_error_response(
                status_code=429,
                code="RATE_LIMITED",
                message="Too many requests. Please retry later.",
                request_id=request_id,
                trace_id=trace_id,
            )

        try:
            response = await call_next(request)
        except ApiSecurityError as exc:
            response = structured_error_response(
                status_code=exc.status_code,
                code=exc.code,
                message=exc.message,
                request_id=request_id,
                trace_id=trace_id,
            )
        except Exception:
            response = structured_error_response(
                status_code=500,
                code="INTERNAL_ERROR",
                message="Request handling failed.",
                request_id=request_id,
                trace_id=trace_id,
            )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id
        return response

    @app.exception_handler(ApiSecurityError)
    async def api_security_exception_handler(request: Request, exc: ApiSecurityError) -> JSONResponse:
        return structured_error_response(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            request_id=getattr(request.state, "request_id", None),
            trace_id=getattr(request.state, "trace_id", None),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return structured_error_response(
            status_code=exc.status_code,
            code="HTTP_ERROR",
            message=detail,
            request_id=getattr(request.state, "request_id", None),
            trace_id=getattr(request.state, "trace_id", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        del exc
        return structured_error_response(
            status_code=422,
            code="VALIDATION_ERROR",
            message="Request validation failed.",
            request_id=getattr(request.state, "request_id", None),
            trace_id=getattr(request.state, "trace_id", None),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        del exc
        return structured_error_response(
            status_code=500,
            code="INTERNAL_ERROR",
            message="Request handling failed.",
            request_id=getattr(request.state, "request_id", None),
            trace_id=getattr(request.state, "trace_id", None),
        )


def structured_error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str | None,
    trace_id: str | None,
) -> Any:
    """Build a consistent JSON error payload."""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
                "trace_id": trace_id,
            }
        },
    )


def _safe_header(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:120]
