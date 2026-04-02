"""FastAPI application for the DAC-3D assistant web client."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any


def create_api_app(assistant: Any) -> Any:
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

    @app.get("/api/knowledge-base/summary")
    def knowledge_base_summary() -> dict[str, Any]:
        return assistant.knowledge_base_summary()

    @app.post("/api/chat")
    def chat(request: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            message, history = _parse_chat_request(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        response = assistant.handle_message(
            message,
            history,
        )
        return response.to_ui_payload()

    @app.post("/api/chat/stream")
    def chat_stream(request: dict[str, Any] = Body(...)) -> StreamingResponse:
        try:
            message, history = _parse_chat_request(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return StreamingResponse(
            _stream_chat_events(assistant, message, history),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/knowledge-base/build")
    async def build_knowledge_base(files: list[UploadFile] | None = File(default=None)) -> dict[str, Any]:
        uploaded_paths: list[Path] = []
        temp_dir: Path | None = None
        if files:
            temp_dir = Path(tempfile.mkdtemp(prefix="dac3d-kb-", dir=assistant.config.base_dir))
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

    if assistant.config.frontend_dist_dir.exists():
        assets_dir = assistant.config.frontend_dist_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

        @app.get("/{static_path:path}")
        async def frontend_static(static_path: str) -> HTMLResponse | FileResponse:
            candidate = assistant.config.frontend_dist_dir / static_path
            if static_path and candidate.exists() and candidate.is_file():
                return FileResponse(candidate)
            return HTMLResponse(assistant.config.frontend_dist_dir.joinpath("index.html").read_text(encoding="utf-8"))

        @app.get("/", response_class=HTMLResponse)
        async def frontend_index() -> HTMLResponse:
            return HTMLResponse(assistant.config.frontend_dist_dir.joinpath("index.html").read_text(encoding="utf-8"))

    else:
        @app.get("/", response_class=HTMLResponse)
        async def frontend_missing() -> HTMLResponse:
            return HTMLResponse(_frontend_hint_html())

    return app


def _stream_chat_events(
    assistant: Any,
    message: str,
    history: Sequence[tuple[str, str]],
) -> Iterator[str]:
    """Yield SSE events for a single chat request."""
    try:
        for event_name, payload in assistant.stream_message(message, history):
            yield _sse_event(event_name, payload)
    except Exception as exc:  # pragma: no cover - defensive streaming guard
        yield _sse_event("error", {"message": str(exc)})


def _parse_chat_request(payload: dict[str, Any]) -> tuple[str, list[tuple[str, str]]]:
    """Normalize the chat request body into the assistant's internal schema."""
    message = str(payload.get("message", "")).strip()
    if not message:
        raise ValueError("The `message` field is required.")

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

    return message, normalized_history


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
