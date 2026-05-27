"""Security tests for the FastAPI API boundary."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import DAC3DAssistant
from tests.test_assistant import make_config
from ui.session import ConfirmationTokenStore
from ui.web_api import create_api_app


def _headers(
    *,
    session_id: str = "security-session",
    operator_id: str = "operator-1",
    roles: str = "operator",
) -> dict[str, str]:
    return {
        "X-DAC3D-Session-ID": session_id,
        "X-DAC3D-Operator-ID": operator_id,
        "X-DAC3D-Roles": roles,
    }


def _preview(client: TestClient, headers: dict[str, str]) -> dict[str, object]:
    response = client.post(
        "/api/commands/preview",
        json={"message": "start online scan"},
        headers=headers,
    )
    assert response.status_code == 200
    return response.json()


def test_missing_session_rejected_for_write(tmp_path) -> None:
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.post(
        "/api/commands/confirm",
        json={"preview_id": "missing", "preview_hash": "missing", "confirmation_token": "missing"},
        headers={"X-DAC3D-Operator-ID": "operator-1", "X-DAC3D-Roles": "operator"},
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "MISSING_SESSION_ID"


def test_wrong_operator_rejected_for_confirmation(tmp_path) -> None:
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))
    preview = _preview(client, _headers(operator_id="operator-1"))
    confirmation = preview["confirmation"]

    response = client.post(
        "/api/commands/confirm",
        json={
            "preview_id": confirmation["preview_id"],
            "preview_hash": confirmation["preview_hash"],
            "confirmation_token": confirmation["confirmation_token"],
        },
        headers=_headers(operator_id="operator-2"),
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "OPERATOR_ID_MISMATCH"


def test_confirmation_token_replay_rejected(tmp_path) -> None:
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))
    headers = _headers()
    preview = _preview(client, headers)
    confirmation = preview["confirmation"]
    confirm_payload = {
        "preview_id": confirmation["preview_id"],
        "preview_hash": confirmation["preview_hash"],
        "confirmation_token": confirmation["confirmation_token"],
    }

    first = client.post("/api/commands/confirm", json=confirm_payload, headers=headers)
    replay = client.post("/api/commands/confirm", json=confirm_payload, headers=headers)

    assert first.status_code == 200
    assert replay.status_code == 403
    assert replay.json()["error"]["code"] == "CONFIRMATION_TOKEN_REPLAYED"


def test_expired_confirmation_token_rejected(tmp_path) -> None:
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    app = create_api_app(assistant)
    app.state.command_confirmations = ConfirmationTokenStore(ttl_seconds=-1)
    client = TestClient(app)
    headers = _headers()
    preview = _preview(client, headers)
    confirmation = preview["confirmation"]

    response = client.post(
        "/api/commands/confirm",
        json={
            "preview_id": confirmation["preview_id"],
            "preview_hash": confirmation["preview_hash"],
            "confirmation_token": confirmation["confirmation_token"],
        },
        headers=headers,
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "CONFIRMATION_TOKEN_EXPIRED"


def test_preview_hash_mismatch_rejected(tmp_path) -> None:
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))
    headers = _headers()
    preview = _preview(client, headers)
    confirmation = preview["confirmation"]

    response = client.post(
        "/api/commands/confirm",
        json={
            "preview_id": confirmation["preview_id"],
            "preview_hash": "0" * 64,
            "confirmation_token": confirmation["confirmation_token"],
        },
        headers=headers,
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "PREVIEW_HASH_MISMATCH"


def test_memory_approve_requires_privileged_role(tmp_path) -> None:
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.post(
        "/api/memory/approve",
        json={"memory_id": "m1"},
        headers=_headers(roles="operator"),
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "PRIVILEGED_ROLE_REQUIRED"


def test_skill_patch_apply_requires_privileged_role(tmp_path) -> None:
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.post(
        "/api/skill-patches/apply",
        json={"patch_id": "p1"},
        headers=_headers(roles="operator"),
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "PRIVILEGED_ROLE_REQUIRED"


def test_knowledge_base_upload_rejects_secret_filename(tmp_path) -> None:
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)

    def fake_build(uploaded_paths: object) -> dict[str, object]:
        del uploaded_paths
        return {"uploaded": 0}

    assistant.build_knowledge_base_from_uploads = fake_build
    client = TestClient(create_api_app(assistant))

    response = client.post(
        "/api/knowledge-base/build",
        files={"files": (".env", b"DAC3D_LLM_API_KEY=secret-value", "text/plain")},
        headers=_headers(roles="operator"),
    )

    assert response.status_code == 400
    assert "PathPolicy" in response.json()["error"]["message"]


def test_stack_trace_not_leaked(tmp_path) -> None:
    assistant = DAC3DAssistant.create(make_config(tmp_path), rebuild_kb=True)

    def boom() -> dict[str, object]:
        raise RuntimeError("super-secret stack trace value")

    assistant.runtime_summary = boom
    client = TestClient(create_api_app(assistant), raise_server_exceptions=False)

    response = client.get("/api/runtime", headers={"X-DAC3D-Session-ID": "security-session"})

    assert response.status_code == 500
    body = response.text
    assert "super-secret" not in body
    assert "Traceback" not in body
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"


def test_production_cors_not_wildcard(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DAC3D_ENV", "prod")
    monkeypatch.setenv("DAC3D_ALLOW_COMMAND_SUBMIT", "false")
    monkeypatch.setenv("DAC3D_CORS_ALLOWED_ORIGINS", "https://dac3d.example")
    monkeypatch.setenv("DAC3D_ALLOWED_INPUT_DIRS", str(tmp_path))
    monkeypatch.setenv("DAC3D_ALLOWED_COMMAND_OUTPUT_DIR", str(tmp_path / "commands"))
    config = make_config(tmp_path)
    config.frontend_dev_url = "*"
    assistant = DAC3DAssistant.create(config, rebuild_kb=True)
    client = TestClient(create_api_app(assistant))

    response = client.options(
        "/api/chat",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers.get("access-control-allow-origin") != "*"
