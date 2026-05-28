"""Static frontend security checks for the React Agent UI."""

from __future__ import annotations

from pathlib import Path


UI2_SRC = Path(__file__).resolve().parents[1] / "ui2" / "src"


def _read(name: str) -> str:
    return (UI2_SRC / name).read_text(encoding="utf-8")


def test_malicious_html_not_executed() -> None:
    app_source = _read("App.tsx")

    assert "dangerouslySetInnerHTML" not in app_source
    assert ".innerHTML" not in app_source
    assert "eval(" not in app_source
    assert "new Function" not in app_source


def test_command_preview_requires_confirm() -> None:
    app_source = _read("App.tsx")
    api_source = _read("api.ts")

    assert "previewCommand" in api_source
    assert "confirmCommand" in api_source
    assert "window.confirm" not in app_source
    assert "CommandApprovalDialog" in app_source
    assert "command-approval-phrase" in app_source
    assert "typedPhrase.trim() === request.phrase" in app_source
    assert "确认执行本次命令" in app_source
    assert "不会授权未来命令" in app_source


def test_blocked_action_displays_reason() -> None:
    api_source = _read("api.ts")

    assert "extractApiErrorMessage" in api_source
    assert "payload.error.code" in api_source
    assert "payload.error.trace_id" in api_source


def test_trace_id_visible() -> None:
    app_source = _read("App.tsx")

    assert "trace_id" in app_source
    assert "trace-chip" in app_source


def test_high_risk_warning_visible() -> None:
    app_source = _read("App.tsx")
    style_source = _read("styles.css")

    assert "high-risk-command" in app_source
    assert "command-warning-list" in app_source
    assert ".high-risk-command" in style_source
