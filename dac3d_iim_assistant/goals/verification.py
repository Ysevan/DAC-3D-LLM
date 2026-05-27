"""Local verification feedback runner for DAC-Agent workspace."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VERIFICATION_STATUSES = {"passed", "failed", "timed_out", "unavailable"}


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(default)
    return payload if isinstance(payload, dict) else dict(default)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = _utc_now_iso()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _clip(value: Any, limit: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _tool_path(name: str) -> str | None:
    local_candidate = Path(sys.executable).with_name(name)
    if local_candidate.exists():
        return str(local_candidate)
    windows_candidate = Path(sys.executable).with_name(f"{name}.exe")
    if windows_candidate.exists():
        return str(windows_candidate)
    return shutil.which(name)


class VerificationRunnerStore:
    """Run and record small local verification presets."""

    def __init__(self, path: Path, base_dir: Path) -> None:
        self.path = path
        self.base_dir = base_dir

    @classmethod
    def from_root(cls, root_dir: Path, base_dir: Path) -> "VerificationRunnerStore":
        return cls(root_dir / "agent_verifications.json", base_dir)

    def presets(self) -> dict[str, Any]:
        pytest_path = _tool_path("pytest")
        ruff_path = _tool_path("ruff")
        presets = [
            {
                "id": "python_compile",
                "label": "Python compileall",
                "command": [sys.executable, "-m", "compileall", "."],
                "cwd": str(self.base_dir),
                "available": True,
                "category": "syntax",
            },
            {
                "id": "python_tests",
                "label": "Pytest suite",
                "command": [pytest_path or "pytest", "tests", "-q"],
                "cwd": str(self.base_dir),
                "available": pytest_path is not None,
                "category": "tests",
            },
            {
                "id": "ruff_check",
                "label": "Ruff lint",
                "command": [ruff_path or "ruff", "check", "."],
                "cwd": str(self.base_dir),
                "available": ruff_path is not None,
                "category": "lint",
            },
        ]
        return {
            "enabled": True,
            "backend": "local_verification_runner",
            "presets": presets,
            "count": len(presets),
            "workflow": "select_preset -> run_command -> record_feedback",
        }

    def run_preset(self, preset_id: str, *, timeout_seconds: int = 120) -> dict[str, Any]:
        preset = self._find_preset(preset_id)
        if preset is None:
            raise ValueError(f"Unknown verification preset: {preset_id}")
        safe_timeout = max(5, min(900, int(timeout_seconds or 120)))
        started_at = _utc_now_iso()
        started = time.monotonic()
        status = "failed"
        returncode: int | None = 1
        stdout = ""
        stderr = ""
        timed_out = False
        if not preset.get("available"):
            status = "unavailable"
            returncode = None
            stderr = f"Verification preset is unavailable: {preset_id}"
        else:
            try:
                result = subprocess.run(
                    list(preset["command"]),
                    cwd=str(preset["cwd"]),
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=safe_timeout,
                )
                returncode = result.returncode
                stdout = result.stdout
                stderr = result.stderr
                status = "passed" if returncode == 0 else "failed"
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                status = "timed_out"
                returncode = None
                stdout = str(exc.stdout or "")
                stderr = str(exc.stderr or "")
            except FileNotFoundError as exc:
                status = "unavailable"
                returncode = None
                stderr = str(exc)
        duration_ms = int((time.monotonic() - started) * 1000)
        run = {
            "id": f"verification-{uuid.uuid4().hex[:12]}",
            "preset_id": preset_id,
            "label": preset["label"],
            "category": preset["category"],
            "command": list(preset["command"]),
            "cwd": str(preset["cwd"]),
            "status": status,
            "returncode": returncode,
            "timed_out": timed_out,
            "timeout_seconds": safe_timeout,
            "duration_ms": duration_ms,
            "started_at": started_at,
            "completed_at": _utc_now_iso(),
            "stdout_tail": _clip(stdout, 2000),
            "stderr_tail": _clip(stderr, 2000),
        }
        payload = self._load()
        runs = self._runs(payload)
        runs.insert(0, run)
        payload["runs"] = runs[:100]
        _write_json(self.path, payload)
        return {"enabled": True, "run": run}

    def list_runs(
        self,
        *,
        status: str | None = None,
        preset_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = self._load()
        runs = self._runs(payload)
        if status:
            normalized_status = str(status or "").strip()
            if normalized_status not in VERIFICATION_STATUSES:
                raise ValueError(f"Unknown verification status: {status}")
            runs = [run for run in runs if str(run.get("status") or "") == normalized_status]
        if preset_id:
            runs = [run for run in runs if str(run.get("preset_id") or "") == preset_id]
        safe_limit = max(1, min(100, int(limit or 50)))
        return {
            "enabled": True,
            "backend": "local_verification_runner",
            "path": str(self.path),
            "runs": runs[:safe_limit],
            "count": len(runs[:safe_limit]),
            "total_count": len(self._runs(payload)),
            "presets": self.presets()["presets"],
            "workflow": "run_preset -> persist_feedback -> workspace_summary",
        }

    def describe(self) -> dict[str, Any]:
        runs = self._runs(self._load())
        by_status: dict[str, int] = {}
        for run in runs:
            status = str(run.get("status") or "unknown")
            by_status[status] = by_status.get(status, 0) + 1
        latest = runs[0] if runs else {}
        return {
            "enabled": True,
            "backend": "local_verification_runner",
            "path": str(self.path),
            "run_count": len(runs),
            "by_status": by_status,
            "latest_run": latest,
            "preset_count": self.presets()["count"],
            "workflow": "verification_preset -> command_result -> agent_feedback",
        }

    def _find_preset(self, preset_id: str) -> dict[str, Any] | None:
        for preset in self.presets()["presets"]:
            if str(preset.get("id") or "") == str(preset_id or ""):
                return preset
        return None

    def _load(self) -> dict[str, Any]:
        payload = _read_json(self.path, {"version": 1, "updated_at": "", "runs": []})
        if not isinstance(payload.get("runs"), list):
            payload["runs"] = []
        return payload

    def _runs(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [run for run in payload.get("runs", []) if isinstance(run, dict)]
