"""Read-only Git workspace context for DAC-Agent Runtime."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any


BRANCH_PATTERN = re.compile(
    r"^## (?P<branch>[^.\s]+|HEAD)(?:\.\.\.(?P<upstream>[^\s]+))?(?: \[(?P<meta>[^\]]+)\])?"
)


def _clip(value: Any, limit: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _repo_root_for(base_dir: Path) -> Path:
    if (base_dir.parent / "dac3d_iim_assistant").exists():
        return base_dir.parent.resolve()
    return base_dir.resolve()


class GitWorkspaceContext:
    """Small read-only wrapper around git status/log/diff/worktree metadata."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    @classmethod
    def from_config_root(cls, base_dir: Path) -> "GitWorkspaceContext":
        return cls(_repo_root_for(base_dir))

    def snapshot(
        self,
        *,
        recent_limit: int = 5,
        include_diff_stat: bool = True,
        include_worktrees: bool = True,
    ) -> dict[str, Any]:
        root_result = self._git("rev-parse", "--show-toplevel")
        if root_result["returncode"] != 0:
            return {
                "enabled": False,
                "backend": "git_workspace_context",
                "repo_root": str(self.repo_root),
                "message": "Directory is not a git repository.",
            }
        actual_root = root_result["stdout"].strip() or str(self.repo_root)
        status_text = self._git("status", "--short", "--branch")["stdout"]
        branch_payload = self._parse_branch(status_text)
        changed_files = self._parse_changed_files(status_text)
        recent_commits = self._recent_commits(recent_limit)
        payload = {
            "enabled": True,
            "backend": "git_workspace_context",
            "repo_root": actual_root,
            "branch": branch_payload["branch"],
            "upstream": branch_payload["upstream"],
            "ahead": branch_payload["ahead"],
            "behind": branch_payload["behind"],
            "clean": not changed_files,
            "changed_file_count": len(changed_files),
            "changed_files": changed_files,
            "recent_commits": recent_commits,
            "workflow": "git_status -> branch_context -> diff_summary -> agent_workspace",
        }
        if include_diff_stat:
            payload["diff_stat"] = self._git("diff", "--stat")["stdout"].strip()
        if include_worktrees:
            payload["worktrees"] = self._worktrees()
        return payload

    def describe(self) -> dict[str, Any]:
        snapshot = self.snapshot(recent_limit=3, include_diff_stat=False, include_worktrees=False)
        if not snapshot.get("enabled"):
            return snapshot
        return {
            "enabled": True,
            "backend": "git_workspace_context",
            "repo_root": snapshot.get("repo_root"),
            "branch": snapshot.get("branch"),
            "upstream": snapshot.get("upstream"),
            "ahead": snapshot.get("ahead"),
            "behind": snapshot.get("behind"),
            "clean": snapshot.get("clean"),
            "changed_file_count": snapshot.get("changed_file_count"),
            "recent_commit_count": len(snapshot.get("recent_commits") or []),
            "workflow": "read_git_context -> workspace_summary -> planning_hint",
        }

    def _git(self, *args: str) -> dict[str, Any]:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.repo_root,
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"returncode": 1, "stdout": "", "stderr": str(exc)}
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def _parse_branch(self, status_text: str) -> dict[str, Any]:
        first_line = status_text.splitlines()[0] if status_text.splitlines() else ""
        match = BRANCH_PATTERN.match(first_line)
        if not match:
            return {"branch": "", "upstream": "", "ahead": 0, "behind": 0}
        meta = match.group("meta") or ""
        ahead = self._extract_count(meta, "ahead")
        behind = self._extract_count(meta, "behind")
        return {
            "branch": match.group("branch") or "",
            "upstream": match.group("upstream") or "",
            "ahead": ahead,
            "behind": behind,
        }

    def _extract_count(self, meta: str, label: str) -> int:
        match = re.search(rf"{label} (\d+)", meta)
        return int(match.group(1)) if match else 0

    def _parse_changed_files(self, status_text: str) -> list[dict[str, str]]:
        files: list[dict[str, str]] = []
        for line in status_text.splitlines()[1:]:
            if not line:
                continue
            status = line[:2]
            path = line[3:] if len(line) > 3 else ""
            files.append({"status": status.strip() or "?", "path": _clip(path, 300)})
        return files

    def _recent_commits(self, limit: int) -> list[dict[str, str]]:
        safe_limit = max(1, min(20, int(limit or 5)))
        result = self._git("log", f"--max-count={safe_limit}", "--pretty=format:%h%x09%s")
        if result["returncode"] != 0:
            return []
        commits = []
        for line in result["stdout"].splitlines():
            commit_hash, _, subject = line.partition("\t")
            if commit_hash:
                commits.append({"hash": commit_hash, "subject": _clip(subject, 200)})
        return commits

    def _worktrees(self) -> list[dict[str, str]]:
        result = self._git("worktree", "list", "--porcelain")
        if result["returncode"] != 0:
            return []
        worktrees: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in result["stdout"].splitlines():
            if not line:
                if current:
                    worktrees.append(current)
                    current = {}
                continue
            key, _, value = line.partition(" ")
            if key in {"worktree", "HEAD", "branch"}:
                current[key] = value
            elif key == "bare":
                current["bare"] = "true"
            elif key == "detached":
                current["detached"] = "true"
        if current:
            worktrees.append(current)
        return worktrees
