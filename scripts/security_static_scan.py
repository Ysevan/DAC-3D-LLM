#!/usr/bin/env python3
"""Lightweight supply-chain and unsafe-code scanner for local/CI checks."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXCLUDES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".tmp",
    "__pycache__",
    "dist",
    "node_modules",
}
SKIPPED_SUFFIXES = {".bin", ".db", ".docx", ".jpg", ".jpeg", ".png", ".pyc", ".sqlite3", ".weights"}
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("api-key", re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}")),
    ("bearer-token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("openai-style-key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("connection-string", re.compile(r"(?i)\b(?:postgresql|postgres|mysql|mongodb|redis)://[^@\s]+:[^@\s]+@")),
)
UNSAFE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("python-eval", re.compile(r"\beval\s*\(")),
    ("python-exec", re.compile(r"\bexec\s*\(")),
    ("subprocess-shell-true", re.compile(r"\bshell\s*=\s*True\b")),
    ("pickle-load", re.compile(r"\bpickle\.(?:load|loads)\s*\(")),
    ("unsafe-yaml-load", re.compile(r"\byaml\.load\s*\((?![^)]*SafeLoader)")),
    ("react-dangerous-html", re.compile(r"dangerouslySetInnerHTML")),
    ("javascript-eval", re.compile(r"\beval\s*\(")),
    ("javascript-new-function", re.compile(r"\bnew\s+Function\s*\(")),
)


@dataclass(slots=True)
class Finding:
    rule: str
    path: Path
    line: int
    snippet: str

    def render(self, root: Path) -> str:
        rel = self.path.relative_to(root) if self.path.is_relative_to(root) else self.path
        return f"{rel}:{self.line}: {self.rule}: {self.snippet.strip()}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan for obvious secrets and unsafe code patterns.")
    parser.add_argument("paths", nargs="*", default=["dac3d_iim_assistant"], help="Paths to scan.")
    args = parser.parse_args()

    root = Path.cwd()
    findings: list[Finding] = []
    for raw_path in args.paths:
        path = Path(raw_path)
        if path.exists():
            findings.extend(scan_path(path))

    if findings:
        for finding in findings:
            print(finding.render(root))
        return 1

    print("security static scan passed")
    return 0


def scan_path(path: Path) -> list[Finding]:
    if path.is_file():
        return scan_file(path)
    findings: list[Finding] = []
    for candidate in path.rglob("*"):
        if candidate.is_file() and should_scan(candidate):
            findings.extend(scan_file(candidate))
    return findings


def should_scan(path: Path) -> bool:
    parts = set(path.parts)
    if parts & DEFAULT_EXCLUDES:
        return False
    if "tests" in parts:
        return False
    if "knowledge_base" in parts and "vector_store" in parts:
        return False
    if path.suffix.lower() in SKIPPED_SUFFIXES:
        return False
    return path.suffix.lower() in {".js", ".jsx", ".md", ".py", ".sh", ".ts", ".tsx", ".toml", ".yaml", ".yml"}


def scan_file(path: Path) -> list[Finding]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return []
    findings: list[Finding] = []
    for line_number, line in enumerate(lines, start=1):
        for rule, pattern in SECRET_PATTERNS + UNSAFE_PATTERNS:
            if pattern.search(line) and not is_allowed_match(path, line):
                findings.append(Finding(rule=rule, path=path, line=line_number, snippet=line[:180]))
    return findings


def is_allowed_match(path: Path, line: str) -> bool:
    if path.name == "security_static_scan.py":
        return True
    if "secrets.py" in str(path) or "redaction.py" in str(path):
        return True
    if any(
        marker in line
        for marker in (
            "self.config.",
            "os.getenv(",
            "secrets.token_urlsafe(",
            "compare_digest(",
            "lifecycle_token(",
        )
    ):
        return True
    return "pragma: allow-security-scan" in line


if __name__ == "__main__":
    sys.exit(main())
