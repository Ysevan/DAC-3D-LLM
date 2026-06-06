"""Filesystem path policy for DAC-Agent Runtime.

The LLM only supplies path-like data. This module decides whether that data can
be used for DAC reads or command-bridge writes.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Iterable
from urllib.parse import unquote, urlparse


WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
SECRET_NAMES = {
    ".env",
    ".netrc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "known_hosts",
}
SECRET_MARKERS = ("secret", "api_key", "apikey", "password", "passwd", "token", "credential", "credentials")
CONFIG_NAMES = {
    ".env",
    "config.py",
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "vite.config.ts",
    "tsconfig.json",
    "AGENTS.md",
}


@dataclass(slots=True)
class PathPolicyDecision:
    """Structured decision for one filesystem path."""

    path: str
    allowed: bool
    reason: str
    matched_root: str | None = None
    operation: str = "read"
    raw_path: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _NormalizedPath:
    display: str
    comparable: str
    flavor: str
    is_network: bool


class PathPolicy:
    """Canonicalize and validate DAC filesystem paths."""

    def __init__(
        self,
        *,
        allowed_read_roots: Iterable[str | Path] | None = None,
        allowed_write_roots: Iterable[str | Path] | None = None,
    ) -> None:
        self._read_roots = [
            self._normalize_path(root)
            for root in (allowed_read_roots or [])
            if str(root or "").strip()
        ]
        self._write_roots = [
            self._normalize_path(root)
            for root in (allowed_write_roots or [])
            if str(root or "").strip()
        ]

    @property
    def allowed_read_roots(self) -> list[str]:
        """Return normalized read roots for diagnostics."""
        return [root.display for root in self._read_roots]

    @property
    def allowed_write_roots(self) -> list[str]:
        """Return normalized command-output roots for diagnostics."""
        return [root.display for root in self._write_roots]

    def normalize(self, path: str | Path) -> str:
        """Return a canonical, local representation of a path-like value."""
        return self._normalize_path(path).display

    def is_allowed_read(self, path: str | Path) -> bool:
        """Return whether a path can be read as DAC input data."""
        return self.validate_input_dir(path).allowed

    def is_allowed_write(self, path: str | Path) -> bool:
        """Return whether a path can be written as DAC command output."""
        return self.validate_command_output(path).allowed

    def validate_input_dir(self, path: str | Path) -> PathPolicyDecision:
        """Validate an input directory or result path against read allowlists."""
        return self._validate(path, roots=self._read_roots, operation="read")

    def validate_command_output(self, path: str | Path) -> PathPolicyDecision:
        """Validate a command output path against write allowlists."""
        return self._validate(
            path,
            roots=self._write_roots,
            operation="write",
            reject_config=True,
        )

    def validate_delete(self, path: str | Path) -> PathPolicyDecision:
        """DAC-Agent never allows filesystem deletion through LLM/tool paths."""
        normalized = self._safe_normalize(path)
        return PathPolicyDecision(
            path=normalized,
            allowed=False,
            reason="Deleting files is forbidden for DAC-Agent tools.",
            operation="delete",
            raw_path=str(path or ""),
        )

    def _validate(
        self,
        path: str | Path,
        *,
        roots: list[_NormalizedPath],
        operation: str,
        reject_config: bool = False,
    ) -> PathPolicyDecision:
        raw_path = str(path or "").strip()
        if not raw_path:
            return PathPolicyDecision(raw_path, False, "Path is empty.", operation=operation, raw_path=raw_path)

        candidate = self._normalize_path(raw_path)
        if self._has_parent_reference(raw_path):
            return PathPolicyDecision(
                candidate.display,
                False,
                "Path traversal using '..' is forbidden.",
                operation=operation,
                raw_path=raw_path,
            )
        if candidate.is_network and not any(root.is_network for root in roots):
            return PathPolicyDecision(
                candidate.display,
                False,
                "Network/UNC paths are forbidden unless explicitly allowlisted.",
                operation=operation,
                raw_path=raw_path,
            )
        if self._is_secret_path(candidate.display):
            return PathPolicyDecision(
                candidate.display,
                False,
                "Secret-bearing paths cannot be read or written by DAC-Agent tools.",
                operation=operation,
                raw_path=raw_path,
            )
        if reject_config and self._is_config_path(candidate.display):
            return PathPolicyDecision(
                candidate.display,
                False,
                "Command output cannot overwrite configuration or project files.",
                operation=operation,
                raw_path=raw_path,
            )
        if not roots:
            return PathPolicyDecision(
                candidate.display,
                False,
                "No allowed directories are configured for this path operation.",
                operation=operation,
                raw_path=raw_path,
            )

        for root in roots:
            if self._contains(root, candidate):
                return PathPolicyDecision(
                    candidate.display,
                    True,
                    "Path is inside an allowed DAC directory.",
                    matched_root=root.display,
                    operation=operation,
                    raw_path=raw_path,
                )
        return PathPolicyDecision(
            candidate.display,
            False,
            "Path is outside allowed DAC directories.",
            operation=operation,
            raw_path=raw_path,
        )

    def _safe_normalize(self, path: str | Path) -> str:
        try:
            return self._normalize_path(path).display
        except Exception:
            return str(path or "")

    def _normalize_path(self, path: str | Path) -> _NormalizedPath:
        raw = self._strip_file_url(str(path or "").strip())
        is_network = self._is_network_path(raw)
        if self._is_windows_path(raw) or is_network:
            normalized = self._normalize_windows(raw)
            return _NormalizedPath(
                display=normalized,
                comparable=self._comparable_windows(normalized),
                flavor="windows",
                is_network=is_network,
            )
        resolved = Path(raw).expanduser().resolve(strict=False)
        display = str(resolved)
        return _NormalizedPath(
            display=display,
            comparable=display,
            flavor="posix",
            is_network=False,
        )

    def _strip_file_url(self, value: str) -> str:
        if not value.lower().startswith("file://"):
            return value
        parsed = urlparse(value)
        path_text = unquote(parsed.path or "")
        if parsed.netloc:
            return f"//{parsed.netloc}{path_text}"
        if path_text.startswith("/") and len(path_text) >= 4 and path_text[2] == ":":
            return path_text[1:]
        return path_text

    def _normalize_windows(self, value: str) -> str:
        normalized = value.replace("/", "\\")
        parts: list[str] = []
        for part in PureWindowsPath(normalized).parts:
            if part in ("", "."):
                continue
            if part == "..":
                parts.append(part)
                continue
            parts.append(part)
        if not parts:
            return normalized
        base = parts[0]
        if base.endswith("\\"):
            return str(PureWindowsPath(base, *parts[1:]))
        return str(PureWindowsPath(*parts))

    def _comparable_windows(self, value: str) -> str:
        return value.rstrip("\\/").casefold()

    def _contains(self, root: _NormalizedPath, candidate: _NormalizedPath) -> bool:
        if root.flavor != candidate.flavor:
            return False
        root_value = root.comparable.rstrip("\\/")
        candidate_value = candidate.comparable.rstrip("\\/")
        if candidate_value == root_value:
            return True
        separator = "\\" if root.flavor == "windows" else "/"
        return candidate_value.startswith(root_value + separator)

    def _has_parent_reference(self, value: str) -> bool:
        path_value = self._strip_file_url(value)
        return any(part == ".." for part in re.split(r"[\\/]+", path_value))

    def _is_windows_path(self, value: str) -> bool:
        return bool(WINDOWS_DRIVE_RE.match(value)) or "\\" in value

    def _is_network_path(self, value: str) -> bool:
        return value.startswith("\\\\") or value.startswith("//")

    def _is_secret_path(self, value: str) -> bool:
        parts = [part.casefold() for part in re.split(r"[\\/]+", value) if part]
        if any(part in SECRET_NAMES for part in parts):
            return True
        leaf = parts[-1] if parts else ""
        return any(marker in leaf for marker in SECRET_MARKERS)

    def _is_config_path(self, value: str) -> bool:
        parts = [part for part in re.split(r"[\\/]+", value) if part]
        lowered = [part.casefold() for part in parts]
        if any(part in {".git", ".ssh", "__pycache__"} for part in lowered):
            return True
        leaf = parts[-1] if parts else ""
        return leaf in CONFIG_NAMES or leaf.casefold() in {name.casefold() for name in CONFIG_NAMES}
