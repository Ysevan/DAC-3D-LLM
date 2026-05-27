"""Static repository context map for DAC-Agent Runtime."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
IGNORED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tmp",
    "__pycache__",
    "build",
    "chroma",
    "dist",
    "node_modules",
    "vector_store",
}
SYMBOL_PATTERN = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
ROUTE_PATTERN = re.compile(r"@app\.(get|post|put|patch|delete)\(\"([^\"]+)\"")


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clip(value: Any, limit: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


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


def _module_for(relative_path: str) -> str:
    parts = relative_path.split("/")
    if len(parts) <= 1:
        return "."
    return parts[0]


def _role_for(relative_path: str, suffix: str) -> str:
    path = relative_path.lower()
    if "/tests/" in f"/{path}" or path.startswith("tests/"):
        return "test"
    if path.endswith(".md") or "/docs/" in f"/{path}":
        return "documentation"
    if path.endswith((".tsx", ".ts", ".css", ".html")) or "/frontend/" in f"/{path}" or "/ui2/" in f"/{path}":
        return "frontend"
    if path.endswith("web_api.py") or "/api" in path:
        return "api"
    if "/goals/" in f"/{path}" or "/memory/" in f"/{path}" or "/context_engineering/" in f"/{path}":
        return "agent_runtime"
    if suffix == ".py":
        return "python"
    return "asset"


class RepoContextMapStore:
    """File-backed project map used by Agent workspace and context tooling."""

    def __init__(self, repo_root: Path, index_path: Path) -> None:
        self.repo_root = repo_root
        self.index_path = index_path

    @classmethod
    def from_config_root(cls, base_dir: Path, memory_dir: Path) -> "RepoContextMapStore":
        repo_root = base_dir.parent if (base_dir.parent / "dac3d_iim_assistant").exists() else base_dir
        return cls(repo_root.resolve(), memory_dir / "repo_context_map.json")

    def build_map(self, *, max_files: int = 1200) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        modules: dict[str, dict[str, Any]] = {}
        extension_counts: Counter[str] = Counter()
        role_counts: Counter[str] = Counter()
        api_routes: list[dict[str, str]] = []
        skipped = 0
        safe_limit = max(50, min(5000, int(max_files or 1200)))

        for path in self._iter_files():
            if len(files) >= safe_limit:
                skipped += 1
                continue
            try:
                stat = path.stat()
            except OSError:
                skipped += 1
                continue
            relative_path = path.relative_to(self.repo_root).as_posix()
            suffix = path.suffix.lower()
            module = _module_for(relative_path)
            role = _role_for(relative_path, suffix)
            extension_counts[suffix or "<none>"] += 1
            role_counts[role] += 1
            module_entry = modules.setdefault(
                module,
                {
                    "name": module,
                    "file_count": 0,
                    "roles": {},
                    "extensions": {},
                    "sample_files": [],
                },
            )
            module_entry["file_count"] += 1
            module_entry["roles"][role] = int(module_entry["roles"].get(role, 0)) + 1
            module_entry["extensions"][suffix or "<none>"] = int(
                module_entry["extensions"].get(suffix or "<none>", 0)
            ) + 1
            if len(module_entry["sample_files"]) < 8:
                module_entry["sample_files"].append(relative_path)

            symbols = self._symbols_for(path) if suffix == ".py" and stat.st_size <= 200_000 else []
            routes = self._routes_for(path) if relative_path.endswith("web_api.py") else []
            api_routes.extend(routes)
            files.append(
                {
                    "path": relative_path,
                    "module": module,
                    "role": role,
                    "extension": suffix,
                    "size_bytes": stat.st_size,
                    "symbols": symbols[:20],
                    "api_routes": routes,
                }
            )

        payload = {
            "version": 1,
            "updated_at": _utc_now_iso(),
            "backend": "static_repo_context_map",
            "repo_root": str(self.repo_root),
            "file_count": len(files),
            "skipped_count": skipped,
            "max_files": safe_limit,
            "modules": sorted(modules.values(), key=lambda item: str(item["name"])),
            "files": files,
            "extension_counts": dict(extension_counts),
            "role_counts": dict(role_counts),
            "api_routes": api_routes[:200],
            "workflow": "scan_repo -> summarize_modules -> search_context -> agent_workspace",
        }
        _write_json(self.index_path, payload)
        return {"enabled": True, **payload}

    def read_map(self) -> dict[str, Any]:
        payload = _read_json(
            self.index_path,
            {
                "version": 1,
                "updated_at": "",
                "backend": "static_repo_context_map",
                "repo_root": str(self.repo_root),
                "file_count": 0,
                "modules": [],
                "files": [],
                "api_routes": [],
            },
        )
        return {"enabled": True, **payload, "path": str(self.index_path)}

    def search(self, query: str, *, limit: int = 20) -> dict[str, Any]:
        payload = self.read_map()
        if not payload.get("updated_at"):
            payload = self.build_map()
        needle = str(query or "").strip().lower()
        if not needle:
            raise ValueError("Search query is required.")
        matches = []
        for file_entry in payload.get("files", []):
            if not isinstance(file_entry, dict):
                continue
            haystack = " ".join(
                [
                    str(file_entry.get("path") or ""),
                    str(file_entry.get("module") or ""),
                    str(file_entry.get("role") or ""),
                    " ".join(str(item) for item in file_entry.get("symbols", [])),
                    json.dumps(file_entry.get("api_routes") or [], ensure_ascii=False),
                ]
            ).lower()
            if needle in haystack:
                matches.append(file_entry)
        safe_limit = max(1, min(100, int(limit or 20)))
        return {
            "enabled": True,
            "backend": "static_repo_context_map",
            "query": query,
            "matches": matches[:safe_limit],
            "count": len(matches[:safe_limit]),
            "total_count": len(matches),
        }

    def describe(self) -> dict[str, Any]:
        payload = self.read_map()
        return {
            "enabled": True,
            "backend": "static_repo_context_map",
            "path": str(self.index_path),
            "repo_root": str(self.repo_root),
            "file_count": int(payload.get("file_count") or 0),
            "module_count": len(payload.get("modules") or []),
            "api_route_count": len(payload.get("api_routes") or []),
            "updated_at": str(payload.get("updated_at") or ""),
            "workflow": "build_repo_map -> search_repo_map -> context_builder_hint",
        }

    def _iter_files(self) -> list[Path]:
        result: list[Path] = []
        for path in self.repo_root.rglob("*"):
            if not path.is_file():
                continue
            relative_parts = path.relative_to(self.repo_root).parts
            if any(part in IGNORED_DIRS for part in relative_parts):
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            result.append(path)
        return sorted(result, key=lambda item: item.relative_to(self.repo_root).as_posix())

    def _symbols_for(self, path: Path) -> list[str]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:80_000]
        except OSError:
            return []
        return [_clip(match, 120) for match in SYMBOL_PATTERN.findall(text)[:30]]

    def _routes_for(self, path: Path) -> list[dict[str, str]]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:120_000]
        except OSError:
            return []
        return [
            {"method": method.upper(), "path": route}
            for method, route in ROUTE_PATTERN.findall(text)
        ]
