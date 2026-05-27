"""Centralized filesystem path policy for local DAC-3D runtime boundaries."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath


class PathPolicyError(ValueError):
    """Raised when a path violates the DAC-3D local filesystem policy."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class PathPolicy:
    """Validate local paths before runtime reads or writes touch the filesystem."""

    allowed_input_dirs: tuple[Path | str, ...] = ()
    allowed_command_output_dir: Path | str | None = None
    allowed_command_filenames: frozenset[str] = field(
        default_factory=lambda: frozenset({"dac3d_assistant_command.json"})
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_input_dirs",
            tuple(
                self._resolve_path(Path(raw), purpose="allowed_input_dir")
                for raw in self.allowed_input_dirs
                if str(raw).strip()
            ),
        )
        output_dir = self.allowed_command_output_dir
        object.__setattr__(
            self,
            "allowed_command_output_dir",
            (
                self._resolve_path(Path(output_dir), purpose="allowed_command_output_dir")
                if output_dir is not None and str(output_dir).strip()
                else None
            ),
        )

    @classmethod
    def from_env(cls, *, base_dir: Path | None = None) -> "PathPolicy":
        """Build a policy from DAC-3D path-related environment variables."""
        allowed_input_dirs = _read_csv("DAC3D_ALLOWED_INPUT_DIRS")
        output_dir = os.getenv("DAC3D_ALLOWED_COMMAND_OUTPUT_DIR")
        if output_dir is None and base_dir is not None and _is_prod_env():
            output_dir = str(base_dir / ".tmp" / "commands")
        return cls(
            allowed_input_dirs=tuple(Path(value) for value in allowed_input_dirs),
            allowed_command_output_dir=Path(output_dir) if output_dir else None,
        )

    @classmethod
    def from_runtime_config(
        cls,
        *,
        allowed_input_dirs: tuple[str, ...],
        allowed_command_output_dir: str,
    ) -> "PathPolicy":
        """Build a policy from already parsed runtime security configuration."""
        return cls(
            allowed_input_dirs=tuple(Path(value) for value in allowed_input_dirs),
            allowed_command_output_dir=Path(allowed_command_output_dir)
            if allowed_command_output_dir
            else None,
        )

    def reject_unsafe_path_syntax(self, raw_path: str | Path, *, purpose: str = "path") -> None:
        """Reject traversal or NUL bytes before doing filesystem operations."""
        text = str(raw_path)
        if "\x00" in text:
            raise PathPolicyError("PATH_NUL_BYTE", f"{purpose} contains a NUL byte.")
        if _contains_parent_reference(text):
            raise PathPolicyError("PATH_TRAVERSAL", f"{purpose} must not contain '..'.")

    def validate_existing_input_dir(self, raw_path: str | Path) -> Path:
        """Return a canonical offline image directory, or reject it."""
        self.reject_unsafe_path_syntax(raw_path, purpose="offline_image_folder")
        path = self._resolve_path(Path(raw_path), purpose="offline_image_folder", strict=True)
        if not path.is_dir():
            raise PathPolicyError("PATH_NOT_DIRECTORY", "offline_image_folder must be a directory.")
        self._reject_secret_path(path, purpose="offline_image_folder")
        if self.allowed_input_dirs:
            self._require_inside_roots(
                path,
                self.allowed_input_dirs,
                code="PATH_OUTSIDE_ALLOWED_ROOTS",
                purpose="offline_image_folder",
            )
        return path

    def validate_status_file_read(self, raw_path: str | Path) -> Path:
        """Return a canonical status-file path, or reject unsafe reads."""
        self.reject_unsafe_path_syntax(raw_path, purpose="status_file")
        candidate = Path(raw_path)
        self._reject_secret_path(candidate, purpose="status_file")
        if candidate.exists() and candidate.is_symlink():
            raise PathPolicyError("PATH_SYMLINK_REJECTED", "status_file must not be a symlink.")
        path = self._resolve_path(candidate, purpose="status_file")
        self._reject_secret_path(path, purpose="status_file")
        if path.exists() and not path.is_file():
            raise PathPolicyError("PATH_NOT_FILE", "status_file must be a regular file.")
        if self.allowed_command_output_dir is not None:
            self._require_inside_roots(
                path,
                (self.allowed_command_output_dir,),
                code="PATH_OUTSIDE_ALLOWED_ROOTS",
                purpose="status_file",
            )
        return path

    def validate_command_output_path(self, raw_path: str | Path) -> Path:
        """Return a canonical command bridge path, or reject unsafe writes."""
        self.reject_unsafe_path_syntax(raw_path, purpose="command_output_path")
        candidate = Path(raw_path)
        self._reject_secret_path(candidate, purpose="command_output_path")
        if candidate.name not in self.allowed_command_filenames:
            raise PathPolicyError(
                "COMMAND_FILENAME_NOT_ALLOWED",
                "command_output_path must use the DAC-3D assistant command filename.",
            )
        parent = self._resolve_path(candidate.parent, purpose="command_output_dir")
        if self.allowed_command_output_dir is not None:
            self._require_inside_roots(
                parent,
                (self.allowed_command_output_dir,),
                code="PATH_OUTSIDE_ALLOWED_ROOTS",
                purpose="command_output_dir",
            )
        destination = parent / candidate.name
        if destination.exists():
            if destination.is_symlink():
                raise PathPolicyError("PATH_SYMLINK_REJECTED", "command_output_path must not be a symlink.")
            if not destination.is_file():
                raise PathPolicyError("PATH_NOT_FILE", "command_output_path must be a regular file.")
        tmp_path = destination.with_suffix(".tmp")
        if tmp_path.exists() and (tmp_path.is_symlink() or not tmp_path.is_file()):
            raise PathPolicyError("PATH_TMP_UNSAFE", "temporary command output path is unsafe.")
        return destination

    def safe_upload_destination(self, upload_root: str | Path, filename: str) -> Path:
        """Return a safe upload destination under a trusted temporary directory."""
        name = _safe_upload_name(filename)
        root = self._resolve_path(Path(upload_root), purpose="upload_root", strict=True)
        destination = root / name
        self._reject_secret_path(destination, purpose="upload_file")
        resolved_destination = self._resolve_path(destination, purpose="upload_file")
        self._require_inside_roots(
            resolved_destination,
            (root,),
            code="UPLOAD_OUTSIDE_ROOT",
            purpose="upload_file",
        )
        if destination.exists() and (destination.is_symlink() or not destination.is_file()):
            raise PathPolicyError("PATH_UPLOAD_UNSAFE", "upload_file destination is unsafe.")
        return destination

    def _resolve_path(self, path: Path, *, purpose: str, strict: bool = False) -> Path:
        try:
            return path.expanduser().resolve(strict=strict)
        except FileNotFoundError as exc:
            raise PathPolicyError("PATH_NOT_FOUND", f"{purpose} does not exist.") from exc
        except (OSError, RuntimeError) as exc:
            raise PathPolicyError("PATH_RESOLUTION_FAILED", f"{purpose} cannot be resolved safely.") from exc

    def _require_inside_roots(
        self,
        path: Path,
        roots: tuple[Path, ...],
        *,
        code: str,
        purpose: str,
    ) -> None:
        if any(_is_relative_to(path, root) for root in roots):
            return
        raise PathPolicyError(code, f"{purpose} is outside the configured allowed directories.")

    def _reject_secret_path(self, path: Path, *, purpose: str) -> None:
        if _looks_like_secret_path(path):
            raise PathPolicyError("SECRET_PATH_REJECTED", f"{purpose} points at a secret-like path.")


def _read_csv(name: str) -> tuple[str, ...]:
    raw_value = os.getenv(name)
    if raw_value is None:
        return ()
    return tuple(part.strip() for part in raw_value.split(",") if part.strip())


def _is_prod_env() -> bool:
    return os.getenv("DAC3D_ENV", "dev").strip().lower() in {"prod", "production"}


def _contains_parent_reference(raw_path: str) -> bool:
    return ".." in Path(raw_path).parts or ".." in PureWindowsPath(raw_path).parts


def _safe_upload_name(filename: str) -> str:
    text = filename.strip()
    if not text:
        raise PathPolicyError("UPLOAD_FILENAME_EMPTY", "upload filename is empty.")
    if "/" in text or "\\" in text:
        raise PathPolicyError("UPLOAD_FILENAME_PATH", "upload filename must not include path separators.")
    if "\x00" in text:
        raise PathPolicyError("PATH_NUL_BYTE", "upload filename contains a NUL byte.")
    if text in {".", ".."} or _contains_parent_reference(text):
        raise PathPolicyError("PATH_TRAVERSAL", "upload filename must not contain '..'.")
    if PureWindowsPath(text).drive:
        raise PathPolicyError("UPLOAD_FILENAME_PATH", "upload filename must not include a drive prefix.")
    if _looks_like_secret_path(Path(text)):
        raise PathPolicyError("SECRET_PATH_REJECTED", "upload filename looks like a secret.")
    return Path(text).name


def _looks_like_secret_path(path: Path) -> bool:
    secret_names = {
        ".env",
        ".env.local",
        ".env.production",
        ".npmrc",
        ".pypirc",
        ".netrc",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
    }
    secret_suffixes = {".pem", ".key", ".p12", ".pfx", ".keystore"}
    for part in path.parts:
        lowered = part.lower()
        if lowered in secret_names or lowered.startswith(".env."):
            return True
    return path.suffix.lower() in secret_suffixes


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
