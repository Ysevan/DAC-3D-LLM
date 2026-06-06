"""Production security configuration validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


class SecurityConfigError(ValueError):
    """Raised when production security configuration is unsafe."""


@dataclass(slots=True)
class ProductionSecurityConfig:
    """Runtime switches that must be safe before production command submission."""

    env: str = "dev"
    allow_command_submit: bool = True
    require_confirmation: bool = True
    allowed_input_dirs: tuple[str, ...] = field(default_factory=tuple)
    allowed_command_output_dir: str = ""
    enable_memory_write: bool = True
    enable_skill_patch_apply: bool = False
    enable_remote_tools: bool = False
    enable_open_world_tools: bool = False
    trace_redaction: bool = True
    cors_allowed_origins: tuple[str, ...] = field(default_factory=tuple)
    rate_limit_enabled: bool = True
    debug_mode: bool = False
    mock_mode: bool = True

    @classmethod
    def from_env(
        cls,
        *,
        base_dir: Path,
        mock_mode: bool,
        debug_mode: bool = False,
    ) -> "ProductionSecurityConfig":
        """Load production security flags from environment variables."""
        return cls(
            env=os.getenv("DAC3D_ENV", "dev").strip().lower() or "dev",
            allow_command_submit=_read_bool("DAC3D_ALLOW_COMMAND_SUBMIT", True),
            require_confirmation=_read_bool("DAC3D_REQUIRE_CONFIRMATION", True),
            allowed_input_dirs=_read_csv("DAC3D_ALLOWED_INPUT_DIRS", ()),
            allowed_command_output_dir=os.getenv(
                "DAC3D_ALLOWED_COMMAND_OUTPUT_DIR",
                str(base_dir / ".tmp" / "commands"),
            ).strip(),
            enable_memory_write=_read_bool("DAC3D_ENABLE_MEMORY_WRITE", True),
            enable_skill_patch_apply=_read_bool("DAC3D_ENABLE_SKILL_PATCH_APPLY", False),
            enable_remote_tools=_read_bool("DAC3D_ENABLE_REMOTE_TOOLS", False),
            enable_open_world_tools=_read_bool("DAC3D_ENABLE_OPEN_WORLD_TOOLS", False),
            trace_redaction=_read_bool("DAC3D_TRACE_REDACTION", True),
            cors_allowed_origins=_read_csv("DAC3D_CORS_ALLOWED_ORIGINS", ()),
            rate_limit_enabled=_read_bool("DAC3D_RATE_LIMIT_ENABLED", True),
            debug_mode=_read_bool("DAC3D_DEBUG_MODE", debug_mode),
            mock_mode=mock_mode,
        )

    @property
    def is_prod(self) -> bool:
        """Return whether this config represents a production runtime."""
        return self.env in {"prod", "production"}

    def validate(self) -> list[str]:
        """Return validation errors. Dev/test can be permissive with warnings."""
        errors: list[str] = []
        if not self.is_prod:
            return errors
        if not self.require_confirmation:
            errors.append("REQUIRE_CONFIRMATION must be true in production.")
        if not self.trace_redaction:
            errors.append("TRACE_REDACTION must be true in production.")
        if "*" in self.cors_allowed_origins:
            errors.append("CORS wildcard is forbidden in production.")
        if not self.cors_allowed_origins:
            errors.append("CORS_ALLOWED_ORIGINS must be set in production.")
        if self.enable_remote_tools:
            errors.append("ENABLE_REMOTE_TOOLS must be false in production unless explicitly reviewed.")
        if self.enable_open_world_tools:
            errors.append("ENABLE_OPEN_WORLD_TOOLS must be false in production unless explicitly reviewed.")
        if not self.allowed_input_dirs:
            errors.append("ALLOWED_INPUT_DIRS must not be empty in production.")
        if not self.allowed_command_output_dir:
            errors.append("ALLOWED_COMMAND_OUTPUT_DIR must not be empty in production.")
        if self.enable_skill_patch_apply:
            errors.append("ENABLE_SKILL_PATCH_APPLY must default to false in production.")
        if self.debug_mode:
            errors.append("Debug mode must be false in production.")
        if not self.rate_limit_enabled:
            errors.append("RATE_LIMIT_ENABLED must be true in production.")
        if self.allow_command_submit and self.mock_mode:
            errors.append("Production command submission cannot use the mock DAC-3D writer.")
        return errors

    def validate_or_raise(self) -> None:
        """Raise if the active security config is unsafe."""
        errors = self.validate()
        if errors:
            raise SecurityConfigError("; ".join(errors))

    def warnings(self) -> list[str]:
        """Return warnings for permissive non-production configurations."""
        warnings: list[str] = []
        if self.is_prod:
            return warnings
        if not self.require_confirmation:
            warnings.append("Confirmation is disabled outside production.")
        if "*" in self.cors_allowed_origins:
            warnings.append("Wildcard CORS is allowed only outside production.")
        if self.enable_remote_tools or self.enable_open_world_tools:
            warnings.append("Remote/open-world tools are enabled outside production.")
        if self.debug_mode:
            warnings.append("Debug mode is enabled outside production.")
        return warnings


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _read_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    values = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    return values or default
