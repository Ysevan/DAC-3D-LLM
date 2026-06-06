"""API authentication and authorization helpers for the DAC-3D web surface."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


SESSION_HEADER = "x-dac3d-session-id"
OPERATOR_HEADER = "x-dac3d-operator-id"
ROLES_HEADER = "x-dac3d-roles"
REQUEST_ID_HEADER = "x-request-id"

READ_ROLES = {"viewer", "operator", "admin", "security_admin"}
WRITE_ROLES = {"operator", "admin", "security_admin"}
PRIVILEGED_ROLES = {"admin", "security_admin"}

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.:-]+")


class ApiSecurityError(RuntimeError):
    """Structured API security failure."""

    def __init__(self, *, code: str, message: str, status_code: int = 403) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class ApiActor:
    """Authenticated API caller context."""

    session_id: str
    operator_id: str | None
    roles: tuple[str, ...]

    @property
    def can_read(self) -> bool:
        return bool(set(self.roles) & READ_ROLES)

    @property
    def can_write(self) -> bool:
        return bool(set(self.roles) & WRITE_ROLES)

    @property
    def can_admin(self) -> bool:
        return bool(set(self.roles) & PRIVILEGED_ROLES)


def normalize_security_id(value: Any, *, field_name: str) -> str:
    """Return a compact identifier safe for logs, traces, and token binding."""
    text = str(value or "").strip()
    if not text:
        raise ApiSecurityError(
            code=f"MISSING_{field_name.upper()}",
            message=f"{field_name} is required.",
            status_code=401,
        )
    normalized = _SAFE_ID_RE.sub("_", text)[:120]
    if not normalized:
        raise ApiSecurityError(
            code=f"INVALID_{field_name.upper()}",
            message=f"{field_name} is invalid.",
            status_code=422,
        )
    return normalized


def require_api_actor(
    headers: Mapping[str, str],
    *,
    payload: Mapping[str, Any] | None = None,
    query_params: Mapping[str, str] | None = None,
    write: bool = False,
    require_operator: bool = False,
    required_roles: set[str] | None = None,
) -> ApiActor:
    """Validate session/operator/role context for one API request."""
    payload = payload or {}
    query_params = query_params or {}
    session_values = [
        _header_value(headers, SESSION_HEADER),
        payload.get("session_id"),
        query_params.get("session_id"),
    ]
    session_id = _single_bound_value(session_values, field_name="session_id")

    operator_values = [
        _header_value(headers, OPERATOR_HEADER),
        payload.get("operator_id"),
        query_params.get("operator_id"),
    ]
    operator_id = _optional_bound_value(operator_values, field_name="operator_id")
    if write or require_operator:
        if not operator_id:
            raise ApiSecurityError(
                code="MISSING_OPERATOR_ID",
                message="operator_id is required for write operations.",
                status_code=403,
            )

    roles = _roles_from_sources(headers, payload)
    actor = ApiActor(session_id=session_id, operator_id=operator_id, roles=roles)
    if not actor.can_read:
        raise ApiSecurityError(
            code="READ_PERMISSION_REQUIRED",
            message="The current actor does not have read permission.",
            status_code=403,
        )
    if write and not actor.can_write:
        raise ApiSecurityError(
            code="WRITE_PERMISSION_REQUIRED",
            message="The current actor does not have write permission.",
            status_code=403,
        )
    if required_roles and not (set(actor.roles) & required_roles):
        raise ApiSecurityError(
            code="PRIVILEGED_ROLE_REQUIRED",
            message="This operation requires a privileged role.",
            status_code=403,
        )
    return actor


def _header_value(headers: Mapping[str, str], key: str) -> str | None:
    for candidate, value in headers.items():
        if candidate.lower() == key:
            return value
    return None


def _single_bound_value(values: list[Any], *, field_name: str) -> str:
    normalized_values = [
        normalize_security_id(value, field_name=field_name)
        for value in values
        if str(value or "").strip()
    ]
    if not normalized_values:
        raise ApiSecurityError(
            code=f"MISSING_{field_name.upper()}",
            message=f"{field_name} is required.",
            status_code=401,
        )
    first = normalized_values[0]
    if any(value != first for value in normalized_values[1:]):
        raise ApiSecurityError(
            code=f"{field_name.upper()}_MISMATCH",
            message=f"{field_name} must match across header, query, and body.",
            status_code=403,
        )
    return first


def _optional_bound_value(values: list[Any], *, field_name: str) -> str | None:
    present = [value for value in values if str(value or "").strip()]
    if not present:
        return None
    return _single_bound_value(present, field_name=field_name)


def _roles_from_sources(
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
) -> tuple[str, ...]:
    raw_roles = payload.get("roles")
    if raw_roles is None:
        raw_roles = _header_value(headers, ROLES_HEADER)
    if raw_roles is None:
        return ("viewer",)
    if isinstance(raw_roles, str):
        values = [part.strip().lower() for part in raw_roles.split(",")]
    elif isinstance(raw_roles, (list, tuple, set)):
        values = [str(part).strip().lower() for part in raw_roles]
    else:
        values = []
    roles = tuple(role for role in values if role)
    return roles or ("viewer",)
