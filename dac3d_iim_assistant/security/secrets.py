"""Secret detection guards for traces and memory writes."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|secret|password|passwd|pwd|token|access[_-]?token|refresh[_-]?token|"
    r"connection[_-]?string|conn[_-]?str|dsn|authorization|bearer)",
    re.IGNORECASE,
)
SECRET_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i)\b(api[_-]?key|secret|password|passwd|pwd|token|access[_-]?token|"
        r"refresh[_-]?token|connection[_-]?string|dsn)\b\s*[:=]\s*['\"]?[^'\"\s,;}]{6,}"
    ),
    re.compile(r"(?i)\bauthorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:postgresql|postgres|mysql|mongodb|redis)://[^@\s]+:[^@\s]+@[^/\s]+", re.IGNORECASE),
    re.compile(r"(?i)\b密码\s*[:=：]\s*\S{4,}"),
)


class SecretDetectedError(ValueError):
    """Raised when a memory or trace payload contains a likely secret."""

    def __init__(self, *, location: str, labels: Sequence[str]) -> None:
        unique_labels = ", ".join(dict.fromkeys(labels))
        super().__init__(f"Secret-like value rejected in {location}: {unique_labels}")
        self.location = location
        self.labels = tuple(dict.fromkeys(labels))


def contains_secret(value: Any) -> bool:
    """Return whether a value contains likely credentials or connection secrets."""
    return bool(find_secret_labels(value))


def assert_no_secrets(value: Any, *, location: str = "payload") -> None:
    """Reject values that look like credentials before persisting them."""
    labels = find_secret_labels(value)
    if labels:
        raise SecretDetectedError(location=location, labels=labels)


def find_secret_labels(value: Any) -> list[str]:
    """Return human-readable labels for likely secrets in nested data."""
    labels: list[str] = []
    _collect_secret_labels(value, labels=labels, path="$")
    return labels


def _collect_secret_labels(value: Any, *, labels: list[str], path: str) -> None:
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            nested_path = f"{path}.{key_text}"
            if SENSITIVE_KEY_RE.search(key_text) and str(nested or "").strip():
                labels.append(nested_path)
            _collect_secret_labels(nested, labels=labels, path=nested_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _collect_secret_labels(nested, labels=labels, path=f"{path}[{index}]")
        return

    text = _stringify(value)
    for pattern in SECRET_TEXT_PATTERNS:
        if pattern.search(text):
            labels.append(path)
            return


def _stringify(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)
