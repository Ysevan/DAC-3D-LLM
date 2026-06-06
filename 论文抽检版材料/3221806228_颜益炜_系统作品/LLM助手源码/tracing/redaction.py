"""Redaction helpers for logs, traces, exceptions, and tool summaries."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from security.secrets import SENSITIVE_KEY_RE


REDACTION = "[REDACTED]"
_TEXT_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)\b(api[_-]?key|secret|password|passwd|pwd|token|access[_-]?token|"
            r"refresh[_-]?token|connection[_-]?string|dsn)\b\s*[:=]\s*['\"]?[^'\"\s,;}]+"
        ),
        r"\1=" + REDACTION,
    ),
    (
        re.compile(r"(?i)\bauthorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~+/=-]+"),
        "authorization: Bearer " + REDACTION,
    ),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer " + REDACTION),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), REDACTION),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), REDACTION),
    (re.compile(r"\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b"), REDACTION),
    (
        re.compile(r"(?i)\b((?:postgresql|postgres|mysql|mongodb|redis)://)[^@\s]+@"),
        r"\1" + REDACTION + "@",
    ),
    (re.compile(r"(?i)\b密码\s*[:=：]\s*\S+"), "密码:" + REDACTION),
)


def redact_text(value: Any, *, max_chars: int | None = None) -> str:
    """Redact secret-like substrings from free text and optionally clip it."""
    text = _stringify(value)
    for pattern, replacement in _TEXT_REDACTIONS:
        text = pattern.sub(replacement, text)
    if max_chars is not None and len(text) > max_chars:
        return f"{text[: max(0, max_chars - 3)]}..."
    return text


def redact_value(value: Any, *, max_string_chars: int = 1000) -> Any:
    """Recursively redact sensitive keys and secret-like text."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, nested in value.items():
            key_text = str(key)
            if SENSITIVE_KEY_RE.search(key_text):
                redacted[key_text] = REDACTION
            else:
                redacted[key_text] = redact_value(nested, max_string_chars=max_string_chars)
        return redacted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_value(item, max_string_chars=max_string_chars) for item in value]
    return redact_text(value, max_chars=max_string_chars)


def summarize_for_trace(value: Any, *, max_chars: int = 1200) -> Any:
    """Return a redacted, bounded summary suitable for audit JSONL."""
    redacted = redact_value(value, max_string_chars=max_chars)
    serialized = json.dumps(redacted, ensure_ascii=False, sort_keys=True, default=str)
    if len(serialized) <= max_chars:
        return redacted
    return redact_text(serialized, max_chars=max_chars)


def redact_exception(exc: BaseException) -> str:
    """Return a redacted exception summary without traceback frames."""
    return redact_text(f"{exc.__class__.__name__}: {exc}", max_chars=500)


def _stringify(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)
