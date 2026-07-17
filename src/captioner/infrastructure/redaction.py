"""Small, deterministic redaction helpers for provider credentials."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from captioner.core.domain.result import JsonValue

REDACTED = "<redacted>"
_SECRET_KEYS = frozenset({"api_key", "authorization", "access_token", "token"})


def redact_text(value: str, secrets: Sequence[str] = ()) -> str:
    """Replace secret values without exposing them in diagnostics."""
    result = value
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        result = result.replace(secret, REDACTED)
    return result


def redact_headers(headers: Mapping[str, str], secrets: Sequence[str] = ()) -> dict[str, str]:
    """Return headers safe for logs; Authorization is always redacted."""
    return {
        key: REDACTED if key.lower() == "authorization" else redact_text(value, secrets)
        for key, value in headers.items()
    }


def redact_json(value: object, secrets: Sequence[str] = ()) -> JsonValue:
    """Recursively copy JSON-like data while removing secret values."""
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {
            str(key): REDACTED
            if str(key).lower().replace("-", "_") in _SECRET_KEYS
            else redact_json(item, secrets)
            for key, item in mapping.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sequence = cast(Sequence[object], value)
        return [redact_json(item, secrets) for item in sequence]
    if isinstance(value, str):
        return redact_text(value, secrets)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return REDACTED


def redact(value: object, secrets: Sequence[str] = ()) -> JsonValue | str:
    """Convenient boundary helper for either text or JSON-like values."""
    if isinstance(value, str):
        return redact_text(value, secrets)
    return redact_json(value, secrets)
