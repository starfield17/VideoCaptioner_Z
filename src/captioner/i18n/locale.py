"""BCP 47-style locale normalization."""

from __future__ import annotations

import re
from collections.abc import Iterable

from captioner.core.domain.errors import AppError

SUPPORTED_LOCALES: tuple[str, ...] = ("en", "zh-CN")
_LOCALE_PART = re.compile(r"^[A-Za-z0-9]+$")


def normalize_locale(value: str, *, strict: bool = True, fallback: str = "en") -> str:
    """Normalize common locale spellings and optionally reject unsupported ones."""
    raw = value.strip().replace("_", "-")
    if not raw:
        if strict:
            raise AppError("i18n.locale_invalid", {"locale": value})
        return fallback

    parts = raw.split("-")
    if any(not part or _LOCALE_PART.fullmatch(part) is None for part in parts):
        if strict:
            raise AppError("i18n.locale_invalid", {"locale": value})
        return fallback

    language = parts[0].lower()
    normalized_parts = [language]
    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            normalized_parts.append(part.title())
        elif len(part) in (2, 3) and part.isalnum():
            normalized_parts.append(part.upper())
        else:
            normalized_parts.append(part)
    normalized = "-".join(normalized_parts)

    if normalized in SUPPORTED_LOCALES:
        return normalized
    if strict:
        raise AppError("i18n.locale_unsupported", {"locale": normalized})
    return normalize_locale(fallback, strict=True)


def available_locales(values: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return unique canonical locale names in stable order."""
    source = SUPPORTED_LOCALES if values is None else values
    result: list[str] = []
    for value in source:
        normalized = normalize_locale(value, strict=True)
        if normalized not in result:
            result.append(normalized)
    return tuple(result)
