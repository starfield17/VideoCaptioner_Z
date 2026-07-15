"""JSON language catalog loading and validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.result import JsonValue
from captioner.i18n.locale import SUPPORTED_LOCALES, normalize_locale
from captioner.i18n.validation import validate_placeholder_pair


@dataclass(frozen=True, slots=True)
class Catalog:
    """Validated catalog data."""

    locale: str
    name: str
    fallback: str | None
    schema_version: int
    messages: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", MappingProxyType(dict(self.messages)))


class _DuplicateKey(ValueError):
    pass


def _object_pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def load_catalog(path: Path, *, expected_locale: str | None = None) -> Catalog:
    """Load one catalog and reject malformed or internally inconsistent data."""
    if not path.is_file():
        raise AppError("i18n.catalog_missing", {"path": str(path)})
    try:
        raw_value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_object_pairs_hook
        )
    except _DuplicateKey as exc:
        raise AppError("i18n.duplicate_key", {"path": str(path), "key": str(exc)}) from exc
    except json.JSONDecodeError as exc:
        raise AppError(
            "i18n.catalog_invalid_json",
            {"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    except OSError as exc:
        raise AppError("i18n.catalog_read_failed", {"path": str(path)}) from exc

    if not isinstance(raw_value, dict):
        raise AppError("i18n.catalog_invalid", {"path": str(path), "reason": "root"})
    raw = cast(dict[str, object], raw_value)
    metadata_value = raw.get("_meta")
    messages_value = raw.get("messages")
    if not isinstance(metadata_value, dict) or not isinstance(messages_value, dict):
        raise AppError("i18n.catalog_invalid", {"path": str(path), "reason": "sections"})

    metadata = cast(dict[str, object], metadata_value)
    locale_value = metadata.get("locale")
    name_value = metadata.get("name")
    fallback_value = metadata.get("fallback")
    schema_value = metadata.get("schema_version")
    if not isinstance(locale_value, str) or not isinstance(name_value, str):
        raise AppError("i18n.catalog_invalid", {"path": str(path), "reason": "metadata"})
    if fallback_value is not None and not isinstance(fallback_value, str):
        raise AppError("i18n.catalog_invalid", {"path": str(path), "reason": "fallback"})
    if schema_value != 1 or not name_value.strip():
        raise AppError("i18n.catalog_invalid", {"path": str(path), "reason": "schema"})

    try:
        locale = normalize_locale(locale_value, strict=True)
    except AppError as exc:
        raise AppError("i18n.catalog_invalid", {"path": str(path), "reason": exc.code}) from exc
    if expected_locale is not None:
        expected = normalize_locale(expected_locale, strict=True)
        if locale != expected:
            raise AppError(
                "i18n.locale_mismatch",
                {"path": str(path), "expected": expected, "actual": locale},
            )
    fallback = None if fallback_value is None else normalize_locale(fallback_value, strict=True)

    messages: dict[str, str] = {}
    raw_messages = cast(dict[object, object], messages_value)
    for key, value in raw_messages.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise AppError("i18n.catalog_invalid", {"path": str(path), "reason": "messages"})
        if not value.strip():
            raise AppError("i18n.empty_translation", {"path": str(path), "key": key})
        messages[key] = value

    return Catalog(
        locale=locale,
        name=name_value,
        fallback=fallback,
        schema_version=cast(int, schema_value),
        messages=messages,
    )


def validate_catalog_pair(english: Catalog, translation: Catalog, *, strict: bool = True) -> None:
    """Validate keys and placeholders against the English source catalog."""
    unknown = sorted(set(translation.messages) - set(english.messages))
    if unknown and strict:
        unknown_values: list[JsonValue] = []
        unknown_values.extend(unknown)
        params: dict[str, JsonValue] = {"locale": translation.locale, "keys": unknown_values}
        raise AppError("i18n.unknown_key", params)
    for key, english_value in english.messages.items():
        translation_value = translation.messages.get(key)
        if translation_value is not None:
            validate_placeholder_pair(key, english_value, translation_value)


def _filename_locale(path: Path) -> str:
    if path.stem not in SUPPORTED_LOCALES:
        raise AppError("i18n.locale_filename_invalid", {"path": str(path)})
    return path.stem


def validate_catalog_directory(resource_dir: Path, *, strict: bool = True) -> tuple[Catalog, ...]:
    """Validate all built-in JSON catalogs, with English as the source."""
    catalog_dir = resource_dir
    english_path = catalog_dir / "en.json"
    english = load_catalog(english_path, expected_locale="en")
    paths = sorted(catalog_dir.glob("*.json"))
    catalogs: list[Catalog] = [english]
    for path in paths:
        if path == english_path:
            continue
        locale = _filename_locale(path)
        catalog = load_catalog(path, expected_locale=locale)
        validate_catalog_pair(english, catalog, strict=strict)
        catalogs.append(catalog)
    return tuple(catalogs)


def catalog_to_json_value(catalog: Catalog) -> dict[str, JsonValue]:
    """Return a JSON-compatible diagnostic representation."""
    return {
        "locale": catalog.locale,
        "name": catalog.name,
        "fallback": catalog.fallback,
        "schema_version": catalog.schema_version,
        "messages": dict(catalog.messages),
    }
