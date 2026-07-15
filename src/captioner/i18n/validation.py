"""Catalog and placeholder validation helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from string import Formatter

from captioner.core.domain.errors import AppError
from captioner.core.domain.result import JsonValue


def placeholder_names(value: str) -> frozenset[str]:
    """Extract root field names using Python's format parser."""
    names: set[str] = set()
    try:
        parsed = Formatter().parse(value)
        for _, field_name, _, _ in parsed:
            if field_name is None:
                continue
            root_name = field_name.split(".", 1)[0].split("[", 1)[0]
            names.add(root_name)
    except ValueError as exc:
        raise AppError("i18n.placeholder_syntax", {"value": value}) from exc
    return frozenset(names)


def validate_placeholder_pair(key: str, english: str, translation: str) -> None:
    """Require the same named placeholder set in both messages."""
    english_names = placeholder_names(english)
    translation_names = placeholder_names(translation)
    if english_names != translation_names:
        english_values = _json_strings(english_names)
        translation_values = _json_strings(translation_names)
        params: dict[str, JsonValue] = {
            "key": key,
            "english": english_values,
            "translation": translation_values,
        }
        raise AppError(
            "i18n.placeholder_mismatch",
            params,
        )

    english_counts = Counter(_field_names(english))
    translation_counts = Counter(_field_names(translation))
    if english_counts != translation_counts:
        english_values = _json_strings(english_counts)
        translation_values = _json_strings(translation_counts)
        params: dict[str, JsonValue] = {
            "key": key,
            "english": english_values,
            "translation": translation_values,
        }
        raise AppError(
            "i18n.placeholder_mismatch",
            params,
        )


def _json_strings(values: Iterable[str]) -> list[JsonValue]:
    result: list[JsonValue] = []
    result.extend(values)
    return result


def _field_names(value: str) -> list[str]:
    result: list[str] = []
    for _, field_name, _, _ in Formatter().parse(value):
        if field_name is not None:
            result.append(field_name.split(".", 1)[0].split("[", 1)[0])
    return result
