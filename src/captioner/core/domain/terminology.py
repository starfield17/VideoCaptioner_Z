"""Immutable, deterministic terminology projections for quality translation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.policies.unicode_metrics import normalize_text


def normalize_term(value: str) -> str:
    """Normalize a source term for deterministic grouping."""
    return normalize_text(value).casefold()


def contains_term(text: str, term: str) -> bool:
    """Match a term on lexical token boundaries, including phrases."""
    source_tokens = _lexical_tokens(text)
    term_tokens = _lexical_tokens(term)
    if not term_tokens or len(term_tokens) > len(source_tokens):
        return False
    width = len(term_tokens)
    return any(
        source_tokens[index : index + width] == term_tokens
        for index in range(len(source_tokens) - width + 1)
    )


def _lexical_tokens(value: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in re.findall(r"[^\W_]+", value, flags=re.UNICODE))


@dataclass(frozen=True, slots=True)
class TerminologyEntry:
    source: str
    normalized_source: str
    target: str
    source_word_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, field in ((self.source, "source"), (self.target, "target")):
            if not value.strip():
                raise AppError("llm.terminology_invalid", {"field": field})
            if normalize_text(value) != value:
                raise AppError("llm.terminology_invalid", {"field": field})
        if self.normalized_source != normalize_term(self.source):
            raise AppError("llm.terminology_invalid", {"field": "normalized_source"})
        word_ids = tuple(self.source_word_ids)
        if (
            not word_ids
            or len(set(word_ids)) != len(word_ids)
            or any(not word_id.strip() for word_id in word_ids)
        ):
            raise AppError("llm.terminology_invalid", {"field": "source_word_ids"})
        object.__setattr__(self, "source_word_ids", word_ids)


@dataclass(frozen=True, slots=True)
class Terminology:
    transcript_id: str
    source_language: str
    target_language: str
    entries: tuple[TerminologyEntry, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        for value, field in (
            (self.transcript_id, "transcript_id"),
            (self.source_language, "source_language"),
            (self.target_language, "target_language"),
        ):
            if not value.strip():
                raise AppError("llm.terminology_invalid", {"field": field})
        if self.schema_version != 1:
            raise AppError("llm.terminology_invalid", {"field": "schema_version"})
        entries = tuple(self.entries)
        normalized = tuple(entry.normalized_source for entry in entries)
        if len(set(normalized)) != len(normalized):
            raise AppError("llm.terminology_invalid", {"field": "entries", "reason": "duplicate"})
        object.__setattr__(self, "entries", entries)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "transcript_id": self.transcript_id,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "entries": [
                {
                    "source": entry.source,
                    "normalized_source": entry.normalized_source,
                    "target": entry.target,
                    "source_word_ids": list(entry.source_word_ids),
                }
                for entry in self.entries
            ],
        }

    @classmethod
    def from_mapping(cls, value: object) -> Terminology:
        if not isinstance(value, dict):
            raise AppError("llm.terminology_invalid", {"reason": "object"})
        raw_value = cast(dict[str, object], value)
        if set(raw_value) != {
            "schema_version",
            "transcript_id",
            "source_language",
            "target_language",
            "entries",
        }:
            raise AppError("llm.terminology_invalid", {"reason": "fields"})
        raw_entries = raw_value["entries"]
        if not isinstance(raw_entries, Sequence) or isinstance(
            raw_entries, (str, bytes, bytearray)
        ):
            raise AppError("llm.terminology_invalid", {"field": "entries"})
        entries: list[TerminologyEntry] = []
        for raw in cast(Sequence[object], raw_entries):
            if not isinstance(raw, dict):
                raise AppError("llm.terminology_invalid", {"field": "entries"})
            typed_raw = cast(dict[str, object], raw)
            if set(typed_raw) != {
                "source",
                "normalized_source",
                "target",
                "source_word_ids",
            }:
                raise AppError("llm.terminology_invalid", {"field": "entries"})
            word_ids = typed_raw["source_word_ids"]
            if not isinstance(word_ids, Sequence) or isinstance(word_ids, (str, bytes, bytearray)):
                raise AppError("llm.terminology_invalid", {"field": "source_word_ids"})
            raw_word_ids = cast(Sequence[object], word_ids)
            if any(not isinstance(word_id, str) for word_id in raw_word_ids):
                raise AppError("llm.terminology_invalid", {"field": "source_word_ids"})
            entries.append(
                TerminologyEntry(
                    _string(typed_raw, "source"),
                    _string(typed_raw, "normalized_source"),
                    _string(typed_raw, "target"),
                    tuple(cast(str, word_id) for word_id in raw_word_ids),
                )
            )
        return cls(
            _string(raw_value, "transcript_id"),
            _string(raw_value, "source_language"),
            _string(raw_value, "target_language"),
            tuple(entries),
            _integer(raw_value, "schema_version"),
        )


def _string(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise AppError("llm.terminology_invalid", {"field": key})
    return item


def _integer(value: Mapping[str, object], key: str) -> int:
    item = value.get(key)
    if type(item) is not int:
        raise AppError("llm.terminology_invalid", {"field": key})
    return item
