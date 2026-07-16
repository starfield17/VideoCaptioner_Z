"""Pure LLM request and strict structured-response domain contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Self, cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.result import JsonValue
from captioner.core.policies.unicode_metrics import normalize_text

LLM_RESPONSE_SCHEMA_VERSION = 1
_FORBIDDEN_SCHEMA_KEYS = frozenset(
    {"start_ms", "end_ms", "timestamp", "timestamps", "duration", "duration_ms"}
)


class LLMTaskKind(StrEnum):
    CORRECT_SOURCE = "correct_source"
    TRANSLATE_FAST = "translate_fast"
    TRANSLATE_QUALITY = "translate_quality"
    REVIEW = "review"
    REPAIR_STRUCTURED = "repair_structured"


@dataclass(frozen=True, slots=True)
class LLMItem:
    """Text sent to a model; timing and Word mapping are deliberately absent."""

    id: str
    source: str

    def __post_init__(self) -> None:
        _canonical_nonempty(self.id, "id")
        _canonical_nonempty(self.source, "source")

    @property
    def text(self) -> str:
        return self.source

    def to_dict(self) -> dict[str, JsonValue]:
        return {"id": self.id, "source": self.source}


LLMRequestItem = LLMItem


@dataclass(frozen=True, slots=True)
class LLMRequest:
    task_kind: str
    items: tuple[LLMItem, ...]
    context: tuple[LLMItem, ...] = ()
    source_language: str | None = None
    target_language: str | None = None
    prompt_id: str = ""
    prompt_version: str = ""
    prompt_content_sha256: str = ""
    prompt_content: str = ""
    metadata: Mapping[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        _canonical_nonempty(self.task_kind, "task_kind")
        items = tuple(self.items)
        context = tuple(self.context)
        if not items:
            raise AppError("llm.request_invalid", {"field": "items", "reason": "empty"})
        item_ids = tuple(item.id for item in items)
        context_ids = tuple(item.id for item in context)
        if len(set(item_ids)) != len(item_ids):
            raise AppError("llm.request_invalid", {"field": "items", "reason": "duplicate_ids"})
        if set(item_ids) & set(context_ids):
            raise AppError("llm.request_invalid", {"field": "context", "reason": "output_id"})
        if len(set(context_ids)) != len(context_ids):
            raise AppError("llm.request_invalid", {"field": "context", "reason": "duplicate_ids"})
        for field in ("source_language", "target_language"):
            value = getattr(self, field)
            if value is not None:
                _canonical_nonempty(value, field)
        for field in ("prompt_id", "prompt_version", "prompt_content_sha256"):
            value = getattr(self, field)
            if value:
                _canonical_nonempty(value, field)
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "context", context)

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.items)

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.context)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "task_kind": self.task_kind,
            "items": [item.to_dict() for item in self.items],
            "context": [item.to_dict() for item in self.context],
            "source_language": self.source_language,
            "target_language": self.target_language,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "prompt_content_sha256": self.prompt_content_sha256,
            "prompt_content": self.prompt_content,
        }


class _StrictResponse:
    _field_names: ClassVar[tuple[str, ...]]
    text_fields: ClassVar[tuple[str, ...]]

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        if not isinstance(value, Mapping):
            raise AppError("llm.response_invalid", {"reason": "object"})
        raw = cast(Mapping[object, object], value)
        expected = set(cls._field_names)
        if set(raw) != expected:
            raise AppError(
                "llm.response_invalid",
                {
                    "reason": "fields",
                    "expected": cast(list[JsonValue], sorted(expected)),
                },
            )
        values: dict[str, object] = {}
        for field in cls._field_names:
            item = raw[field]
            if not isinstance(item, str):
                raise AppError("llm.response_invalid", {"reason": field})
            values[field] = _canonical_nonempty(item, field)
        return cls(**values)

    @classmethod
    def from_json(cls, value: str | bytes) -> Self:
        try:
            parsed = json.loads(
                value,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise AppError("llm.response_invalid", {"reason": "json"}) from exc
        return cls.from_mapping(parsed)

    @classmethod
    def json_schema(cls) -> dict[str, JsonValue]:
        properties = {field: {"type": "string", "minLength": 1} for field in cls._field_names}
        return cast(
            dict[str, JsonValue],
            {
                "type": "object",
                "additionalProperties": False,
                "required": list(cls._field_names),
                "properties": properties,
            },
        )

    @classmethod
    def model_json_schema(cls) -> dict[str, JsonValue]:
        return cls.json_schema()

    @classmethod
    def schema(cls) -> dict[str, JsonValue]:
        return cls.json_schema()

    def to_dict(self) -> dict[str, JsonValue]:
        return {field: cast(str, getattr(self, field)) for field in self._field_names}


@dataclass(frozen=True, slots=True)
class SourceCorrectionResponse(_StrictResponse):
    id: str
    corrected_source: str

    _field_names: ClassVar[tuple[str, ...]] = ("id", "corrected_source")
    text_fields: ClassVar[tuple[str, ...]] = ("corrected_source",)

    def __post_init__(self) -> None:
        _canonical_nonempty(self.id, "id")
        _canonical_nonempty(self.corrected_source, "corrected_source")


@dataclass(frozen=True, slots=True)
class FastTranslationResponse(_StrictResponse):
    id: str
    corrected_source: str
    translated_text: str

    _field_names: ClassVar[tuple[str, ...]] = ("id", "corrected_source", "translated_text")
    text_fields: ClassVar[tuple[str, ...]] = ("corrected_source", "translated_text")

    def __post_init__(self) -> None:
        _canonical_nonempty(self.id, "id")
        _canonical_nonempty(self.corrected_source, "corrected_source")
        _canonical_nonempty(self.translated_text, "translated_text")


@dataclass(frozen=True, slots=True)
class QualityTranslationResponse(_StrictResponse):
    id: str
    translated_text: str

    _field_names: ClassVar[tuple[str, ...]] = ("id", "translated_text")
    text_fields: ClassVar[tuple[str, ...]] = ("translated_text",)

    def __post_init__(self) -> None:
        _canonical_nonempty(self.id, "id")
        _canonical_nonempty(self.translated_text, "translated_text")


@dataclass(frozen=True, slots=True)
class ReviewResponse(_StrictResponse):
    id: str
    translated_text: str

    _field_names: ClassVar[tuple[str, ...]] = ("id", "translated_text")
    text_fields: ClassVar[tuple[str, ...]] = ("translated_text",)

    def __post_init__(self) -> None:
        _canonical_nonempty(self.id, "id")
        _canonical_nonempty(self.translated_text, "translated_text")


# Names used by application code and external contract tests are both kept
# descriptive and stable.
CorrectSourceResponse = SourceCorrectionResponse
SourceCorrection = SourceCorrectionResponse
FastTranslation = FastTranslationResponse
QualityTranslation = QualityTranslationResponse
Review = ReviewResponse
ReviewTranslationResponse = ReviewResponse


def response_schema_for(response_schema: type[object]) -> dict[str, JsonValue]:
    """Return a schema and fail closed if a response adds a timing field."""
    schema_method = getattr(response_schema, "json_schema", None)
    if not callable(schema_method):
        raise AppError("llm.schema_invalid", {"reason": "response_schema"})
    schema = cast(dict[str, JsonValue], schema_method())
    _assert_no_timing_fields(schema)
    return schema


def _assert_no_timing_fields(value: object) -> None:
    if isinstance(value, Mapping):
        typed = cast(Mapping[object, object], value)
        for key, item in typed.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_SCHEMA_KEYS:
                raise AppError("llm.schema_invalid", {"field": key})
            _assert_no_timing_fields(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in cast(Sequence[object], value):
            _assert_no_timing_fields(item)


def _canonical_nonempty(value: str, field: str) -> str:
    if not value.strip():
        raise AppError("llm.response_invalid", {"field": field, "reason": "empty"})
    try:
        canonical = normalize_text(value)
    except AppError as exc:
        raise AppError("llm.response_invalid", {"field": field, "reason": "control"}) from exc
    if canonical != value:
        raise AppError("llm.response_invalid", {"field": field, "reason": "not_canonical"})
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non_finite_json_value:{value}")
