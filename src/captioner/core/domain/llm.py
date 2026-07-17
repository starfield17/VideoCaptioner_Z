"""Pure LLM request and strict structured-response domain contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Self, cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.result import (
    FrozenJsonValue,
    JsonValue,
    freeze_json_value,
    thaw_json_value,
)
from captioner.core.policies.unicode_metrics import normalize_text

LLM_RESPONSE_SCHEMA_VERSION = 1
_FORBIDDEN_SCHEMA_KEYS = frozenset(
    {"start_ms", "end_ms", "timestamp", "timestamps", "duration", "duration_ms"}
)
_CONTEXT_PAYLOAD_FIELDS = frozenset({"terminology", "anomalies", "nearby_cues"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class LLMTaskKind(StrEnum):
    CORRECT_SOURCE = "correct_source"
    TRANSLATE_FAST = "translate_fast"
    TRANSLATE_QUALITY = "translate_quality"
    REVIEW = "review"
    TERMINOLOGY = "terminology"
    REPAIR_STRUCTURED = "repair_structured"


# Stable OpenAI-compatible json_schema.name values. Never derive these from
# Python class __qualname__ (dynamic batch classes embed "<locals>").
_PROVIDER_SCHEMA_BASE_NAMES: Mapping[str, str] = {
    LLMTaskKind.CORRECT_SOURCE.value: "captioner_correct_source_batch",
    LLMTaskKind.TRANSLATE_FAST.value: "captioner_translate_fast_batch",
    LLMTaskKind.TRANSLATE_QUALITY.value: "captioner_translate_quality_batch",
    LLMTaskKind.REVIEW.value: "captioner_review_batch",
    LLMTaskKind.TERMINOLOGY.value: "captioner_terminology_batch",
    LLMTaskKind.REPAIR_STRUCTURED.value: "captioner_repair_structured_batch",
}
_SCHEMA_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


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
    context_payload: Mapping[str, JsonValue] | None = None
    repair_prompt_id: str = ""
    repair_prompt_version: str = ""
    repair_prompt_content_sha256: str = ""
    repair_prompt_content: str = ""

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
        for field in (
            "prompt_id",
            "prompt_version",
            "prompt_content_sha256",
            "prompt_content",
            "repair_prompt_id",
            "repair_prompt_version",
            "repair_prompt_content_sha256",
            "repair_prompt_content",
        ):
            value = getattr(self, field)
            if not isinstance(value, str):
                raise AppError("llm.request_invalid", {"field": field})
        prompt_identity = (self.prompt_id, self.prompt_version, self.prompt_content_sha256)
        if any(prompt_identity) and not all(prompt_identity):
            raise AppError("llm.request_invalid", {"field": "prompt", "reason": "identity"})
        if self.prompt_content and not all(prompt_identity):
            raise AppError("llm.request_invalid", {"field": "prompt_content"})
        for field in ("prompt_id", "prompt_version"):
            value = getattr(self, field)
            if value:
                _canonical_nonempty(value, field)
        if self.prompt_content_sha256 and _SHA256_RE.fullmatch(self.prompt_content_sha256) is None:
            raise AppError("llm.request_invalid", {"field": "prompt_content_sha256"})
        repair_identity = (
            self.repair_prompt_id,
            self.repair_prompt_version,
            self.repair_prompt_content_sha256,
        )
        if any(repair_identity) and not all(repair_identity):
            raise AppError("llm.request_invalid", {"field": "repair_prompt", "reason": "identity"})
        for field, value in zip(
            ("repair_prompt_id", "repair_prompt_version", "repair_prompt_content_sha256"),
            repair_identity,
            strict=True,
        ):
            if value:
                _canonical_nonempty(value, field)
        if (
            self.repair_prompt_content_sha256
            and _SHA256_RE.fullmatch(self.repair_prompt_content_sha256) is None
        ):
            raise AppError("llm.request_invalid", {"field": "repair_prompt_content_sha256"})
        if (
            self.prompt_content
            and self.prompt_content_sha256
            and hashlib.sha256(self.prompt_content.encode("utf-8")).hexdigest()
            != self.prompt_content_sha256
        ):
            raise AppError("llm.request_invalid", {"field": "prompt_content_sha256"})
        if (
            self.repair_prompt_content
            and self.repair_prompt_content_sha256
            and hashlib.sha256(self.repair_prompt_content.encode("utf-8")).hexdigest()
            != self.repair_prompt_content_sha256
        ):
            raise AppError("llm.request_invalid", {"field": "repair_prompt_content_sha256"})
        if self.context_payload is not None:
            try:
                frozen_payload = freeze_json_value(self.context_payload)
                if (
                    not isinstance(frozen_payload, Mapping)
                    or _contains_forbidden_context_key(frozen_payload)
                    or _contains_secret_context_key(frozen_payload)
                ):
                    raise AppError("llm.request_invalid", {"field": "context_payload"})
                validate_context_payload(frozen_payload)
                # Validate finite JSON and detach the caller's mutable mapping.
                json.dumps(
                    thaw_json_value(cast(Mapping[str, FrozenJsonValue], frozen_payload)),
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise AppError("llm.request_invalid", {"field": "context_payload"}) from exc
            object.__setattr__(
                self,
                "context_payload",
                cast(Mapping[str, JsonValue], frozen_payload),
            )
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
            "context_payload": (
                None
                if self.context_payload is None
                else thaw_json_value(cast(FrozenJsonValue, self.context_payload))
            ),
            "repair_prompt_id": self.repair_prompt_id,
            "repair_prompt_version": self.repair_prompt_version,
            "repair_prompt_content_sha256": self.repair_prompt_content_sha256,
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
class TerminologyTerm:
    source_term: str
    target_term: str

    def __post_init__(self) -> None:
        _canonical_nonempty(self.source_term, "source_term")
        _canonical_nonempty(self.target_term, "target_term")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"source_term": self.source_term, "target_term": self.target_term}


@dataclass(frozen=True, slots=True, init=False)
class TerminologyResponse(_StrictResponse):
    """Sparse terminology output; an input unit may contain no terms."""

    id: str
    terms: tuple[TerminologyTerm, ...]

    _field_names: ClassVar[tuple[str, ...]] = ("id", "terms")
    text_fields: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        id: str,
        terms: Sequence[TerminologyTerm | Mapping[str, object]],
    ) -> None:
        _canonical_nonempty(id, "id")
        converted_terms: list[TerminologyTerm] = []
        for term in terms:
            if isinstance(term, TerminologyTerm):
                converted_terms.append(term)
            else:
                raw_value = cast(object, term)
                if not isinstance(raw_value, Mapping):
                    raise AppError("llm.response_invalid", {"reason": "terms"})
                raw = cast(Mapping[str, object], raw_value)
                if set(raw) != {"source_term", "target_term"}:
                    raise AppError("llm.response_invalid", {"reason": "terms"})
                source = raw.get("source_term")
                target = raw.get("target_term")
                if not isinstance(source, str) or not isinstance(target, str):
                    raise AppError("llm.response_invalid", {"reason": "terms"})
                converted_terms.append(TerminologyTerm(source, target))
        converted = tuple(converted_terms)
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "terms", converted)

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        if not isinstance(value, Mapping):
            raise AppError("llm.response_invalid", {"reason": "object"})
        raw = cast(Mapping[str, object], value)
        if set(raw) != {"id", "terms"}:
            raise AppError("llm.response_invalid", {"reason": "fields"})
        identifier = raw.get("id")
        terms = raw.get("terms")
        if (
            not isinstance(identifier, str)
            or not isinstance(terms, Sequence)
            or isinstance(terms, (str, bytes, bytearray))
        ):
            raise AppError("llm.response_invalid", {"reason": "terms"})
        return cls(identifier, cast(Sequence[Mapping[str, object]], terms))

    @classmethod
    def json_schema(cls) -> dict[str, JsonValue]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "terms"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["source_term", "target_term"],
                        "properties": {
                            "source_term": {"type": "string", "minLength": 1},
                            "target_term": {"type": "string", "minLength": 1},
                        },
                    },
                },
            },
        }

    @classmethod
    def model_json_schema(cls) -> dict[str, JsonValue]:
        return cls.json_schema()

    @classmethod
    def schema(cls) -> dict[str, JsonValue]:
        return cls.json_schema()

    def to_dict(self) -> dict[str, JsonValue]:
        return {"id": self.id, "terms": [term.to_dict() for term in self.terms]}

    @property
    def source_term(self) -> str:
        return self.terms[0].source_term if self.terms else ""

    @property
    def target_term(self) -> str:
        return self.terms[0].target_term if self.terms else ""


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
Terminology = TerminologyResponse
FastTranslation = FastTranslationResponse
QualityTranslation = QualityTranslationResponse
Review = ReviewResponse
ReviewTranslationResponse = ReviewResponse


@dataclass(frozen=True, slots=True)
class StructuredResponseBatch:
    """Strict array wrapper used when one request covers multiple Chunk items."""

    responses: tuple[object, ...]

    def __post_init__(self) -> None:
        responses = tuple(self.responses)
        if not responses:
            raise AppError("llm.response_invalid", {"reason": "empty_batch"})
        object.__setattr__(self, "responses", responses)

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        raise NotImplementedError

    @classmethod
    def from_json(cls, value: str | bytes) -> Self:
        raise NotImplementedError

    def to_dict(self) -> list[JsonValue]:
        result: list[JsonValue] = []
        for response in self.responses:
            to_dict = getattr(response, "to_dict", None)
            if not callable(to_dict):
                raise AppError("llm.response_invalid", {"reason": "batch_item"})
            value = to_dict()
            if not isinstance(value, dict):
                raise AppError("llm.response_invalid", {"reason": "batch_item"})
            result.append(cast(dict[str, JsonValue], value))
        return result


def response_batch_schema[T](item_schema: type[T]) -> type[StructuredResponseBatch]:
    """Create a schema class for a non-empty array of one strict item schema."""

    class BatchResponse(StructuredResponseBatch):
        @classmethod
        def from_mapping(cls, value: object) -> StructuredResponseBatch:
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
                raise AppError("llm.response_invalid", {"reason": "batch_array"})
            responses: list[object] = []
            for item in cast(Sequence[object], value):
                parser = getattr(item_schema, "from_mapping", None)
                if not callable(parser):
                    raise AppError("llm.schema_invalid", {"reason": "batch_item_schema"})
                responses.append(parser(item))
            return cls(tuple(responses))

        @classmethod
        def from_json(cls, value: str | bytes) -> StructuredResponseBatch:
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
            return {
                "type": "array",
                "minItems": 1,
                "items": response_schema_for(cast(type[object], item_schema)),
            }

        @classmethod
        def model_json_schema(cls) -> dict[str, JsonValue]:
            return cls.json_schema()

        @classmethod
        def schema(cls) -> dict[str, JsonValue]:
            return cls.json_schema()

    BatchResponse.__name__ = f"{item_schema.__name__}BatchResponse"
    return BatchResponse


batch_response_schema = response_batch_schema
make_batch_response_schema = response_batch_schema


def response_schema_for(response_schema: type[object]) -> dict[str, JsonValue]:
    """Return a schema and fail closed if a response adds a timing field."""
    schema_method = getattr(response_schema, "json_schema", None)
    if not callable(schema_method):
        raise AppError("llm.schema_invalid", {"reason": "response_schema"})
    schema = cast(dict[str, JsonValue], schema_method())
    _assert_no_timing_fields(schema)
    return schema


def encode_llm_request(
    request: LLMRequest,
    model: str,
    temperature: float,
    response_schema: type[object],
    *,
    response_schema_version: int = LLM_RESPONSE_SCHEMA_VERSION,
) -> bytes:
    """Serialize the exact provider request shape used by the adapter.

    Keeping this representation in Core lets the token-budget estimator and the
    OpenAI-compatible adapter reason about the same complete request without
    importing HTTP types into the domain. Schema *body* still comes from the
    response class; schema *name* is a stable task-based identity.
    """
    schema = response_schema_for(response_schema)
    payload: dict[str, JsonValue] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": request.prompt_content or "Return only the requested JSON object.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    request.to_dict(),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": provider_response_schema_name(request.task_kind, response_schema_version),
                "strict": True,
                "schema": schema,
            },
        },
    }
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )


def provider_response_schema_name(
    task_kind: str,
    response_schema_version: int = LLM_RESPONSE_SCHEMA_VERSION,
) -> str:
    """Return a stable OpenAI-compatible json_schema.name for one task kind.

    Names are derived only from durable business identity. They never depend on
    Python class ``__name__``, ``__qualname__``, module path, or runtime state.
    """
    if type(response_schema_version) is not int or response_schema_version < 1:
        raise AppError(
            "llm.schema_name_invalid",
            {"reason": "version", "version": response_schema_version},
        )
    if not task_kind.strip():
        raise AppError("llm.schema_name_invalid", {"reason": "task_kind"})
    base = _PROVIDER_SCHEMA_BASE_NAMES.get(task_kind.strip())
    if base is None:
        raise AppError("llm.schema_name_invalid", {"task_kind": task_kind})
    name = f"{base}_v{response_schema_version}"
    if _SCHEMA_NAME_RE.fullmatch(name) is None:
        raise AppError("llm.schema_name_invalid", {"name": name})
    return name


def response_schema_name(
    response_schema: type[object] | None = None,
    task_kind: str = "",
    *,
    response_schema_version: int = LLM_RESPONSE_SCHEMA_VERSION,
) -> str:
    """Return a stable schema identity shared by cache and request encoding.

    ``response_schema`` is accepted for call-site compatibility but is ignored;
    identity is task-based only so dynamic batch classes never leak ``<locals>``.
    """
    del response_schema
    return provider_response_schema_name(task_kind, response_schema_version)


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


def _contains_forbidden_context_key(value: object) -> bool:
    forbidden = {
        "start_ms",
        "end_ms",
        "timestamp",
        "timestamps",
        "duration",
        "duration_ms",
        "source_word_ids",
        "word_mapping",
    }
    if isinstance(value, Mapping):
        typed = cast(Mapping[object, object], value)
        return any(
            (isinstance(key, str) and key.lower() in forbidden)
            or _contains_forbidden_context_key(item)
            for key, item in typed.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_forbidden_context_key(item) for item in cast(Sequence[object], value))
    return False


def _contains_secret_context_key(value: object) -> bool:
    forbidden = {"api_key", "authorization", "access_token", "token", "secret", "password"}
    if isinstance(value, Mapping):
        typed = cast(Mapping[object, object], value)
        return any(
            (isinstance(key, str) and key.lower().replace("-", "_") in forbidden)
            or _contains_secret_context_key(item)
            for key, item in typed.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_secret_context_key(item) for item in cast(Sequence[object], value))
    return False


def validate_context_payload(value: object) -> None:
    """Validate the finite dynamic context envelope shared by LLM stages."""
    if not isinstance(value, Mapping):
        raise AppError("llm.request_invalid", {"field": "context_payload"})
    raw = cast(Mapping[object, object], value)
    if any(not isinstance(key, str) or key not in _CONTEXT_PAYLOAD_FIELDS for key in raw):
        raise AppError("llm.request_invalid", {"field": "context_payload"})
    _validate_context_entries(
        raw.get("terminology"),
        {"source_term", "target_term"},
        ("source_term", "target_term"),
    )
    _validate_context_entries(raw.get("anomalies"), {"cue_id", "reasons"}, ("cue_id",))
    _validate_context_entries(
        raw.get("nearby_cues"),
        {"cue_id", "source_text", "translated_text"},
        ("cue_id", "source_text", "translated_text"),
    )


def _validate_context_entries(
    value: object,
    fields: set[str],
    text_fields: tuple[str, ...],
) -> None:
    if value is None:
        return
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise AppError("llm.request_invalid", {"field": "context_payload"})
    for entry in cast(Sequence[object], value):
        if not isinstance(entry, Mapping):
            raise AppError("llm.request_invalid", {"field": "context_payload"})
        raw = cast(Mapping[object, object], entry)
        if set(raw) != fields:
            raise AppError("llm.request_invalid", {"field": "context_payload"})
        for field_name in text_fields:
            field_value = raw.get(field_name)
            if not isinstance(field_value, str):
                raise AppError("llm.request_invalid", {"field": "context_payload"})
            try:
                _canonical_nonempty(field_value, field_name)
            except AppError as exc:
                raise AppError("llm.request_invalid", {"field": "context_payload"}) from exc
        if fields == {"cue_id", "reasons"}:
            reasons = raw.get("reasons")
            if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes, bytearray)):
                raise AppError("llm.request_invalid", {"field": "context_payload"})
            typed_reasons = cast(Sequence[object], reasons)
            if not typed_reasons or any(not isinstance(reason, str) for reason in typed_reasons):
                raise AppError("llm.request_invalid", {"field": "context_payload"})
            for reason in typed_reasons:
                try:
                    _canonical_nonempty(reason, "reason")
                except AppError as exc:
                    raise AppError("llm.request_invalid", {"field": "context_payload"}) from exc


def _canonical_nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
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
