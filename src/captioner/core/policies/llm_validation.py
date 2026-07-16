"""Deterministic validation for structured LLM responses."""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.llm import (
    FastTranslationResponse,
    QualityTranslationResponse,
    ReviewResponse,
    SourceCorrectionResponse,
    response_schema_for,
)
from captioner.core.domain.result import JsonValue
from captioner.core.policies.protected_spans import find_protected_spans
from captioner.core.policies.unicode_metrics import normalize_text


@dataclass(frozen=True, slots=True)
class ProtectedNumericToken:
    text: str
    digits: str
    percent: bool


def validate_responses(
    responses: Sequence[object],
    expected_ids: Sequence[str],
    *,
    context_ids: Sequence[str] = (),
    source_texts: Mapping[str, str] | None = None,
    target_language: str | None = None,
) -> tuple[object, ...]:
    """Validate IDs and every model-controlled text field in stable order."""
    expected = tuple(expected_ids)
    if len(set(expected)) != len(expected):
        raise AppError("llm.validation_config_invalid", {"reason": "duplicate_expected_ids"})
    contexts = set(context_ids)
    actual: list[str] = []
    for response in responses:
        response_id = _response_id(response)
        if response_id in actual:
            raise AppError("llm.duplicate_id", {"id": response_id})
        if response_id in contexts:
            raise AppError("llm.context_id_returned", {"id": response_id})
        actual.append(response_id)
    if set(actual) != set(expected):
        missing = [item_id for item_id in expected if item_id not in actual]
        extra = [item_id for item_id in actual if item_id not in expected]
        if missing:
            raise AppError("llm.missing_id", {"ids": cast(list[JsonValue], missing)})
        raise AppError("llm.extra_id", {"ids": cast(list[JsonValue], extra)})
    source_map: Mapping[str, str] = {} if source_texts is None else source_texts
    for response in responses:
        response_texts = _response_texts(response)
        for text in response_texts:
            _validate_text(text)
        response_id = _response_id(response)
        source = source_map.get(response_id)
        if source is not None:
            _validate_protected_numbers(source, " ".join(_protected_output_texts(response)))
        if target_language is not None:
            for text in _language_texts(response):
                if is_obvious_wrong_language(text, target_language):
                    raise AppError(
                        "llm.wrong_language",
                        {"id": response_id, "language": target_language},
                    )
    return tuple(responses)


def validate_response(
    response: object,
    expected_ids: Sequence[str],
    *,
    context_ids: Sequence[str] = (),
    source_texts: Mapping[str, str] | None = None,
    target_language: str | None = None,
) -> object:
    """Singular convenience wrapper used by adapters and contract tests."""
    return validate_responses(
        (response,),
        expected_ids,
        context_ids=context_ids,
        source_texts=source_texts,
        target_language=target_language,
    )[0]


def validate_llm_response(
    responses: Sequence[object],
    expected_ids: Sequence[str],
    *,
    context_ids: Sequence[str] = (),
    source_texts: Mapping[str, str] | None = None,
    target_language: str | None = None,
) -> tuple[object, ...]:
    return validate_responses(
        responses,
        expected_ids,
        context_ids=context_ids,
        source_texts=source_texts,
        target_language=target_language,
    )


def validate_response_schema(response: object, response_schema: type[object]) -> object:
    """Check the response shape through the schema class before ID checks."""
    if not hasattr(response_schema, "from_mapping"):
        raise AppError("llm.schema_invalid", {"reason": "response_schema"})
    schema = cast(type[_ResponseSchema], response_schema)
    return schema.from_mapping(response)


def response_schema_has_no_timestamps(response_schema: type[object]) -> bool:
    schema = response_schema_for(response_schema)
    return not _contains_forbidden_key(schema)


def protected_numeric_tokens(text: str) -> tuple[ProtectedNumericToken, ...]:
    tokens: list[ProtectedNumericToken] = []
    for span in find_protected_spans(text):
        digits = "".join(character for character in span.text if character.isdigit())
        if digits:
            tokens.append(ProtectedNumericToken(span.text, digits, "%" in span.text))
    return tuple(tokens)


def script_heuristic(text: str) -> str:
    counts: Counter[str] = Counter()
    for character in text:
        if character.isspace() or unicodedata.category(character).startswith("P"):
            continue
        if "一" <= character <= "鿿":
            counts["cjk"] += 1
        elif "぀" <= character <= "ヿ":
            counts["kana"] += 1
        elif "가" <= character <= "힯":
            counts["hangul"] += 1
        elif "؀" <= character <= "ۿ":
            counts["arabic"] += 1
        elif "Ѐ" <= character <= "ӿ":
            counts["cyrillic"] += 1
        elif character.isalpha():
            counts["latin"] += 1
    if not counts:
        return "other"
    if len(counts) > 1:
        return "mixed"
    return next(iter(counts))


def is_obvious_wrong_language(text: str, language: str) -> bool:
    script = script_heuristic(text)
    normalized = language.lower().replace("_", "-")
    if script in {"other", "mixed"}:
        return False
    if normalized.startswith(("zh",)):
        return script not in {"cjk"}
    if normalized.startswith("ja"):
        return script not in {"cjk", "kana"}
    if normalized.startswith("ko"):
        return script != "hangul"
    if normalized.startswith(("ar", "fa", "ur")):
        return script != "arabic"
    if normalized.startswith(("ru", "uk", "bg", "sr")):
        return script != "cyrillic"
    if normalized.startswith(("en", "de", "fr", "es", "it", "pt", "nl")):
        return script in {"cjk", "kana", "hangul", "arabic", "cyrillic"}
    return False


def _response_id(response: object) -> str:
    value: object
    if isinstance(response, Mapping):
        value = cast(Mapping[str, object], response).get("id")
    else:
        value = getattr(response, "id", None)
    if not isinstance(value, str) or not value.strip():
        raise AppError("llm.response_invalid", {"reason": "id"})
    return value


def _response_texts(response: object) -> tuple[str, ...]:
    if isinstance(response, Mapping):
        raw = cast(Mapping[str, object], response)
        if set(raw) - {"id", "corrected_source", "translated_text"}:
            raise AppError("llm.response_invalid", {"reason": "fields"})
        values: tuple[object, ...] = tuple(value for key, value in raw.items() if key != "id")
    else:
        if isinstance(
            response,
            (
                SourceCorrectionResponse,
                FastTranslationResponse,
                QualityTranslationResponse,
                ReviewResponse,
            ),
        ):
            fields = response.text_fields
        else:
            fields = tuple(
                name for name in ("corrected_source", "translated_text") if hasattr(response, name)
            )
        values = tuple(getattr(response, field) for field in fields)
    if not values or any(not isinstance(value, str) for value in values):
        raise AppError("llm.response_invalid", {"reason": "text"})
    return tuple(cast(str, value) for value in values)


def _language_texts(response: object) -> tuple[str, ...]:
    if isinstance(response, Mapping):
        value = cast(Mapping[str, object], response).get("translated_text")
        return (value,) if isinstance(value, str) else ()
    value = getattr(response, "translated_text", None)
    return (value,) if isinstance(value, str) else ()


def _protected_output_texts(response: object) -> tuple[str, ...]:
    if isinstance(response, Mapping):
        raw = cast(Mapping[str, object], response)
        translated = raw.get("translated_text")
        if isinstance(translated, str):
            return (translated,)
        corrected = raw.get("corrected_source")
        return (corrected,) if isinstance(corrected, str) else ()
    translated = getattr(response, "translated_text", None)
    if isinstance(translated, str):
        return (translated,)
    corrected = getattr(response, "corrected_source", None)
    return (corrected,) if isinstance(corrected, str) else ()


def _validate_text(text: str) -> None:
    if not text.strip():
        raise AppError("llm.empty_text")
    try:
        canonical = normalize_text(text)
    except AppError as exc:
        raise AppError("llm.non_canonical_text", {"reason": "control"}) from exc
    if canonical != text:
        raise AppError("llm.non_canonical_text")


def _validate_protected_numbers(source: str, output: str) -> None:
    source_tokens = protected_numeric_tokens(source)
    output_digits = "".join(character for character in output if character.isdigit())
    cursor = 0
    for token in source_tokens:
        position = output_digits.find(token.digits, cursor)
        if position < 0:
            raise AppError("llm.protected_token_lost", {"token": token.text})
        if token.percent and "%" not in output:
            raise AppError("llm.protected_token_lost", {"token": token.text})
        cursor = position + len(token.digits)


def _contains_forbidden_key(value: object) -> bool:
    forbidden = {"start_ms", "end_ms", "timestamp", "timestamps", "duration", "duration_ms"}
    if isinstance(value, Mapping):
        typed = cast(Mapping[object, object], value)
        return any(
            (isinstance(key, str) and key.lower() in forbidden) or _contains_forbidden_key(item)
            for key, item in typed.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_forbidden_key(item) for item in cast(Sequence[object], value))
    return False


class _ResponseSchema(Protocol):
    @classmethod
    def from_mapping(cls, value: object) -> object: ...
