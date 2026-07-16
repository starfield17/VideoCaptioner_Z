"""Immutable integer-millisecond transcript domain models."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.result import (
    FrozenJsonValue,
    JsonValue,
    freeze_json_value,
    thaw_json_value,
)


def _text(value: str, field: str) -> None:
    if not value.strip():
        raise AppError("transcript.invalid", {"field": field, "reason": "empty"})


def _integer_ms(value: object, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AppError("transcript.invalid", {"field": field, "reason": "integer_ms"})


def _time_range(start_ms: int, end_ms: int, field: str) -> None:
    _integer_ms(start_ms, f"{field}.start_ms")
    _integer_ms(end_ms, f"{field}.end_ms")
    if start_ms < 0 or end_ms <= start_ms:
        raise AppError(
            "transcript.invalid",
            {"field": field, "reason": "timestamp", "start_ms": start_ms, "end_ms": end_ms},
        )


def _confidence(value: object, field: str) -> None:
    if value is not None and (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise AppError("transcript.invalid", {"field": field, "reason": "confidence"})


def _metadata(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    try:
        frozen = cast(Mapping[str, FrozenJsonValue], freeze_json_value(value))
    except (TypeError, ValueError) as exc:
        raise AppError("transcript.invalid", {"field": "metadata", "reason": str(exc)}) from exc
    return cast(Mapping[str, JsonValue], frozen)


@dataclass(frozen=True, slots=True)
class WordToken:
    id: str
    text: str
    start_ms: int
    end_ms: int
    confidence: float | None = None
    speaker_id: str | None = None

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _text(self.text, "text")
        _time_range(self.start_ms, self.end_ms, "word")
        _confidence(self.confidence, "confidence")


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    id: str
    word_ids: tuple[str, ...]
    raw_text: str
    start_ms: int
    end_ms: int
    confidence: float | None

    def __post_init__(self) -> None:
        _text(self.id, "id")
        word_ids = tuple(self.word_ids)
        if (
            not word_ids
            or len(set(word_ids)) != len(word_ids)
            or any(not item.strip() for item in word_ids)
        ):
            raise AppError(
                "transcript.invalid", {"field": "word_ids", "reason": "duplicate_or_empty"}
            )
        object.__setattr__(self, "word_ids", word_ids)
        _text(self.raw_text, "raw_text")
        _time_range(self.start_ms, self.end_ms, "segment")
        _confidence(self.confidence, "confidence")


@dataclass(frozen=True, slots=True)
class Transcript:
    id: str
    language: str
    words: tuple[WordToken, ...]
    segments: tuple[TranscriptSegment, ...]
    engine_id: str
    model_id: str
    metadata: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _text(self.language, "language")
        _text(self.engine_id, "engine_id")
        _text(self.model_id, "model_id")
        words = tuple(self.words)
        segments = tuple(self.segments)
        if not words:
            raise AppError("transcript.invalid", {"field": "words", "reason": "empty"})
        if not segments:
            raise AppError("transcript.invalid", {"field": "segments", "reason": "empty"})
        word_ids = [word.id for word in words]
        segment_ids = [segment.id for segment in segments]
        if len(set(word_ids)) != len(word_ids):
            raise AppError("transcript.invalid", {"field": "words", "reason": "duplicate_ids"})
        if len(set(segment_ids)) != len(segment_ids):
            raise AppError("transcript.invalid", {"field": "segments", "reason": "duplicate_ids"})
        for previous, current in pairwise(words):
            if current.start_ms < previous.start_ms or current.start_ms < previous.end_ms:
                raise AppError(
                    "transcript.invalid", {"field": "words", "reason": "overlap_or_order"}
                )
        for previous, current in pairwise(segments):
            if current.start_ms < previous.start_ms or current.start_ms < previous.end_ms:
                raise AppError(
                    "transcript.invalid", {"field": "segments", "reason": "overlap_or_order"}
                )
        by_id = {word.id: word for word in words}
        assigned: set[str] = set()
        for segment in segments:
            for word_id in segment.word_ids:
                if word_id not in by_id:
                    raise AppError(
                        "transcript.invalid", {"field": "word_ids", "reason": "missing_reference"}
                    )
                if word_id in assigned:
                    raise AppError(
                        "transcript.invalid", {"field": "word_ids", "reason": "multiple_assignment"}
                    )
                word = by_id[word_id]
                if word.start_ms < segment.start_ms or word.end_ms > segment.end_ms:
                    raise AppError(
                        "transcript.invalid",
                        {"field": "word_ids", "reason": "word_outside_segment"},
                    )
                assigned.add(word_id)
        if assigned != set(word_ids):
            raise AppError("transcript.invalid", {"field": "word_ids", "reason": "unassigned_word"})
        object.__setattr__(self, "words", words)
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "metadata", _metadata(self.metadata))


def derive_transcript_id(
    *,
    language: str,
    words: Sequence[WordToken],
    segments: Sequence[TranscriptSegment],
    engine_id: str,
    model_id: str,
    metadata: Mapping[str, JsonValue],
) -> str:
    """Derive an ID from canonical transcript content, excluding source paths."""
    payload: dict[str, object] = {
        "language": language,
        "engine_id": engine_id,
        "model_id": model_id,
        "words": [_word_to_dict(word) for word in words],
        "segments": [_segment_to_dict(segment) for segment in segments],
        "metadata": thaw_json_value(freeze_json_value(metadata)),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"transcript-{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


def _word_to_dict(word: WordToken) -> dict[str, object]:
    return {
        "id": word.id,
        "text": word.text,
        "start_ms": word.start_ms,
        "end_ms": word.end_ms,
        "confidence": word.confidence,
        "speaker_id": word.speaker_id,
    }


def _segment_to_dict(segment: TranscriptSegment) -> dict[str, object]:
    return {
        "id": segment.id,
        "word_ids": list(segment.word_ids),
        "raw_text": segment.raw_text,
        "start_ms": segment.start_ms,
        "end_ms": segment.end_ms,
        "confidence": segment.confidence,
    }
