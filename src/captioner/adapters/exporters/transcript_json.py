"""Deterministic UTF-8 Transcript JSON serialization."""

from __future__ import annotations

import json
from typing import cast

from captioner.core.domain.result import FrozenJsonValue, JsonValue, thaw_json_value
from captioner.core.domain.transcript import Transcript


def transcript_to_dict(transcript: Transcript) -> dict[str, JsonValue]:
    """Return the documented schema without mutating the domain object."""
    words: list[JsonValue] = []
    for word in transcript.words:
        words.append(
            {
                "id": word.id,
                "text": word.text,
                "start_ms": word.start_ms,
                "end_ms": word.end_ms,
                "confidence": word.confidence,
                "speaker_id": word.speaker_id,
            }
        )
    segments: list[JsonValue] = []
    for segment in transcript.segments:
        segments.append(
            {
                "id": segment.id,
                "word_ids": list(segment.word_ids),
                "raw_text": segment.raw_text,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "confidence": segment.confidence,
            }
        )
    transcript_value: dict[str, JsonValue] = {
        "id": transcript.id,
        "language": transcript.language,
        "engine_id": transcript.engine_id,
        "model_id": transcript.model_id,
        "words": words,
        "segments": segments,
        "metadata": thaw_json_value(cast(FrozenJsonValue, transcript.metadata)),
    }
    return {"schema_version": 1, "transcript": transcript_value}


def serialize(transcript: Transcript) -> str:
    """Serialize one transcript with stable ordering and a final newline."""
    return (
        json.dumps(
            transcript_to_dict(transcript),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def serialize_bytes(transcript: Transcript) -> bytes:
    return serialize(transcript).encode("utf-8")
