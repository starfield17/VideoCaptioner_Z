"""Immutable subtitle track domain models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

from captioner.core.domain.errors import AppError


def _text(value: str, field: str) -> None:
    if not value.strip():
        raise AppError("subtitle.invalid", {"field": field, "reason": "empty"})


def _integer_ms(value: object, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AppError("subtitle.invalid", {"field": field, "reason": "integer_ms"})


def _integer_revision(value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AppError("subtitle.invalid", {"field": "revision", "reason": "integer"})


def _time_range(start_ms: int, end_ms: int) -> None:
    _integer_ms(start_ms, "start_ms")
    _integer_ms(end_ms, "end_ms")
    if start_ms < 0 or end_ms <= start_ms:
        raise AppError(
            "subtitle.invalid",
            {"reason": "timestamp", "start_ms": start_ms, "end_ms": end_ms},
        )


@dataclass(frozen=True, slots=True)
class SubtitleCue:
    id: str
    start_ms: int
    end_ms: int
    source_word_ids: tuple[str, ...]
    source_text: str
    translated_text: str | None
    lines: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _time_range(self.start_ms, self.end_ms)
        word_ids = tuple(self.source_word_ids)
        if not word_ids or len(set(word_ids)) != len(word_ids):
            raise AppError(
                "subtitle.invalid",
                {"field": "source_word_ids", "reason": "duplicate_or_empty"},
            )
        object.__setattr__(self, "source_word_ids", word_ids)
        _text(self.source_text, "source_text")
        if self.translated_text is not None:
            raise AppError(
                "subtitle.invalid", {"field": "translated_text", "reason": "phase1_forbidden"}
            )
        lines = tuple(self.lines)
        if not lines or any(not line.strip() for line in lines):
            raise AppError("subtitle.invalid", {"field": "lines", "reason": "empty"})
        object.__setattr__(self, "lines", lines)


@dataclass(frozen=True, slots=True)
class SubtitleTrack:
    id: str
    source_transcript_id: str
    language: str
    cues: tuple[SubtitleCue, ...]
    revision: int

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _text(self.source_transcript_id, "source_transcript_id")
        _text(self.language, "language")
        cues = tuple(self.cues)
        _integer_revision(self.revision)
        if self.revision != 0:
            raise AppError(
                "subtitle.invalid", {"field": "revision", "reason": "phase1_must_be_zero"}
            )
        cue_ids = [cue.id for cue in cues]
        if len(set(cue_ids)) != len(cue_ids):
            raise AppError("subtitle.invalid", {"field": "cues", "reason": "duplicate_ids"})
        assigned: set[str] = set()
        for previous, current in pairwise(cues):
            if current.start_ms < previous.end_ms:
                raise AppError("subtitle.invalid", {"field": "cues", "reason": "overlap_or_order"})
        for cue in cues:
            for word_id in cue.source_word_ids:
                if word_id in assigned:
                    raise AppError(
                        "subtitle.invalid",
                        {"field": "source_word_ids", "reason": "multiple_assignment"},
                    )
                assigned.add(word_id)
        object.__setattr__(self, "cues", cues)


def derive_subtitle_track_id(
    transcript_id: str,
    language: str,
    cues: Sequence[SubtitleCue],
    config: Mapping[str, int],
) -> str:
    """Derive a stable track ID from source transcript, config and cue content."""
    payload = {
        "transcript_id": transcript_id,
        "language": language,
        "config": dict(config),
        "cues": [
            {
                "id": cue.id,
                "start_ms": cue.start_ms,
                "end_ms": cue.end_ms,
                "source_word_ids": list(cue.source_word_ids),
                "source_text": cue.source_text,
                "translated_text": cue.translated_text,
                "lines": list(cue.lines),
            }
            for cue in cues
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"track-{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"
