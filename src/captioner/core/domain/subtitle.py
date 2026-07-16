"""Immutable subtitle track domain models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.policies.unicode_metrics import normalize_text

LEGACY_POLICY_SIGNATURE = "legacy-policy-unknown"


def _text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
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
    translated_text: str | None = None
    lines: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _time_range(self.start_ms, self.end_ms)
        raw_word_ids = tuple(cast(tuple[object, ...], self.source_word_ids))
        if (
            not raw_word_ids
            or len(set(raw_word_ids)) != len(raw_word_ids)
            or any(not isinstance(word_id, str) or not word_id.strip() for word_id in raw_word_ids)
        ):
            raise AppError(
                "subtitle.invalid",
                {"field": "source_word_ids", "reason": "duplicate_or_empty"},
            )
        object.__setattr__(self, "source_word_ids", cast(tuple[str, ...], raw_word_ids))
        _text(self.source_text, "source_text")
        if normalize_text(self.source_text) != self.source_text:
            raise AppError("subtitle.invalid", {"field": "source_text", "reason": "not_canonical"})
        if self.translated_text is not None:
            raise AppError(
                "subtitle.invalid", {"field": "translated_text", "reason": "phase1_forbidden"}
            )
        raw_lines = tuple(cast(tuple[object, ...], self.lines))
        if (
            not raw_lines
            or len(raw_lines) > 2
            or any(
                not isinstance(line, str) or not line.strip() or "\n" in line or "\r" in line
                for line in raw_lines
            )
        ):
            raise AppError("subtitle.invalid", {"field": "lines", "reason": "empty"})
        lines = cast(tuple[str, ...], raw_lines)
        if any(normalize_text(line) != line for line in lines):
            raise AppError("subtitle.invalid", {"field": "lines", "reason": "not_canonical"})
        object.__setattr__(self, "lines", lines)


@dataclass(frozen=True, slots=True)
class SubtitleTrack:
    id: str
    source_transcript_id: str
    language: str | None
    cues: tuple[SubtitleCue, ...]
    revision: int = 0
    policy_signature: str = ""

    def __post_init__(self) -> None:
        _text(self.id, "id")
        _text(self.source_transcript_id, "source_transcript_id")
        if self.language is not None:
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
        _text(self.policy_signature, "policy_signature")


def derive_subtitle_track_id(
    transcript_id: str,
    language: str | None,
    cues: Sequence[SubtitleCue],
    config: Mapping[str, object],
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
