"""Small deterministic Phase 1 transcript-to-cue policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from captioner.core.domain.errors import AppError
from captioner.core.domain.subtitle import (
    SubtitleCue,
    SubtitleTrack,
    derive_subtitle_track_id,
)
from captioner.core.domain.transcript import Transcript, TranscriptSegment, WordToken

_PUNCTUATION = frozenset(".,!?;:\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001")


@dataclass(frozen=True, slots=True)
class SimpleSegmentationConfig:
    max_duration_ms: int = 7_000
    max_text_units: int = 84
    hard_gap_ms: int = 700

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> SimpleSegmentationConfig:
        expected = {"max_duration_ms", "max_text_units", "hard_gap_ms"}
        if set(values) != expected:
            raise AppError("job.config_invalid", {"field": "segmentation"})
        max_duration_ms = values["max_duration_ms"]
        max_text_units = values["max_text_units"]
        hard_gap_ms = values["hard_gap_ms"]
        if (
            not isinstance(max_duration_ms, int)
            or isinstance(max_duration_ms, bool)
            or not isinstance(max_text_units, int)
            or isinstance(max_text_units, bool)
            or not isinstance(hard_gap_ms, int)
            or isinstance(hard_gap_ms, bool)
        ):
            raise AppError("job.config_invalid", {"field": "segmentation"})
        try:
            result = cls(
                max_duration_ms,
                max_text_units,
                hard_gap_ms,
            )
        except (TypeError, ValueError) as exc:
            raise AppError("job.config_invalid", {"field": "segmentation"}) from exc
        if result.max_duration_ms < 1 or result.max_text_units < 1 or result.hard_gap_ms < 0:
            raise AppError("job.config_invalid", {"field": "segmentation"})
        return result

    def __post_init__(self) -> None:
        if self.max_duration_ms <= 0 or self.max_text_units <= 0 or self.hard_gap_ms < 0:
            raise ValueError


def segment_transcript(
    transcript: Transcript,
    config: SimpleSegmentationConfig | None = None,
) -> SubtitleTrack:
    """Segment each source transcript segment greedily at word boundaries."""
    settings = SimpleSegmentationConfig() if config is None else config
    words_by_id = {word.id: word for word in transcript.words}
    cues: list[SubtitleCue] = []
    next_cue_number = 1
    assigned: set[str] = set()
    for segment in transcript.segments:
        words = _resolve_words(segment, words_by_id)
        while words:
            end = _choose_end(words, 0, settings)
            selected = words[:end]
            source_text = _join_words(selected)
            word_ids = tuple(word.id for word in selected)
            if not source_text or assigned.intersection(word_ids):
                raise AppError("subtitle.segmentation_failed", {"segment_id": segment.id})
            assigned.update(word_ids)
            cues.append(
                SubtitleCue(
                    id=f"cue-{next_cue_number:06d}",
                    start_ms=selected[0].start_ms,
                    end_ms=selected[-1].end_ms,
                    source_word_ids=word_ids,
                    source_text=source_text,
                    translated_text=None,
                    lines=(source_text,),
                )
            )
            next_cue_number += 1
            words = words[end:]

    expected = {word.id for word in transcript.words}
    if assigned != expected:
        raise AppError("subtitle.segmentation_failed", {"reason": "dropped_word"})
    config_values = {
        "max_duration_ms": settings.max_duration_ms,
        "max_text_units": settings.max_text_units,
        "hard_gap_ms": settings.hard_gap_ms,
    }
    track_id = derive_subtitle_track_id(transcript.id, transcript.language, cues, config_values)
    return SubtitleTrack(
        id=track_id,
        source_transcript_id=transcript.id,
        language=transcript.language,
        cues=tuple(cues),
        revision=0,
    )


def _resolve_words(
    segment: TranscriptSegment, words_by_id: Mapping[str, WordToken]
) -> list[WordToken]:
    try:
        words = [words_by_id[word_id] for word_id in segment.word_ids]
    except KeyError as exc:
        raise AppError("subtitle.segmentation_failed", {"reason": "missing_word"}) from exc
    return sorted(words, key=lambda word: (word.start_ms, word.end_ms, word.id))


def _choose_end(words: list[WordToken], start: int, config: SimpleSegmentationConfig) -> int:
    fit_end = start
    for end in range(start + 1, len(words) + 1):
        candidate = words[start:end]
        text = _join_words(candidate)
        duration = candidate[-1].end_ms - candidate[0].start_ms
        if end > start + 1 and (
            duration > config.max_duration_ms or len(text) > config.max_text_units
        ):
            break
        fit_end = end
        if duration > config.max_duration_ms or len(text) > config.max_text_units:
            break
    if fit_end <= start:
        return start + 1

    if fit_end == len(words):
        return fit_end

    preferred: list[int] = []
    for end in range(start + 1, fit_end + 1):
        current = words[end - 1]
        punctuation = current.text.rstrip().endswith(tuple(_PUNCTUATION))
        silence = end < len(words) and words[end].start_ms - current.end_ms >= config.hard_gap_ms
        if punctuation or silence:
            preferred.append(end)
    if preferred:
        return preferred[-1]
    return fit_end


def _join_words(words: list[WordToken]) -> str:
    return "".join(word.text for word in words).strip()
