"""Compatibility facade for the Phase 2 segmentation API."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from captioner.core.domain.errors import AppError
from captioner.core.domain.subtitle import SubtitleCue, SubtitleTrack, derive_subtitle_track_id
from captioner.core.domain.transcript import Transcript, TranscriptSegment, WordToken
from captioner.core.policies.segmentation import segment_transcript_dp
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig

__all__ = ["SegmentationPolicyConfig", "SimpleSegmentationConfig", "segment_transcript"]


@dataclass(frozen=True, slots=True)
class SimpleSegmentationConfig:
    """Legacy three-field constructor mapped to the complete Phase 3 policy."""

    max_duration_ms: int = 7_000
    max_text_units: int = 84
    hard_gap_ms: int = 700
    policy: SegmentationPolicyConfig | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> SimpleSegmentationConfig:
        if set(values) != {"max_duration_ms", "max_text_units", "hard_gap_ms"}:
            policy = SegmentationPolicyConfig.from_mapping(values)
            return cls(policy.max_duration_ms, policy.max_cue_width, policy.hard_gap_ms, policy)
        max_duration_ms = values.get("max_duration_ms")
        max_text_units = values.get("max_text_units")
        hard_gap_ms = values.get("hard_gap_ms")
        if (
            not isinstance(max_duration_ms, int)
            or isinstance(max_duration_ms, bool)
            or not isinstance(max_text_units, int)
            or isinstance(max_text_units, bool)
            or not isinstance(hard_gap_ms, int)
            or isinstance(hard_gap_ms, bool)
            or max_duration_ms <= 0
            or max_text_units <= 0
            or hard_gap_ms < 0
        ):
            raise AppError("job.config_invalid", {"field": "segmentation"})
        return cls(max_duration_ms, max_text_units, hard_gap_ms)

    def __post_init__(self) -> None:
        if self.max_duration_ms <= 0 or self.max_text_units <= 0 or self.hard_gap_ms < 0:
            raise ValueError

    def to_policy_config(self) -> SegmentationPolicyConfig:
        if self.policy is not None:
            return self.policy
        return SegmentationPolicyConfig.from_mapping(
            {
                "max_duration_ms": self.max_duration_ms,
                "max_text_units": self.max_text_units,
                "hard_gap_ms": self.hard_gap_ms,
            }
        )


def segment_transcript(
    transcript: Transcript,
    config: SimpleSegmentationConfig | SegmentationPolicyConfig | None = None,
    progress: Callable[[], None] | None = None,
) -> SubtitleTrack:
    if config is None:
        settings = SegmentationPolicyConfig()
    elif isinstance(config, SimpleSegmentationConfig):
        if config.policy is None:
            return _legacy_segment_transcript(transcript, config, progress)
        settings = config.policy
    else:
        settings = config
    return segment_transcript_dp(transcript, settings, progress)


def _legacy_segment_transcript(
    transcript: Transcript,
    config: SimpleSegmentationConfig,
    progress: Callable[[], None] | None,
) -> SubtitleTrack:
    words_by_id = {word.id: word for word in transcript.words}
    cues: list[SubtitleCue] = []
    assigned: set[str] = set()
    number = 1
    for segment_index, segment in enumerate(transcript.segments):
        words = _resolve_words(segment, words_by_id)
        while words:
            end = _choose_end(words, config)
            selected = words[:end]
            source_text = "".join(word.text for word in selected).strip()
            word_ids = tuple(word.id for word in selected)
            if not source_text or assigned.intersection(word_ids):
                raise AppError("subtitle.segmentation_failed", {"segment_id": segment.id})
            assigned.update(word_ids)
            cues.append(
                SubtitleCue(
                    id=f"cue-{number:06d}",
                    start_ms=selected[0].start_ms,
                    end_ms=selected[-1].end_ms,
                    source_word_ids=word_ids,
                    source_text=source_text,
                    translated_text=None,
                    lines=(source_text,),
                )
            )
            number += 1
            if progress is not None and (
                words[end:] or segment_index < len(transcript.segments) - 1
            ):
                progress()
            words = words[end:]
    if assigned != {word.id for word in transcript.words}:
        raise AppError("subtitle.segmentation_failed", {"reason": "dropped_word"})
    config_values = {
        "max_duration_ms": config.max_duration_ms,
        "max_text_units": config.max_text_units,
        "hard_gap_ms": config.hard_gap_ms,
    }
    return SubtitleTrack(
        id=derive_subtitle_track_id(transcript.id, transcript.language, cues, config_values),
        source_transcript_id=transcript.id,
        language=transcript.language,
        cues=tuple(cues),
        revision=0,
    )


def _resolve_words(
    segment: TranscriptSegment, words_by_id: Mapping[str, WordToken]
) -> list[WordToken]:
    try:
        return sorted(
            [words_by_id[word_id] for word_id in segment.word_ids],
            key=lambda word: (word.start_ms, word.end_ms, word.id),
        )
    except KeyError as exc:
        raise AppError("subtitle.segmentation_failed", {"reason": "missing_word"}) from exc


def _choose_end(words: list[WordToken], config: SimpleSegmentationConfig) -> int:
    fit_end = 0
    for end in range(1, len(words) + 1):
        candidate = words[:end]
        text = "".join(word.text for word in candidate).strip()
        duration = candidate[-1].end_ms - candidate[0].start_ms
        if end > 1 and (duration > config.max_duration_ms or len(text) > config.max_text_units):
            break
        fit_end = end
        if duration > config.max_duration_ms or len(text) > config.max_text_units:
            break
    if fit_end == 0:
        return 1
    if fit_end == len(words):
        return fit_end
    preferred: list[int] = []
    for end in range(1, fit_end + 1):
        current = words[end - 1]
        punctuation = current.text.rstrip().endswith(
            tuple(".,!?;:\uff0c\u3002\uff01\uff1f\uff1b\uff1a")
        )
        silence = (
            end < len(words)
            and words[end].start_ms - current.end_ms >= config.hard_gap_ms
        )
        if punctuation or silence:
            preferred.append(end)
    return preferred[-1] if preferred else fit_end
