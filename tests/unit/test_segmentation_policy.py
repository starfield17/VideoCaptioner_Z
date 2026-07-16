from __future__ import annotations

import pytest

from captioner.core.domain.errors import AppError
from captioner.core.domain.transcript import Transcript, TranscriptSegment, WordToken
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.simple_segmentation import segment_transcript


def test_legacy_configuration_is_completed_deterministically() -> None:
    policy = SegmentationPolicyConfig.from_mapping(
        {"max_duration_ms": 7_000, "max_text_units": 84, "hard_gap_ms": 700}
    )
    assert policy.max_cue_width == 84
    assert policy.to_mapping()["schema_version"] == 1
    assert policy.signature == SegmentationPolicyConfig.from_mapping(policy.to_mapping()).signature


def test_policy_rejects_partial_unknown_configuration() -> None:
    with pytest.raises(AppError, match=r"job\.config_invalid"):
        SegmentationPolicyConfig.from_mapping({"max_duration_ms": 7_000})


def _transcript(texts: tuple[str, ...], starts: tuple[int, ...]) -> Transcript:
    words = tuple(
        WordToken(f"word-{index:06d}", text, start, start + 100)
        for index, (text, start) in enumerate(zip(texts, starts, strict=True), start=1)
    )
    segment = TranscriptSegment(
        "segment-000001",
        tuple(word.id for word in words),
        "".join(texts).strip(),
        min(word.start_ms for word in words),
        max(word.end_ms for word in words),
        None,
    )
    return Transcript("segmentation-policy", "en", words, (segment,), "fake", "model", {})


def test_hard_silence_forces_a_boundary() -> None:
    transcript = _transcript(("one ", "two ", "three"), (0, 100, 1_000))
    track = segment_transcript(transcript, SegmentationPolicyConfig())
    assert [cue.source_text for cue in track.cues] == ["one two", "three"]


def test_preferred_silence_and_sentence_punctuation_affect_ties() -> None:
    config = SegmentationPolicyConfig(
        min_duration_ms=1,
        target_duration_ms=200,
        max_duration_ms=1_000,
        target_cps_milli=100_000,
        max_cps_milli=100_000,
        max_line_width=42,
        max_cue_width=84,
    )
    punctuation = _transcript(("Hello. ", "next"), (0, 100))
    no_punctuation = _transcript(("Hello ", "next"), (0, 100))
    assert len(segment_transcript(punctuation, config).cues) == 2
    assert len(segment_transcript(no_punctuation, config).cues) == 1

    silence = _transcript(("one ", "two ", "three"), (0, 400, 500))
    assert [cue.source_text for cue in segment_transcript(silence, config).cues] == [
        "one",
        "two three",
    ]
