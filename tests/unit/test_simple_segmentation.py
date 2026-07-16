from __future__ import annotations

from tests.support import make_transcript

from captioner.core.domain.transcript import Transcript, TranscriptSegment, WordToken
from captioner.core.policies.simple_segmentation import (
    SimpleSegmentationConfig,
    segment_transcript,
)


def test_segmentation_splits_at_limits_and_preserves_token_whitespace() -> None:
    transcript = make_transcript(("Hello, ", "world. ", "这是", "测试"), language="zh-CN")
    track = segment_transcript(
        transcript,
        SimpleSegmentationConfig(max_duration_ms=3_000, max_text_units=20, hard_gap_ms=700),
    )
    completed = SimpleSegmentationConfig(
        max_duration_ms=3_000, max_text_units=20, hard_gap_ms=700
    ).to_policy_config()
    assert track == segment_transcript(transcript, completed)
    assert [word_id for cue in track.cues for word_id in cue.source_word_ids] == [
        word.id for word in transcript.words
    ]


def test_segmentation_terminates_for_one_oversized_word_and_is_deterministic() -> None:
    transcript = make_transcript(("x" * 100,))
    config = SimpleSegmentationConfig(max_duration_ms=1, max_text_units=4)
    first = segment_transcript(transcript, config)
    second = segment_transcript(transcript, config)
    assert len(first.cues) == 1
    assert first == second


def _transcript_with_gaps(texts: tuple[str, ...], gaps: tuple[int, ...]) -> Transcript:
    words: list[WordToken] = []
    cursor = 0
    for number, text in enumerate(texts):
        word = WordToken(f"word-{number + 1:06d}", text, cursor, cursor + 100)
        words.append(word)
        cursor = word.end_ms + gaps[number]
    segment = TranscriptSegment(
        "segment-000001",
        tuple(word.id for word in words),
        "".join(texts).strip(),
        words[0].start_ms,
        words[-1].end_ms,
        None,
    )
    return Transcript(
        "transcript-segmentation",
        "en",
        tuple(words),
        (segment,),
        "fake-asr",
        "test-model",
        {},
    )


def test_oversized_candidate_prefers_punctuation_boundary() -> None:
    transcript = _transcript_with_gaps(("one, ", "two ", "three"), (100, 100, 0))
    track = segment_transcript(
        transcript,
        SimpleSegmentationConfig(max_duration_ms=10_000, max_text_units=10),
    )
    assert [cue.source_text for cue in track.cues] == ["one,", "two three"]


def test_oversized_candidate_prefers_latest_silence_boundary() -> None:
    transcript = _transcript_with_gaps(
        ("one ", "two ", "three ", "four ", "five"), (100, 100, 900, 100, 0)
    )
    track = segment_transcript(
        transcript,
        SimpleSegmentationConfig(max_duration_ms=10_000, max_text_units=19, hard_gap_ms=700),
    )
    assert [cue.source_text for cue in track.cues] == ["one two three", "four five"]


def test_latest_preferred_boundary_wins() -> None:
    transcript = _transcript_with_gaps(("one, ", "two? ", "three ", "four"), (100, 100, 100, 0))
    track = segment_transcript(
        transcript,
        SimpleSegmentationConfig(max_duration_ms=10_000, max_text_units=16),
    )
    assert track.cues[0].source_text == "one, two?"


def test_preferred_boundaries_outside_fitting_range_are_ignored() -> None:
    punctuation = _transcript_with_gaps(("one ", "two ", "three, ", "four"), (100, 100, 100, 0))
    silence = _transcript_with_gaps(("one ", "two ", "three ", "four"), (100, 100, 900, 0))
    config = SimpleSegmentationConfig(max_duration_ms=10_000, max_text_units=10, hard_gap_ms=700)
    assert segment_transcript(punctuation, config).cues[0].source_text == "one two"
    assert segment_transcript(silence, config).cues[0].source_text == "one"


def test_no_preferred_boundary_falls_back_to_latest_fitting_word() -> None:
    transcript = _transcript_with_gaps(("one ", "two ", "three"), (100, 100, 0))
    track = segment_transcript(
        transcript,
        SimpleSegmentationConfig(max_duration_ms=10_000, max_text_units=8),
    )
    assert [cue.source_text for cue in track.cues] == ["one", "two three"]


def test_all_remaining_words_fit_as_one_cue() -> None:
    transcript = make_transcript(("one, ", "two ", "three"))
    track = segment_transcript(
        transcript,
        SimpleSegmentationConfig(max_duration_ms=10_000, max_text_units=50),
    )
    assert [cue.source_text for cue in track.cues] == ["one,", "two three"]


def test_legacy_mapping_and_completed_policy_produce_identical_track() -> None:
    transcript = make_transcript(("one ", "two ", "three"))
    legacy_mapping = {
        "max_duration_ms": 7_000,
        "max_text_units": 84,
        "hard_gap_ms": 700,
    }
    legacy = SimpleSegmentationConfig.from_mapping(legacy_mapping)
    completed = legacy.to_policy_config()
    assert segment_transcript(transcript, legacy) == segment_transcript(transcript, completed)


def test_legacy_config_applies_line_breaking_and_has_policy_identity() -> None:
    transcript = make_transcript(("one ", "two ", "three ", "four ", "five"))
    config = SimpleSegmentationConfig(max_duration_ms=7_000, max_text_units=12, hard_gap_ms=700)
    track = segment_transcript(transcript, config)
    assert track.policy_signature == config.to_policy_config().signature
    assert all(1 <= len(cue.lines) <= 2 for cue in track.cues)
