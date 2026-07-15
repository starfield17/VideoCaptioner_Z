from __future__ import annotations

from tests.support import make_transcript

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
    assert [cue.source_text for cue in track.cues] == ["Hello, world. 这是测试"]
    assert track.cues[0].source_word_ids == tuple(word.id for word in transcript.words)


def test_segmentation_terminates_for_one_oversized_word_and_is_deterministic() -> None:
    transcript = make_transcript(("x" * 100,))
    config = SimpleSegmentationConfig(max_duration_ms=1, max_text_units=4)
    first = segment_transcript(transcript, config)
    second = segment_transcript(transcript, config)
    assert len(first.cues) == 1
    assert first == second
