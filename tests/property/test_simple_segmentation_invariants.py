from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st
from tests.support import make_transcript

from captioner.core.policies.segmentation import canonical_words
from captioner.core.policies.simple_segmentation import SimpleSegmentationConfig, segment_transcript


@given(st.lists(st.integers(min_value=1, max_value=12), min_size=1, max_size=20))
def test_segmentation_assigns_every_word_once_and_never_overlaps(lengths: list[int]) -> None:
    texts = tuple("x" * length + " " for length in lengths)
    transcript = make_transcript(texts)
    config = SimpleSegmentationConfig(max_duration_ms=800, max_text_units=20, hard_gap_ms=700)
    track = segment_transcript(transcript, config)
    assigned = [word_id for cue in track.cues for word_id in cue.source_word_ids]
    assert assigned == [word.id for word in transcript.words]
    assert len(set(assigned)) == len(assigned)
    assert all(
        left.end_ms <= right.start_ms
        for left, right in zip(track.cues, track.cues[1:], strict=False)
    )
    assert all(
        len(cue.source_word_ids) == 1
        or (
            cue.end_ms - cue.start_ms <= config.max_duration_ms
            and len(cue.source_text) <= config.max_text_units
        )
        for cue in track.cues
    )
    assert segment_transcript(transcript, config) == segment_transcript(transcript, config)
    assert [word_id for cue in track.cues for word_id in cue.source_word_ids] == [
        word.id for word in canonical_words(transcript.words)
    ]
