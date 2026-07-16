from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from tests.support import make_transcript

from captioner.core.domain.subtitle_validation import validate_subtitle_track
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.simple_segmentation import segment_transcript


@settings(deadline=None)
@given(st.lists(st.integers(min_value=1, max_value=12), min_size=1, max_size=24))
def test_every_generated_word_is_assigned_once(lengths: list[int]) -> None:
    transcript = make_transcript(tuple("x" * length + " " for length in lengths))
    config = SegmentationPolicyConfig()
    track = segment_transcript(transcript, config)
    assigned = [word_id for cue in track.cues for word_id in cue.source_word_ids]
    assert assigned == [word.id for word in transcript.words]
    assert validate_subtitle_track(track, transcript, config).is_valid


def test_track_id_changes_for_policy_and_source_identity() -> None:
    transcript = make_transcript()
    first = segment_transcript(transcript)
    changed_policy = segment_transcript(
        transcript,
        SegmentationPolicyConfig(max_line_width=40, max_cue_width=80),
    )
    changed_source = segment_transcript(make_transcript(language="zh-CN"))
    assert first.id != changed_policy.id
    assert first.id != changed_source.id
