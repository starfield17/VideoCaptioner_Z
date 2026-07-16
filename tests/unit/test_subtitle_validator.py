from __future__ import annotations

from tests.support import make_transcript

from captioner.core.domain.subtitle import SubtitleCue, SubtitleTrack, derive_subtitle_track_id
from captioner.core.domain.subtitle_validation import validate_subtitle_track
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig


def test_validator_reports_missing_word_without_mutating_track() -> None:
    transcript = make_transcript(("hello ", "world"))
    config = SegmentationPolicyConfig()
    cue = SubtitleCue("cue-000001", 0, 500, ("word-000001",), "hello", None, ("hello",))
    track = SubtitleTrack(
        derive_subtitle_track_id(transcript.id, transcript.language, (cue,), config.to_mapping()),
        transcript.id,
        transcript.language,
        (cue,),
        0,
        config.signature,
    )
    before = track
    report = validate_subtitle_track(track, transcript, config)
    assert not report.is_valid
    assert any(issue.code == "subtitle.word_missing" for issue in report.issues)
    assert track == before
