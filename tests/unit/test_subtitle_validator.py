from __future__ import annotations

from dataclasses import replace

from tests.support import make_transcript

from captioner.core.domain.subtitle import SubtitleCue, SubtitleTrack, derive_subtitle_track_id
from captioner.core.domain.subtitle_validation import validate_subtitle_track
from captioner.core.domain.transcript import Transcript, TranscriptSegment, WordToken
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.simple_segmentation import segment_transcript


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


def test_validator_warns_when_a_cue_break_splits_a_protected_unit() -> None:
    words = (
        WordToken("word-000001", "10 ", 0, 1_000),
        WordToken("word-000002", "kg", 1_000, 2_000),
    )
    transcript = Transcript(
        "transcript-units",
        "en",
        words,
        (
            TranscriptSegment(
                "segment-000001", ("word-000001", "word-000002"), "10 kg", 0, 2_000, None
            ),
        ),
        "fake",
        "model",
        {},
    )
    config = SegmentationPolicyConfig()
    cues = (
        SubtitleCue("cue-000001", 0, 1_000, ("word-000001",), "10", None, ("10",)),
        SubtitleCue("cue-000002", 1_000, 2_000, ("word-000002",), "kg", None, ("kg",)),
    )
    track = SubtitleTrack(
        derive_subtitle_track_id(transcript.id, transcript.language, cues, config.to_mapping()),
        transcript.id,
        transcript.language,
        cues,
        0,
        config.signature,
    )
    report = validate_subtitle_track(track, transcript, config)
    assert any(issue.code == "subtitle.protected_span_broken" for issue in report.issues)


def test_validator_rejects_policy_signature_and_track_id_mismatch() -> None:
    transcript = make_transcript(("hello ", "world"))
    config = SegmentationPolicyConfig()
    track = segment_transcript(transcript, config)
    altered = replace(track, id="track-arbitrary", policy_signature="policy-" + "0" * 64)
    report = validate_subtitle_track(altered, transcript, config)
    assert {issue.code for issue in report.issues} >= {
        "subtitle.policy_signature_invalid",
        "subtitle.track_id_invalid",
    }


def test_validator_rejects_language_mismatch() -> None:
    transcript = make_transcript(("hello ", "world"), language="en")
    config = SegmentationPolicyConfig()
    track = segment_transcript(transcript, config)
    altered = replace(track, language="zh-CN")
    report = validate_subtitle_track(altered, transcript, config)
    assert any(issue.code == "subtitle.language_mismatch" for issue in report.issues)


def test_validator_rejects_reversed_word_order() -> None:
    transcript = make_transcript(("one ", "two ", "three"))
    config = SegmentationPolicyConfig()
    cues = (
        SubtitleCue("cue-000001", 0, 1_100, ("word-000002",), "two", None, ("two",)),
        SubtitleCue("cue-000002", 1_100, 1_600, ("word-000001",), "one", None, ("one",)),
        SubtitleCue("cue-000003", 1_600, 2_200, ("word-000003",), "three", None, ("three",)),
    )
    track = SubtitleTrack(
        derive_subtitle_track_id(transcript.id, transcript.language, cues, config.to_mapping()),
        transcript.id,
        transcript.language,
        cues,
        0,
        config.signature,
    )
    report = validate_subtitle_track(track, transcript, config)
    assert any(issue.code == "subtitle.word_order_invalid" for issue in report.issues)


def test_validator_rejects_noncontiguous_word_span() -> None:
    transcript = make_transcript(("one ", "two ", "three"))
    config = SegmentationPolicyConfig()
    cues = (
        SubtitleCue(
            "cue-000001",
            0,
            1_100,
            ("word-000001", "word-000003"),
            "one three",
            None,
            ("one three",),
        ),
        SubtitleCue("cue-000002", 1_100, 1_700, ("word-000002",), "two", None, ("two",)),
    )
    track = SubtitleTrack(
        derive_subtitle_track_id(transcript.id, transcript.language, cues, config.to_mapping()),
        transcript.id,
        transcript.language,
        cues,
        0,
        config.signature,
    )
    report = validate_subtitle_track(track, transcript, config)
    assert any(issue.code == "subtitle.word_span_noncontiguous" for issue in report.issues)
