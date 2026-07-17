from __future__ import annotations

import pytest
from tests.support import make_transcript

from captioner.adapters.persistence.domain_codecs import decode_track, encode_track
from captioner.core.domain.subtitle import SubtitleCue, SubtitleTrack, derive_subtitle_track_id
from captioner.core.domain.subtitle_validation import validate_translated_track
from captioner.core.policies.line_breaking import break_lines
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig


def _track(
    translated_text: str = "你好 10",
    *,
    language: str = "zh-CN",
    lines: tuple[str, ...] | None = None,
) -> SubtitleTrack:
    transcript = make_transcript(("hello 10",))
    config = SegmentationPolicyConfig()
    cue = SubtitleCue(
        "cue-000001",
        0,
        500,
        ("word-000001",),
        "hello 10",
        translated_text,
        break_lines(translated_text, config) if lines is None else lines,
    )
    return SubtitleTrack(
        derive_subtitle_track_id(transcript.id, language, (cue,), config.to_mapping()),
        transcript.id,
        language,
        (cue,),
        1,
        config.signature,
    )


def test_translated_track_round_trip_keeps_source_mapping_and_schema_v3() -> None:
    transcript = make_transcript(("hello 10",))
    config = SegmentationPolicyConfig()
    track = _track()
    report = validate_translated_track(track, transcript, config, "zh-CN")
    assert report.is_valid
    decoded = decode_track(encode_track(track))
    assert decoded == track
    assert decoded.cues[0].start_ms == 0
    assert decoded.cues[0].end_ms == 500
    assert decoded.cues[0].source_word_ids == ("word-000001",)


@pytest.mark.parametrize(
    ("track", "code"),
    [
        (_track("你好", language="zh-CN"), "subtitle.protected_token_lost"),
        (_track("hello 10", language="zh-CN"), "subtitle.language_mismatch"),
        (_track("你好 10", language="zh-CN", lines=("10 你好",)), "subtitle.lines_not_derived"),
    ],
)
def test_translated_track_rejects_display_contract_violations(
    track: SubtitleTrack, code: str
) -> None:
    transcript = make_transcript(("hello 10",))
    report = validate_translated_track(track, transcript, SegmentationPolicyConfig(), "zh-CN")
    assert any(issue.code == code for issue in report.issues)
