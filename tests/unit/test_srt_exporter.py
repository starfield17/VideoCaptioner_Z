from __future__ import annotations

import pytest
from scripts.validate_subtitle import parse_srt
from tests.support import POLICY_SIGNATURE

from captioner.adapters.exporters.srt import format_timestamp, serialize
from captioner.core.domain.errors import AppError
from captioner.core.domain.subtitle import SubtitleCue, SubtitleTrack


def test_srt_supports_unicode_and_durations_above_one_hour() -> None:
    track = SubtitleTrack(
        "track-1",
        "transcript-1",
        "zh-CN",
        (SubtitleCue("cue-1", 3_723_456, 3_724_456, ("word-1",), "你好", None, ("你好",)),),
        0,
        POLICY_SIGNATURE,
    )
    rendered = serialize(track)
    assert "01:02:03,456 --> 01:02:04,456" in rendered
    assert rendered.endswith("\n")
    assert not rendered.endswith("\n\n")
    assert parse_srt(rendered)[0].text == ("你好",)
    assert format_timestamp(3_723_456) == "01:02:03,456"


def test_srt_validator_rejects_bad_input() -> None:
    for value in (
        "2\n00:00:00,000 --> 00:00:01,000\ntext\n",
        "1\n00:00:01,000 --> 00:00:00,000\ntext\n",
        "1\nnot a timestamp\ntext\n",
        "1\n00:00:00,000 --> 00:00:01,000\n\n2\n00:00:00,500 --> 00:00:02,000\ntext\n",
    ):
        with pytest.raises(ValueError):
            parse_srt(value)


def test_exporter_rejects_invalid_order_if_a_track_like_object_is_corrupted() -> None:
    cue = SubtitleCue("cue-1", 0, 100, ("word-1",), "text", None, ("text",))
    track = SubtitleTrack("track-1", "transcript-1", "en", (cue,), 0, POLICY_SIGNATURE)
    object.__setattr__(cue, "start_ms", -1)
    with pytest.raises(AppError, match="srt_invalid"):
        serialize(track)
