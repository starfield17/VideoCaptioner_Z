from __future__ import annotations

import pytest

from captioner.adapters.subtitles import webvtt
from captioner.core.domain.errors import AppError
from captioner.core.domain.subtitle import SubtitleCue, SubtitleTrack


def _track() -> SubtitleTrack:
    return SubtitleTrack(
        "track-1",
        "transcript-1",
        "en",
        (
            SubtitleCue(
                "cue-000001",
                1_000,
                3_500,
                ("word-1",),
                "hello world",
                None,
                ("hello", "world"),
            ),
        ),
        0,
    )


def test_webvtt_round_trip_is_exact_and_uses_lf() -> None:
    data = webvtt.serialize_bytes(_track())
    assert data.startswith(b"WEBVTT\n\n")
    assert b"\r" not in data
    parsed = webvtt.parse(data)
    assert parsed.cues[0].start_ms == 1_000
    assert parsed.cues[0].end_ms == 3_500
    assert parsed.cues[0].lines == ("hello", "world")
    assert webvtt.serialize_bytes(_track()) == data


def test_webvtt_escapes_plain_text_and_rejects_bad_timestamp() -> None:
    track = SubtitleTrack(
        "track-1",
        "transcript-1",
        "en",
        (SubtitleCue("cue-000001", 0, 100, ("word-1",), "a & b", None, ("a & b",)),),
        0,
    )
    data = webvtt.serialize_bytes(track)
    assert b"a &amp; b" in data
    assert webvtt.parse(data).cues[0].lines == ("a & b",)
    with pytest.raises(AppError, match=r"export\.webvtt_invalid"):
        webvtt.parse(b"WEBVTT\n\n00:99:00.000 --> 00:00:01.000\nx\n")
