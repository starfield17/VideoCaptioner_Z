from __future__ import annotations

from captioner.adapters.subtitles import ass
from captioner.core.domain.subtitle import SubtitleCue, SubtitleTrack


def test_ass_round_trip_has_at_most_ten_millisecond_timing_error() -> None:
    track = SubtitleTrack(
        "track-1",
        "transcript-1",
        "en",
        (
            SubtitleCue(
                "cue-000001",
                1_004,
                3_506,
                ("word-1",),
                "hello",
                None,
                ("hello", "world"),
            ),
        ),
        0,
    )
    data = ass.serialize_bytes(track)
    assert data.startswith(b"[Script Info]\n")
    assert b"\r" not in data
    parsed = ass.parse(data)
    cue = parsed.cues[0]
    assert abs(cue.start_ms - 1_004) <= 10
    assert abs(cue.end_ms - 3_506) <= 10
    assert cue.lines == ("hello", "world")


def test_ass_escapes_override_braces_and_backslashes() -> None:
    text = r"literal \N {not-a-tag}"
    track = SubtitleTrack(
        "track-1",
        "transcript-1",
        "en",
        (SubtitleCue("cue-000001", 0, 100, ("word-1",), text, None, (text,)),),
        0,
    )
    assert ass.parse(ass.serialize_bytes(track)).cues[0].lines == (text,)
