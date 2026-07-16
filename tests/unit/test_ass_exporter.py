from __future__ import annotations

import pytest
from tests.support import POLICY_SIGNATURE

from captioner.adapters.subtitles import ass
from captioner.core.domain.errors import AppError
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
        POLICY_SIGNATURE,
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
        POLICY_SIGNATURE,
    )
    assert ass.parse(ass.serialize_bytes(track)).cues[0].lines == (text,)


def test_ass_parser_rejects_unescaped_override_tags() -> None:
    track = SubtitleTrack(
        "track-1",
        "transcript-1",
        "en",
        (SubtitleCue("cue-000001", 0, 1, ("word-1",), "x", None, ("x",)),),
        0,
        POLICY_SIGNATURE,
    )
    data = ass.serialize(track).replace("x\n", "{\\i1}x\n").encode("utf-8")
    with pytest.raises(AppError, match=r"export\.ass_invalid"):
        ass.parse(data)


def test_ass_quantization_preserves_adjacent_representable_cue_order() -> None:
    track = SubtitleTrack(
        "track-1",
        "transcript-1",
        "en",
        (
            SubtitleCue("cue-000001", 0, 10, ("word-1",), "one", None, ("one",)),
            SubtitleCue("cue-000002", 10, 20, ("word-2",), "two", None, ("two",)),
        ),
        0,
        POLICY_SIGNATURE,
    )
    parsed = ass.parse(ass.serialize_bytes(track))
    assert [(cue.start_ms, cue.end_ms) for cue in parsed.cues] == [(0, 10), (10, 20)]


def test_ass_rejects_unrepresentable_adjacent_one_millisecond_cues() -> None:
    track = SubtitleTrack(
        "track-1",
        "transcript-1",
        "en",
        (
            SubtitleCue("cue-000001", 0, 1, ("word-1",), "one", None, ("one",)),
            SubtitleCue("cue-000002", 1, 2, ("word-2",), "two", None, ("two",)),
        ),
        0,
        POLICY_SIGNATURE,
    )
    with pytest.raises(AppError, match=r"export\.ass_unrepresentable"):
        ass.serialize_bytes(track)


def test_ass_parser_rejects_overlapping_and_out_of_order_dialogue_rows() -> None:
    valid = ass.serialize(
        SubtitleTrack(
            "track-1",
            "transcript-1",
            "en",
            (
                SubtitleCue("cue-000001", 0, 1_000, ("word-1",), "one", None, ("one",)),
                SubtitleCue("cue-000002", 1_000, 2_000, ("word-2",), "two", None, ("two",)),
            ),
            0,
            POLICY_SIGNATURE,
        )
    )
    first, second = valid.rstrip("\n").split("\n")[-2:]
    overlap = valid.replace("0:00:01.00,0:00:02.00", "0:00:00.50,0:00:01.50")
    out_of_order = "\n".join((*valid.rstrip("\n").split("\n")[:-2], second, first, ""))
    for data in (overlap, out_of_order):
        with pytest.raises(AppError, match=r"export\.ass_invalid"):
            ass.parse(data)
