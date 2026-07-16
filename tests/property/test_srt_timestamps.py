from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st
from scripts.validate_subtitle import parse_srt
from tests.support import POLICY_SIGNATURE

from captioner.adapters.exporters.srt import format_timestamp, serialize
from captioner.core.domain.subtitle import SubtitleCue, SubtitleTrack


@given(st.integers(min_value=0, max_value=10_000_000), st.integers(min_value=1, max_value=100_000))
def test_srt_timestamp_format_round_trips(start_ms: int, duration_ms: int) -> None:
    end_ms = start_ms + duration_ms
    track = SubtitleTrack(
        "track-test",
        "transcript-test",
        "en",
        (SubtitleCue("cue-1", start_ms, end_ms, ("word-1",), "text", None, ("text",)),),
        0,
        POLICY_SIGNATURE,
    )
    parsed = parse_srt(serialize(track))
    assert parsed[0].start_ms == start_ms
    assert parsed[0].end_ms == end_ms
    assert format_timestamp(start_ms).count(":") == 2
