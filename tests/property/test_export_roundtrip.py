from __future__ import annotations

from tests.support import make_transcript

from captioner.adapters.exporters import srt
from captioner.adapters.persistence.domain_codecs import decode_track
from captioner.adapters.subtitles import ass, json_track, webvtt
from captioner.core.policies.simple_segmentation import segment_transcript


def test_repeated_export_is_byte_identical_and_observationally_pure() -> None:
    track = segment_transcript(make_transcript(("hello ", "世界 ", "emoji")))
    before = track
    outputs = (
        json_track.serialize(track),
        srt.serialize_bytes(track),
        webvtt.serialize_bytes(track),
        ass.serialize_bytes(track),
    )
    repeated = (
        json_track.serialize(track),
        srt.serialize_bytes(track),
        webvtt.serialize_bytes(track),
        ass.serialize_bytes(track),
    )
    assert outputs == repeated
    assert track == before
    assert decode_track(outputs[0]) == track
    assert srt.parse(outputs[1]).cues[0].lines == track.cues[0].lines
    assert webvtt.parse(outputs[2]).cues[-1].end_ms == track.cues[-1].end_ms
    assert abs(ass.parse(outputs[3]).cues[0].start_ms - track.cues[0].start_ms) <= 10
