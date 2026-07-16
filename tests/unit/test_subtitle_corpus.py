from __future__ import annotations

from pathlib import Path

from captioner.adapters.exporters import srt
from captioner.adapters.persistence.domain_codecs import (
    decode_transcript,
    encode_track,
)
from captioner.adapters.subtitles import ass, webvtt
from captioner.core.application.subtitle_corpus import CorpusFormat, run_subtitle_corpus
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig


def test_corpus_runner_requires_an_actual_json_track_round_trip() -> None:
    fixture = Path("tests/fixtures/transcripts/short_words.json")

    report = run_subtitle_corpus(
        fixture.parent,
        decode_transcript=decode_transcript,
        encode_track=encode_track,
        decode_track=lambda _data: object(),  # type: ignore[return-value]  # injected bad decoder
        formats=(
            CorpusFormat("srt", srt.serialize_bytes, srt.parse),
            CorpusFormat("webvtt", webvtt.serialize_bytes, webvtt.parse),
            CorpusFormat("ass", ass.serialize_bytes, ass.parse, tolerance_ms=10),
        ),
        config=SegmentationPolicyConfig(),
    )
    assert report.failed == 14
    assert all(result.errors == ("corpus.json_round_trip_failed",) for result in report.fixtures)
