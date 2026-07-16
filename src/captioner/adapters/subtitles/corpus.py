"""Adapter wiring for the application subtitle corpus service."""

from __future__ import annotations

from pathlib import Path

from captioner.adapters.exporters import srt
from captioner.adapters.persistence.domain_codecs import (
    decode_track,
    decode_transcript,
    encode_track,
)
from captioner.adapters.subtitles import ass, webvtt
from captioner.core.application.subtitle_corpus import (
    CorpusFormat,
    CorpusReport,
    run_subtitle_corpus,
)
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig


def run_project_subtitle_corpus(
    root: Path, *, config: SegmentationPolicyConfig | None = None
) -> CorpusReport:
    return run_subtitle_corpus(
        root,
        decode_transcript=decode_transcript,
        encode_track=encode_track,
        decode_track=decode_track,
        formats=(
            CorpusFormat("srt", srt.serialize_bytes, srt.parse),
            CorpusFormat("webvtt", webvtt.serialize_bytes, webvtt.parse),
            CorpusFormat("ass", ass.serialize_bytes, ass.parse, tolerance_ms=10),
        ),
        config=config,
    )
