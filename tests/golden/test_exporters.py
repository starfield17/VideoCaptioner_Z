from __future__ import annotations

from pathlib import Path

from tests.golden.manifest import verify_manifest

from captioner.adapters.exporters import srt
from captioner.adapters.persistence.domain_codecs import decode_transcript
from captioner.adapters.subtitles import ass, json_track, webvtt
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.simple_segmentation import segment_transcript

ROOT = Path(__file__).parent / "data"
FIXTURES = Path(__file__).parents[1] / "fixtures" / "transcripts"


def test_exporter_goldens_are_reviewed_and_byte_exact() -> None:
    verify_manifest(ROOT, FIXTURES)
    exporters = {
        "track.json": json_track.serialize,
        "srt": srt.serialize_bytes,
        "vtt": webvtt.serialize_bytes,
        "ass": ass.serialize_bytes,
    }
    for fixture in sorted(FIXTURES.glob("*.json")):
        track = segment_transcript(
            decode_transcript(fixture.read_bytes()), SegmentationPolicyConfig()
        )
        for suffix, exporter in exporters.items():
            assert exporter(track) == (ROOT / f"{fixture.stem}.{suffix}").read_bytes(), (
                fixture.name,
                suffix,
            )
