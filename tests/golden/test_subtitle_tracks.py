from __future__ import annotations

from pathlib import Path

from captioner.adapters.persistence.domain_codecs import decode_transcript
from captioner.adapters.subtitles.json_track import serialize
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.simple_segmentation import segment_transcript

ROOT = Path(__file__).parent / "data"
FIXTURES = Path(__file__).parents[1] / "fixtures" / "transcripts"


def test_subtitle_track_goldens_are_reviewed_and_byte_exact() -> None:
    for fixture in sorted(FIXTURES.glob("*.json")):
        transcript = decode_transcript(fixture.read_bytes())
        track = segment_transcript(transcript, SegmentationPolicyConfig())
        expected = (ROOT / f"{fixture.stem}.track.json").read_bytes()
        assert serialize(track) == expected, fixture.name
