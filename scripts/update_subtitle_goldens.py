"""Update reviewed deterministic subtitle goldens after explicit acknowledgement."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from captioner.adapters.exporters import srt
from captioner.adapters.persistence.domain_codecs import decode_transcript
from captioner.adapters.subtitles import ass, json_track, webvtt
from captioner.core.domain.subtitle import SubtitleTrack
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.simple_segmentation import segment_transcript

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = ROOT / "tests" / "fixtures" / "transcripts"
GOLDEN_ROOT = ROOT / "tests" / "golden" / "data"
ACKNOWLEDGEMENT = "PHASE3_GOLDENS_REVIEWED"
EXPORTER_VERSIONS = {
    "track_json": "track-json-v2",
    "srt": "srt-v2",
    "webvtt": "webvtt-v1",
    "ass": "ass-v1",
}
Exporter = Callable[[SubtitleTrack], bytes]


class GoldenError(Exception):
    """Raised when the fixture set cannot be converted to goldens."""


def _serialize_outputs(track: SubtitleTrack) -> dict[str, bytes]:
    exporters: tuple[tuple[str, Exporter], ...] = (
        ("track.json", json_track.serialize),
        ("srt", srt.serialize_bytes),
        ("vtt", webvtt.serialize_bytes),
        ("ass", ass.serialize_bytes),
    )
    return {suffix: exporter(track) for suffix, exporter in exporters}


def _fixture_paths(root: Path) -> tuple[Path, ...]:
    paths = tuple(sorted(root.glob("*.json"), key=lambda path: path.as_posix()))
    names = [path.stem.casefold() for path in paths]
    if len(names) != len(set(names)):
        raise GoldenError
    if not paths:
        raise GoldenError
    return paths


def _build_goldens(fixtures: tuple[Path, ...]) -> dict[Path, bytes]:
    outputs: dict[Path, bytes] = {}
    for fixture in fixtures:
        transcript = decode_transcript(fixture.read_bytes())
        track = segment_transcript(transcript, SegmentationPolicyConfig())
        for suffix, data in _serialize_outputs(track).items():
            outputs[GOLDEN_ROOT / f"{fixture.stem}.{suffix}"] = data
    return outputs


def _manifest(outputs: dict[Path, bytes]) -> bytes:
    hashes = {
        path.name: hashlib.sha256(data).hexdigest()
        for path, data in sorted(outputs.items(), key=lambda item: item[0].name)
    }
    value = {
        "schema_version": 1,
        "policy_signature": SegmentationPolicyConfig().signature,
        "exporter_versions": EXPORTER_VERSIONS,
        "golden_sha256": hashes,
    }
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{serialized}\n".encode()


def _changes(outputs: dict[Path, bytes], manifest: bytes) -> tuple[tuple[Path, bytes], ...]:
    all_outputs = {**outputs, GOLDEN_ROOT / "manifest.json": manifest}
    return tuple(
        (path, data)
        for path, data in sorted(all_outputs.items(), key=lambda item: item[0].as_posix())
        if not path.exists() or path.read_bytes() != data
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--accept", default="")
    arguments = parser.parse_args(None if argv is None else list(argv))
    try:
        fixtures = _fixture_paths(arguments.fixtures)
        outputs = _build_goldens(fixtures)
        manifest = _manifest(outputs)
        changes = _changes(outputs, manifest)
    except (GoldenError, OSError) as exc:
        print(f"golden preparation failed: {exc}", file=sys.stderr)
        return 1

    for path, _ in changes:
        print(f"would update {path.relative_to(ROOT)}")
    if arguments.accept != ACKNOWLEDGEMENT:
        print(
            "No files were changed. Re-run with --accept "
            f"{ACKNOWLEDGEMENT} after reviewing the proposed goldens.",
            file=sys.stderr,
        )
        return 2

    GOLDEN_ROOT.mkdir(parents=True, exist_ok=True)
    if not changes:
        print("goldens are already current")
        return 0
    for path, data in changes:
        print(f"updating {path.relative_to(ROOT)}")
        path.write_bytes(data)
    print("Review every changed golden before committing it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
