"""Run the deterministic subtitle processing corpus without network access."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from captioner.adapters.exporters import srt
from captioner.adapters.persistence.domain_codecs import decode_transcript, encode_track
from captioner.adapters.subtitles import ass, webvtt
from captioner.core.domain.subtitle import SubtitleTrack
from captioner.core.domain.subtitle_validation import validate_subtitle_track
from captioner.core.policies.reading_speed import reading_speed
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.simple_segmentation import segment_transcript
from captioner.core.policies.unicode_metrics import measure_text
from captioner.core.ports.subtitle_exporter import ParsedSubtitle


class CorpusError(Exception):
    """Raised when a corpus fixture cannot complete its round trip."""


def _fixture_paths(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        raise CorpusError
    paths = tuple(sorted(root.glob("*.json"), key=lambda path: path.as_posix()))
    names = [path.stem.casefold() for path in paths]
    if len(names) != len(set(names)):
        raise CorpusError
    if not paths:
        raise CorpusError
    return paths


def _assert_parsed(
    track: SubtitleTrack, parsed: ParsedSubtitle, *, ass_tolerance: bool = False
) -> None:
    cues = parsed.cues
    if len(cues) != len(track.cues):
        raise CorpusError
    for expected, actual in zip(track.cues, cues, strict=True):
        timing_delta: tuple[int, int] = (
            abs(actual.start_ms - expected.start_ms),
            abs(actual.end_ms - expected.end_ms),
        )
        if ass_tolerance:
            if max(timing_delta) > 10:
                raise CorpusError
        elif timing_delta != (0, 0):
            raise CorpusError
        if tuple(actual.lines) != expected.lines:
            raise CorpusError


def _run_fixture(path: Path, config: SegmentationPolicyConfig) -> tuple[int, int, int, int, int]:
    transcript = decode_transcript(path.read_bytes())
    track = segment_transcript(transcript, config)
    report = validate_subtitle_track(track, transcript, config)
    if not report.is_valid:
        raise CorpusError
    json_bytes = encode_track(track)
    if encode_track(track) != json_bytes:
        raise CorpusError
    _assert_parsed(track, srt.parse(srt.serialize_bytes(track)))
    _assert_parsed(track, webvtt.parse(webvtt.serialize_bytes(track)))
    _assert_parsed(track, ass.parse(ass.serialize_bytes(track)), ass_tolerance=True)
    max_cps = max(
        (reading_speed(cue.source_text, cue.end_ms - cue.start_ms).cps_milli for cue in track.cues),
        default=0,
    )
    max_width = max(
        (measure_text(line).display_columns for cue in track.cues for line in cue.lines),
        default=0,
    )
    warnings = sum(issue.severity.value == "warning" for issue in report.issues)
    errors = sum(issue.severity.value == "error" for issue in report.issues)
    return len(track.cues), max_cps, max_width, warnings, errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_directory", type=Path)
    arguments = parser.parse_args(None if argv is None else list(argv))
    try:
        paths = _fixture_paths(arguments.fixture_directory)
    except (CorpusError, OSError) as exc:
        print(f"corpus failed: {exc}", file=sys.stderr)
        return 1

    config = SegmentationPolicyConfig()
    failures = 0
    for path in paths:
        try:
            cues, max_cps, max_width, warnings, errors = _run_fixture(path, config)
            print(
                f"{path.as_posix()}: cues={cues} max_cps_milli={max_cps} "
                f"max_line_width={max_width} warnings={warnings} errors={errors}"
            )
            failures += int(errors > 0)
        except Exception as exc:
            print(f"{path.as_posix()}: errors=1 ({type(exc).__name__})", file=sys.stderr)
            failures += 1
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
