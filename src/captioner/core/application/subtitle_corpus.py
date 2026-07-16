"""Application service for deterministic subtitle corpus validation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from captioner.core.domain.errors import AppError
from captioner.core.domain.subtitle import SubtitleTrack
from captioner.core.domain.subtitle_validation import validate_subtitle_track
from captioner.core.domain.transcript import Transcript
from captioner.core.policies.reading_speed import reading_speed
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.simple_segmentation import segment_transcript
from captioner.core.policies.unicode_metrics import measure_text
from captioner.core.ports.subtitle_exporter import ParsedSubtitle


class CorpusError(Exception):
    """Raised for a corpus root that cannot be enumerated."""


@dataclass(frozen=True, slots=True)
class CorpusFormat:
    name: str
    serialize: Callable[[SubtitleTrack], bytes]
    parse: Callable[[bytes], ParsedSubtitle]
    tolerance_ms: int = 0


@dataclass(frozen=True, slots=True)
class CorpusFixtureResult:
    name: str
    cue_count: int
    max_cps_milli: int
    max_line_width: int
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "cue_count": self.cue_count,
            "max_cps_milli": self.max_cps_milli,
            "max_line_width": self.max_line_width,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class CorpusReport:
    fixture_count: int
    passed: int
    failed: int
    fixtures: tuple[CorpusFixtureResult, ...]
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "fixture_count": self.fixture_count,
            "passed": self.passed,
            "failed": self.failed,
            "fixtures": [fixture.to_dict() for fixture in self.fixtures],
            "errors": list(self.errors),
        }


def run_subtitle_corpus(
    root: Path,
    *,
    decode_transcript: Callable[[bytes], Transcript],
    encode_track: Callable[[SubtitleTrack], bytes],
    decode_track: Callable[[bytes], SubtitleTrack],
    formats: Sequence[CorpusFormat],
    config: SegmentationPolicyConfig | None = None,
) -> CorpusReport:
    settings = SegmentationPolicyConfig() if config is None else config
    try:
        paths = _fixture_paths(root)
    except CorpusError as exc:
        return CorpusReport(0, 0, 1, (), (str(exc),))

    results: list[CorpusFixtureResult] = []
    for path in paths:
        try:
            results.append(
                _run_fixture(
                    path,
                    settings,
                    decode_transcript=decode_transcript,
                    encode_track=encode_track,
                    decode_track=decode_track,
                    formats=formats,
                )
            )
        except AppError as exc:
            results.append(CorpusFixtureResult(path.stem, 0, 0, 0, (), (exc.code,)))
        except Exception:
            results.append(CorpusFixtureResult(path.stem, 0, 0, 0, (), ("corpus.fixture_failed",)))
    passed = sum(result.passed for result in results)
    return CorpusReport(len(paths), passed, len(results) - passed, tuple(results))


def _fixture_paths(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        raise CorpusError("corpus.fixture_directory_invalid")
    paths = tuple(sorted(root.glob("*.json"), key=lambda path: path.as_posix()))
    names = [path.stem.casefold() for path in paths]
    if len(names) != len(set(names)):
        raise CorpusError("corpus.duplicate_fixture")
    if not paths:
        raise CorpusError("corpus.no_fixtures")
    return paths


def _run_fixture(
    path: Path,
    config: SegmentationPolicyConfig,
    *,
    decode_transcript: Callable[[bytes], Transcript],
    encode_track: Callable[[SubtitleTrack], bytes],
    decode_track: Callable[[bytes], SubtitleTrack],
    formats: Sequence[CorpusFormat],
) -> CorpusFixtureResult:
    transcript = decode_transcript(path.read_bytes())
    track = segment_transcript(transcript, config)
    report = validate_subtitle_track(track, transcript, config)
    errors = tuple(issue.code for issue in report.issues if issue.severity.value == "error")
    warnings = tuple(issue.code for issue in report.issues if issue.severity.value == "warning")
    if errors:
        return CorpusFixtureResult(path.stem, len(track.cues), 0, 0, warnings, errors)

    json_bytes = encode_track(track)
    decoded = decode_track(json_bytes)
    if decoded != track or encode_track(decoded) != json_bytes:
        raise AppError("corpus.json_round_trip_failed")
    for output_format in formats:
        data = output_format.serialize(track)
        _assert_parsed(track, output_format.parse(data), output_format.tolerance_ms)

    max_cps = max(
        (reading_speed(cue.source_text, cue.end_ms - cue.start_ms).cps_milli for cue in track.cues),
        default=0,
    )
    max_width = max(
        (measure_text(line).display_columns for cue in track.cues for line in cue.lines),
        default=0,
    )
    return CorpusFixtureResult(path.stem, len(track.cues), max_cps, max_width, warnings, ())


def _assert_parsed(track: SubtitleTrack, parsed: ParsedSubtitle, tolerance_ms: int) -> None:
    if len(parsed.cues) != len(track.cues):
        raise AppError("corpus.round_trip_failed")
    for expected, actual in zip(track.cues, parsed.cues, strict=True):
        if (
            abs(actual.start_ms - expected.start_ms) > tolerance_ms
            or abs(actual.end_ms - expected.end_ms) > tolerance_ms
            or actual.lines != expected.lines
        ):
            raise AppError("corpus.round_trip_failed")
