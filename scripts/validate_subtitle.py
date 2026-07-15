"""Validate generated SRT files without an external subtitle library."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_TIMESTAMP = re.compile(r"^(\d{2,}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2,}):(\d{2}):(\d{2}),(\d{3})$")


class SubtitleValidationError(ValueError):
    """Raised when an SRT violates the small validator grammar."""


@dataclass(frozen=True, slots=True)
class ParsedCue:
    index: int
    start_ms: int
    end_ms: int
    text: tuple[str, ...]


def parse_srt(text: str) -> tuple[ParsedCue, ...]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return ()
    cues: list[ParsedCue] = []
    previous_end = -1
    for expected_index, block in enumerate(normalized.strip().split("\n\n"), start=1):
        lines = block.split("\n")
        if len(lines) < 3:
            raise SubtitleValidationError("missing cue text or timestamp")
        try:
            index = int(lines[0])
        except ValueError as exc:
            raise SubtitleValidationError("invalid cue index") from exc
        if index != expected_index:
            raise SubtitleValidationError("invalid cue index order")
        match = _TIMESTAMP.fullmatch(lines[1])
        if match is None:
            raise SubtitleValidationError("malformed timestamp")
        start = _timestamp_to_ms(match.group(1), match.group(2), match.group(3), match.group(4))
        end = _timestamp_to_ms(match.group(5), match.group(6), match.group(7), match.group(8))
        if end <= start:
            raise SubtitleValidationError("end must be greater than start")
        if start < previous_end:
            raise SubtitleValidationError("overlapping or non-monotonic cues")
        cue_text = tuple(lines[2:])
        if not cue_text or any(not line.strip() for line in cue_text):
            raise SubtitleValidationError("missing cue text")
        cues.append(ParsedCue(index=index, start_ms=start, end_ms=end, text=cue_text))
        previous_end = end
    return tuple(cues)


def validate_file(path: Path) -> tuple[ParsedCue, ...]:
    return parse_srt(path.read_text(encoding="utf-8"))


def _timestamp_to_ms(hours: str, minutes: str, seconds: str, milliseconds: str) -> int:
    minute_value = int(minutes)
    second_value = int(seconds)
    if minute_value > 59 or second_value > 59:
        raise SubtitleValidationError("invalid timestamp component")
    return int(hours) * 3_600_000 + minute_value * 60_000 + second_value * 1_000 + int(milliseconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an SRT file")
    parser.add_argument("path", type=Path)
    namespace = parser.parse_args(argv)
    try:
        cues = validate_file(namespace.path)
    except (OSError, ValueError) as exc:
        print(f"invalid SRT: {exc}", file=sys.stderr)
        return 1
    print(f"valid SRT: {len(cues)} cue(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
