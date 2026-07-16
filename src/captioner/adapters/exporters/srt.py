"""Strict deterministic SubRip exporter."""

from __future__ import annotations

import re

from captioner.core.domain.errors import AppError
from captioner.core.domain.subtitle import SubtitleTrack
from captioner.core.ports.subtitle_exporter import ParsedCue, ParsedSubtitle

_TIMESTAMP = re.compile(r"^(\d{2,}):(\d{2}):(\d{2}),(\d{3})$")


def _validate_integer_timestamp(value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AppError("export.srt_invalid", {"reason": "integer_timestamp"})


def format_timestamp(milliseconds: int) -> str:
    """Format non-negative integer milliseconds as ``HH:MM:SS,mmm``."""
    _validate_integer_timestamp(milliseconds)
    if milliseconds < 0:
        raise AppError("export.srt_invalid", {"reason": "negative_timestamp"})
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def serialize(track: SubtitleTrack) -> str:
    """Render a validated track without repairing invalid ordering."""
    blocks: list[str] = []
    previous_start = -1
    previous_end = -1
    for number, cue in enumerate(track.cues, start=1):
        if (
            cue.start_ms <= previous_start
            or cue.start_ms < previous_end
            or cue.end_ms <= cue.start_ms
        ):
            raise AppError("export.srt_invalid", {"reason": "cue_order", "cue_id": cue.id})
        if not cue.lines or any(not line.strip() for line in cue.lines):
            raise AppError("export.srt_invalid", {"reason": "cue_text", "cue_id": cue.id})
        blocks.append(
            "\n".join(
                (
                    str(number),
                    f"{format_timestamp(cue.start_ms)} --> {format_timestamp(cue.end_ms)}",
                    *cue.lines,
                )
            )
        )
        previous_start = cue.start_ms
        previous_end = cue.end_ms
    return "\n\n".join(blocks) + "\n" if blocks else ""


def serialize_bytes(track: SubtitleTrack) -> bytes:
    return serialize(track).encode("utf-8")


def parse(data: bytes | str) -> ParsedSubtitle:
    text = _decode(data)
    if "\r" in text or not text.endswith("\n"):
        raise AppError("export.srt_invalid", {"reason": "line_endings"})
    body = text[:-1]
    if not body:
        return ParsedSubtitle(())
    blocks = body.split("\n\n")
    cues: list[ParsedCue] = []
    previous_end = -1
    for number, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        if len(lines) < 3 or lines[0] != str(number) or " --> " not in lines[1]:
            raise AppError("export.srt_invalid", {"reason": "cue"})
        start, end = _parse_range(lines[1])
        cue_lines = tuple(lines[2:])
        if len(cue_lines) > 2 or any(not line.strip() for line in cue_lines):
            raise AppError("export.srt_invalid", {"reason": "cue_text"})
        if start < previous_end or end <= start:
            raise AppError("export.srt_invalid", {"reason": "cue_order"})
        cues.append(ParsedCue(start, end, cue_lines))
        previous_end = end
    return ParsedSubtitle(tuple(cues))


def _decode(data: bytes | str) -> str:
    if isinstance(data, bytes):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AppError("export.srt_invalid", {"reason": "utf8"}) from exc
    return data


def _parse_range(value: str) -> tuple[int, int]:
    start_text, end_text = value.split(" --> ", maxsplit=1)
    return _parse_timestamp(start_text), _parse_timestamp(end_text)


def _parse_timestamp(value: str) -> int:
    match = _TIMESTAMP.fullmatch(value)
    if match is None:
        raise AppError("export.srt_invalid", {"reason": "timestamp"})
    hours, minutes, seconds, millis = (int(item) for item in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise AppError("export.srt_invalid", {"reason": "timestamp"})
    return ((hours * 60 + minutes) * 60 + seconds) * 1_000 + millis
