"""Canonical WebVTT exporter and parser."""

from __future__ import annotations

import html
import re

from captioner.core.domain.errors import AppError
from captioner.core.domain.subtitle import SubtitleTrack
from captioner.core.ports.subtitle_exporter import ParsedCue, ParsedSubtitle

_TIMESTAMP = re.compile(r"^(\d+):(\d{2}):(\d{2})\.(\d{3})$")


def format_timestamp(milliseconds: int) -> str:
    milliseconds = _validated_timestamp(milliseconds)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def serialize(track: SubtitleTrack) -> str:
    blocks: list[str] = []
    previous_end = -1
    for cue in track.cues:
        if cue.start_ms < previous_end or cue.end_ms <= cue.start_ms:
            raise AppError("export.webvtt_invalid", {"reason": "cue_order", "cue_id": cue.id})
        if not cue.lines or any(not line.strip() for line in cue.lines):
            raise AppError("export.webvtt_invalid", {"reason": "cue_text", "cue_id": cue.id})
        blocks.append(
            "\n".join(
                (
                    f"{format_timestamp(cue.start_ms)} --> {format_timestamp(cue.end_ms)}",
                    *(_escape(line) for line in cue.lines),
                )
            )
        )
        previous_end = cue.end_ms
    return "WEBVTT\n\n" if not blocks else "WEBVTT\n\n" + "\n\n".join(blocks) + "\n"


def serialize_bytes(track: SubtitleTrack) -> bytes:
    return serialize(track).encode("utf-8")


def parse(data: bytes | str) -> ParsedSubtitle:
    text = _decode(data)
    if not text.startswith("WEBVTT\n\n") or "\r" in text or not text.endswith("\n"):
        raise AppError("export.webvtt_invalid", {"reason": "header"})
    body = text[len("WEBVTT\n\n") : -1]
    if not body:
        return ParsedSubtitle(())
    blocks = body.split("\n\n")
    cues: list[ParsedCue] = []
    previous_end = -1
    for block in blocks:
        lines = block.split("\n")
        if len(lines) < 2 or " --> " not in lines[0]:
            raise AppError("export.webvtt_invalid", {"reason": "cue"})
        start, end = _parse_range(lines[0])
        cue_lines = tuple(html.unescape(line) for line in lines[1:])
        if len(cue_lines) > 2 or any(not line.strip() for line in cue_lines):
            raise AppError("export.webvtt_invalid", {"reason": "lines"})
        if start < previous_end or end <= start:
            raise AppError("export.webvtt_invalid", {"reason": "cue_order"})
        cues.append(ParsedCue(start, end, cue_lines))
        previous_end = end
    return ParsedSubtitle(tuple(cues))


def _escape(line: str) -> str:
    return html.escape(line, quote=False)


def _decode(data: bytes | str) -> str:
    if isinstance(data, bytes):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AppError("export.webvtt_invalid", {"reason": "utf8"}) from exc
    return data


def _parse_range(value: str) -> tuple[int, int]:
    start_text, end_text = value.split(" --> ", maxsplit=1)
    return _parse_timestamp(start_text), _parse_timestamp(end_text)


def _parse_timestamp(value: str) -> int:
    match = _TIMESTAMP.fullmatch(value)
    if match is None:
        raise AppError("export.webvtt_invalid", {"reason": "timestamp"})
    hours, minutes, seconds, millis = (int(item) for item in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise AppError("export.webvtt_invalid", {"reason": "timestamp"})
    return ((hours * 60 + minutes) * 60 + seconds) * 1_000 + millis


def _validated_timestamp(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AppError("export.webvtt_invalid", {"reason": "timestamp"})
    return value
