"""Canonical ASS subset exporter and parser."""

from __future__ import annotations

import re

from captioner.core.domain.errors import AppError
from captioner.core.domain.subtitle import SubtitleTrack
from captioner.core.ports.subtitle_exporter import ParsedCue, ParsedSubtitle

_ASS_TIMESTAMP = re.compile(r"^(\d+):(\d{2}):(\d{2})\.(\d{2})$")
_HEADER = (
    "[Script Info]\n"
    "ScriptType: v4.00+\n"
    "PlayResX: 1920\n"
    "PlayResY: 1080\n"
    "\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
    "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,"
    "100,100,0,0,1,2,0,2,40,40,40,1\n"
    "\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
)


def format_timestamp(milliseconds: int) -> str:
    milliseconds = _validated_timestamp(milliseconds)
    return _format_centiseconds(_rounded_centiseconds(milliseconds))


def serialize(track: SubtitleTrack) -> str:
    previous_end_centiseconds = 0
    rows: list[str] = []
    for cue in track.cues:
        if cue.start_ms < 0 or cue.end_ms <= cue.start_ms:
            raise AppError("export.ass_invalid", {"reason": "cue_order", "cue_id": cue.id})
        natural_start = _rounded_centiseconds(cue.start_ms)
        natural_end = _rounded_centiseconds(cue.end_ms)
        start_centiseconds = max(natural_start, previous_end_centiseconds)
        end_centiseconds = max(natural_end, start_centiseconds + 1)
        if (
            abs(start_centiseconds * 10 - cue.start_ms) > 10
            or abs(end_centiseconds * 10 - cue.end_ms) > 10
        ):
            raise AppError("export.ass_unrepresentable", {"cue_id": cue.id})
        text = r"\N".join(_escape(line) for line in cue.lines)
        rows.append(
            "Dialogue: 0,"
            f"{_format_centiseconds(start_centiseconds)},{_format_centiseconds(end_centiseconds)},"
            f"Default,,0,0,0,,{text}"
        )
        previous_end_centiseconds = end_centiseconds
    return _HEADER + "\n".join(rows) + "\n"


def serialize_bytes(track: SubtitleTrack) -> bytes:
    return serialize(track).encode("utf-8")


def parse(data: bytes | str) -> ParsedSubtitle:
    text = _decode(data)
    if "\r" in text or not text.startswith(_HEADER) or not text.endswith("\n"):
        raise AppError("export.ass_invalid", {"reason": "header"})
    rows = [line for line in text[len(_HEADER) : -1].split("\n") if line]
    cues: list[ParsedCue] = []
    previous_end = -1
    for row in rows:
        if not row.startswith("Dialogue: "):
            raise AppError("export.ass_invalid", {"reason": "event"})
        fields = row[len("Dialogue: ") :].split(",", maxsplit=9)
        if len(fields) != 10 or fields[0] != "0" or fields[3:8] != ["Default", "", "0", "0", "0"]:
            raise AppError("export.ass_invalid", {"reason": "event_fields"})
        start = _parse_timestamp(fields[1])
        end = _parse_timestamp(fields[2])
        lines = _split_ass_lines(fields[9])
        if start < previous_end or end <= start or not lines or len(lines) > 2:
            raise AppError("export.ass_invalid", {"reason": "cue_order"})
        cues.append(ParsedCue(start, end, lines))
        previous_end = end
    return ParsedSubtitle(tuple(cues))


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _split_ass_lines(value: str) -> tuple[str, ...]:
    raw_lines: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character == "\\" and index + 1 < len(value):
            next_character = value[index + 1]
            if next_character == "\\":
                current.extend((character, next_character))
                index += 2
                continue
            if next_character == "N":
                raw_lines.append("".join(current))
                current = []
                index += 2
                continue
            if next_character in "{}":
                current.extend((character, next_character))
                index += 2
                continue
        if character == "\\":
            raise AppError("export.ass_invalid", {"reason": "override_tag"})
        current.append(character)
        index += 1
    raw_lines.append("".join(current))
    return tuple(_unescape(line) for line in raw_lines)


def _unescape(value: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(value):
        if value[index] == "\\" and index + 1 < len(value) and value[index + 1] in "\\{}":
            result.append(value[index + 1])
            index += 2
        elif value[index] == "\\" or value[index] in "{}":
            raise AppError("export.ass_invalid", {"reason": "override_tag"})
        else:
            result.append(value[index])
            index += 1
    return "".join(result)


def _decode(data: bytes | str) -> str:
    if isinstance(data, bytes):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AppError("export.ass_invalid", {"reason": "utf8"}) from exc
    return data


def _parse_timestamp(value: str) -> int:
    match = _ASS_TIMESTAMP.fullmatch(value)
    if match is None:
        raise AppError("export.ass_invalid", {"reason": "timestamp"})
    hours, minutes, seconds, centiseconds = (int(item) for item in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise AppError("export.ass_invalid", {"reason": "timestamp"})
    return ((hours * 60 + minutes) * 60 + seconds) * 1_000 + centiseconds * 10


def _validated_timestamp(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AppError("export.ass_invalid", {"reason": "timestamp"})
    return value


def _rounded_centiseconds(milliseconds: int) -> int:
    return (milliseconds + 5) // 10


def _format_centiseconds(centiseconds: int) -> str:
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centis:02d}"
