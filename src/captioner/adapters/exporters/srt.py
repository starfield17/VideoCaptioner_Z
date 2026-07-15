"""Strict deterministic SubRip exporter."""

from __future__ import annotations

from captioner.core.domain.errors import AppError
from captioner.core.domain.subtitle import SubtitleTrack


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
