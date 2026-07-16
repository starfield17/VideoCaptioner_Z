"""Pure subtitle exporter and parser boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from captioner.core.domain.subtitle import SubtitleTrack


@dataclass(frozen=True, slots=True)
class ParsedCue:
    start_ms: int
    end_ms: int
    lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedSubtitle:
    cues: tuple[ParsedCue, ...]


class SubtitleExporter(Protocol):
    format_name: str
    version: str
    media_type: str
    extension: str

    def serialize(self, track: SubtitleTrack) -> bytes: ...
