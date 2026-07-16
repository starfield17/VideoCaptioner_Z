"""Exact integer reading-speed calculations for subtitle cues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from captioner.core.policies.unicode_metrics import measure_text


@dataclass(frozen=True, slots=True)
class ReadingSpeed:
    characters: int
    duration_ms: int
    cps_milli: int
    status: Literal["ok", "warning", "error"]


def reading_speed(
    text: str,
    duration_ms: int,
    *,
    target_cps_milli: int = 17_000,
    max_cps_milli: int = 20_000,
) -> ReadingSpeed:
    characters = measure_text(text).reading_characters
    if duration_ms <= 0:
        return ReadingSpeed(characters, duration_ms, 0, "error")
    cps_milli = characters * 1_000_000 // duration_ms
    if characters * 1_000_000 > max_cps_milli * duration_ms:
        status: Literal["ok", "warning", "error"] = "error"
    elif characters * 1_000_000 > target_cps_milli * duration_ms:
        status = "warning"
    else:
        status = "ok"
    return ReadingSpeed(characters, duration_ms, cps_milli, status)


def cps_within_limit(characters: int, duration_ms: int, max_cps_milli: int) -> bool:
    return duration_ms > 0 and characters * 1_000_000 <= max_cps_milli * duration_ms
