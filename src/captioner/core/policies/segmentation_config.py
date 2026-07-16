"""Canonical immutable configuration for deterministic subtitle policies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from captioner.core.domain.errors import AppError

_FIELDS = frozenset(
    {
        "schema_version",
        "min_duration_ms",
        "target_duration_ms",
        "max_duration_ms",
        "preferred_gap_ms",
        "hard_gap_ms",
        "target_cps_milli",
        "max_cps_milli",
        "max_lines",
        "max_line_width",
        "max_cue_width",
        "punctuation_bonus",
        "silence_bonus",
        "protected_break_penalty",
        "overflow_penalty",
    }
)
_LEGACY_FIELDS = frozenset({"max_duration_ms", "max_text_units", "hard_gap_ms"})


@dataclass(frozen=True, slots=True)
class SegmentationPolicyConfig:
    """All integer knobs used by segmentation, reading speed and line breaking."""

    schema_version: int = 1
    min_duration_ms: int = 800
    target_duration_ms: int = 3_500
    max_duration_ms: int = 7_000
    preferred_gap_ms: int = 300
    hard_gap_ms: int = 700
    target_cps_milli: int = 17_000
    max_cps_milli: int = 20_000
    max_lines: int = 2
    max_line_width: int = 42
    max_cue_width: int = 84
    punctuation_bonus: int = 800
    silence_bonus: int = 1_000
    protected_break_penalty: int = 10_000
    overflow_penalty: int = 1_000_000

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError
        positive = (
            self.min_duration_ms,
            self.target_duration_ms,
            self.max_duration_ms,
            self.max_line_width,
            self.max_cue_width,
            self.target_cps_milli,
            self.max_cps_milli,
        )
        if any(value <= 0 for value in positive):
            raise ValueError
        nonnegative = (
            self.preferred_gap_ms,
            self.hard_gap_ms,
            self.punctuation_bonus,
            self.silence_bonus,
            self.protected_break_penalty,
            self.overflow_penalty,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError
        if not 1 <= self.max_lines <= 2:
            raise ValueError
        if self.min_duration_ms > self.max_duration_ms:
            raise ValueError
        if self.target_duration_ms > self.max_duration_ms:
            raise ValueError
        if self.preferred_gap_ms > self.hard_gap_ms:
            raise ValueError
        if self.target_cps_milli > self.max_cps_milli:
            raise ValueError
        if self.max_cue_width < self.max_line_width:
            raise ValueError

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> SegmentationPolicyConfig:
        keys = set(values)
        if keys == set(_LEGACY_FIELDS):
            max_duration = _positive_int(values, "max_duration_ms")
            max_text_units = _positive_int(values, "max_text_units")
            hard_gap = _nonnegative_int(values, "hard_gap_ms")
            return cls(
                min_duration_ms=min(800, max_duration),
                max_duration_ms=max_duration,
                target_duration_ms=min(3_500, max_duration),
                preferred_gap_ms=min(300, hard_gap),
                hard_gap_ms=hard_gap,
                max_line_width=min(42, max_text_units),
                max_cue_width=max_text_units,
            )
        if keys != set(_FIELDS):
            raise AppError("job.config_invalid", {"field": "segmentation"})
        try:
            result = cls(
                schema_version=_mapping_int(values, "schema_version"),
                min_duration_ms=_mapping_int(values, "min_duration_ms"),
                target_duration_ms=_mapping_int(values, "target_duration_ms"),
                max_duration_ms=_mapping_int(values, "max_duration_ms"),
                preferred_gap_ms=_mapping_int(values, "preferred_gap_ms"),
                hard_gap_ms=_mapping_int(values, "hard_gap_ms"),
                target_cps_milli=_mapping_int(values, "target_cps_milli"),
                max_cps_milli=_mapping_int(values, "max_cps_milli"),
                max_lines=_mapping_int(values, "max_lines"),
                max_line_width=_mapping_int(values, "max_line_width"),
                max_cue_width=_mapping_int(values, "max_cue_width"),
                punctuation_bonus=_mapping_int(values, "punctuation_bonus"),
                silence_bonus=_mapping_int(values, "silence_bonus"),
                protected_break_penalty=_mapping_int(values, "protected_break_penalty"),
                overflow_penalty=_mapping_int(values, "overflow_penalty"),
            )
        except (TypeError, ValueError) as exc:
            raise AppError("job.config_invalid", {"field": "segmentation"}) from exc
        return result

    def to_mapping(self) -> dict[str, int]:
        return {
            "schema_version": self.schema_version,
            "min_duration_ms": self.min_duration_ms,
            "target_duration_ms": self.target_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "preferred_gap_ms": self.preferred_gap_ms,
            "hard_gap_ms": self.hard_gap_ms,
            "target_cps_milli": self.target_cps_milli,
            "max_cps_milli": self.max_cps_milli,
            "max_lines": self.max_lines,
            "max_line_width": self.max_line_width,
            "max_cue_width": self.max_cue_width,
            "punctuation_bonus": self.punctuation_bonus,
            "silence_bonus": self.silence_bonus,
            "protected_break_penalty": self.protected_break_penalty,
            "overflow_penalty": self.overflow_penalty,
        }

    @property
    def signature(self) -> str:
        payload = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))
        return f"policy-{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _positive_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise AppError("job.config_invalid", {"field": "segmentation"})
    return value


def _mapping_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError
    return value


def _nonnegative_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AppError("job.config_invalid", {"field": "segmentation"})
    return value
