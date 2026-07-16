"""Compatibility facade for the Phase 2 segmentation API."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from captioner.core.domain.errors import AppError
from captioner.core.domain.subtitle import SubtitleTrack
from captioner.core.domain.transcript import Transcript
from captioner.core.policies.segmentation import segment_transcript_dp
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig

__all__ = ["SegmentationPolicyConfig", "SimpleSegmentationConfig", "segment_transcript"]


@dataclass(frozen=True, slots=True)
class SimpleSegmentationConfig:
    """Legacy three-field constructor mapped to the complete Phase 3 policy."""

    max_duration_ms: int = 7_000
    max_text_units: int = 84
    hard_gap_ms: int = 700
    policy: SegmentationPolicyConfig | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> SimpleSegmentationConfig:
        if set(values) != {"max_duration_ms", "max_text_units", "hard_gap_ms"}:
            policy = SegmentationPolicyConfig.from_mapping(values)
            return cls(policy.max_duration_ms, policy.max_cue_width, policy.hard_gap_ms, policy)
        max_duration_ms = values.get("max_duration_ms")
        max_text_units = values.get("max_text_units")
        hard_gap_ms = values.get("hard_gap_ms")
        if (
            not isinstance(max_duration_ms, int)
            or isinstance(max_duration_ms, bool)
            or not isinstance(max_text_units, int)
            or isinstance(max_text_units, bool)
            or not isinstance(hard_gap_ms, int)
            or isinstance(hard_gap_ms, bool)
            or max_duration_ms <= 0
            or max_text_units <= 0
            or hard_gap_ms < 0
        ):
            raise AppError("job.config_invalid", {"field": "segmentation"})
        return cls(max_duration_ms, max_text_units, hard_gap_ms)

    def __post_init__(self) -> None:
        if self.max_duration_ms <= 0 or self.max_text_units <= 0 or self.hard_gap_ms < 0:
            raise ValueError

    def to_policy_config(self) -> SegmentationPolicyConfig:
        if self.policy is not None:
            return self.policy
        return SegmentationPolicyConfig.from_mapping(
            {
                "max_duration_ms": self.max_duration_ms,
                "max_text_units": self.max_text_units,
                "hard_gap_ms": self.hard_gap_ms,
            }
        )


def segment_transcript(
    transcript: Transcript,
    config: SimpleSegmentationConfig | SegmentationPolicyConfig | None = None,
    progress: Callable[[], None] | None = None,
) -> SubtitleTrack:
    if config is None:
        settings = SegmentationPolicyConfig()
    elif isinstance(config, SimpleSegmentationConfig):
        settings = config.to_policy_config()
    else:
        settings = config
    return segment_transcript_dp(transcript, settings, progress)
