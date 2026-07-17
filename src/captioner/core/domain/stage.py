"""Pipeline stages and profile-specific immutable stage plans."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from captioner.core.domain.artifact import ArtifactRef


class StageName(StrEnum):
    INSPECT = "inspect"
    NORMALIZE = "normalize"
    TRANSCRIBE = "transcribe"
    CORRECT_SOURCE = "correct_source"
    SEGMENT = "segment"
    TRANSLATE = "translate"
    REVIEW = "review"
    EXPORT = "export"
    PUBLISH = "publish"


class PipelineProfile(StrEnum):
    DETERMINISTIC = "deterministic"
    FAST = "fast"
    QUALITY = "quality"


_DETERMINISTIC_PLAN: tuple[StageName, ...] = (
    StageName.INSPECT,
    StageName.NORMALIZE,
    StageName.TRANSCRIBE,
    StageName.SEGMENT,
    StageName.EXPORT,
    StageName.PUBLISH,
)
_FAST_PLAN: tuple[StageName, ...] = (
    StageName.INSPECT,
    StageName.NORMALIZE,
    StageName.TRANSCRIBE,
    StageName.SEGMENT,
    StageName.TRANSLATE,
    StageName.EXPORT,
    StageName.PUBLISH,
)
_QUALITY_PLAN: tuple[StageName, ...] = (
    StageName.INSPECT,
    StageName.NORMALIZE,
    StageName.TRANSCRIBE,
    StageName.CORRECT_SOURCE,
    StageName.SEGMENT,
    StageName.TRANSLATE,
    StageName.REVIEW,
    StageName.EXPORT,
    StageName.PUBLISH,
)

# Compatibility name for Phase 2/3 callers.  It intentionally is not derived
# from ``tuple(StageName)``: the latter is the complete vocabulary, not a Job
# execution plan.
STAGE_PLAN = _DETERMINISTIC_PLAN


def stage_plan_for(profile: PipelineProfile | str) -> tuple[StageName, ...]:
    """Return the exact durable plan for one pipeline profile."""
    try:
        selected = PipelineProfile(profile)
    except ValueError as exc:
        raise ValueError(f"unknown_pipeline_profile:{profile}") from exc
    return {
        PipelineProfile.DETERMINISTIC: _DETERMINISTIC_PLAN,
        PipelineProfile.FAST: _FAST_PLAN,
        PipelineProfile.QUALITY: _QUALITY_PLAN,
    }[selected]


def stage_versions_for(profile: PipelineProfile | str) -> dict[str, str]:
    """Return the complete version map for the selected durable plan."""
    selected = PipelineProfile(profile)
    versions = {
        StageName.INSPECT: "inspect-v1",
        StageName.NORMALIZE: "normalize-v1",
        StageName.TRANSCRIBE: "transcribe-v1",
        StageName.CORRECT_SOURCE: "correct-source-v2",
        StageName.SEGMENT: "segment-v2",
        StageName.TRANSLATE: "translate-quality-v2"
        if selected is PipelineProfile.QUALITY
        else "translate-v2",
        StageName.REVIEW: "review-v2",
        StageName.EXPORT: "export-v3",
        StageName.PUBLISH: "publish-v3",
    }
    return {stage.value: versions[stage] for stage in stage_plan_for(selected)}


class StageState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMMITTED = "committed"
    INVALIDATED = "invalidated"


@dataclass(frozen=True, slots=True)
class StageProjection:
    name: StageName
    state: StageState = StageState.PENDING
    attempt: int = 0
    cache_key: str | None = None
    artifacts: tuple[ArtifactRef, ...] = ()


def dependencies(
    stage: StageName,
    plan: Sequence[StageName] = STAGE_PLAN,
) -> tuple[StageName, ...]:
    index = plan.index(stage)
    return () if index == 0 else (plan[index - 1],)


def stage_suffix(
    stage: StageName,
    plan: Sequence[StageName] = STAGE_PLAN,
) -> tuple[StageName, ...]:
    return tuple(plan[plan.index(stage) :])
