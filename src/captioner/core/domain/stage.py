"""Fixed Phase 2 stage plan and immutable stage projection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from captioner.core.domain.artifact import ArtifactRef


class StageName(StrEnum):
    INSPECT = "inspect"
    NORMALIZE = "normalize"
    TRANSCRIBE = "transcribe"
    SEGMENT = "segment"
    EXPORT = "export"
    PUBLISH = "publish"


STAGE_PLAN: tuple[StageName, ...] = tuple(StageName)


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


def dependencies(stage: StageName) -> tuple[StageName, ...]:
    index = STAGE_PLAN.index(stage)
    return () if index == 0 else (STAGE_PLAN[index - 1],)


def stage_suffix(stage: StageName) -> tuple[StageName, ...]:
    return STAGE_PLAN[STAGE_PLAN.index(stage) :]
