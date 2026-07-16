"""Stage execution boundary; runners cannot commit durable state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from captioner.core.domain.artifact import ArtifactRef
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.job import JobConfig
from captioner.core.domain.stage import StageName


@dataclass(frozen=True, slots=True)
class ProducedArtifact:
    kind: str
    media_type: str
    logical_name: str
    source_path: Path | None = None
    data: bytes | None = None

    def __post_init__(self) -> None:
        if (self.source_path is None) == (self.data is None):
            raise ValueError


@dataclass(frozen=True, slots=True)
class StageExecutionRequest:
    batch_id: str
    job_id: str
    input_path: Path
    config: JobConfig
    input_artifacts: tuple[ArtifactRef, ...]


@dataclass(frozen=True, slots=True)
class StageExecutionContext:
    execution: ExecutionContext
    workspace: Path


class StageRunner(Protocol):
    @property
    def name(self) -> StageName: ...

    @property
    def version(self) -> str: ...

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]: ...
