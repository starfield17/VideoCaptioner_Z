"""Durable Job configuration and projection."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.result import FrozenJsonValue, JsonValue, freeze_json_value
from captioner.core.domain.stage import STAGE_PLAN, StageName, StageProjection

JOB_CONFIG_SCHEMA_VERSION = 1
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def validate_identifier(value: str, *, field: str) -> str:
    if _ID_RE.fullmatch(value) is None or ".." in value:
        raise AppError("job.identity_invalid", {"field": field})
    return value


class JobState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"


@dataclass(frozen=True, slots=True)
class JobConfig:
    model_ref: str
    model_identity: str
    device: str
    compute_type: str
    language: str | None
    vad_filter: bool
    ffmpeg_bin: str
    ffprobe_bin: str
    normalization: Mapping[str, FrozenJsonValue]
    segmentation: Mapping[str, FrozenJsonValue]
    output_dir: str
    overwrite: bool
    stage_versions: Mapping[str, FrozenJsonValue]
    schema_version: int = JOB_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != JOB_CONFIG_SCHEMA_VERSION:
            raise AppError("job.config_invalid", {"field": "schema_version"})
        required = (self.model_ref, self.model_identity, self.device, self.compute_type)
        if any(not value.strip() for value in required):
            raise AppError("job.config_invalid", {"field": "model"})
        if self.language is not None and not self.language.strip():
            raise AppError("job.config_invalid", {"field": "language"})
        if not self.ffmpeg_bin.strip() or not self.ffprobe_bin.strip():
            raise AppError("job.config_invalid", {"field": "executables"})
        if not Path(self.output_dir).is_absolute():
            raise AppError("job.config_invalid", {"field": "output_dir"})
        for name in ("normalization", "segmentation", "stage_versions"):
            frozen = freeze_json_value(getattr(self, name))
            object.__setattr__(self, name, cast(Mapping[str, FrozenJsonValue], frozen))

    def to_dict(self) -> dict[str, JsonValue]:
        from captioner.core.domain.result import thaw_json_value

        return {
            "schema_version": self.schema_version,
            "model_ref": self.model_ref,
            "model_identity": self.model_identity,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language,
            "vad_filter": self.vad_filter,
            "ffmpeg_bin": self.ffmpeg_bin,
            "ffprobe_bin": self.ffprobe_bin,
            "normalization": thaw_json_value(self.normalization),
            "segmentation": thaw_json_value(self.segmentation),
            "output_dir": self.output_dir,
            "overwrite": self.overwrite,
            "stage_versions": thaw_json_value(self.stage_versions),
        }


@dataclass(frozen=True, slots=True)
class JobProjection:
    job_id: str
    input_path: str
    config: JobConfig
    state: JobState = JobState.PENDING
    stages: tuple[StageProjection, ...] = tuple(StageProjection(name) for name in STAGE_PLAN)

    def __post_init__(self) -> None:
        validate_identifier(self.job_id, field="job_id")
        if not Path(self.input_path).is_absolute():
            raise AppError("job.config_invalid", {"field": "input_path"})

    def stage(self, name: StageName) -> StageProjection:
        return self.stages[STAGE_PLAN.index(name)]
