"""Durable Job configuration and projection."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.llm_job_config import LLMJobSnapshot
from captioner.core.domain.result import (
    FrozenJsonValue,
    JsonValue,
    freeze_json_value,
    thaw_json_value,
)
from captioner.core.domain.stage import (
    PipelineProfile,
    StageName,
    StageProjection,
    stage_plan_for,
)

JOB_CONFIG_SCHEMA_VERSION = 2
LEGACY_JOB_CONFIG_SCHEMA_VERSION = 1
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def validate_identifier(value: str, *, field: str) -> str:
    if (
        not value
        or value != value.strip()
        or _ID_RE.fullmatch(value) is None
        or ".." in value
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PureWindowsPath(value).drive
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
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
    pipeline_profile: PipelineProfile = PipelineProfile.DETERMINISTIC
    llm: Mapping[str, FrozenJsonValue] | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in {
            LEGACY_JOB_CONFIG_SCHEMA_VERSION,
            JOB_CONFIG_SCHEMA_VERSION,
        }:
            raise AppError("job.config_invalid", {"field": "schema_version"})
        try:
            profile = PipelineProfile(self.pipeline_profile)
        except ValueError as exc:
            raise AppError("job.config_invalid", {"field": "pipeline_profile"}) from exc
        if self.schema_version == LEGACY_JOB_CONFIG_SCHEMA_VERSION:
            profile = PipelineProfile.DETERMINISTIC
            if self.llm is not None:
                raise AppError("job.config_invalid", {"field": "llm"})
        object.__setattr__(self, "pipeline_profile", profile)
        required = (self.model_ref, self.model_identity, self.device, self.compute_type)
        if any(not value.strip() for value in required):
            raise AppError("job.config_invalid", {"field": "model"})
        if self.language is not None and not self.language.strip():
            raise AppError("job.config_invalid", {"field": "language"})
        if not self.ffmpeg_bin.strip() or not self.ffprobe_bin.strip():
            raise AppError("job.config_invalid", {"field": "executables"})
        if not Path(self.output_dir).is_absolute():
            raise AppError("job.config_invalid", {"field": "output_dir"})
        expected_stage_names = {stage.value for stage in stage_plan_for(profile)}
        if set(self.stage_versions) != expected_stage_names:
            raise AppError("job.config_invalid", {"field": "stage_versions"})
        if any(
            not isinstance(version, str) or not version.strip()
            for version in self.stage_versions.values()
        ):
            raise AppError("job.config_invalid", {"field": "stage_versions"})
        for name in ("normalization", "segmentation", "stage_versions"):
            try:
                frozen = freeze_json_value(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise AppError("job.config_invalid", {"field": name}) from exc
            object.__setattr__(self, name, cast(Mapping[str, FrozenJsonValue], frozen))
        if profile is PipelineProfile.DETERMINISTIC and self.llm is not None:
            raise AppError("job.config_invalid", {"field": "llm"})
        if profile is not PipelineProfile.DETERMINISTIC and self.llm is None:
            raise AppError("job.config_invalid", {"field": "llm"})
        if self.llm is not None:
            try:
                frozen_llm = freeze_json_value(self.llm)
            except (TypeError, ValueError) as exc:
                raise AppError("job.config_invalid", {"field": "llm"}) from exc
            if not isinstance(frozen_llm, Mapping):
                raise AppError("job.config_invalid", {"field": "llm"})
            if _contains_secret_key(frozen_llm):
                raise AppError("job.config_invalid", {"field": "llm"})
            if profile is not PipelineProfile.DETERMINISTIC:
                snapshot = LLMJobSnapshot.from_mapping(thaw_json_value(frozen_llm))
                if snapshot.profile is not profile:
                    raise AppError("job.config_invalid", {"field": "llm.profile"})
                frozen_llm = snapshot.to_mapping()
            object.__setattr__(
                self,
                "llm",
                cast(Mapping[str, FrozenJsonValue], frozen_llm),
            )

    def to_dict(self) -> dict[str, JsonValue]:
        from captioner.core.domain.result import thaw_json_value

        value: dict[str, JsonValue] = {
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
        if self.schema_version == JOB_CONFIG_SCHEMA_VERSION:
            value["pipeline_profile"] = self.pipeline_profile.value
            value["llm"] = None if self.llm is None else thaw_json_value(self.llm)
        return value

    @property
    def runtime_signature(self) -> tuple[object, ...]:
        return (
            self.model_ref,
            self.model_identity,
            self.device,
            self.compute_type,
            self.language,
            self.vad_filter,
            self.ffmpeg_bin,
            self.ffprobe_bin,
            _hashable_json(self.normalization),
            _hashable_json(self.segmentation),
            _hashable_json(self.stage_versions),
            self.pipeline_profile.value,
            _hashable_json(self.llm) if self.llm is not None else None,
        )

    @property
    def target_language(self) -> str | None:
        return _llm_string(self.llm, "target_language")

    @property
    def provider_profile(self) -> str | None:
        return _llm_string(self.llm, "provider_profile")

    @property
    def llm_model(self) -> str | None:
        return _llm_string(self.llm, "model")


@dataclass(frozen=True, slots=True)
class JobProjection:
    job_id: str
    input_path: str
    config: JobConfig
    state: JobState = JobState.PENDING
    stages: tuple[StageProjection, ...] = ()

    def __post_init__(self) -> None:
        validate_identifier(self.job_id, field="job_id")
        if not Path(self.input_path).is_absolute():
            raise AppError("job.config_invalid", {"field": "input_path"})
        expected = stage_plan_for(self.config.pipeline_profile)
        stages = tuple(self.stages)
        if not stages:
            stages = tuple(StageProjection(name) for name in expected)
        if tuple(stage.name for stage in stages) != expected:
            raise AppError("job.stage_plan_invalid", {"job_id": self.job_id})
        object.__setattr__(self, "stages", stages)

    def stage(self, name: StageName) -> StageProjection:
        try:
            index = next(index for index, stage in enumerate(self.stages) if stage.name is name)
            return self.stages[index]
        except StopIteration as exc:
            raise AppError(
                "job.stage_unavailable",
                {"job_id": self.job_id, "stage_name": name.value},
            ) from exc


def _hashable_json(value: FrozenJsonValue) -> object:
    if isinstance(value, Mapping):
        return tuple(sorted((key, _hashable_json(item)) for key, item in value.items()))
    if isinstance(value, tuple):
        return tuple(_hashable_json(item) for item in value)
    return value


def _contains_secret_key(value: FrozenJsonValue) -> bool:
    if isinstance(value, Mapping):
        return any(
            key.lower().replace("-", "_")
            in {"api_key", "authorization", "access_token", "token", "secret", "password"}
            or _contains_secret_key(item)
            for key, item in value.items()
        )
    if isinstance(value, tuple):
        return any(_contains_secret_key(item) for item in value)
    return False


def _llm_string(value: Mapping[str, FrozenJsonValue] | None, key: str) -> str | None:
    if value is None:
        return None
    item = value.get(key)
    return item if isinstance(item, str) else None
