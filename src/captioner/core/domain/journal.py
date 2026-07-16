"""Journal event schema and pure immutable replay."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Never, cast

from captioner.core.domain.artifact import ArtifactRef
from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobConfig, JobProjection, JobState, validate_identifier
from captioner.core.domain.result import (
    FrozenJsonValue,
    JsonValue,
    freeze_json_value,
    thaw_json_value,
)
from captioner.core.domain.stage import (
    STAGE_PLAN,
    StageName,
    StageProjection,
    StageState,
    dependencies,
    stage_suffix,
)

JOURNAL_SCHEMA_VERSION = 1
EVENT_TYPES = frozenset(
    {
        "batch.created",
        "batch.config_updated",
        "job.created",
        "job.config_updated",
        "job.retry_requested",
        "stage.started",
        "stage.interrupted",
        "stage.committed",
        "stage.failed",
        "stage.cancelled",
        "stage.invalidated",
        "job.succeeded",
        "job.failed",
        "job.cancelled",
    }
)


@dataclass(frozen=True, slots=True)
class JournalEvent:
    seq: int
    event_id: str
    timestamp_utc: str
    batch_id: str
    type: str
    payload: Mapping[str, FrozenJsonValue]
    schema_version: int = JOURNAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != JOURNAL_SCHEMA_VERSION or self.seq < 1:
            raise AppError("journal.event_invalid", {"field": "schema"})
        validate_identifier(self.event_id, field="event_id")
        validate_identifier(self.batch_id, field="batch_id")
        if self.type not in EVENT_TYPES:
            raise AppError("journal.event_invalid", {"field": "type"})
        try:
            parsed = datetime.fromisoformat(self.timestamp_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise AppError("journal.event_invalid", {"field": "timestamp_utc"}) from exc
        if parsed.tzinfo is None:
            raise AppError("journal.event_invalid", {"field": "timestamp_utc"})
        frozen = freeze_json_value(self.payload)
        object.__setattr__(self, "payload", cast(Mapping[str, FrozenJsonValue], frozen))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "seq": self.seq,
            "event_id": self.event_id,
            "timestamp_utc": self.timestamp_utc,
            "batch_id": self.batch_id,
            "type": self.type,
            "payload": thaw_json_value(self.payload),
        }

    @classmethod
    def from_dict(cls, value: object) -> JournalEvent:
        if not isinstance(value, Mapping):
            raise AppError("journal.corrupt", {"reason": "event_root"})
        raw = cast(Mapping[object, object], value)
        expected = {
            "schema_version",
            "seq",
            "event_id",
            "timestamp_utc",
            "batch_id",
            "type",
            "payload",
        }
        if set(raw) != expected:
            raise AppError("journal.corrupt", {"reason": "event_fields"})
        try:
            schema_version = raw["schema_version"]
            seq = raw["seq"]
            event_id = raw["event_id"]
            timestamp_utc = raw["timestamp_utc"]
            batch_id = raw["batch_id"]
            event_type = raw["type"]
            payload = raw["payload"]
            valid_types = (
                not isinstance(schema_version, int)
                or isinstance(schema_version, bool)
                or not isinstance(seq, int)
                or isinstance(seq, bool)
                or not isinstance(event_id, str)
                or not isinstance(timestamp_utc, str)
                or not isinstance(batch_id, str)
                or not isinstance(event_type, str)
                or not isinstance(payload, Mapping)
            )
            if valid_types:
                return _invalid_event_types()
            payload_mapping = cast(Mapping[object, object], payload)
            frozen = cast(Mapping[str, FrozenJsonValue], freeze_json_value(payload_mapping))
            return cls(
                cast(int, seq),
                cast(str, event_id),
                cast(str, timestamp_utc),
                cast(str, batch_id),
                cast(str, event_type),
                frozen,
                cast(int, schema_version),
            )
        except (TypeError, ValueError, AppError) as exc:
            if isinstance(exc, AppError) and exc.code == "journal.corrupt":
                raise
            raise AppError("journal.corrupt", {"reason": "event_schema"}) from exc


def _invalid_event_types() -> Never:
    raise TypeError


def replay(events: Iterable[JournalEvent]) -> BatchProjection:
    projection: BatchProjection | None = None
    for event in events:
        projection = apply_event(projection, event)
    if projection is None:
        raise AppError("journal.empty")
    return projection


def apply_event(
    projection: BatchProjection | None,
    event: JournalEvent,
) -> BatchProjection:
    if projection is None:
        if event.type != "batch.created" or event.seq != 1:
            raise AppError("journal.transition_invalid", {"reason": "batch_not_created"})
        return BatchProjection(
            event.batch_id, last_event_seq=1, event_ids=frozenset({event.event_id})
        )
    _validate_envelope(projection, event)
    if event.type == "batch.created":
        raise AppError("journal.transition_invalid", {"reason": "duplicate_batch"})
    updated = _apply_payload(projection, event)
    return replace(
        updated,
        last_event_seq=event.seq,
        event_ids=updated.event_ids | {event.event_id},
    )


def _validate_envelope(projection: BatchProjection, event: JournalEvent) -> None:
    if event.batch_id != projection.batch_id:
        raise AppError("journal.transition_invalid", {"reason": "batch_mismatch"})
    if event.seq != projection.last_event_seq + 1:
        raise AppError("journal.transition_invalid", {"reason": "sequence_gap"})
    if event.event_id in projection.event_ids:
        raise AppError("journal.transition_invalid", {"reason": "duplicate_event_id"})


def _apply_payload(projection: BatchProjection, event: JournalEvent) -> BatchProjection:
    if event.type == "batch.config_updated":
        return _update_batch_config(projection, event.payload)
    if event.type == "job.created":
        return _create_job(projection, event.payload)
    job_id = _required_str(event.payload, "job_id")
    job_index, job = _find_job(projection, job_id)
    if event.type == "job.config_updated":
        job = replace(job, config=_config_from_value(event.payload.get("config")))
    elif event.type == "job.retry_requested":
        job = _retry_job(job, event.payload)
    elif event.type.startswith("stage."):
        job = _apply_stage_event(job, event.type, event.payload)
    else:
        job = _apply_job_terminal(job, event.type)
    jobs = list(projection.jobs)
    jobs[job_index] = job
    return replace(projection, jobs=tuple(jobs))


def _update_batch_config(
    projection: BatchProjection,
    payload: Mapping[str, FrozenJsonValue],
) -> BatchProjection:
    if set(payload) != {"config", "earliest_stage"}:
        raise AppError("journal.transition_invalid", {"reason": "batch_config_fields"})
    config = _config_from_value(payload.get("config"))
    try:
        earliest_stage = StageName(_required_str(payload, "earliest_stage"))
    except ValueError as exc:
        raise AppError("journal.transition_invalid", {"reason": "stage_name"}) from exc
    jobs: list[JobProjection] = []
    for job in projection.jobs:
        stages = _invalidate_suffix(list(job.stages), earliest_stage)
        state = job.state
        if state not in {JobState.FAILED, JobState.CANCELLED}:
            state = JobState.PENDING
        jobs.append(replace(job, config=config, state=state, stages=tuple(stages)))
    return replace(projection, jobs=tuple(jobs))


def _create_job(
    projection: BatchProjection,
    payload: Mapping[str, FrozenJsonValue],
) -> BatchProjection:
    job_id = _required_str(payload, "job_id")
    if any(job.job_id == job_id for job in projection.jobs):
        raise AppError("journal.transition_invalid", {"reason": "duplicate_job"})
    input_path = _required_str(payload, "input_path")
    config = _config_from_value(payload.get("config"))
    return replace(projection, jobs=(*projection.jobs, JobProjection(job_id, input_path, config)))


def _apply_stage_event(
    job: JobProjection,
    event_type: str,
    payload: Mapping[str, FrozenJsonValue],
) -> JobProjection:
    try:
        stage_name = StageName(_required_str(payload, "stage_name"))
    except ValueError as exc:
        raise AppError("journal.transition_invalid", {"reason": "stage_name"}) from exc
    attempt = _required_int(payload, "attempt")
    current = job.stage(stage_name)
    stages = list(job.stages)
    if event_type == "stage.started":
        if attempt <= current.attempt or current.state is StageState.RUNNING:
            raise AppError("journal.transition_invalid", {"reason": "attempt_reused"})
        if any(
            job.stage(dep).state is not StageState.COMMITTED for dep in dependencies(stage_name)
        ):
            raise AppError("journal.transition_invalid", {"reason": "dependency_not_committed"})
        replacement = StageProjection(stage_name, StageState.RUNNING, attempt)
        job_state = JobState.RUNNING
    elif event_type == "stage.invalidated":
        if attempt != current.attempt or current.state is not StageState.COMMITTED:
            raise AppError("journal.transition_invalid", {"reason": "not_committed"})
        stages = _invalidate_suffix(stages, stage_name)
        return replace(job, state=JobState.PENDING, stages=tuple(stages))
    else:
        if current.state is not StageState.RUNNING or current.attempt != attempt:
            raise AppError("journal.transition_invalid", {"reason": "attempt_not_running"})
        replacement, job_state = _terminal_stage(event_type, current, payload)
    stages[STAGE_PLAN.index(stage_name)] = replacement
    return replace(job, state=job_state, stages=tuple(stages))


def _retry_job(job: JobProjection, payload: Mapping[str, FrozenJsonValue]) -> JobProjection:
    try:
        stage_name = StageName(_required_str(payload, "stage_name"))
    except ValueError as exc:
        raise AppError("journal.transition_invalid", {"reason": "stage_name"}) from exc
    current = job.stage(stage_name)
    if current.state is StageState.PENDING and job.state not in {
        JobState.FAILED,
        JobState.CANCELLED,
    }:
        raise AppError("journal.transition_invalid", {"reason": "retry_pending"})
    if any(job.stage(dep).state is not StageState.COMMITTED for dep in dependencies(stage_name)):
        raise AppError("journal.transition_invalid", {"reason": "retry_dependency"})
    stages = list(job.stages)
    for name in stage_suffix(stage_name):
        prior = stages[STAGE_PLAN.index(name)]
        stages[STAGE_PLAN.index(name)] = StageProjection(name, StageState.PENDING, prior.attempt)
    return replace(job, state=JobState.PENDING, stages=tuple(stages))


def _terminal_stage(
    event_type: str,
    current: StageProjection,
    payload: Mapping[str, FrozenJsonValue],
) -> tuple[StageProjection, JobState]:
    states = {
        "stage.interrupted": (StageState.INTERRUPTED, JobState.INTERRUPTED),
        "stage.failed": (StageState.FAILED, JobState.FAILED),
        "stage.cancelled": (StageState.CANCELLED, JobState.CANCELLED),
    }
    if event_type in states:
        stage_state, job_state = states[event_type]
        return replace(current, state=stage_state), job_state
    if event_type != "stage.committed":
        raise AppError("journal.transition_invalid", {"reason": "stage_event"})
    cache_key = _required_str(payload, "cache_key")
    if not cache_key.startswith("sha256:") or len(cache_key) != 71:
        raise AppError("journal.transition_invalid", {"reason": "cache_key"})
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, tuple):
        raise AppError("journal.transition_invalid", {"reason": "artifacts"})
    artifacts = tuple(ArtifactRef.from_dict(thaw_json_value(item)) for item in raw_artifacts)
    if not artifacts:
        raise AppError("journal.transition_invalid", {"reason": "artifacts"})
    return replace(
        current,
        state=StageState.COMMITTED,
        cache_key=cache_key,
        artifacts=artifacts,
    ), JobState.RUNNING


def _invalidate_suffix(
    stages: list[StageProjection],
    stage_name: StageName,
) -> list[StageProjection]:
    for name in stage_suffix(stage_name):
        current = stages[STAGE_PLAN.index(name)]
        if current.state is StageState.COMMITTED:
            stages[STAGE_PLAN.index(name)] = replace(current, state=StageState.INVALIDATED)
    return stages


def _apply_job_terminal(job: JobProjection, event_type: str) -> JobProjection:
    if job.state is JobState.SUCCEEDED:
        raise AppError("journal.transition_invalid", {"reason": "job_terminal"})
    if event_type == "job.succeeded":
        if any(stage.state is not StageState.COMMITTED for stage in job.stages):
            raise AppError("journal.transition_invalid", {"reason": "stages_uncommitted"})
        return replace(job, state=JobState.SUCCEEDED)
    if event_type == "job.cancelled":
        if job.state not in {
            JobState.PENDING,
            JobState.RUNNING,
            JobState.INTERRUPTED,
            JobState.CANCELLED,
        }:
            raise AppError("journal.transition_invalid", {"reason": "job_not_cancelled"})
        return replace(job, state=JobState.CANCELLED)
    if event_type == "job.failed":
        if job.state not in {JobState.PENDING, JobState.RUNNING, JobState.FAILED}:
            raise AppError("journal.transition_invalid", {"reason": "job_not_failed"})
        return replace(job, state=JobState.FAILED)
    raise AppError("journal.transition_invalid", {"reason": "job_event"})


def _find_job(projection: BatchProjection, job_id: str) -> tuple[int, JobProjection]:
    for index, job in enumerate(projection.jobs):
        if job.job_id == job_id:
            return index, job
    raise AppError("journal.transition_invalid", {"reason": "job_missing"})


def _required_str(payload: Mapping[str, FrozenJsonValue], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AppError("journal.transition_invalid", {"reason": key})
    return value


def _required_int(payload: Mapping[str, FrozenJsonValue], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise AppError("journal.transition_invalid", {"reason": key})
    return value


def _config_from_value(value: FrozenJsonValue | None) -> JobConfig:
    raw = thaw_json_value(value)
    if not isinstance(raw, dict):
        raise AppError("journal.transition_invalid", {"reason": "config"})
    expected = {
        "schema_version",
        "model_ref",
        "model_identity",
        "device",
        "compute_type",
        "language",
        "vad_filter",
        "ffmpeg_bin",
        "ffprobe_bin",
        "normalization",
        "segmentation",
        "output_dir",
        "overwrite",
        "stage_versions",
    }
    if set(raw) != expected:
        raise AppError("journal.transition_invalid", {"reason": "config_fields"})
    try:
        return JobConfig(
            model_ref=cast(str, raw["model_ref"]),
            model_identity=cast(str, raw["model_identity"]),
            device=cast(str, raw["device"]),
            compute_type=cast(str, raw["compute_type"]),
            language=cast(str | None, raw["language"]),
            vad_filter=cast(bool, raw["vad_filter"]),
            ffmpeg_bin=cast(str, raw["ffmpeg_bin"]),
            ffprobe_bin=cast(str, raw["ffprobe_bin"]),
            normalization=cast(Mapping[str, FrozenJsonValue], raw["normalization"]),
            segmentation=cast(Mapping[str, FrozenJsonValue], raw["segmentation"]),
            output_dir=cast(str, raw["output_dir"]),
            overwrite=cast(bool, raw["overwrite"]),
            stage_versions=cast(Mapping[str, FrozenJsonValue], raw["stage_versions"]),
            schema_version=cast(int, raw["schema_version"]),
        )
    except (TypeError, ValueError) as exc:
        raise AppError("journal.transition_invalid", {"reason": "config_types"}) from exc
