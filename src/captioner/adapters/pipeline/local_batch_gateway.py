"""Local durable Batch gateway for GUI-owned serial Pipeline execution."""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from captioner.adapters.persistence.batch_lease import inspect_batch_lease
from captioner.adapters.persistence.filesystem_batch_catalog import FilesystemBatchCatalog
from captioner.adapters.persistence.json_manifest_store import JsonManifestStore
from captioner.adapters.persistence.jsonl_journal import JsonlJournal
from captioner.core.application.durable_pipeline import (
    clear_pause_marker,
    write_cancel_marker,
    write_pause_marker,
)
from captioner.core.application.input_selection import BatchDraft
from captioner.core.application.job_detail import resolve_earliest_retry_stage
from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobConfig, JobState, validate_identifier
from captioner.core.domain.journal import JournalEvent, replay
from captioner.core.domain.result import FrozenJsonValue
from captioner.core.domain.stage import PipelineProfile, StageName, StageState
from captioner.core.ports.batch_catalog import LeaseExecutionState
from captioner.core.ports.batch_gateway import (
    CreatedBatch,
    JobDetailSource,
    RecoveryReadResult,
    RecoverySource,
    RecoverySourceIssue,
)
from captioner.infrastructure.app_paths import AppPaths, resolve_safe_child
from captioner.infrastructure.ids import new_id

_OUTPUT_SUFFIXES = (
    ".transcript.json",
    ".subtitle.json",
    ".srt",
    ".vtt",
    ".ass",
)
_TERMINAL_JOB_STATES = frozenset(
    {
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.CANCELLED,
    }
)


@dataclass(slots=True)
class LocalBatchGateway:
    """Durable Batch operations for the GUI Application boundary."""

    paths: AppPaths
    _shared_runtime: Any | None = field(default=None, init=False, repr=False)
    _shared_runtime_snapshot: Mapping[str, FrozenJsonValue] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def create_batch(self, draft: BatchDraft) -> CreatedBatch:
        batch_id = new_id("batch-")
        batch_dir = resolve_safe_child(self.paths.batches_dir, batch_id, field="batch_id")
        if batch_dir.exists():
            raise AppError("batch.exists", {"batch_id": batch_id})
        try:
            batch_dir.mkdir(parents=True, exist_ok=False)
            (batch_dir / "control").mkdir(parents=True, exist_ok=True)
            jobs = self._build_jobs(draft, batch_id=batch_id)
            llm = jobs[0][2].llm
            # Lazy bootstrap import keeps gui_bootstrap lightweight.
            from captioner.bootstrap import build_durable_service

            bundle = build_durable_service(
                batch_id,
                model_ref=draft.model_ref,
                device=draft.device,
                compute_type=draft.compute_type,
                language=draft.source_language,
                ffmpeg_bin=draft.ffmpeg_bin,
                ffprobe_bin=draft.ffprobe_bin,
                paths=self.paths,
                pipeline_profile=draft.pipeline_profile,
                llm=llm,
                initialize_runtime=False,
            )
            bundle.service.create(batch_id, jobs)
            return CreatedBatch(
                batch_id=batch_id,
                job_ids=tuple(job_id for job_id, _, _ in jobs),
            )
        except Exception:
            self._cleanup_empty_batch_dir(batch_dir)
            raise

    def execute_created_batch(self, batch_id: str) -> None:
        self._run_with_lease(batch_id, operation="execute")

    def validate_resume(self, batch_id: str) -> None:
        """Read-only Resume preflight; never repairs or mutates Journal."""
        validate_identifier(batch_id, field="batch_id")
        projection = self._read_projection(batch_id, repair=False)
        self._validate_resume_state(batch_id, projection)

    def resume_batch(self, batch_id: str) -> None:
        # Lease-protected path revalidates after repair under the writer lease.
        self._run_with_lease(batch_id, operation="resume")

    def resolve_retry_stage(self, batch_id: str, job_id: str) -> StageName:
        """Read-only Retry preflight; never repairs or mutates Journal."""
        validate_identifier(batch_id, field="batch_id")
        validate_identifier(job_id, field="job_id")
        projection = self._read_projection(batch_id, repair=False)
        return self._resolve_retry_stage_from_projection(batch_id, projection, job_id)

    def retry_job(self, batch_id: str, job_id: str, stage: StageName) -> None:
        validate_identifier(batch_id, field="batch_id")
        validate_identifier(job_id, field="job_id")
        # Lease-protected path revalidates that stage remains earliest retryable.
        self._run_with_lease(batch_id, operation="retry", job_id=job_id, stage=stage)

    def request_cancel(
        self,
        batch_id: str,
        *,
        job_id: str | None,
        execution_scheduled: bool,
    ) -> None:
        validate_identifier(batch_id, field="batch_id")
        if job_id is not None:
            validate_identifier(job_id, field="job_id")
        batch_dir = resolve_safe_child(self.paths.batches_dir, batch_id, field="batch_id")
        projection = self._read_projection(batch_id, repair=False)
        if job_id is not None:
            job = projection.job(job_id)
            if job.state in _TERMINAL_JOB_STATES:
                raise AppError("batch.cancel_invalid", {"reason": "terminal"})
        elif all(job.state in _TERMINAL_JOB_STATES for job in projection.jobs):
            raise AppError("batch.cancel_invalid", {"reason": "terminal"})

        control = batch_dir / "control"
        write_cancel_marker(control, job_id=job_id)
        if job_id is None:
            clear_pause_marker(control)

        if execution_scheduled:
            # Active/queued: marker acknowledgement only; executor cooperates.
            return

        # Inactive: finalize cancellation under a Batch lease when possible.
        lease_state = inspect_batch_lease(batch_dir / "lease.json")
        if lease_state in {"active_local", "active_remote"}:
            return
        self._finalize_cancel(batch_id)

    def request_pause(
        self,
        batch_id: str,
        *,
        execution_scheduled: bool,
    ) -> None:
        del execution_scheduled
        validate_identifier(batch_id, field="batch_id")
        projection = self._read_projection(batch_id, repair=False)
        if all(job.state in _TERMINAL_JOB_STATES for job in projection.jobs):
            raise AppError("batch.pause_invalid", {"reason": "terminal"})
        batch_dir = resolve_safe_child(self.paths.batches_dir, batch_id, field="batch_id")
        if (batch_dir / "control" / "cancel-batch").exists():
            raise AppError("batch.pause_invalid", {"reason": "cancel_requested"})
        write_pause_marker(batch_dir / "control")

    def create_run_again(self, batch_id: str, job_id: str) -> CreatedBatch:
        validate_identifier(batch_id, field="batch_id")
        validate_identifier(job_id, field="job_id")
        source = self._read_projection(batch_id, repair=False)
        job = source.job(job_id)
        if job.state not in _TERMINAL_JOB_STATES:
            raise AppError("batch.run_again_invalid", {"reason": "not_terminal"})
        input_path = Path(job.input_path)
        if not input_path.is_file():
            raise AppError("recovery.input_missing", {"path": job.input_path})

        new_batch_id = new_id("batch-")
        new_job_id = "job-000001"
        original_output = Path(job.config.output_dir)
        new_output = (original_output / "run-again" / new_batch_id / new_job_id).resolve()
        new_output.mkdir(parents=True, exist_ok=True)
        new_config = replace(
            job.config,
            output_dir=str(new_output),
            overwrite=False,
        )
        batch_dir = resolve_safe_child(self.paths.batches_dir, new_batch_id, field="batch_id")
        try:
            batch_dir.mkdir(parents=True, exist_ok=False)
            (batch_dir / "control").mkdir(parents=True, exist_ok=True)
            from captioner.bootstrap import build_durable_service

            bundle = build_durable_service(
                new_batch_id,
                model_ref=new_config.model_ref,
                device=new_config.device,
                compute_type=new_config.compute_type,
                language=new_config.language,
                ffmpeg_bin=new_config.ffmpeg_bin,
                ffprobe_bin=new_config.ffprobe_bin,
                paths=self.paths,
                segmentation=new_config.segmentation,
                pipeline_profile=new_config.pipeline_profile,
                llm=new_config.llm,
                initialize_runtime=False,
            )
            bundle.service.create(
                new_batch_id,
                ((new_job_id, input_path.resolve(), new_config),),
            )
            return CreatedBatch(batch_id=new_batch_id, job_ids=(new_job_id,))
        except Exception:
            self._cleanup_empty_batch_dir(batch_dir)
            raise

    def read_job_detail_source(self, batch_id: str, job_id: str) -> JobDetailSource:
        validate_identifier(batch_id, field="batch_id")
        validate_identifier(job_id, field="job_id")
        batch_dir = resolve_safe_child(self.paths.batches_dir, batch_id, field="batch_id")
        journal = JsonlJournal(batch_dir / "journal.jsonl")
        snapshot = journal.read_snapshot()
        if not snapshot.events:
            raise AppError("batch.not_found", {"batch_id": batch_id})
        projection = replay(snapshot.events)
        job = projection.job(job_id)
        lease_state = inspect_batch_lease(batch_dir / "lease.json")
        control = batch_dir / "control"
        batch_cancel = (control / "cancel-batch").is_file()
        job_cancel = (control / f"cancel-{job_id}").is_file()
        pause_requested = (control / "pause-batch").is_file()
        cancel_requested = batch_cancel or job_cancel
        state = _project_job_state(job.state, lease_state)
        active_stage, active_stage_state, attempt = _active_stage(job)
        if state is JobState.INTERRUPTED and active_stage_state is StageState.RUNNING:
            active_stage_state = StageState.INTERRUPTED
        manifest_status = JsonManifestStore(batch_dir / "manifest.json").inspect(projection)
        nonterminal = [item for item in projection.jobs if item.state not in _TERMINAL_JOB_STATES]
        batch_inputs_available = all(Path(item.input_path).is_file() for item in nonterminal)
        return JobDetailSource(
            batch_id=batch_id,
            job_id=job_id,
            input_path=job.input_path,
            output_dir=job.config.output_dir,
            state=state,
            active_stage=active_stage,
            active_stage_state=active_stage_state,
            active_stage_attempt=attempt,
            lease_state=lease_state,
            cancel_requested=cancel_requested,
            pause_requested=pause_requested,
            input_exists=Path(job.input_path).is_file(),
            batch_inputs_available=batch_inputs_available,
            batch_has_nonterminal=bool(nonterminal),
            batch_cancel_requested=batch_cancel,
            job_cancel_requested=job_cancel,
            events=snapshot.events,
            journal_tail_status=snapshot.tail_status,
            manifest_status=manifest_status,
            stage_states=tuple((stage.name, stage.state) for stage in job.stages),
            pipeline_profile=job.config.pipeline_profile.value,
        )

    def read_recovery_sources(self) -> RecoveryReadResult:
        catalog = FilesystemBatchCatalog(self.paths.batches_dir).read_snapshot()
        sources: list[RecoverySource] = []
        for entry in catalog.batches:
            missing: list[str] = []
            for job in entry.projection.jobs:
                if job.state in _TERMINAL_JOB_STATES:
                    continue
                if not Path(job.input_path).is_file():
                    missing.append(job.input_path)
            sources.append(
                RecoverySource(
                    batch_id=entry.batch_id,
                    created_at_utc=entry.created_at_utc,
                    state=entry.projection.state,
                    job_count=len(entry.projection.jobs),
                    pause_requested=entry.batch_pause_requested,
                    missing_input_paths=tuple(missing),
                    last_event_seq=entry.projection.last_event_seq,
                    lease_state=entry.lease_state,
                    projection=entry.projection,
                )
            )
        issues = tuple(
            RecoverySourceIssue(batch_name=issue.batch_name, code=issue.code)
            for issue in catalog.issues
        )
        return RecoveryReadResult(sources=tuple(sources), issues=issues)

    def close_shared_runtime(self) -> None:
        runtime = self._shared_runtime
        self._shared_runtime = None
        self._shared_runtime_snapshot = None
        if runtime is None:
            return
        try:
            asyncio.run(runtime.close())
        except RuntimeError:
            # Nested event loop — close synchronously via client if possible.
            close = getattr(runtime, "close", None)
            if close is not None:
                try:
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(close())
                    finally:
                        loop.close()
                except Exception:
                    return

    def _build_jobs(
        self,
        draft: BatchDraft,
        *,
        batch_id: str,
    ) -> tuple[tuple[str, Path, JobConfig], ...]:
        from captioner.bootstrap import create_job_config, create_llm_job_snapshot

        output_root = Path(draft.output_root).expanduser()
        try:
            output_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AppError("output.directory_failed", {"path": str(output_root)}) from exc
        if not output_root.is_dir() or output_root.is_symlink():
            raise AppError("output.directory_invalid", {"path": str(output_root)})
        resolved_root = output_root.resolve()

        llm_snapshot = None
        if draft.pipeline_profile in {PipelineProfile.FAST, PipelineProfile.QUALITY}:
            if draft.target_language is None:
                raise AppError("llm.target_language_missing")
            llm_snapshot = create_llm_job_snapshot(
                target_language=draft.target_language,
                provider_profile=draft.provider_profile,
                source_language=draft.source_language,
                paths=self.paths,
                pipeline_profile=draft.pipeline_profile,
            )

        jobs: list[tuple[str, Path, JobConfig]] = []
        collision_targets: set[str] = set()
        for index, path_text in enumerate(draft.input_paths, start=1):
            source = Path(path_text).expanduser().resolve()
            if not source.is_file():
                raise AppError("media.input_missing", {"path": str(source)})
            job_id = f"job-{index:06d}"
            if draft.collision_policy == "unique_subdir":
                job_output = (resolved_root / batch_id / job_id).resolve()
                job_output.mkdir(parents=True, exist_ok=True)
                overwrite = False
            else:
                job_output = resolved_root
                overwrite = draft.collision_policy == "overwrite"
            config = create_job_config(
                model_ref=draft.model_ref,
                device=draft.device,
                compute_type=draft.compute_type,
                language=draft.source_language,
                ffmpeg_bin=draft.ffmpeg_bin,
                ffprobe_bin=draft.ffprobe_bin,
                output_dir=job_output,
                overwrite=overwrite,
                paths=self.paths,
                pipeline_profile=draft.pipeline_profile,
                llm=llm_snapshot,
            )
            for suffix in _OUTPUT_SUFFIXES:
                target = job_output / f"{source.stem}{suffix}"
                key = os.path.normcase(str(target))
                if key in collision_targets:
                    raise AppError("batch.output_collision", {"logical_name": target.name})
                collision_targets.add(key)
                if draft.collision_policy == "fail" and target.exists():
                    raise AppError("batch.output_exists", {"logical_name": target.name})
            jobs.append((job_id, source, config))
        return tuple(jobs)

    def _run_with_lease(
        self,
        batch_id: str,
        *,
        operation: str,
        job_id: str | None = None,
        stage: StageName | None = None,
    ) -> None:
        from captioner.bootstrap import build_durable_service, create_batch_lease

        validate_identifier(batch_id, field="batch_id")
        batch_dir = resolve_safe_child(self.paths.batches_dir, batch_id, field="batch_id")
        # Writer lease must precede any Journal repair (repair truncates incomplete tails).
        lease = create_batch_lease(batch_dir)
        lease.acquire()
        try:
            projection = self._read_projection(
                batch_id,
                repair=operation != "execute",
            )
            if operation == "resume":
                self._validate_resume_state(batch_id, projection, check_lease=False)
            elif operation == "retry":
                if job_id is None or stage is None:
                    raise AppError("batch.retry_invalid", {"reason": "missing_stage"})
                resolved = self._resolve_retry_stage_from_projection(
                    batch_id,
                    projection,
                    job_id,
                    check_lease=False,
                )
                if resolved is not stage:
                    raise AppError("batch.retry_invalid", {"reason": "stage_changed"})
            config = projection.jobs[0].config
            runtime = self._runtime_for(config)
            bundle = build_durable_service(
                batch_id,
                model_ref=config.model_ref,
                device=config.device,
                compute_type=config.compute_type,
                language=config.language,
                ffmpeg_bin=config.ffmpeg_bin,
                ffprobe_bin=config.ffprobe_bin,
                paths=self.paths,
                segmentation=config.segmentation,
                pipeline_profile=config.pipeline_profile,
                llm=config.llm,
                llm_runtime=runtime,
                initialize_runtime=runtime is not None
                or config.pipeline_profile is PipelineProfile.DETERMINISTIC,
            )
            # Do not close shared runtime via bundle.close().
            if operation == "execute":
                asyncio.run(bundle.service.run(projection))
            elif operation == "resume":
                asyncio.run(bundle.service.resume())
            elif operation == "retry":
                if job_id is None or stage is None:
                    raise AppError("batch.retry_invalid", {"reason": "missing_stage"})
                asyncio.run(bundle.service.retry(job_id, stage))
            else:
                raise AppError("batch.command_invalid", {"field": "operation"})
        finally:
            lease.release()

    def _runtime_for(self, config: JobConfig) -> Any | None:
        if config.pipeline_profile is PipelineProfile.DETERMINISTIC:
            return None
        if config.llm is None:
            raise AppError("llm.config_missing", {"reason": "job_snapshot"})
        from captioner.bootstrap import build_llm_runtime, validate_llm_runtime_snapshot

        snapshot = config.llm
        if self._shared_runtime is not None and self._shared_runtime_snapshot is not None:
            try:
                validate_llm_runtime_snapshot(self._shared_runtime, snapshot)
            except AppError:
                self.close_shared_runtime()
            else:
                return self._shared_runtime
        provider_profile = snapshot.get("provider_profile", "default")
        if not isinstance(provider_profile, str):
            provider_profile = "default"
        runtime = build_llm_runtime(
            provider_profile=provider_profile,
            paths=self.paths,
            expected_snapshot=snapshot,
        )
        self._shared_runtime = runtime
        self._shared_runtime_snapshot = snapshot
        return runtime

    def _finalize_cancel(self, batch_id: str) -> None:
        from captioner.bootstrap import build_durable_service, create_batch_lease

        batch_dir = resolve_safe_child(self.paths.batches_dir, batch_id, field="batch_id")
        # Durable cancellation acknowledgement repairs Journal under the writer lease.
        lease = create_batch_lease(batch_dir)
        lease.acquire()
        try:
            projection = self._read_projection(batch_id, repair=True)
            config = projection.jobs[0].config
            bundle = build_durable_service(
                batch_id,
                model_ref=config.model_ref,
                device=config.device,
                compute_type=config.compute_type,
                language=config.language,
                ffmpeg_bin=config.ffmpeg_bin,
                ffprobe_bin=config.ffprobe_bin,
                paths=self.paths,
                segmentation=config.segmentation,
                pipeline_profile=config.pipeline_profile,
                llm=config.llm,
                initialize_runtime=False,
            )
            updated = bundle.service.acknowledge_cancel_requests(
                projection,
                active_job_id=None,
            )
            del updated
        finally:
            lease.release()

    def _validate_resume_state(
        self,
        batch_id: str,
        projection: BatchProjection,
        *,
        check_lease: bool = True,
    ) -> None:
        batch_dir = resolve_safe_child(self.paths.batches_dir, batch_id, field="batch_id")
        if check_lease:
            lease_state = inspect_batch_lease(batch_dir / "lease.json")
            if lease_state in {"active_local", "active_remote"}:
                raise AppError("batch.busy", {"batch_id": batch_id, "lease": lease_state})
        if (batch_dir / "control" / "cancel-batch").exists():
            raise AppError("batch.resume_invalid", {"reason": "cancel_requested"})
        nonterminal = [job for job in projection.jobs if job.state not in _TERMINAL_JOB_STATES]
        if not nonterminal:
            raise AppError("batch.resume_invalid", {"reason": "terminal"})
        missing = [job.input_path for job in nonterminal if not Path(job.input_path).is_file()]
        if missing:
            raise AppError("recovery.input_missing", {"paths": list(missing)})

    def _resolve_retry_stage_from_projection(
        self,
        batch_id: str,
        projection: BatchProjection,
        job_id: str,
        *,
        check_lease: bool = True,
    ) -> StageName:
        batch_dir = resolve_safe_child(self.paths.batches_dir, batch_id, field="batch_id")
        if check_lease:
            lease_state = inspect_batch_lease(batch_dir / "lease.json")
            if lease_state in {"active_local", "active_remote"}:
                raise AppError("batch.busy", {"batch_id": batch_id, "lease": lease_state})
        job = projection.job(job_id)
        if job.state not in {JobState.FAILED, JobState.CANCELLED, JobState.INTERRUPTED}:
            raise AppError("batch.retry_invalid", {"reason": "job_state"})
        if not Path(job.input_path).is_file():
            raise AppError("recovery.input_missing", {"paths": [job.input_path]})
        return resolve_earliest_retry_stage(
            tuple((stage.name, stage.state) for stage in job.stages),
            job.config.pipeline_profile.value,
        )

    def _validate_inputs_present(self, batch_id: str, *, job_id: str | None = None) -> None:
        projection = self._read_projection(batch_id, repair=False)
        jobs = projection.jobs if job_id is None else (projection.job(job_id),)
        missing = [
            job.input_path
            for job in jobs
            if job.state not in _TERMINAL_JOB_STATES and not Path(job.input_path).is_file()
        ]
        if missing:
            raise AppError("recovery.input_missing", {"paths": list(missing)})

    def _read_projection(self, batch_id: str, *, repair: bool) -> BatchProjection:
        batch_dir = resolve_safe_child(self.paths.batches_dir, batch_id, field="batch_id")
        journal = JsonlJournal(batch_dir / "journal.jsonl")
        events: tuple[JournalEvent, ...]
        events = journal.repair_and_read() if repair else journal.read_snapshot().events
        if not events:
            raise AppError("batch.not_found", {"batch_id": batch_id})
        return replay(events)

    def _cleanup_empty_batch_dir(self, batch_dir: Path) -> None:
        if not batch_dir.exists():
            return
        try:
            # Only remove if Journal was never made durable.
            journal = batch_dir / "journal.jsonl"
            if journal.exists() and journal.stat().st_size > 0:
                return
            shutil.rmtree(batch_dir)
        except OSError as exc:
            raise AppError("batch.cleanup_failed", {"batch_id": batch_dir.name}) from exc


def _project_job_state(state: JobState, lease_state: LeaseExecutionState) -> JobState:
    if state is JobState.RUNNING and lease_state in {"missing", "stale", "invalid"}:
        return JobState.INTERRUPTED
    return state


def _active_stage(
    job: Any,
) -> tuple[StageName | None, StageState | None, int]:
    for stage in job.stages:
        if stage.state is StageState.RUNNING:
            return stage.name, stage.state, stage.attempt
    for stage in job.stages:
        if stage.state in {
            StageState.INTERRUPTED,
            StageState.FAILED,
            StageState.CANCELLED,
        }:
            return stage.name, stage.state, stage.attempt
    for stage in job.stages:
        if stage.state is not StageState.COMMITTED:
            return stage.name, stage.state, stage.attempt
    return None, None, 0


__all__ = ["LocalBatchGateway"]
