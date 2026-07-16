"""Phase 1 composition root for the one-shot CLI workflow."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from captioner.adapters.asr.faster_whisper import FasterWhisperConfig, FasterWhisperEngine
from captioner.adapters.exporters.srt import serialize_bytes as serialize_srt
from captioner.adapters.exporters.transcript_json import serialize_bytes as serialize_transcript
from captioner.adapters.media.ffmpeg_audio import FFmpegAudioNormalizer
from captioner.adapters.media.ffprobe import FFprobeMediaInspector
from captioner.adapters.persistence.batch_lease import BatchLease
from captioner.adapters.persistence.content_addressed_artifact_store import (
    ContentAddressedArtifactStore,
)
from captioner.adapters.persistence.json_manifest_store import JsonManifestStore
from captioner.adapters.persistence.jsonl_journal import JsonlJournal
from captioner.adapters.persistence.local_artifact_store import LocalArtifactStore
from captioner.adapters.pipeline.stages import (
    ExportStage,
    InspectStage,
    NormalizeStage,
    PublishStage,
    SegmentStage,
    TranscribeStage,
)
from captioner.adapters.process.asyncio_subprocess import AsyncioSubprocessRunner
from captioner.adapters.testing.fault_injector import ScriptedFaultInjector
from captioner.core.application.durable_pipeline import DurablePipelineService
from captioner.core.application.run_single import RunSingleService
from captioner.core.application.stage_executor import EventFactory, StageExecutor
from captioner.core.domain.job import JobConfig
from captioner.core.domain.journal import replay
from captioner.core.domain.stage import StageName
from captioner.core.policies.simple_segmentation import SimpleSegmentationConfig
from captioner.infrastructure.app_paths import (
    AppPaths,
    ensure_runtime_layout,
    resolve_app_paths,
    resolve_safe_child,
)
from captioner.infrastructure.ids import new_id


def build_run_service(
    *,
    model_ref: str = "tiny",
    device: str,
    compute_type: str,
    language: str | None,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    paths: AppPaths | None = None,
    model_id: str | None = None,
) -> RunSingleService:
    """Assemble concrete adapters for one CLI invocation."""
    if model_id is not None:
        if model_ref != "tiny" and model_ref != model_id:
            raise ValueError
        model_ref = model_id
    application_paths = resolve_app_paths() if paths is None else paths
    ensure_runtime_layout(application_paths)
    process = AsyncioSubprocessRunner()
    inspector = FFprobeMediaInspector(process, executable=ffprobe_bin)
    normalizer = FFmpegAudioNormalizer(process, executable=ffmpeg_bin)
    engine = FasterWhisperEngine(
        FasterWhisperConfig(
            model_ref=model_ref,
            device=device,
            compute_type=compute_type,
            language=language,
        )
    )
    return RunSingleService(
        inspector=inspector,
        normalizer=normalizer,
        asr_engine=engine,
        artifact_store_factory=LocalArtifactStore,
        transcript_serializer=serialize_transcript,
        subtitle_serializer=serialize_srt,
        temp_root=application_paths.temp_dir,
    )


@dataclass(frozen=True, slots=True)
class DurableServiceBundle:
    service: DurablePipelineService
    batch_dir: Path


def create_job_config(
    *,
    model_ref: str,
    device: str,
    compute_type: str,
    language: str | None,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    output_dir: Path,
    overwrite: bool,
) -> JobConfig:
    model = FasterWhisperConfig(model_ref, device, compute_type, language)
    return JobConfig(
        model.model_ref,
        model.model_identity,
        device,
        compute_type,
        language,
        True,
        ffmpeg_bin,
        ffprobe_bin,
        {"codec": "pcm_s16le", "sample_rate": 16000, "channels": 1},
        {"max_duration_ms": 7000, "max_text_units": 84, "hard_gap_ms": 700},
        str(output_dir.expanduser().resolve()),
        overwrite,
        {stage.value: "1" for stage in StageName},
    )


def create_batch_lease(batch_dir: Path) -> BatchLease:
    return BatchLease(
        batch_dir / "lease.json",
        new_id("lease-"),
        os.getpid(),
        socket.gethostname(),
        datetime.now(UTC).isoformat(),
        _pid_alive,
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_batch_config(batch_id: str, *, paths: AppPaths | None = None) -> JobConfig:
    application_paths = resolve_app_paths() if paths is None else paths
    batch_dir = resolve_safe_child(application_paths.batches_dir, batch_id, field="batch_id")
    events = JsonlJournal(batch_dir / "journal.jsonl").read_snapshot().events
    if not events:
        from captioner.core.domain.errors import AppError

        raise AppError("batch.not_found", {"batch_id": batch_id})
    projection = replay(events)
    if not projection.jobs:
        from captioner.core.domain.errors import AppError

        raise AppError("batch.invalid", {"batch_id": batch_id})
    return projection.jobs[0].config


def build_durable_service(
    batch_id: str,
    *,
    model_ref: str,
    device: str,
    compute_type: str,
    language: str | None,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    paths: AppPaths | None = None,
) -> DurableServiceBundle:
    application_paths = resolve_app_paths() if paths is None else paths
    ensure_runtime_layout(application_paths)
    batch_dir = resolve_safe_child(application_paths.batches_dir, batch_id, field="batch_id")
    process = AsyncioSubprocessRunner()
    durable = ContentAddressedArtifactStore(application_paths.artifacts_dir)
    config = FasterWhisperConfig(model_ref, device, compute_type, language)
    engine = FasterWhisperEngine(config)
    runners = {
        StageName.INSPECT: InspectStage(FFprobeMediaInspector(process, executable=ffprobe_bin)),
        StageName.NORMALIZE: NormalizeStage(
            FFmpegAudioNormalizer(process, executable=ffmpeg_bin), durable
        ),
        StageName.TRANSCRIBE: TranscribeStage(engine, durable),
        StageName.SEGMENT: SegmentStage(durable, SimpleSegmentationConfig()),
        StageName.EXPORT: ExportStage(durable),
        StageName.PUBLISH: PublishStage(durable),
    }
    journal = JsonlJournal(batch_dir / "journal.jsonl")
    manifest = JsonManifestStore(batch_dir / "manifest.json")
    event_factory = EventFactory(
        lambda: new_id("event-"),
        lambda: datetime.now(UTC).isoformat(),
    )
    executor = StageExecutor(journal, manifest, durable, event_factory, batch_dir / "work")
    fault_spec = os.environ.get("CAPTIONER_FAULT_POINT")
    if fault_spec is not None:
        if os.environ.get("CAPTIONER_ENABLE_FAULT_INJECTION") != "1":
            from captioner.core.domain.errors import AppError

            raise AppError("fault_injection.disabled")
        try:
            stage_name, point = fault_spec.split(":", maxsplit=1)
            StageName(stage_name)
            _validate_fault_point(point)
        except ValueError as exc:
            from captioner.core.domain.errors import AppError

            raise AppError("fault_injection.invalid") from exc
        executor.fault_injector = ScriptedFaultInjector(stage_name, point)
    return DurableServiceBundle(
        DurablePipelineService(
            journal, manifest, executor, event_factory, runners, batch_dir / "control"
        ),
        batch_dir,
    )


def _validate_fault_point(point: str) -> None:
    if point not in {
        "before_execute",
        "mid_execute",
        "after_artifact_write",
        "before_journal_commit",
        "after_journal_commit",
        "before_manifest_projection",
    }:
        raise ValueError
