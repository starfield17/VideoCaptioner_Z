"""Phase 1 composition root for the one-shot CLI workflow."""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from captioner.adapters.asr.faster_whisper import FasterWhisperConfig, FasterWhisperEngine
from captioner.adapters.exporters.srt import serialize_bytes as serialize_srt
from captioner.adapters.exporters.transcript_json import serialize_bytes as serialize_transcript
from captioner.adapters.llm.http_transport import HTTPTransport
from captioner.adapters.llm.openai_compatible import OpenAICompatibleClient
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
    verify_publication,
)
from captioner.adapters.process.asyncio_subprocess import AsyncioSubprocessRunner
from captioner.adapters.subtitles.ass import serialize_bytes as serialize_ass
from captioner.adapters.subtitles.json_track import serialize as serialize_track_json
from captioner.adapters.subtitles.webvtt import serialize_bytes as serialize_webvtt
from captioner.adapters.testing.fault_injector import ScriptedFaultInjector
from captioner.core.application.durable_pipeline import DurablePipelineService
from captioner.core.application.run_single import RunSingleService
from captioner.core.application.stage_executor import EventFactory, StageExecutor
from captioner.core.application.structured_llm_service import Sleep, StructuredLLMService
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobConfig
from captioner.core.domain.journal import replay
from captioner.core.domain.result import FrozenJsonValue, freeze_json_value
from captioner.core.domain.stage import PipelineProfile, StageName, stage_plan_for
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.simple_segmentation import SimpleSegmentationConfig
from captioner.infrastructure.app_paths import (
    AppPaths,
    ensure_runtime_layout,
    resolve_app_paths,
    resolve_safe_child,
)
from captioner.infrastructure.config import OpenAICompatibleProvider, load_provider_config
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
        subtitle_json_serializer=serialize_track_json,
        webvtt_serializer=serialize_webvtt,
        ass_serializer=serialize_ass,
    )


@dataclass(frozen=True, slots=True)
class DurableServiceBundle:
    service: DurablePipelineService
    batch_dir: Path


@dataclass(slots=True)
class LLMRuntime:
    """One application-wide provider client, semaphore, and retry service."""

    provider: OpenAICompatibleProvider
    semaphore: asyncio.Semaphore
    client: OpenAICompatibleClient
    service: StructuredLLMService

    async def close(self) -> None:
        await self.client.close()


def build_llm_runtime(
    *,
    provider_profile: str = "default",
    paths: AppPaths | None = None,
    transport: HTTPTransport | None = None,
    sleep: Sleep | None = None,
    max_response_bytes: int = 2 * 1024 * 1024,
) -> LLMRuntime:
    """Create the single shared LLM runtime at the composition root."""
    application_paths = resolve_app_paths() if paths is None else paths
    ensure_runtime_layout(application_paths)
    provider = load_provider_config(application_paths.config_dir, provider_profile)
    semaphore = asyncio.Semaphore(provider.max_concurrency)
    client = OpenAICompatibleClient(
        provider,
        transport=transport,
        semaphore=semaphore,
        max_response_bytes=max_response_bytes,
    )
    service = StructuredLLMService(
        client,
        max_retries=provider.max_retries,
        sleep=asyncio.sleep if sleep is None else sleep,
    )
    return LLMRuntime(provider, semaphore, client, service)


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
    pipeline_profile: PipelineProfile = PipelineProfile.DETERMINISTIC,
    llm: Mapping[str, object] | None = None,
    target_language: str | None = None,
    provider_profile: str | None = None,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    temperature: float | None = None,
    timeout_sec: int | None = None,
    max_retries: int | None = None,
    chunk: Mapping[str, object] | None = None,
    prompt_identity: Mapping[str, object] | None = None,
) -> JobConfig:
    model = FasterWhisperConfig(model_ref, device, compute_type, language)
    snapshot: Mapping[str, object] | None = llm
    if snapshot is None and any(
        item is not None
        for item in (
            target_language,
            provider_profile,
            llm_base_url,
            llm_model,
            temperature,
            timeout_sec,
            max_retries,
            chunk,
            prompt_identity,
        )
    ):
        snapshot = {
            **({"target_language": target_language} if target_language is not None else {}),
            **({"provider_profile": provider_profile} if provider_profile is not None else {}),
            **({"base_url": llm_base_url} if llm_base_url is not None else {}),
            **({"model": llm_model} if llm_model is not None else {}),
            **({"temperature": temperature} if temperature is not None else {}),
            **({"timeout_sec": timeout_sec} if timeout_sec is not None else {}),
            **({"max_retries": max_retries} if max_retries is not None else {}),
            **({"chunk": chunk} if chunk is not None else {}),
            **({"prompt": prompt_identity} if prompt_identity is not None else {}),
        }
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
        SimpleSegmentationConfig().to_policy_config().to_mapping(),
        str(output_dir.expanduser().resolve()),
        overwrite,
        {
            stage.value: {
                StageName.SEGMENT.value: "segment-v2",
                StageName.EXPORT.value: "export-v2",
                StageName.PUBLISH.value: "publish-v2",
            }.get(stage.value, "1")
            for stage in stage_plan_for(pipeline_profile)
        },
        pipeline_profile=pipeline_profile,
        llm=None if snapshot is None else _frozen_mapping(snapshot),
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


def _frozen_mapping(value: Mapping[str, object]) -> Mapping[str, FrozenJsonValue]:
    frozen = freeze_json_value(value)
    if not isinstance(frozen, Mapping):
        raise AppError("job.config_invalid", {"field": "llm"})
    return frozen


def load_batch_config(batch_id: str, *, paths: AppPaths | None = None) -> JobConfig:
    application_paths = resolve_app_paths() if paths is None else paths
    batch_dir = resolve_safe_child(application_paths.batches_dir, batch_id, field="batch_id")
    events = JsonlJournal(batch_dir / "journal.jsonl").read_snapshot().events
    if not events:
        raise AppError("batch.not_found", {"batch_id": batch_id})
    projection = replay(events)
    if not projection.jobs:
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
    segmentation: Mapping[str, object] | None = None,
    initialize_runtime: bool = True,
    pipeline_profile: PipelineProfile = PipelineProfile.DETERMINISTIC,
) -> DurableServiceBundle:
    if PipelineProfile(pipeline_profile) is not PipelineProfile.DETERMINISTIC:
        raise AppError(
            "pipeline.profile_unavailable",
            {"profile": PipelineProfile(pipeline_profile).value},
        )
    application_paths = resolve_app_paths() if paths is None else paths
    if initialize_runtime:
        ensure_runtime_layout(application_paths)
    batch_dir = resolve_safe_child(application_paths.batches_dir, batch_id, field="batch_id")
    process = AsyncioSubprocessRunner()
    durable = ContentAddressedArtifactStore(
        application_paths.artifacts_dir, initialize=initialize_runtime
    )
    engine_config = FasterWhisperConfig(model_ref, device, compute_type, language)
    engine = FasterWhisperEngine(engine_config)
    policy = SegmentationPolicyConfig.from_mapping(
        segmentation or SegmentationPolicyConfig().to_mapping()
    )
    runners = {
        StageName.INSPECT: InspectStage(FFprobeMediaInspector(process, executable=ffprobe_bin)),
        StageName.NORMALIZE: NormalizeStage(
            FFmpegAudioNormalizer(process, executable=ffmpeg_bin), durable
        ),
        StageName.TRANSCRIBE: TranscribeStage(engine, durable),
        StageName.SEGMENT: SegmentStage(durable, policy),
        StageName.EXPORT: ExportStage(durable, policy),
        StageName.PUBLISH: PublishStage(durable),
    }
    journal = JsonlJournal(batch_dir / "journal.jsonl")
    manifest = JsonManifestStore(batch_dir / "manifest.json")
    event_factory = EventFactory(
        lambda: new_id("event-"),
        lambda: datetime.now(UTC).isoformat(),
    )
    executor = StageExecutor(journal, manifest, durable, event_factory, batch_dir / "work")

    def verify_committed(job: object, stage: object) -> None:
        from captioner.core.domain.job import JobProjection
        from captioner.core.domain.stage import StageProjection

        if not isinstance(job, JobProjection) or not isinstance(stage, StageProjection):
            raise TypeError
        if stage.name is not StageName.PUBLISH:
            return
        receipt_ref = next(
            (ref for ref in stage.artifacts if ref.logical_name == "publication-receipt.json"),
            None,
        )
        if receipt_ref is None:
            raise AppError("output.publication_invalid", {"reason": "receipt_missing"})
        export_refs = job.stage(StageName.EXPORT).artifacts
        verify_publication(
            durable.read_bytes(receipt_ref),
            output_dir=Path(job.config.output_dir),
            input_path=Path(job.input_path),
            export_refs=export_refs,
            publication_version=runners[stage.name].version,
        )

    executor.committed_verifier = verify_committed
    fault_spec = os.environ.get("CAPTIONER_FAULT_POINT")
    if fault_spec is not None:
        if os.environ.get("CAPTIONER_ENABLE_FAULT_INJECTION") != "1":
            raise AppError("fault_injection.disabled")
        try:
            stage_name, point = fault_spec.split(":", maxsplit=1)
            if stage_name != "batch-config":
                StageName(stage_name)
            _validate_fault_point(point)
        except ValueError as exc:
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
        "before_batch_config_commit",
        "after_batch_config_commit",
        "before_batch_config_manifest",
        "after_batch_config_manifest",
    }:
        raise ValueError
