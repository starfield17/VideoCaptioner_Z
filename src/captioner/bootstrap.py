"""Phase-aware composition root for durable subtitle pipelines."""

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
from captioner.adapters.llm.token_counter import ModelTokenCounter, resolve_tokenizer_id
from captioner.adapters.media.ffmpeg_audio import FFmpegAudioNormalizer
from captioner.adapters.media.ffprobe import FFprobeMediaInspector
from captioner.adapters.persistence.batch_lease import BatchLease
from captioner.adapters.persistence.content_addressed_artifact_store import (
    ContentAddressedArtifactStore,
)
from captioner.adapters.persistence.filesystem_llm_cache import FilesystemLLMCache
from captioner.adapters.persistence.json_manifest_store import JsonManifestStore
from captioner.adapters.persistence.jsonl_journal import JsonlJournal
from captioner.adapters.persistence.local_artifact_store import LocalArtifactStore
from captioner.adapters.pipeline.stages import (
    CorrectSourceStage,
    ExportStage,
    InspectStage,
    NormalizeStage,
    PublishStage,
    QualityTranslateStage,
    ReviewStage,
    SegmentStage,
    TranscribeStage,
    TranslateStage,
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
from captioner.core.domain.llm_job_config import (
    LLMJobSnapshot,
    PromptSnapshot,
    ProviderPublicSnapshot,
    required_prompts_for,
)
from captioner.core.domain.result import FrozenJsonValue, freeze_json_value, thaw_json_value
from captioner.core.domain.stage import (
    PipelineProfile,
    StageName,
    stage_versions_for,
)
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.simple_segmentation import SimpleSegmentationConfig
from captioner.core.ports.llm_cache import LLMCachePort
from captioner.core.ports.stage_runner import StageRunner
from captioner.core.ports.token_counter import TokenCounter
from captioner.infrastructure.app_paths import (
    AppPaths,
    ensure_runtime_layout,
    resolve_app_paths,
    resolve_safe_child,
)
from captioner.infrastructure.config import OpenAICompatibleProvider, load_provider_config
from captioner.infrastructure.ids import new_id
from captioner.infrastructure.prompts import PromptIdentity, PromptLoader


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
    runtime: LLMRuntime | None = None

    async def close(self) -> None:
        if self.runtime is not None:
            await self.runtime.close()


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
    expected_snapshot: Mapping[str, FrozenJsonValue] | None = None,
) -> LLMRuntime:
    """Create the single shared LLM runtime at the composition root."""
    application_paths = resolve_app_paths() if paths is None else paths
    ensure_runtime_layout(application_paths)
    provider = load_provider_config(application_paths.config_dir, provider_profile)
    _validate_provider_snapshot(provider, expected_snapshot)
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


def validate_llm_runtime_snapshot(
    runtime: LLMRuntime,
    snapshot: Mapping[str, FrozenJsonValue],
) -> None:
    """Fail before a request when a resumed runtime changed public identity."""
    _validate_provider_snapshot(runtime.provider, snapshot)


def _validate_provider_snapshot(
    provider: OpenAICompatibleProvider,
    snapshot: Mapping[str, FrozenJsonValue] | None,
) -> None:
    if snapshot is None:
        return
    try:
        durable = ProviderPublicSnapshot.from_mapping(
            {
                field: snapshot[field]
                for field in (
                    "kind",
                    "provider_profile",
                    "base_url",
                    "model",
                    "max_concurrency",
                    "request_timeout_sec",
                    "max_retries",
                    "temperature",
                    "tokenizer",
                )
            }
        )
    except (KeyError, TypeError, AppError) as exc:
        raise AppError("llm.provider_snapshot_mismatch", {"fields": ["snapshot"]}) from exc
    current = ProviderPublicSnapshot.from_mapping(provider.to_snapshot())
    changed = durable.changed_fields(current)
    if changed:
        raise AppError("llm.provider_snapshot_mismatch", {"fields": list(changed)})


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
    selected_profile = PipelineProfile(pipeline_profile)
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
        raise AppError(
            "llm.config_invalid",
            {"reason": "complete_snapshot_required"},
        )
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
        stage_versions_for(selected_profile),
        pipeline_profile=selected_profile,
        llm=None if snapshot is None else _frozen_mapping(snapshot),
    )


def create_llm_job_snapshot(
    *,
    target_language: str,
    provider_profile: str,
    source_language: str | None,
    paths: AppPaths,
    pipeline_profile: PipelineProfile = PipelineProfile.FAST,
) -> Mapping[str, FrozenJsonValue]:
    """Build a redacted LLM snapshot from runtime config and versioned prompts."""
    if not target_language.strip():
        raise AppError("llm.target_language_missing")
    if target_language != target_language.strip() or any(
        not (character.isalnum() or character in "-_") for character in target_language
    ):
        raise AppError("llm.target_language_invalid")
    provider = load_provider_config(paths.config_dir, provider_profile)
    selected_profile = PipelineProfile(pipeline_profile)
    loader = PromptLoader(paths.prompt_resource_dir)
    if selected_profile is PipelineProfile.DETERMINISTIC:
        raise AppError("llm.config_invalid", {"field": "pipeline_profile"})
    prompt_versions = {
        "terminology": "v2",
    }
    prompts = {
        prompt_id: loader.load(prompt_id, prompt_versions.get(prompt_id, "v1"))
        for prompt_id in required_prompts_for(selected_profile)
    }
    snapshot = LLMJobSnapshot(
        profile=selected_profile,
        provider=ProviderPublicSnapshot.from_mapping(provider.to_snapshot()),
        source_language=source_language,
        target_language=target_language.strip(),
        chunk={
            "max_items": 32,
            "max_input_tokens": 4096,
            "context_before_items": 1,
            "context_after_items": 1,
            "max_audio_context_duration_ms": 120_000,
        },
        prompts={
            prompt_id: PromptSnapshot(
                identity.prompt_id,
                identity.prompt_version,
                identity.content_sha256,
            )
            for prompt_id, identity in prompts.items()
        },
        response_schema_version=1,
    )
    return snapshot.to_mapping()


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
    llm: Mapping[str, FrozenJsonValue] | None = None,
    llm_runtime: LLMRuntime | None = None,
    llm_cache: LLMCachePort | None = None,
    token_counter: TokenCounter | None = None,
) -> DurableServiceBundle:
    selected_profile = PipelineProfile(pipeline_profile)
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
    runners: dict[StageName, StageRunner] = {
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

    runtime = llm_runtime
    cache = llm_cache
    counter = token_counter
    if selected_profile in {PipelineProfile.FAST, PipelineProfile.QUALITY}:
        if llm is None:
            raise AppError("llm.config_missing", {"reason": "job_snapshot"})
        snapshot = LLMJobSnapshot.from_mapping(thaw_json_value(llm))
        if snapshot.profile is not selected_profile:
            raise AppError("llm.snapshot_invalid", {"reason": "profile"})
        if runtime is None and initialize_runtime:
            runtime = build_llm_runtime(
                provider_profile=_snapshot_string(llm, "provider_profile", "default"),
                paths=application_paths,
                expected_snapshot=llm,
            )
        elif runtime is not None:
            validate_llm_runtime_snapshot(runtime, llm)
        if runtime is not None:
            cache = cache or FilesystemLLMCache(application_paths.cache_dir)
            if counter is None:
                tokenizer_id = resolve_tokenizer_id(
                    runtime.provider.tokenizer, runtime.provider.model
                )
                counter = ModelTokenCounter(tokenizer_id)
            if selected_profile is PipelineProfile.FAST:
                prompt = _prompt_for_snapshot(application_paths, llm, "translate_fast")
                repair_prompt = _prompt_for_snapshot(application_paths, llm, "repair_structured")
                runners[StageName.TRANSLATE] = TranslateStage(
                    durable,
                    runtime.service,
                    cache,
                    counter,
                    prompt,
                    policy,
                    repair_prompt=repair_prompt,
                )
            else:
                terminology_prompt = _prompt_for_snapshot(application_paths, llm, "terminology")
                correction_prompt = _prompt_for_snapshot(application_paths, llm, "correct_source")
                quality_prompt = _prompt_for_snapshot(application_paths, llm, "translate_quality")
                review_prompt = _prompt_for_snapshot(application_paths, llm, "review_anomalies")
                repair_prompt = _prompt_for_snapshot(application_paths, llm, "repair_structured")
                runners[StageName.CORRECT_SOURCE] = CorrectSourceStage(
                    durable,
                    runtime.service,
                    cache,
                    counter,
                    terminology_prompt,
                    correction_prompt,
                    policy,
                    repair_prompt=repair_prompt,
                )
                runners[StageName.TRANSLATE] = QualityTranslateStage(
                    durable,
                    runtime.service,
                    cache,
                    counter,
                    quality_prompt,
                    policy,
                    repair_prompt=repair_prompt,
                )
                runners[StageName.REVIEW] = ReviewStage(
                    durable,
                    runtime.service,
                    cache,
                    counter,
                    review_prompt,
                    policy,
                    repair_prompt=repair_prompt,
                )

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
        runtime,
    )


def _snapshot_string(values: Mapping[str, FrozenJsonValue], key: str, default: str) -> str:
    value = values.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise AppError("llm.config_invalid", {"field": key})
    return value


def _prompt_for_snapshot(
    paths: AppPaths,
    snapshot: Mapping[str, FrozenJsonValue],
    prompt_id: str,
) -> PromptIdentity:
    prompt_value: object = None
    prompts = snapshot.get("prompts")
    if isinstance(prompts, Mapping):
        prompt_value = prompts.get(prompt_id)
    if not isinstance(prompt_value, Mapping):
        raise AppError("prompt.identity_missing", {"prompt_id": prompt_id})
    version = prompt_value.get("prompt_version")
    content_hash = prompt_value.get("content_sha256")
    if not isinstance(version, str) or not isinstance(content_hash, str):
        raise AppError("prompt.identity_invalid", {"prompt_id": prompt_id})
    prompt = PromptLoader(paths.prompt_resource_dir).load(prompt_id, version)
    if prompt.content_sha256 != content_hash:
        raise AppError("prompt.identity_mismatch", {"prompt_id": prompt_id})
    return prompt


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
