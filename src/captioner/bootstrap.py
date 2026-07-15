"""Phase 1 composition root for the one-shot CLI workflow."""

from __future__ import annotations

from captioner.adapters.asr.faster_whisper import FasterWhisperConfig, FasterWhisperEngine
from captioner.adapters.exporters.srt import serialize_bytes as serialize_srt
from captioner.adapters.exporters.transcript_json import serialize_bytes as serialize_transcript
from captioner.adapters.media.ffmpeg_audio import FFmpegAudioNormalizer
from captioner.adapters.media.ffprobe import FFprobeMediaInspector
from captioner.adapters.persistence.local_artifact_store import LocalArtifactStore
from captioner.adapters.process.asyncio_subprocess import AsyncioSubprocessRunner
from captioner.core.application.run_single import RunSingleService
from captioner.infrastructure.app_paths import AppPaths, ensure_runtime_layout, resolve_app_paths


def build_run_service(
    *,
    model_id: str,
    device: str,
    compute_type: str,
    language: str | None,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    paths: AppPaths | None = None,
) -> RunSingleService:
    """Assemble concrete adapters for one CLI invocation."""
    application_paths = resolve_app_paths() if paths is None else paths
    ensure_runtime_layout(application_paths)
    process = AsyncioSubprocessRunner()
    inspector = FFprobeMediaInspector(process, executable=ffprobe_bin)
    normalizer = FFmpegAudioNormalizer(process, executable=ffmpeg_bin)
    engine = FasterWhisperEngine(
        FasterWhisperConfig(
            model_id=model_id,
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
