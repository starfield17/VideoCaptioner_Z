"""One-shot application service for the Phase 1 vertical slice."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.media import MediaAsset
from captioner.core.domain.subtitle import SubtitleTrack
from captioner.core.domain.transcript import Transcript
from captioner.core.policies.simple_segmentation import (
    SimpleSegmentationConfig,
    segment_transcript,
)
from captioner.core.ports.artifact_store import ArtifactStorePort, StagedArtifact
from captioner.core.ports.asr import ASREngine, TranscriptionRequest
from captioner.core.ports.media import AudioNormalizer, MediaInspector


@dataclass(frozen=True, slots=True)
class RunSingleRequest:
    input_path: Path
    output_dir: Path
    language: str | None
    overwrite: bool


@dataclass(frozen=True, slots=True)
class RunSingleResult:
    media_id: str
    transcript_id: str
    transcript_path: Path
    subtitle_path: Path
    detected_language: str
    word_count: int
    cue_count: int


ArtifactStoreFactory = Callable[[Path], ArtifactStorePort]
TranscriptSerializer = Callable[[Transcript], bytes]
SubtitleSerializer = Callable[[SubtitleTrack], bytes]


@dataclass(slots=True)
class RunSingleService:
    inspector: MediaInspector
    normalizer: AudioNormalizer
    asr_engine: ASREngine
    artifact_store_factory: ArtifactStoreFactory
    transcript_serializer: TranscriptSerializer
    subtitle_serializer: SubtitleSerializer
    temp_root: Path
    segmentation_config: SimpleSegmentationConfig = field(default_factory=SimpleSegmentationConfig)

    async def run(
        self, request: RunSingleRequest, context: ExecutionContext | None = None
    ) -> RunSingleResult:
        execution = ExecutionContext() if context is None else context
        source = request.input_path.expanduser().resolve()
        _validate_input(source)
        output_dir = request.output_dir.expanduser().resolve()
        _prepare_output_dir(output_dir)
        execution.raise_if_cancelled()
        asset = await self.inspector.inspect(source, execution)
        execution.raise_if_cancelled()
        self.temp_root.expanduser().resolve().mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="captioner-", dir=self.temp_root) as workspace_name:
            workspace = Path(workspace_name)
            audio = await self.normalizer.normalize(asset, workspace, execution)
            execution.raise_if_cancelled()
            transcript = await self.asr_engine.transcribe(
                TranscriptionRequest(audio=audio, language=request.language), execution
            )
            execution.raise_if_cancelled()
            track = segment_transcript(transcript, self.segmentation_config)
            execution.raise_if_cancelled()
            transcript_bytes = self.transcript_serializer(transcript)
            srt_bytes = self.subtitle_serializer(track)
            execution.raise_if_cancelled()
            return _commit_outputs(
                store=self.artifact_store_factory(output_dir),
                asset=asset,
                transcript=transcript,
                track=track,
                transcript_bytes=transcript_bytes,
                srt_bytes=srt_bytes,
                overwrite=request.overwrite,
                context=execution,
            )


def _validate_input(source: Path) -> None:
    if not source.exists():
        raise AppError("media.input_missing", {"path": str(source)})
    if not source.is_file():
        raise AppError("media.input_not_file", {"path": str(source)})


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise AppError("output.not_directory", {"path": str(output_dir)})
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AppError("output.directory_create_failed", {"path": str(output_dir)}) from exc


def _commit_outputs(
    *,
    store: ArtifactStorePort,
    asset: MediaAsset,
    transcript: Transcript,
    track: SubtitleTrack,
    transcript_bytes: bytes,
    srt_bytes: bytes,
    overwrite: bool,
    context: ExecutionContext,
) -> RunSingleResult:
    transcript_key = f"{asset.source_path.stem}.transcript.json"
    subtitle_key = f"{asset.source_path.stem}.srt"
    keys = (transcript_key, subtitle_key)
    previous: dict[str, bytes | None] = {
        key: store.read_bytes(key) if store.exists(key) else None for key in keys
    }
    staged: list[StagedArtifact] = []
    committed: list[str] = []
    transcript_path = store.root / transcript_key
    subtitle_path = store.root / subtitle_key
    staging_cleaned = False
    try:
        staged.append(store.stage_bytes(transcript_key, transcript_bytes))
        staged.append(store.stage_bytes(subtitle_key, srt_bytes))
        context.raise_if_cancelled()
        transcript_path = _commit_staged(staged[0], overwrite=overwrite, committed=committed)
        context.raise_if_cancelled()
        subtitle_path = _commit_staged(staged[1], overwrite=overwrite, committed=committed)
        context.raise_if_cancelled()
        result = RunSingleResult(
            media_id=asset.id,
            transcript_id=transcript.id,
            transcript_path=transcript_path,
            subtitle_path=subtitle_path,
            detected_language=transcript.language,
            word_count=len(transcript.words),
            cue_count=len(track.cues),
        )
        for artifact in reversed(staged):
            artifact.discard()
        staging_cleaned = True
        context.raise_if_cancelled()
    except BaseException as exc:
        _record_committed_stages(staged, committed)
        _rollback(store, committed, previous, exc)
        raise
    finally:
        if not staging_cleaned:
            for artifact in reversed(staged):
                artifact.discard()
    return result


def _commit_staged(artifact: StagedArtifact, *, overwrite: bool, committed: list[str]) -> Path:
    try:
        path = artifact.commit(overwrite=overwrite)
    except BaseException:
        if artifact.committed and artifact.key not in committed:
            committed.append(artifact.key)
        raise
    committed.append(artifact.key)
    return path


def _record_committed_stages(staged: list[StagedArtifact], committed: list[str]) -> None:
    for artifact in staged:
        if artifact.committed and artifact.key not in committed:
            committed.append(artifact.key)


def _rollback(
    store: ArtifactStorePort,
    committed: list[str],
    previous: dict[str, bytes | None],
    original: BaseException,
) -> None:
    try:
        for key in reversed(committed):
            old_value = previous[key]
            if old_value is None:
                store.delete(key)
            else:
                store.write_bytes(key, old_value, overwrite=True)
    except BaseException as rollback_error:
        reason = (
            rollback_error.code
            if isinstance(rollback_error, AppError)
            else type(rollback_error).__name__
        )
        raise AppError("output.rollback_failed", {"reason": reason}) from original
