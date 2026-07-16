"""One-shot application service with deterministic Phase 3 subtitle export."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from captioner.core.application.output_transaction import commit_output_pair, commit_output_set
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.media import MediaAsset
from captioner.core.domain.subtitle import SubtitleTrack
from captioner.core.domain.transcript import Transcript
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.simple_segmentation import (
    segment_transcript,
)
from captioner.core.ports.artifact_store import ArtifactStorePort
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
    subtitle_json_path: Path | None = None
    vtt_path: Path | None = None
    ass_path: Path | None = None


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
    segmentation_config: SegmentationPolicyConfig = field(default_factory=SegmentationPolicyConfig)
    subtitle_json_serializer: SubtitleSerializer | None = None
    webvtt_serializer: SubtitleSerializer | None = None
    ass_serializer: SubtitleSerializer | None = None

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
            extra_serializers = (
                self.subtitle_json_serializer,
                self.webvtt_serializer,
                self.ass_serializer,
            )
            if any(serializer is not None for serializer in extra_serializers) and not all(
                serializer is not None for serializer in extra_serializers
            ):
                raise AppError("output.serializer_invalid")
            if all(serializer is not None for serializer in extra_serializers):
                json_serializer = self.subtitle_json_serializer
                webvtt_serializer = self.webvtt_serializer
                ass_serializer = self.ass_serializer
                assert json_serializer is not None
                assert webvtt_serializer is not None
                assert ass_serializer is not None
                extra_outputs: tuple[tuple[str, bytes], ...] | None = (
                    (
                        f"{asset.source_path.stem}.subtitle.json",
                        json_serializer(track),
                    ),
                    (
                        f"{asset.source_path.stem}.vtt",
                        webvtt_serializer(track),
                    ),
                    (
                        f"{asset.source_path.stem}.ass",
                        ass_serializer(track),
                    ),
                )
            else:
                extra_outputs = None
            return _commit_outputs(
                store=self.artifact_store_factory(output_dir),
                asset=asset,
                transcript=transcript,
                track=track,
                transcript_bytes=transcript_bytes,
                srt_bytes=srt_bytes,
                overwrite=request.overwrite,
                context=execution,
                extra_outputs=extra_outputs,
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
    extra_outputs: tuple[tuple[str, bytes], ...] | None,
) -> RunSingleResult:
    transcript_key = f"{asset.source_path.stem}.transcript.json"
    subtitle_key = f"{asset.source_path.stem}.srt"
    if extra_outputs is None:
        transcript_path, subtitle_path = commit_output_pair(
            store,
            ((transcript_key, transcript_bytes), (subtitle_key, srt_bytes)),
            overwrite=overwrite,
            context=context,
        )
        subtitle_json_path = vtt_path = ass_path = None
    else:
        paths = commit_output_set(
            store,
            (
                (transcript_key, transcript_bytes),
                *extra_outputs[:1],
                (subtitle_key, srt_bytes),
                *extra_outputs[1:],
            ),
            overwrite=overwrite,
            context=context,
        )
        transcript_path, subtitle_json_path, subtitle_path, vtt_path, ass_path = paths
    return RunSingleResult(
        media_id=asset.id,
        transcript_id=transcript.id,
        transcript_path=transcript_path,
        subtitle_path=subtitle_path,
        detected_language=transcript.language,
        word_count=len(transcript.words),
        cue_count=len(track.cues),
        subtitle_json_path=subtitle_json_path,
        vtt_path=vtt_path,
        ass_path=ass_path,
    )
