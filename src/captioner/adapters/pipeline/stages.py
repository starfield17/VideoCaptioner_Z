"""Six concrete Stage runners composed from Phase 1 adapters."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from captioner.adapters.exporters.srt import serialize_bytes as serialize_srt
from captioner.adapters.persistence.domain_codecs import (
    decode_audio,
    decode_media,
    decode_track,
    decode_transcript,
    encode_audio,
    encode_json,
    encode_media,
    encode_track,
    encode_transcript,
)
from captioner.adapters.persistence.local_artifact_store import LocalArtifactStore
from captioner.core.domain.artifact import ArtifactRef
from captioner.core.domain.errors import AppError
from captioner.core.domain.stage import StageName
from captioner.core.policies.simple_segmentation import SimpleSegmentationConfig, segment_transcript
from captioner.core.ports.asr import ASREngine, TranscriptionRequest
from captioner.core.ports.durable_artifact_store import DurableArtifactStorePort
from captioner.core.ports.media import AudioNormalizer, MediaInspector
from captioner.core.ports.stage_runner import (
    ProducedArtifact,
    StageExecutionContext,
    StageExecutionRequest,
)


@dataclass(slots=True)
class InspectStage:
    inspector: MediaInspector
    name: StageName = StageName.INSPECT
    version: str = "inspect-v1"

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        asset = await self.inspector.inspect(request.input_path, context.execution)
        return (
            ProducedArtifact(
                "media-json", "application/json", "media.json", data=encode_media(asset)
            ),
        )


@dataclass(slots=True)
class NormalizeStage:
    normalizer: AudioNormalizer
    artifacts: DurableArtifactStorePort
    name: StageName = StageName.NORMALIZE
    version: str = "normalize-v1"

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        media = decode_media(self.artifacts.read_bytes(_ref(request, "media.json")))
        audio = await self.normalizer.normalize(media, context.workspace, context.execution)
        return (
            ProducedArtifact(
                "normalized-wav", "audio/wav", "normalized.wav", source_path=audio.path
            ),
            ProducedArtifact(
                "normalized-audio-json",
                "application/json",
                "normalized-audio.json",
                data=encode_audio(audio),
            ),
        )


@dataclass(slots=True)
class TranscribeStage:
    engine: ASREngine
    artifacts: DurableArtifactStorePort
    name: StageName = StageName.TRANSCRIBE
    version: str = "transcribe-v1"

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        wav_ref = _ref(request, "normalized.wav")
        audio = decode_audio(
            self.artifacts.read_bytes(_ref(request, "normalized-audio.json")),
            path=str(self.artifacts.resolve(wav_ref)),
        )
        transcript = await self.engine.transcribe(
            TranscriptionRequest(audio, request.config.language), context.execution
        )
        return (
            ProducedArtifact(
                "transcript-json",
                "application/json",
                "transcript.json",
                data=encode_transcript(transcript),
            ),
        )


@dataclass(slots=True)
class SegmentStage:
    artifacts: DurableArtifactStorePort
    config: SimpleSegmentationConfig
    name: StageName = StageName.SEGMENT
    version: str = "segment-v1"

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        context.execution.raise_if_cancelled()
        transcript = decode_transcript(self.artifacts.read_bytes(_ref(request, "transcript.json")))
        track = segment_transcript(transcript, self.config)
        return (
            ProducedArtifact(
                "subtitle-track-json",
                "application/json",
                "subtitle-track.json",
                data=encode_track(track),
            ),
        )


@dataclass(slots=True)
class ExportStage:
    artifacts: DurableArtifactStorePort
    name: StageName = StageName.EXPORT
    version: str = "export-v1"

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        context.execution.raise_if_cancelled()
        transcript_bytes = self.artifacts.read_bytes(_ref(request, "transcript.json"))
        track = decode_track(self.artifacts.read_bytes(_ref(request, "subtitle-track.json")))
        return (
            ProducedArtifact(
                "final-transcript-json",
                "application/json",
                "final-transcript.json",
                data=transcript_bytes,
            ),
            ProducedArtifact(
                "final-subtitle-srt",
                "application/x-subrip",
                "final-subtitle.srt",
                data=serialize_srt(track),
            ),
        )


@dataclass(slots=True)
class PublishStage:
    artifacts: DurableArtifactStorePort
    name: StageName = StageName.PUBLISH
    version: str = "publish-v1"

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        context.execution.raise_if_cancelled()
        transcript_ref = _ref(request, "final-transcript.json")
        subtitle_ref = _ref(request, "final-subtitle.srt")
        stem = request.input_path.stem
        targets = ((f"{stem}.transcript.json", transcript_ref), (f"{stem}.srt", subtitle_ref))
        store = LocalArtifactStore(Path(request.config.output_dir))
        previous = {key: store.read_bytes(key) if store.exists(key) else None for key, _ in targets}
        staged = [store.stage_bytes(key, self.artifacts.read_bytes(ref)) for key, ref in targets]
        committed: list[str] = []
        try:
            context.execution.raise_if_cancelled()
            for artifact in staged:
                artifact.commit(overwrite=request.config.overwrite)
                committed.append(artifact.key)
                context.execution.raise_if_cancelled()
        except BaseException:
            for key in reversed(committed):
                old = previous[key]
                if old is None:
                    store.delete(key)
                else:
                    store.write_bytes(key, old, overwrite=True)
            raise
        finally:
            for artifact in reversed(staged):
                artifact.discard()
        for key, ref in targets:
            published = Path(request.config.output_dir) / key
            if published.stat().st_size != ref.size_bytes or _sha256(published) != ref.sha256:
                raise AppError("output.publication_invalid", {"logical_name": key})
        receipt = {
            "schema_version": 1,
            "targets": [
                {
                    "path": str((Path(request.config.output_dir) / key).resolve()),
                    "sha256": ref.sha256,
                    "size_bytes": ref.size_bytes,
                }
                for key, ref in targets
            ],
            "output_generation": hashlib.sha256(
                "".join(ref.sha256 for _, ref in targets).encode()
            ).hexdigest(),
        }
        return (
            ProducedArtifact(
                "publication-receipt-json",
                "application/json",
                "publication-receipt.json",
                data=encode_json(receipt),
            ),
        )


def _ref(request: StageExecutionRequest, logical_name: str) -> ArtifactRef:
    for ref in request.input_artifacts:
        if ref.logical_name == logical_name:
            return ref
    raise AppError("stage.input_missing", {"logical_name": logical_name})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
