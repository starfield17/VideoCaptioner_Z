"""Six concrete Stage runners composed from Phase 1 adapters."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from captioner.adapters.exporters.srt import serialize_bytes as serialize_srt
from captioner.adapters.persistence.domain_codecs import (
    decode_audio,
    decode_media,
    decode_publication_receipt,
    decode_track,
    decode_transcript,
    encode_audio,
    encode_media,
    encode_publication_receipt,
    encode_track,
    encode_transcript,
)
from captioner.adapters.persistence.local_artifact_store import LocalArtifactStore
from captioner.adapters.subtitles.ass import serialize_bytes as serialize_ass
from captioner.adapters.subtitles.json_track import serialize as serialize_track_json
from captioner.adapters.subtitles.webvtt import serialize_bytes as serialize_webvtt
from captioner.core.application.output_transaction import commit_output_set
from captioner.core.domain.artifact import ArtifactRef
from captioner.core.domain.errors import AppError
from captioner.core.domain.publication import PublicationReceipt, PublishedTarget
from captioner.core.domain.stage import StageName
from captioner.core.domain.subtitle_validation import validate_subtitle_track
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
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
        context.checkpoint("mid_execute")
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
    config: SegmentationPolicyConfig | SimpleSegmentationConfig
    name: StageName = StageName.SEGMENT
    version: str = "segment-v2"

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        context.execution.raise_if_cancelled()
        transcript = decode_transcript(self.artifacts.read_bytes(_ref(request, "transcript.json")))

        midpoint_emitted = False

        def midpoint() -> None:
            nonlocal midpoint_emitted
            if not midpoint_emitted:
                midpoint_emitted = True
                context.checkpoint("mid_execute")

        track = segment_transcript(
            transcript,
            self.config,
            progress=midpoint,
        )
        report = validate_subtitle_track(track, transcript, _policy_config(self.config))
        if not report.is_valid:
            first = next(issue for issue in report.issues if issue.severity.value == "error")
            raise AppError("subtitle.validation_failed", {"reason": first.code})
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
    config: SegmentationPolicyConfig | SimpleSegmentationConfig | None = None
    name: StageName = StageName.EXPORT
    version: str = "export-v2"

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        context.execution.raise_if_cancelled()
        transcript = decode_transcript(self.artifacts.read_bytes(_ref(request, "transcript.json")))
        track = decode_track(self.artifacts.read_bytes(_ref(request, "subtitle-track.json")))
        config = _policy_config(self.config)
        report = validate_subtitle_track(track, transcript, config)
        if not report.is_valid:
            first = next(issue for issue in report.issues if issue.severity.value == "error")
            raise AppError("subtitle.validation_failed", {"reason": first.code})
        transcript_bytes = encode_transcript(transcript)
        context.checkpoint("mid_execute")
        track_json = serialize_track_json(track)
        subtitle_bytes = serialize_srt(track)
        webvtt_bytes = serialize_webvtt(track)
        ass_bytes = serialize_ass(track)
        return (
            ProducedArtifact(
                "final-transcript-json",
                "application/json",
                "final-transcript.json",
                data=transcript_bytes,
            ),
            ProducedArtifact(
                "final-subtitle-json",
                "application/json",
                "final-subtitle.json",
                data=track_json,
            ),
            ProducedArtifact(
                "final-subtitle-srt",
                "application/x-subrip",
                "final-subtitle.srt",
                data=subtitle_bytes,
            ),
            ProducedArtifact(
                "final-subtitle-vtt",
                "text/vtt",
                "final-subtitle.vtt",
                data=webvtt_bytes,
            ),
            ProducedArtifact(
                "final-subtitle-ass",
                "text/x-ass",
                "final-subtitle.ass",
                data=ass_bytes,
            ),
        )


@dataclass(slots=True)
class PublishStage:
    artifacts: DurableArtifactStorePort
    name: StageName = StageName.PUBLISH
    version: str = "publish-v2"

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        context.execution.raise_if_cancelled()
        target_specs = _publication_specs(request.input_path, request.input_artifacts)
        targets = tuple((target_name, ref) for _, target_name, ref in target_specs)
        store = LocalArtifactStore(Path(request.config.output_dir))
        if not all(
            _published_matches(Path(request.config.output_dir) / key, ref)
            for key, ref in targets
        ):
            commit_output_set(
                store,
                tuple((key, self.artifacts.read_bytes(ref)) for key, ref in targets),
                overwrite=request.config.overwrite or request.recovery,
                context=context.execution,
            )
        for key, ref in targets:
            published = Path(request.config.output_dir) / key
            _verify_published_target(published, ref, key)
        receipt = PublicationReceipt(
            hashlib.sha256("".join(ref.sha256 for _, ref in targets).encode()).hexdigest(),
            tuple(
            PublishedTarget(
                str((Path(request.config.output_dir) / key).resolve()),
                ref.sha256,
                ref.size_bytes,
                key,
            )
            for key, ref in targets
            ),
        )
        return (
            ProducedArtifact(
                "publication-receipt-json",
                "application/json",
                "publication-receipt.json",
                data=encode_publication_receipt(receipt),
            ),
        )


def _ref(request: StageExecutionRequest, logical_name: str) -> ArtifactRef:
    matches = [ref for ref in request.input_artifacts if ref.logical_name == logical_name]
    if len(matches) == 1:
        return matches[0]
    raise AppError("stage.input_missing", {"logical_name": logical_name})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _published_matches(path: Path, ref: ArtifactRef) -> bool:
    try:
        _verify_published_target(path, ref, path.name)
    except AppError:
        return False
    return True


def _verify_published_target(path: Path, ref: ArtifactRef, logical_name: str) -> None:
    try:
        if path.is_symlink() or not path.is_file():
            raise AppError("output.publication_invalid", {"logical_name": logical_name})
        if path.stat().st_size != ref.size_bytes or _sha256(path) != ref.sha256:
            raise AppError("output.publication_invalid", {"logical_name": logical_name})
    except OSError as exc:
        raise AppError("output.publication_invalid", {"logical_name": logical_name}) from exc


def verify_publication(
    receipt_bytes: bytes,
    *,
    output_dir: Path,
    input_path: Path,
    export_refs: tuple[ArtifactRef, ...],
) -> None:
    receipt = decode_publication_receipt(receipt_bytes)
    target_specs = _publication_specs(input_path, export_refs)
    expected_generation = hashlib.sha256(
        "".join(ref.sha256 for _, _, ref in target_specs).encode()
    ).hexdigest()
    expected_targets = {
        str((output_dir / target_name).resolve()): (ref, target_name)
        for _, target_name, ref in target_specs
    }
    if (
        receipt.output_generation != expected_generation
        or len(receipt.targets) != len(expected_targets)
        or {target.path for target in receipt.targets} != set(expected_targets)
        or tuple(target.logical_name for target in receipt.targets)
        != tuple(target_name for _, target_name, _ in target_specs)
    ):
        raise AppError("output.publication_invalid", {"reason": "receipt"})
    for target in receipt.targets:
        expected = expected_targets.get(target.path)
        if (
            expected is None
            or target.logical_name != expected[1]
            or target.sha256 != expected[0].sha256
            or target.size_bytes != expected[0].size_bytes
        ):
            raise AppError("output.publication_invalid", {"reason": "target_metadata"})
        path = Path(target.path)
        _verify_published_target(path, expected[0], target.logical_name)


def _policy_config(
    config: SegmentationPolicyConfig | SimpleSegmentationConfig | None,
) -> SegmentationPolicyConfig:
    if config is None:
        return SegmentationPolicyConfig()
    if isinstance(config, SimpleSegmentationConfig):
        return config.to_policy_config()
    return config


def _publication_specs(
    input_path: Path, export_refs: tuple[ArtifactRef, ...]
) -> tuple[tuple[str, str, ArtifactRef], ...]:
    export_refs = tuple(ref for ref in export_refs if ref.logical_name.startswith("final-"))
    export_by_name = {ref.logical_name: ref for ref in export_refs}
    if len(export_by_name) != len(export_refs):
        raise AppError("output.publication_invalid", {"reason": "export_refs"})
    if set(export_by_name) == {
        "final-transcript.json",
        "final-subtitle.json",
        "final-subtitle.srt",
        "final-subtitle.vtt",
        "final-subtitle.ass",
    }:
        names = (
            ("final-transcript.json", f"{input_path.stem}.transcript.json"),
            ("final-subtitle.json", f"{input_path.stem}.subtitle.json"),
            ("final-subtitle.srt", f"{input_path.stem}.srt"),
            ("final-subtitle.vtt", f"{input_path.stem}.vtt"),
            ("final-subtitle.ass", f"{input_path.stem}.ass"),
        )
    elif set(export_by_name) == {"final-transcript.json", "final-subtitle.srt"}:
        names = (
            ("final-transcript.json", f"{input_path.stem}.transcript.json"),
            ("final-subtitle.srt", f"{input_path.stem}.srt"),
        )
    else:
        raise AppError("output.publication_invalid", {"reason": "export_refs"})
    return tuple(
        (logical_name, target_name, export_by_name[logical_name])
        for logical_name, target_name in names
    )
