"""Profile-aware Phase 4 Stage runners composed from application ports."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from captioner.adapters.exporters.srt import serialize_bytes as serialize_srt
from captioner.adapters.persistence.domain_codecs import (
    decode_audio,
    decode_corrected_transcript,
    decode_media,
    decode_publication_receipt,
    decode_terminology,
    decode_track,
    decode_transcript,
    encode_audio,
    encode_corrected_transcript,
    encode_json,
    encode_media,
    encode_publication_receipt,
    encode_terminology,
    encode_track,
    encode_transcript,
)
from captioner.adapters.persistence.local_artifact_store import LocalArtifactStore
from captioner.adapters.subtitles.ass import serialize_bytes as serialize_ass
from captioner.adapters.subtitles.json_track import serialize as serialize_track_json
from captioner.adapters.subtitles.webvtt import serialize_bytes as serialize_webvtt
from captioner.core.application.anomaly_review import build_reviewed_track, review_report
from captioner.core.application.llm_chunk_executor import (
    LLMChunkExecutionConfig,
    LLMChunkExecutor,
)
from captioner.core.application.output_transaction import commit_output_set
from captioner.core.application.source_correction import (
    build_corrected_transcript,
    build_terminology_units,
    merge_terminology,
)
from captioner.core.domain.artifact import ArtifactRef
from captioner.core.domain.errors import AppError
from captioner.core.domain.llm import (
    FastTranslationResponse,
    LLMTaskKind,
    QualityTranslationResponse,
    ReviewResponse,
    SourceCorrectionResponse,
    TerminologyResponse,
)
from captioner.core.domain.publication import PublicationReceipt, PublishedTarget
from captioner.core.domain.result import JsonValue
from captioner.core.domain.stage import StageName
from captioner.core.domain.subtitle import SubtitleCue, SubtitleTrack, derive_subtitle_track_id
from captioner.core.domain.subtitle_validation import (
    validate_subtitle_track,
    validate_translated_mapping,
)
from captioner.core.domain.terminology import Terminology
from captioner.core.domain.transcript import CorrectedTranscript
from captioner.core.policies.line_breaking import break_lines
from captioner.core.policies.llm_anomalies import (
    AnomalyChunkPlanner,
    SubtitleAnomaly,
    detect_anomalies,
)
from captioner.core.policies.llm_chunking import ChunkingConfig, ChunkItem, ChunkPlanner, LLMChunk
from captioner.core.policies.segmentation import canonical_words
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.simple_segmentation import SimpleSegmentationConfig, segment_transcript
from captioner.core.ports.asr import ASREngine, TranscriptionRequest
from captioner.core.ports.durable_artifact_store import DurableArtifactStorePort
from captioner.core.ports.llm import LLMClient
from captioner.core.ports.llm_cache import LLMCachePort
from captioner.core.ports.media import AudioNormalizer, MediaInspector
from captioner.core.ports.stage_runner import (
    ProducedArtifact,
    StageExecutionContext,
    StageExecutionRequest,
)
from captioner.core.ports.token_counter import TokenCounter
from captioner.infrastructure.prompts import PromptIdentity

_PHASE3_EXPORT_NAMES = (
    "final-transcript.json",
    "final-subtitle.json",
    "final-subtitle.srt",
    "final-subtitle.vtt",
    "final-subtitle.ass",
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
class CorrectSourceStage:
    """Extract terminology and correct one immutable source Word at a time."""

    artifacts: DurableArtifactStorePort
    client: LLMClient
    cache: LLMCachePort
    token_counter: TokenCounter
    terminology_prompt: PromptIdentity
    correction_prompt: PromptIdentity
    config: SegmentationPolicyConfig | SimpleSegmentationConfig | None = None
    name: StageName = StageName.CORRECT_SOURCE
    version: str = "correct-source-v1"
    repair_prompt: PromptIdentity | None = None

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        target_language = request.config.target_language
        if target_language is None:
            raise AppError("llm.target_language_missing")
        _validate_target_language(target_language)
        transcript = decode_transcript(self.artifacts.read_bytes(_ref(request, "transcript.json")))
        words = canonical_words(transcript.words)
        terminology_units = build_terminology_units(transcript)
        terminology_items = tuple(
            ChunkItem(unit.id, unit.text, unit.start_ms, unit.end_ms) for unit in terminology_units
        )
        correction_items = tuple(
            ChunkItem(
                word.id,
                _canonical_word_text(word.text),
                word.start_ms,
                word.end_ms,
            )
            for word in words
        )
        chunking = _chunking_from_snapshot(request.config.llm)
        terminology_config = _stage_execution_config(
            request.config.llm,
            "terminology",
            transcript.language,
            target_language,
            self.terminology_prompt,
            chunking,
            "quality",
        )
        terminology_executor = LLMChunkExecutor(
            self.client,
            self.cache,
            ChunkPlanner(self.token_counter, chunking),
            terminology_config,
        )
        terminology_responses = await terminology_executor.execute(
            terminology_items,
            TerminologyResponse,
            context.execution,
        )
        terminology = merge_terminology(
            transcript,
            transcript.language,
            target_language,
            terminology_responses,
            units=terminology_units,
        )
        correction_config = _stage_execution_config(
            request.config.llm,
            LLMTaskKind.CORRECT_SOURCE.value,
            transcript.language,
            target_language,
            self.correction_prompt,
            chunking,
            "quality",
            self.repair_prompt,
            lambda chunk: _terminology_context(
                terminology,
                {item_id for item in chunk.items for item_id in (item.id,)},
            ),
        )
        correction_executor = LLMChunkExecutor(
            self.client,
            self.cache,
            ChunkPlanner(self.token_counter, chunking),
            correction_config,
        )
        correction_responses = await correction_executor.execute(
            correction_items,
            SourceCorrectionResponse,
            context.execution,
        )
        corrected = build_corrected_transcript(transcript, correction_responses)
        return (
            ProducedArtifact(
                "terminology-json",
                "application/json",
                "terminology.json",
                data=encode_terminology(terminology),
            ),
            ProducedArtifact(
                "corrected-transcript-json",
                "application/json",
                "corrected-transcript.json",
                data=encode_corrected_transcript(corrected),
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
        corrected: CorrectedTranscript | None = None
        corrected_ref = _optional_ref(request, "corrected-transcript.json")
        if corrected_ref is not None:
            corrected = decode_corrected_transcript(self.artifacts.read_bytes(corrected_ref))
            if corrected.transcript_id != transcript.id:
                raise AppError("transcript.correction_invalid", {"reason": "transcript_id"})
        corrected_mapping = None if corrected is None else corrected.corrected_text_by_word_id

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
            corrected_text_by_word_id=corrected_mapping,
        )
        report = validate_subtitle_track(
            track,
            transcript,
            _policy_config(self.config),
            corrected_text_by_word_id=corrected_mapping,
        )
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
class TranslateStage:
    """Translate each source Cue without allowing the model to edit its timing."""

    artifacts: DurableArtifactStorePort
    client: LLMClient
    cache: LLMCachePort
    token_counter: TokenCounter
    prompt: PromptIdentity
    config: SegmentationPolicyConfig | SimpleSegmentationConfig | None = None
    name: StageName = StageName.TRANSLATE
    version: str = "translate-v1"
    repair_prompt: PromptIdentity | None = None

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        target_language = request.config.target_language
        if target_language is None:
            raise AppError("llm.target_language_missing")
        _validate_target_language(target_language)
        transcript = decode_transcript(self.artifacts.read_bytes(_ref(request, "transcript.json")))
        source_track = decode_track(self.artifacts.read_bytes(_ref(request, "subtitle-track.json")))
        policy = _policy_config(self.config)
        source_report = validate_subtitle_track(source_track, transcript, policy)
        if not source_report.is_valid:
            first = next(issue for issue in source_report.issues if issue.severity.value == "error")
            raise AppError("subtitle.validation_failed", {"reason": first.code})

        items = tuple(
            ChunkItem(cue.id, cue.source_text, cue.start_ms, cue.end_ms)
            for cue in source_track.cues
        )
        chunking = _chunking_from_snapshot(request.config.llm)
        execution_config = _translation_execution_config(
            request.config.llm,
            transcript.language,
            target_language,
            self.prompt,
            chunking,
            self.repair_prompt,
        )
        executor = LLMChunkExecutor(
            self.client,
            self.cache,
            ChunkPlanner(self.token_counter, chunking),
            execution_config,
        )
        responses = await executor.execute(
            items,
            FastTranslationResponse,
            context.execution,
        )
        response_by_id = {_response_id(response): response for response in responses}
        cues: list[SubtitleCue] = []
        for cue in source_track.cues:
            response = response_by_id[cue.id]
            translated = _fast_response(response)
            cues.append(
                SubtitleCue(
                    cue.id,
                    cue.start_ms,
                    cue.end_ms,
                    cue.source_word_ids,
                    translated.corrected_source,
                    translated.translated_text,
                    break_lines(translated.translated_text, policy),
                )
            )
        track_id = derive_subtitle_track_id(
            transcript.id,
            target_language,
            cues,
            policy.to_mapping(),
        )
        translated_track = SubtitleTrack(
            track_id,
            transcript.id,
            target_language,
            tuple(cues),
            1,
            policy.signature,
        )
        report = validate_subtitle_track(
            translated_track,
            transcript,
            policy,
            target_language,
        )
        if not report.is_valid:
            first = next(issue for issue in report.issues if issue.severity.value == "error")
            raise AppError("subtitle.validation_failed", {"reason": first.code})
        target_name = _translated_track_name(target_language)
        return (
            ProducedArtifact(
                "translated-subtitle-track-json",
                "application/json",
                target_name,
                data=encode_track(translated_track),
            ),
            ProducedArtifact(
                "translation-report-json",
                "application/json",
                "translation-report.json",
                data=encode_json(
                    {
                        "schema_version": 1,
                        "profile": "fast",
                        "source_track_id": source_track.id,
                        "translated_track_id": translated_track.id,
                        "source_language": transcript.language,
                        "target_language": target_language,
                        "cue_count": len(cues),
                        "validated": True,
                    }
                ),
            ),
        )


@dataclass(slots=True)
class QualityTranslateStage:
    """Translate corrected source cues without changing their mapping."""

    artifacts: DurableArtifactStorePort
    client: LLMClient
    cache: LLMCachePort
    token_counter: TokenCounter
    prompt: PromptIdentity
    config: SegmentationPolicyConfig | SimpleSegmentationConfig | None = None
    name: StageName = StageName.TRANSLATE
    version: str = "translate-quality-v1"
    repair_prompt: PromptIdentity | None = None

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        target_language = request.config.target_language
        if target_language is None:
            raise AppError("llm.target_language_missing")
        _validate_target_language(target_language)
        transcript = decode_transcript(self.artifacts.read_bytes(_ref(request, "transcript.json")))
        source_track = decode_track(self.artifacts.read_bytes(_ref(request, "subtitle-track.json")))
        policy = _policy_config(self.config)
        corrected = _read_corrected(self.artifacts, request, transcript)
        corrected_mapping = None if corrected is None else corrected.corrected_text_by_word_id
        source_report = validate_subtitle_track(
            source_track,
            transcript,
            policy,
            corrected_text_by_word_id=corrected_mapping,
        )
        if not source_report.is_valid:
            first = next(issue for issue in source_report.issues if issue.severity.value == "error")
            raise AppError("subtitle.validation_failed", {"reason": first.code})
        terminology = decode_terminology(
            self.artifacts.read_bytes(_ref(request, "terminology.json"))
        )
        if (
            terminology.transcript_id != transcript.id
            or terminology.target_language != target_language
        ):
            raise AppError("llm.terminology_invalid", {"reason": "transcript_or_language"})
        chunking = _chunking_from_snapshot(request.config.llm)
        execution_config = _stage_execution_config(
            request.config.llm,
            LLMTaskKind.TRANSLATE_QUALITY.value,
            transcript.language,
            target_language,
            self.prompt,
            chunking,
            "quality",
            self.repair_prompt,
            lambda chunk: _terminology_context_for_cues(
                terminology,
                {item.id for item in chunk.items},
                source_track,
            ),
        )
        executor = LLMChunkExecutor(
            self.client,
            self.cache,
            ChunkPlanner(self.token_counter, chunking),
            execution_config,
        )
        items = tuple(
            ChunkItem(cue.id, cue.source_text, cue.start_ms, cue.end_ms)
            for cue in source_track.cues
        )
        responses = await executor.execute(items, QualityTranslationResponse, context.execution)
        response_by_id = {_response_id(response): response for response in responses}
        cues: list[SubtitleCue] = []
        for cue in source_track.cues:
            response = response_by_id[cue.id]
            cues.append(
                SubtitleCue(
                    cue.id,
                    cue.start_ms,
                    cue.end_ms,
                    cue.source_word_ids,
                    cue.source_text,
                    response.translated_text,
                    break_lines(response.translated_text, policy),
                )
            )
        translated = SubtitleTrack(
            derive_subtitle_track_id(
                transcript.id,
                target_language,
                cues,
                policy.to_mapping(),
            ),
            transcript.id,
            target_language,
            tuple(cues),
            1,
            policy.signature,
        )
        report = validate_translated_mapping(
            translated,
            transcript,
            policy,
            target_language,
            corrected_mapping,
        )
        if not report.is_valid:
            first = next(issue for issue in report.issues if issue.severity.value == "error")
            raise AppError("subtitle.validation_failed", {"reason": first.code})
        return (
            ProducedArtifact(
                "translated-subtitle-track-json",
                "application/json",
                _translated_track_name(target_language),
                data=encode_track(translated),
            ),
            ProducedArtifact(
                "translation-report-json",
                "application/json",
                "translation-report.json",
                data=encode_json(
                    {
                        "schema_version": 1,
                        "profile": "quality",
                        "source_track_id": source_track.id,
                        "translated_track_id": translated.id,
                        "source_language": transcript.language,
                        "target_language": target_language,
                        "cue_count": len(cues),
                        "validated": True,
                    }
                ),
            ),
        )


@dataclass(slots=True)
class ReviewStage:
    """Review only deterministic anomalies and preserve every source field."""

    artifacts: DurableArtifactStorePort
    client: LLMClient
    cache: LLMCachePort
    token_counter: TokenCounter
    prompt: PromptIdentity
    config: SegmentationPolicyConfig | SimpleSegmentationConfig | None = None
    name: StageName = StageName.REVIEW
    version: str = "review-v1"
    repair_prompt: PromptIdentity | None = None

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        target_language = request.config.target_language
        if target_language is None:
            raise AppError("llm.target_language_missing")
        _validate_target_language(target_language)
        transcript = decode_transcript(self.artifacts.read_bytes(_ref(request, "transcript.json")))
        track = decode_track(
            self.artifacts.read_bytes(_ref(request, _translated_track_name(target_language)))
        )
        policy = _policy_config(self.config)
        corrected = _read_corrected(self.artifacts, request, transcript)
        corrected_mapping = None if corrected is None else corrected.corrected_text_by_word_id
        report = validate_translated_mapping(
            track,
            transcript,
            policy,
            target_language,
            corrected_mapping,
        )
        if not report.is_valid:
            first = next(issue for issue in report.issues if issue.severity.value == "error")
            raise AppError("subtitle.validation_failed", {"reason": first.code})
        terminology_ref = _optional_ref(request, "terminology.json")
        terminology = (
            None
            if terminology_ref is None
            else decode_terminology(self.artifacts.read_bytes(terminology_ref))
        )
        anomalies = detect_anomalies(track, transcript, target_language, policy, terminology)
        responses: tuple[object, ...] = ()
        if anomalies:
            all_items = tuple(
                ChunkItem(cue.id, _review_item_text(cue), cue.start_ms, cue.end_ms)
                for cue in track.cues
            )
            anomaly_ids = {anomaly.cue_id for anomaly in anomalies}
            items = tuple(item for item in all_items if item.id in anomaly_ids)
            chunking = _chunking_from_snapshot(request.config.llm)
            execution_config = _stage_execution_config(
                request.config.llm,
                LLMTaskKind.REVIEW.value,
                transcript.language,
                target_language,
                self.prompt,
                chunking,
                "quality",
                self.repair_prompt,
                lambda chunk: _review_context(
                    anomalies,
                    terminology,
                    track,
                    chunk,
                ),
            )
            executor = LLMChunkExecutor(
                self.client,
                self.cache,
                AnomalyChunkPlanner(self.token_counter, all_items),
                execution_config,
            )
            responses = await executor.execute(
                items,
                ReviewResponse,
                context.execution,
                validation_source_texts={cue.id: cue.source_text for cue in track.cues},
            )
        reviewed = build_reviewed_track(
            track,
            transcript,
            target_language,
            policy,
            anomalies,
            responses,
            terminology,
            corrected_mapping,
        )
        changed_cue_ids = tuple(
            before.id
            for before, after in zip(track.cues, reviewed.cues, strict=True)
            if before.translated_text != after.translated_text
        )
        return (
            ProducedArtifact(
                "reviewed-subtitle-track-json",
                "application/json",
                f"reviewed-track.{target_language}.json",
                data=encode_track(reviewed),
            ),
            ProducedArtifact(
                "review-report-json",
                "application/json",
                "review-report.json",
                data=encode_json(
                    review_report(
                        track,
                        anomalies,
                        terminology,
                        output_track_id=reviewed.id,
                        target_language=target_language,
                        changed_cue_ids=changed_cue_ids,
                        llm_called=bool(anomalies),
                        validated=True,
                        prompt=self.prompt,
                    )
                ),
            ),
        )


@dataclass(slots=True)
class ExportStage:
    artifacts: DurableArtifactStorePort
    config: SegmentationPolicyConfig | SimpleSegmentationConfig | None = None
    name: StageName = StageName.EXPORT
    version: str = "export-v3"

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        context.execution.raise_if_cancelled()
        transcript = decode_transcript(self.artifacts.read_bytes(_ref(request, "transcript.json")))
        track_name = "subtitle-track.json"
        corrected_mapping = None
        if request.config.pipeline_profile.value in {"fast", "quality"}:
            target_language = request.config.target_language
            if target_language is None:
                raise AppError("llm.target_language_missing")
            track_name = (
                _reviewed_track_name(target_language)
                if request.config.pipeline_profile.value == "quality"
                else _translated_track_name(target_language)
            )
            corrected = _read_corrected(self.artifacts, request, transcript)
            corrected_mapping = None if corrected is None else corrected.corrected_text_by_word_id
        track = decode_track(self.artifacts.read_bytes(_ref(request, track_name)))
        config = _policy_config(self.config)
        report = validate_subtitle_track(
            track,
            transcript,
            config,
            request.config.target_language
            if request.config.pipeline_profile.value != "deterministic"
            else None,
            corrected_mapping,
        )
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
    version: str = "publish-v3"

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        context.execution.raise_if_cancelled()
        export_refs = tuple(
            ref for ref in request.input_artifacts if ref.logical_name.startswith("final-")
        )
        target_specs = _publication_specs(
            request.input_path, export_refs, publication_version=self.version
        )
        targets = tuple((target_name, ref) for _, target_name, ref in target_specs)
        store = LocalArtifactStore(Path(request.config.output_dir))
        if not all(
            _published_matches(Path(request.config.output_dir) / key, ref) for key, ref in targets
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
                    str(_target_path(Path(request.config.output_dir), key)),
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


def _optional_ref(request: StageExecutionRequest, logical_name: str) -> ArtifactRef | None:
    matches = [ref for ref in request.input_artifacts if ref.logical_name == logical_name]
    if len(matches) > 1:
        raise AppError("stage.input_duplicate", {"logical_name": logical_name})
    return matches[0] if matches else None


def _read_corrected(
    artifacts: DurableArtifactStorePort,
    request: StageExecutionRequest,
    transcript: object,
) -> CorrectedTranscript | None:
    corrected_ref = _optional_ref(request, "corrected-transcript.json")
    if corrected_ref is None:
        return None
    corrected = decode_corrected_transcript(artifacts.read_bytes(corrected_ref))
    transcript_id = getattr(transcript, "id", None)
    if corrected.transcript_id != transcript_id:
        raise AppError("transcript.correction_invalid", {"reason": "transcript_id"})
    return corrected


def _canonical_word_text(value: str) -> str:
    from captioner.core.policies.unicode_metrics import normalize_text

    return normalize_text(value)


def _terminology_context(terminology: Terminology, word_ids: set[str]) -> Mapping[str, JsonValue]:
    entries = terminology.entries
    values = [
        {"source_term": entry.source, "target_term": entry.target}
        for entry in entries
        if set(entry.source_word_ids) & word_ids
    ]
    return cast(Mapping[str, JsonValue], {"terminology": values})


def _terminology_context_for_cues(
    terminology: Terminology,
    cue_ids: set[str],
    track: SubtitleTrack,
) -> Mapping[str, JsonValue]:
    cue_words = {
        word_id for cue in track.cues if cue.id in cue_ids for word_id in cue.source_word_ids
    }
    return _terminology_context(terminology, cue_words)


def _review_context(
    anomalies: tuple[SubtitleAnomaly, ...],
    terminology: Terminology | None,
    track: SubtitleTrack,
    chunk: LLMChunk,
) -> Mapping[str, JsonValue]:
    core_ids = {item.id for item in chunk.items}
    context_ids = {item.id for item in chunk.context}
    reasons = [
        {"cue_id": anomaly.cue_id, "reasons": list(anomaly.reasons)}
        for anomaly in anomalies
        if anomaly.cue_id in core_ids
    ]
    nearby = [
        {
            "cue_id": cue.id,
            "source_text": cue.source_text,
            "translated_text": cue.translated_text,
        }
        for cue in track.cues
        if cue.id in context_ids
    ]
    cue_words = {
        word_id
        for cue in track.cues
        if cue.id in core_ids | context_ids
        for word_id in cue.source_word_ids
    }
    terms = []
    if terminology is not None:
        terms = [
            {"source_term": entry.source, "target_term": entry.target}
            for entry in terminology.entries
            if set(entry.source_word_ids) & cue_words
        ]
    return cast(
        Mapping[str, JsonValue],
        {"anomalies": reasons, "nearby_cues": nearby, "terminology": terms},
    )


def _review_item_text(cue: SubtitleCue) -> str:
    return (
        encode_json(
            {
                "source_text": cue.source_text,
                "translated_text": cue.translated_text,
            }
        )
        .decode("utf-8")
        .rstrip("\n")
    )


def _reviewed_track_name(target_language: str) -> str:
    _validate_target_language(target_language)
    return f"reviewed-track.{target_language}.json"


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
    publication_version: str = "publish-v2",
) -> None:
    receipt = decode_publication_receipt(receipt_bytes)
    target_specs = _publication_specs(
        input_path, export_refs, publication_version=publication_version
    )
    expected_generation = hashlib.sha256(
        "".join(ref.sha256 for _, _, ref in target_specs).encode()
    ).hexdigest()
    expected_targets = {
        str(_target_path(output_dir, target_name)): (ref, target_name)
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


def _chunking_from_snapshot(
    llm: Mapping[str, object] | None,
) -> ChunkingConfig:
    defaults = ChunkingConfig(context_before_items=1, context_after_items=1)
    if llm is None or llm.get("chunk") is None:
        return defaults
    raw_value = llm.get("chunk")
    if not isinstance(raw_value, Mapping):
        raise AppError("llm.chunk_config_invalid", {"reason": "object"})
    raw = cast(Mapping[str, object], raw_value)
    fields = {
        "max_items",
        "max_input_tokens",
        "context_before_items",
        "context_after_items",
        "max_audio_context_duration_ms",
    }
    if set(raw) - fields:
        raise AppError("llm.chunk_config_invalid", {"reason": "fields"})
    max_items = _snapshot_int(raw, "max_items", defaults.max_items)
    max_input_tokens = _snapshot_int(raw, "max_input_tokens", defaults.max_input_tokens)
    context_before = _snapshot_int(raw, "context_before_items", defaults.context_before_items)
    context_after = _snapshot_int(raw, "context_after_items", defaults.context_after_items)
    duration = raw.get("max_audio_context_duration_ms", defaults.max_audio_context_duration_ms)
    if duration is not None and type(duration) is not int:
        raise AppError("llm.chunk_config_invalid", {"field": "max_audio_context_duration_ms"})
    try:
        return ChunkingConfig(
            max_items,
            max_input_tokens,
            context_before,
            context_after,
            duration,
        )
    except AppError:
        raise
    except (TypeError, ValueError) as exc:
        raise AppError("llm.chunk_config_invalid", {"reason": "values"}) from exc


def _translation_execution_config(
    llm: Mapping[str, object] | None,
    source_language: str,
    target_language: str,
    prompt: PromptIdentity,
    chunking: ChunkingConfig,
    repair_prompt: PromptIdentity | None = None,
) -> LLMChunkExecutionConfig:
    return _stage_execution_config(
        llm,
        LLMTaskKind.TRANSLATE_FAST.value,
        source_language,
        target_language,
        prompt,
        chunking,
        "fast",
        repair_prompt,
    )


def _stage_execution_config(
    llm: Mapping[str, object] | None,
    task_kind: str,
    source_language: str,
    target_language: str | None,
    prompt: PromptIdentity,
    chunking: ChunkingConfig,
    profile: str,
    repair_prompt: PromptIdentity | None = None,
    context_payload_factory: Callable[[LLMChunk], Mapping[str, JsonValue]] | None = None,
) -> LLMChunkExecutionConfig:
    values = {} if llm is None else dict(llm)
    provider_kind = _snapshot_string(values, "kind", "openai-compatible")
    provider_identity = _snapshot_string(values, "provider_profile", "default")
    base_url = _snapshot_string(values, "base_url", "https://unconfigured.invalid/v1")
    model = _snapshot_string(values, "model", "unit-test-model")
    temperature = values.get("temperature", 0.1)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise AppError("llm.config_invalid", {"field": "temperature"})
    schema_version = values.get("response_schema_version", 1)
    if type(schema_version) is not int or schema_version < 1:
        raise AppError("llm.config_invalid", {"field": "response_schema_version"})
    return LLMChunkExecutionConfig(
        task_kind=task_kind,
        provider_kind=provider_kind,
        provider_identity=provider_identity,
        base_url_identity=base_url,
        model=model,
        temperature=float(temperature),
        source_language=source_language,
        target_language=target_language,
        profile=profile,
        prompt_id=prompt.prompt_id,
        prompt_version=prompt.prompt_version,
        prompt_content_sha256=prompt.content_sha256,
        prompt_content=prompt.content,
        chunking=chunking,
        response_schema_version=schema_version,
        repair_prompt_id="" if repair_prompt is None else repair_prompt.prompt_id,
        repair_prompt_version="" if repair_prompt is None else repair_prompt.prompt_version,
        repair_prompt_content_sha256="" if repair_prompt is None else repair_prompt.content_sha256,
        repair_prompt_content="" if repair_prompt is None else repair_prompt.content,
        context_payload_factory=context_payload_factory,
    )


def _snapshot_string(values: Mapping[str, object], key: str, default: str) -> str:
    value = values.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise AppError("llm.config_invalid", {"field": key})
    return value.strip()


def _snapshot_int(values: Mapping[str, object], key: str, default: int) -> int:
    value = values.get(key, default)
    if type(value) is not int:
        raise AppError("llm.chunk_config_invalid", {"field": key})
    return value


def _fast_response(value: object) -> FastTranslationResponse:
    if isinstance(value, FastTranslationResponse):
        return value
    return FastTranslationResponse.from_mapping(value)


def _response_id(value: object) -> str:
    mapping = cast(Mapping[str, object], value) if isinstance(value, Mapping) else None
    item = mapping.get("id") if mapping is not None else getattr(cast(object, value), "id", None)
    if not isinstance(item, str) or not item:
        raise AppError("llm.response_invalid", {"reason": "id"})
    return item


def _validate_target_language(value: str) -> None:
    if (
        not value
        or value != value.strip()
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise AppError("llm.target_language_invalid")


def _translated_track_name(target_language: str) -> str:
    _validate_target_language(target_language)
    return f"translated-track.{target_language}.json"


def _publication_specs(
    input_path: Path,
    export_refs: tuple[ArtifactRef, ...],
    *,
    publication_version: str,
) -> tuple[tuple[str, str, ArtifactRef], ...]:
    if any(not ref.logical_name.startswith("final-") for ref in export_refs):
        raise AppError("output.publication_invalid", {"reason": "export_refs"})
    export_by_name = {ref.logical_name: ref for ref in export_refs}
    if len(export_by_name) != len(export_refs):
        raise AppError("output.publication_invalid", {"reason": "export_refs"})
    if publication_version in {"publish-v2", "publish-v3"} and set(export_by_name) == set(
        _PHASE3_EXPORT_NAMES
    ):
        names = (
            ("final-transcript.json", f"{input_path.stem}.transcript.json"),
            ("final-subtitle.json", f"{input_path.stem}.subtitle.json"),
            ("final-subtitle.srt", f"{input_path.stem}.srt"),
            ("final-subtitle.vtt", f"{input_path.stem}.vtt"),
            ("final-subtitle.ass", f"{input_path.stem}.ass"),
        )
    elif publication_version == "publish-v1" and set(export_by_name) == {
        "final-transcript.json",
        "final-subtitle.srt",
    }:
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


def _target_path(output_dir: Path, target_name: str) -> Path:
    """Resolve only the output root; keep the final target path lexical."""
    return output_dir.expanduser().resolve() / target_name
