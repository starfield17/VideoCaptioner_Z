from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.support import make_transcript

from captioner.adapters.llm.scripted import ScriptedLLMAdapter
from captioner.adapters.persistence.content_addressed_artifact_store import (
    ContentAddressedArtifactStore,
)
from captioner.adapters.persistence.domain_codecs import (
    decode_corrected_transcript,
    decode_terminology,
    decode_track,
    encode_transcript,
)
from captioner.adapters.persistence.filesystem_llm_cache import FilesystemLLMCache
from captioner.adapters.pipeline.stages import (
    CorrectSourceStage,
    QualityTranslateStage,
    ReviewStage,
    SegmentStage,
)
from captioner.core.domain.artifact import ArtifactRef
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.job import JobConfig
from captioner.core.domain.llm import LLMRequest
from captioner.core.domain.stage import PipelineProfile, stage_plan_for
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.ports.stage_runner import (
    StageExecutionContext,
    StageExecutionRequest,
)
from captioner.infrastructure.prompts import PromptLoader


class CharacterCounter:
    def count(self, text: str) -> int:
        return len(text)


def _parse(schema: type[object], payload: object) -> object:
    parser = getattr(schema, "from_mapping", None)
    if not callable(parser):
        raise TypeError
    return parser(payload)


def _config(tmp_path: Path) -> JobConfig:
    policy = SegmentationPolicyConfig(hard_gap_ms=100, preferred_gap_ms=100)
    profile = PipelineProfile.QUALITY
    return JobConfig(
        "tiny",
        "faster-whisper:tiny",
        "cpu",
        "int8",
        "en",
        True,
        "ffmpeg",
        "ffprobe",
        {"rate": 16000},
        policy.to_mapping(),
        str((tmp_path / "output").resolve()),
        False,
        {stage.value: "1" for stage in stage_plan_for(profile)},
        pipeline_profile=profile,
        llm={
            "kind": "openai-compatible",
            "provider_profile": "default",
            "base_url": "https://provider.example/v1",
            "model": "unit-test-model",
            "temperature": 0.1,
            "target_language": "zh-CN",
            "chunk": {
                "max_items": 32,
                "max_input_tokens": 4096,
                "context_before_items": 1,
                "context_after_items": 1,
                "max_audio_context_duration_ms": 120_000,
            },
        },
    )


def _put(store: ContentAddressedArtifactStore, data: bytes, name: str) -> ArtifactRef:
    return store.put_bytes(data, kind=name, media_type="application/json", logical_name=name)


def _request(
    tmp_path: Path,
    config: JobConfig,
    refs: tuple[ArtifactRef, ...],
) -> StageExecutionRequest:
    return StageExecutionRequest(
        "batch-quality",
        "job-000001",
        (tmp_path / "input.wav").resolve(),
        config,
        refs,
    )


def _context(tmp_path: Path, name: str) -> StageExecutionContext:
    workspace = tmp_path / name
    workspace.mkdir()
    return StageExecutionContext(ExecutionContext(), workspace)


def _terminology_and_correction(
    request: LLMRequest,
    response_schema: type[object],
    _context: ExecutionContext,
) -> object:
    if request.task_kind == "terminology":
        payload = [
            {
                "id": item.id,
                "source_term": item.source,
                "target_term": "你好 10" if item.id == "word-000001" else "世界",
            }
            for item in request.items
        ]
    else:
        payload = [{"id": item.id, "corrected_source": item.source} for item in request.items]
    return _parse(response_schema, payload)


def _translation_response(
    anomalous: bool,
):
    def respond(
        request: LLMRequest,
        response_schema: type[object],
        _context: ExecutionContext,
    ) -> object:
        payload: list[dict[str, str]] = []
        for item in request.items:
            text = (
                "问候 10"
                if anomalous and item.id == "cue-000001"
                else ("你好 10" if item.id == "cue-000001" else "世界")
            )
            payload.append({"id": item.id, "translated_text": text})
        return _parse(response_schema, payload)

    return respond


def _review_response(
    request: LLMRequest,
    response_schema: type[object],
    _context: ExecutionContext,
) -> object:
    return _parse(
        response_schema, [{"id": item.id, "translated_text": "你好 10"} for item in request.items]
    )


def _number_loss_translation_response(
    request: LLMRequest,
    response_schema: type[object],
    _context: ExecutionContext,
) -> object:
    return _parse(
        response_schema,
        [
            {
                "id": item.id,
                "translated_text": "你好" if item.id == "cue-000001" else "世界",
            }
            for item in request.items
        ],
    )


def _correct_and_segment(
    tmp_path: Path,
) -> tuple[
    ContentAddressedArtifactStore,
    JobConfig,
    tuple[ArtifactRef, ArtifactRef, ArtifactRef, ArtifactRef],
]:
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    config = _config(tmp_path)
    transcript = make_transcript(("hello 10 ", "world"))
    transcript_ref = _put(store, encode_transcript(transcript), "transcript.json")
    prompt_loader = PromptLoader(Path("resources/prompts"))
    correct_stage = CorrectSourceStage(
        store,
        ScriptedLLMAdapter(structured_responses=(_terminology_and_correction,) * 2),
        FilesystemLLMCache(tmp_path / "cache"),
        CharacterCounter(),
        prompt_loader.load("terminology", "v1"),
        prompt_loader.load("correct_source", "v1"),
        SegmentationPolicyConfig(hard_gap_ms=100, preferred_gap_ms=100),
    )
    produced = asyncio.run(
        correct_stage.execute(
            _request(tmp_path, config, (transcript_ref,)),
            _context(tmp_path, "correct-source"),
        )
    )
    terminology_ref = _put(
        store,
        next(item.data for item in produced if item.logical_name == "terminology.json") or b"",
        "terminology.json",
    )
    corrected_ref = _put(
        store,
        next(item.data for item in produced if item.logical_name == "corrected-transcript.json")
        or b"",
        "corrected-transcript.json",
    )
    segment = SegmentStage(
        store,
        SegmentationPolicyConfig(hard_gap_ms=100, preferred_gap_ms=100),
    )
    segmented = asyncio.run(
        segment.execute(
            _request(tmp_path, config, (transcript_ref, corrected_ref, terminology_ref)),
            _context(tmp_path, "segment"),
        )
    )
    track_ref = _put(store, segmented[0].data or b"", "subtitle-track.json")
    return store, config, (transcript_ref, terminology_ref, corrected_ref, track_ref)


def test_quality_correction_and_segmentation_preserve_original_mapping(tmp_path: Path) -> None:
    store, _, refs = _correct_and_segment(tmp_path)
    corrected = decode_corrected_transcript(store.read_bytes(refs[2]))
    terminology = decode_terminology(store.read_bytes(refs[1]))
    track = decode_track(store.read_bytes(refs[3]))

    assert corrected.transcript_id == "transcript-test"
    assert corrected.source_word_ids == ("word-000001", "word-000002")
    assert [entry.target for entry in terminology.entries] == ["你好 10", "世界"]
    assert [cue.source_word_ids for cue in track.cues] == [
        ("word-000001",),
        ("word-000002",),
    ]
    assert [cue.source_text for cue in track.cues] == ["hello 10", "world"]


def test_quality_translation_and_anomaly_free_review_do_not_call_review_llm(
    tmp_path: Path,
) -> None:
    store, config, refs = _correct_and_segment(tmp_path)
    prompt_loader = PromptLoader(Path("resources/prompts"))
    translate = QualityTranslateStage(
        store,
        ScriptedLLMAdapter(structured_responses=(_translation_response(False),)),
        FilesystemLLMCache(tmp_path / "cache-translate"),
        CharacterCounter(),
        prompt_loader.load("translate_quality", "v1"),
        SegmentationPolicyConfig(hard_gap_ms=100, preferred_gap_ms=100),
    )
    translated = asyncio.run(
        translate.execute(
            _request(tmp_path, config, refs),
            _context(tmp_path, "translate"),
        )
    )
    translated_ref = _put(store, translated[0].data or b"", "translated-track.zh-CN.json")
    review_adapter = ScriptedLLMAdapter()
    review = ReviewStage(
        store,
        review_adapter,
        FilesystemLLMCache(tmp_path / "cache-review"),
        CharacterCounter(),
        prompt_loader.load("review_anomalies", "v1"),
        SegmentationPolicyConfig(hard_gap_ms=100, preferred_gap_ms=100),
    )
    reviewed = asyncio.run(
        review.execute(
            _request(tmp_path, config, (*refs, translated_ref)),
            _context(tmp_path, "review-clean"),
        )
    )
    assert review_adapter.structured_calls == []
    assert reviewed[0].logical_name == "reviewed-track.zh-CN.json"
    assert (
        decode_track(
            store.read_bytes(_put(store, reviewed[0].data or b"", reviewed[0].logical_name))
        ).revision
        == 1
    )


def test_quality_review_sends_only_anomalies_and_adjacent_context(tmp_path: Path) -> None:
    store, config, refs = _correct_and_segment(tmp_path)
    prompt_loader = PromptLoader(Path("resources/prompts"))
    translate = QualityTranslateStage(
        store,
        ScriptedLLMAdapter(structured_responses=(_translation_response(True),)),
        FilesystemLLMCache(tmp_path / "cache-translate"),
        CharacterCounter(),
        prompt_loader.load("translate_quality", "v1"),
        SegmentationPolicyConfig(hard_gap_ms=100, preferred_gap_ms=100),
    )
    translated = asyncio.run(
        translate.execute(_request(tmp_path, config, refs), _context(tmp_path, "translate-bad"))
    )
    translated_ref = _put(store, translated[0].data or b"", "translated-track.zh-CN.json")
    review_adapter = ScriptedLLMAdapter(structured_responses=(_review_response,))
    review = ReviewStage(
        store,
        review_adapter,
        FilesystemLLMCache(tmp_path / "cache-review"),
        CharacterCounter(),
        prompt_loader.load("review_anomalies", "v1"),
        SegmentationPolicyConfig(hard_gap_ms=100, preferred_gap_ms=100),
    )
    asyncio.run(
        review.execute(
            _request(tmp_path, config, (*refs, translated_ref)),
            _context(tmp_path, "review-bad"),
        )
    )
    assert len(review_adapter.structured_calls) == 1
    assert review_adapter.structured_calls[0].item_ids == ("cue-000001",)
    assert review_adapter.structured_calls[0].context_ids == ("cue-000002",)


def test_quality_number_loss_is_rejected_before_cache_write(tmp_path: Path) -> None:
    store, config, refs = _correct_and_segment(tmp_path)
    prompt_loader = PromptLoader(Path("resources/prompts"))
    cache = FilesystemLLMCache(tmp_path / "cache-translate")
    translate = QualityTranslateStage(
        store,
        ScriptedLLMAdapter(
            structured_responses=(
                _number_loss_translation_response,
                _number_loss_translation_response,
            )
        ),
        cache,
        CharacterCounter(),
        prompt_loader.load("translate_quality", "v1"),
        SegmentationPolicyConfig(hard_gap_ms=100, preferred_gap_ms=100),
    )
    with pytest.raises(AppError, match=r"llm\.protected_token_lost"):
        asyncio.run(
            translate.execute(
                _request(tmp_path, config, refs),
                _context(tmp_path, "translate-number"),
            )
        )
    assert not list(cache.root.rglob("*.json"))
