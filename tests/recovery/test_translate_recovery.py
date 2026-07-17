from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.support import llm_snapshot

from captioner.adapters.llm.scripted import ScriptedLLMAdapter
from captioner.adapters.persistence.content_addressed_artifact_store import (
    ContentAddressedArtifactStore,
)
from captioner.adapters.persistence.domain_codecs import encode_track, encode_transcript
from captioner.adapters.persistence.filesystem_llm_cache import FilesystemLLMCache
from captioner.adapters.pipeline.stages import TranslateStage
from captioner.core.domain.artifact import ArtifactRef
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.job import JobConfig
from captioner.core.domain.llm import FastTranslationResponse, LLMRequest, response_batch_schema
from captioner.core.domain.stage import PipelineProfile, stage_plan_for
from captioner.core.domain.subtitle import SubtitleCue, SubtitleTrack, derive_subtitle_track_id
from captioner.core.domain.transcript import Transcript, TranscriptSegment, WordToken
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.ports.stage_runner import (
    ProducedArtifact,
    StageExecutionContext,
    StageExecutionRequest,
)
from captioner.infrastructure.prompts import PromptLoader


class CharacterCounter:
    def count(self, text: str) -> int:
        return len(text)


def _fixture(
    tmp_path: Path,
) -> tuple[ContentAddressedArtifactStore, JobConfig, tuple[ArtifactRef, ...]]:
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    words = (
        WordToken("word-000001", "hello 10", 0, 500),
        WordToken("word-000002", "world", 1_000, 1_500),
    )
    transcript = Transcript(
        "transcript-recovery",
        "en",
        words,
        (
            TranscriptSegment(
                "segment-000001",
                ("word-000001", "word-000002"),
                "hello 10 world",
                0,
                1_500,
                None,
            ),
        ),
        "fixture",
        "fixture",
        {},
    )
    config = SegmentationPolicyConfig()
    cues = (
        SubtitleCue("cue-000001", 0, 500, ("word-000001",), "hello 10", None, ("hello 10",)),
        SubtitleCue("cue-000002", 1_000, 1_500, ("word-000002",), "world", None, ("world",)),
    )
    track = SubtitleTrack(
        derive_subtitle_track_id(transcript.id, transcript.language, cues, config.to_mapping()),
        transcript.id,
        transcript.language,
        cues,
        0,
        config.signature,
    )
    transcript_ref = store.put_bytes(
        encode_transcript(transcript),
        kind="transcript",
        media_type="application/json",
        logical_name="transcript.json",
    )
    track_ref = store.put_bytes(
        encode_track(track),
        kind="track",
        media_type="application/json",
        logical_name="subtitle-track.json",
    )
    profile = PipelineProfile.FAST
    job_config = JobConfig(
        "tiny",
        "faster-whisper:tiny",
        "cpu",
        "int8",
        "en",
        True,
        "ffmpeg",
        "ffprobe",
        {"rate": 16000},
        config.to_mapping(),
        str((tmp_path / "output").resolve()),
        False,
        {stage.value: "1" for stage in stage_plan_for(profile)},
        pipeline_profile=profile,
        llm=llm_snapshot(
            profile,
            chunk={
                "max_items": 1,
                "max_input_tokens": 4096,
                "context_before_items": 0,
                "context_after_items": 0,
                "max_audio_context_duration_ms": None,
            },
        ),
    )
    return store, job_config, (transcript_ref, track_ref)


def _response(request: LLMRequest, _schema: type[object], _context: ExecutionContext) -> object:
    payload = [
        {
            "id": item.id,
            "corrected_source": item.source,
            "translated_text": "你好 10" if item.id == "cue-000001" else "世界",
        }
        for item in request.items
    ]
    return response_batch_schema(FastTranslationResponse).from_mapping(payload)


def _run(
    stage: TranslateStage,
    tmp_path: Path,
    config: JobConfig,
    refs: tuple[ArtifactRef, ...],
) -> tuple[ProducedArtifact, ...]:
    workspace = tmp_path / f"workspace-{id(stage)}"
    workspace.mkdir()
    request = StageExecutionRequest(
        "batch-a", "job-000001", (tmp_path / "input.wav").resolve(), config, refs
    )
    return asyncio.run(stage.execute(request, StageExecutionContext(ExecutionContext(), workspace)))


def test_translate_retry_reuses_successful_chunk_cache(tmp_path: Path) -> None:
    store, config, refs = _fixture(tmp_path)
    prompt = PromptLoader(Path("resources/prompts")).load("translate_fast", "v1")
    cache = FilesystemLLMCache(tmp_path / "cache")
    first = ScriptedLLMAdapter(
        structured_responses=(_response, AppError("llm.upstream_unavailable", retryable=True))
    )
    stage = TranslateStage(store, first, cache, CharacterCounter(), prompt)
    with pytest.raises(AppError, match=r"llm\.upstream_unavailable"):
        _run(stage, tmp_path, config, refs)
    assert len(first.structured_calls) == 2

    second = ScriptedLLMAdapter(structured_responses=(_response,))
    result = _run(
        TranslateStage(store, second, cache, CharacterCounter(), prompt),
        tmp_path,
        config,
        refs,
    )
    assert len(second.structured_calls) == 1
    assert second.structured_calls[0].items[0].id == "cue-000002"
    assert result[0].data is not None
