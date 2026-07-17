from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tests.support import llm_snapshot, make_transcript

from captioner.adapters.llm.scripted import ScriptedLLMAdapter
from captioner.adapters.persistence.content_addressed_artifact_store import (
    ContentAddressedArtifactStore,
)
from captioner.adapters.persistence.domain_codecs import (
    decode_track,
    encode_track,
    encode_transcript,
)
from captioner.adapters.persistence.filesystem_llm_cache import FilesystemLLMCache
from captioner.adapters.pipeline.stages import ExportStage, TranslateStage
from captioner.core.domain.artifact import ArtifactRef
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.job import JobConfig
from captioner.core.domain.llm import (
    FastTranslationResponse,
    LLMRequest,
    response_batch_schema,
)
from captioner.core.domain.stage import PipelineProfile, stage_plan_for
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.simple_segmentation import segment_transcript
from captioner.core.ports.stage_runner import StageExecutionContext, StageExecutionRequest
from captioner.infrastructure.prompts import PromptLoader


class CharacterCounter:
    def count(self, text: str) -> int:
        return len(text)


def _config(tmp_path: Path) -> JobConfig:
    profile = PipelineProfile.FAST
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
        SegmentationPolicyConfig().to_mapping(),
        str((tmp_path / "output").resolve()),
        False,
        {stage.value: "1" for stage in stage_plan_for(profile)},
        pipeline_profile=profile,
        llm=llm_snapshot(profile),
    )


def _put(store: ContentAddressedArtifactStore, data: bytes, name: str):
    return store.put_bytes(data, kind=name, media_type="application/json", logical_name=name)


def _request(
    tmp_path: Path,
    config: JobConfig,
    refs: tuple[ArtifactRef, ...],
) -> StageExecutionRequest:
    return StageExecutionRequest(
        "batch-a",
        "job-000001",
        (tmp_path / "input.wav").resolve(),
        config,
        refs,
    )


def _context(tmp_path: Path) -> StageExecutionContext:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return StageExecutionContext(ExecutionContext(), workspace)


def test_fast_translation_preserves_cue_mapping_and_renders_target_text(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    transcript = make_transcript(("hello 10 ", "world"), language="ja")
    source_track = segment_transcript(transcript)
    transcript_ref = _put(store, encode_transcript(transcript), "transcript.json")
    source_ref = _put(store, encode_track(source_track), "subtitle-track.json")

    seen_languages: list[str | None] = []

    def response(request: LLMRequest, _schema: type[object], _context: ExecutionContext) -> object:
        seen_languages.append(request.source_language)
        items = request.items
        payload = [
            {
                "id": item.id,
                "corrected_source": item.source,
                "translated_text": "你好 10 世界",
            }
            for item in items
        ]
        return response_batch_schema(FastTranslationResponse).from_mapping({"responses": payload})

    prompt = PromptLoader(Path("resources/prompts")).load("translate_fast", "v1")
    config = _config(tmp_path)
    adapter = ScriptedLLMAdapter(structured_responses=(response,))
    stage = TranslateStage(
        store,
        adapter,
        FilesystemLLMCache(tmp_path / "cache"),
        CharacterCounter(),
        prompt,
        SegmentationPolicyConfig(),
    )
    produced = asyncio.run(
        stage.execute(
            _request(tmp_path, config, (transcript_ref, source_ref)),
            _context(tmp_path),
        )
    )
    translated_ref = _put(store, produced[0].data or b"", produced[0].logical_name)
    translated = decode_track(store.read_bytes(translated_ref))

    assert produced[0].logical_name == "translated-track.zh-CN.json"
    assert produced[1].logical_name == "translation-report.json"
    assert json.loads(produced[0].data or b"")["schema_version"] == 3
    assert seen_languages == ["ja"]
    assert config.language == "en"
    assert translated.revision == 1
    assert translated.language == "zh-CN"
    assert translated.cues[0].translated_text == "你好 10 世界"
    assert translated.cues[0].lines == ("你好 10 世界",)
    assert translated.cues[0].source_word_ids == source_track.cues[0].source_word_ids
    assert translated.cues[0].start_ms == source_track.cues[0].start_ms
    assert translated.cues[0].end_ms == source_track.cues[0].end_ms

    export = asyncio.run(
        ExportStage(store, SegmentationPolicyConfig()).execute(
            _request(tmp_path, _config(tmp_path), (transcript_ref, source_ref, translated_ref)),
            _context(tmp_path),
        )
    )
    srt = next(item.data for item in export if item.logical_name == "final-subtitle.srt")
    assert "你好 10 世界".encode() in (srt or b"")
