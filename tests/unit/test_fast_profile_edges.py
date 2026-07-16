from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from pathlib import Path
from typing import cast

import pytest

from captioner.bootstrap import create_job_config
from captioner.core.domain.artifact import ArtifactRef
from captioner.core.domain.errors import AppError
from captioner.core.domain.llm import FastTranslationResponse
from captioner.core.policies.llm_chunking import ChunkingConfig
from captioner.infrastructure.prompts import PromptLoader

stages_private = import_module("captioner.adapters.pipeline.stages")
chunking_from_snapshot = cast(
    Callable[[Mapping[str, object] | None], ChunkingConfig],
    stages_private._chunking_from_snapshot,
)
fast_response = cast(Callable[[object], FastTranslationResponse], stages_private._fast_response)
publication_specs = cast(Callable[..., object], stages_private._publication_specs)
response_id = cast(Callable[[object], str], stages_private._response_id)
translation_execution_config = cast(
    Callable[..., object], stages_private._translation_execution_config
)
validate_target_language = cast(Callable[[str], None], stages_private._validate_target_language)


def _prompt():
    return PromptLoader(Path("resources/prompts")).load("translate_fast", "v1")


def test_create_job_config_builds_a_redacted_llm_snapshot(tmp_path: Path) -> None:
    config = create_job_config(
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        output_dir=tmp_path / "output",
        overwrite=False,
        target_language="zh-CN",
        provider_profile="default",
        llm_base_url="https://provider.example/v1",
        llm_model="unit-test-model",
        temperature=0.1,
        timeout_sec=30,
        max_retries=2,
        chunk={"max_items": 2},
        prompt_identity={"prompt_id": "translate_fast", "prompt_version": "v1"},
    )

    assert config.schema_version == 2
    assert config.target_language == "zh-CN"
    assert config.provider_profile == "default"
    assert config.llm_model == "unit-test-model"
    assert "api_key" not in repr(config.to_dict())


@pytest.mark.parametrize(
    "snapshot",
    [
        {"chunk": "not-an-object"},
        {"chunk": {"unknown": 1}},
        {"chunk": {"max_items": "2"}},
        {"chunk": {"max_items": 0}},
        {"chunk": {"max_audio_context_duration_ms": True}},
    ],
)
def test_chunk_snapshot_rejects_invalid_configuration(snapshot: Mapping[str, object]) -> None:
    with pytest.raises(AppError, match=r"llm\.chunk_config_invalid"):
        chunking_from_snapshot(snapshot)


def test_chunk_snapshot_uses_defaults_and_preserves_audio_limit() -> None:
    assert chunking_from_snapshot(None) == ChunkingConfig(
        context_before_items=1, context_after_items=1
    )
    selected = chunking_from_snapshot(
        {
            "chunk": {
                "max_items": 2,
                "max_input_tokens": 20,
                "context_before_items": 0,
                "context_after_items": 1,
                "max_audio_context_duration_ms": 5000,
            }
        }
    )
    assert selected == ChunkingConfig(2, 20, 0, 1, 5000)


@pytest.mark.parametrize(
    "snapshot",
    [
        {"temperature": True},
        {"temperature": "0.1"},
        {"response_schema_version": 0},
        {"model": " "},
    ],
)
def test_translation_execution_config_rejects_invalid_values(
    snapshot: Mapping[str, object],
) -> None:
    with pytest.raises(AppError, match=r"llm\.(config|response_schema)_invalid"):
        translation_execution_config(
            snapshot,
            "en",
            "zh-CN",
            _prompt(),
            ChunkingConfig(),
        )


def test_response_helpers_accept_mapping_and_domain_values() -> None:
    mapping = {
        "id": "cue-000001",
        "corrected_source": "hello",
        "translated_text": "你好",
    }
    response = FastTranslationResponse.from_mapping(mapping)
    assert fast_response(mapping) == response
    assert fast_response(response) is response
    assert response_id(mapping) == "cue-000001"
    assert response_id(response) == "cue-000001"

    with pytest.raises(AppError, match=r"llm\.response_invalid"):
        response_id(object())


@pytest.mark.parametrize("value", ["", " zh-CN", "zh CN", "zh.CN"])
def test_target_language_is_strictly_validated(value: str) -> None:
    with pytest.raises(AppError, match=r"llm\.target_language_invalid"):
        validate_target_language(value)


def test_publication_specs_accept_v3_and_reject_wrong_export_sets(tmp_path: Path) -> None:
    refs = tuple(
        ArtifactRef("0" * 64, 0, "test", "application/octet-stream", name)
        for name in (
            "final-transcript.json",
            "final-subtitle.json",
            "final-subtitle.srt",
            "final-subtitle.vtt",
            "final-subtitle.ass",
        )
    )
    specs = cast(
        tuple[tuple[str, str, ArtifactRef], ...],
        publication_specs(tmp_path / "input.wav", refs, publication_version="publish-v3"),
    )
    assert [target for _, target, _ in specs] == [
        "input.transcript.json",
        "input.subtitle.json",
        "input.srt",
        "input.vtt",
        "input.ass",
    ]
    with pytest.raises(AppError, match=r"output\.publication_invalid"):
        publication_specs(tmp_path / "input.wav", refs[:1], publication_version="publish-v3")
    with pytest.raises(AppError, match=r"output\.publication_invalid"):
        publication_specs(
            tmp_path / "input.wav",
            (refs[0], ArtifactRef("0" * 64, 0, "test", "application/octet-stream", "bad.txt")),
            publication_version="publish-v3",
        )
