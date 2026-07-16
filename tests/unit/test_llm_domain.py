from __future__ import annotations

import json
from pathlib import Path

import pytest

from captioner.core.domain.errors import AppError
from captioner.core.domain.llm import (
    FastTranslationResponse,
    LLMItem,
    LLMRequest,
    QualityTranslationResponse,
    ReviewResponse,
    SourceCorrectionResponse,
    response_schema_for,
)
from captioner.core.domain.transcript import CorrectedSpan, CorrectedTranscript
from captioner.core.policies.llm_chunking import ChunkingConfig, ChunkItem, ChunkPlanner
from captioner.core.policies.llm_validation import (
    is_obvious_wrong_language,
    response_schema_has_no_timestamps,
    script_heuristic,
    validate_response,
    validate_response_schema,
    validate_responses,
)
from captioner.infrastructure.prompts import PromptLoader


def test_response_parsing_and_serialization_paths() -> None:
    responses = (
        SourceCorrectionResponse("unit", "corrected"),
        FastTranslationResponse("unit", "corrected", "translated"),
        QualityTranslationResponse("unit", "translated"),
        ReviewResponse("unit", "translated"),
    )
    for response in responses:
        assert type(response).from_mapping(response.to_dict()) == response
        assert (
            type(response).from_json(json.dumps(response.to_dict(), ensure_ascii=False)) == response
        )
        assert type(response).model_json_schema() == type(response).schema()

    with pytest.raises(AppError, match=r"llm\.response_invalid"):
        SourceCorrectionResponse.from_mapping(None)
    with pytest.raises(AppError, match=r"llm\.response_invalid"):
        SourceCorrectionResponse.from_mapping({"id": "unit"})
    with pytest.raises(AppError, match=r"llm\.response_invalid"):
        SourceCorrectionResponse.from_mapping({"id": "unit", "corrected_source": 1})
    with pytest.raises(AppError, match=r"llm\.response_invalid"):
        SourceCorrectionResponse.from_json("not-json")
    with pytest.raises(AppError, match=r"llm\.response_invalid"):
        SourceCorrectionResponse.from_json('{"id":"unit","corrected_source":NaN}')
    with pytest.raises(AppError, match=r"llm\.response_invalid"):
        SourceCorrectionResponse("unit", "not canonical ")


def test_request_validation_and_response_schema_guard() -> None:
    with pytest.raises(AppError, match=r"llm\.request_invalid"):
        LLMRequest("translate", ())
    with pytest.raises(AppError, match=r"llm\.request_invalid"):
        LLMRequest("translate", (LLMItem("unit", "text"), LLMItem("unit", "text")))
    with pytest.raises(AppError, match=r"llm\.request_invalid"):
        LLMRequest("translate", (LLMItem("unit", "text"),), (LLMItem("unit", "text"),))
    with pytest.raises(AppError, match=r"llm\.response_invalid"):
        LLMRequest("translate", (LLMItem("unit", "text"),), source_language=" ")
    request = LLMRequest(
        "translate",
        (LLMItem("unit", "text"),),
        prompt_id="prompt",
        prompt_version="v1",
        prompt_content_sha256="hash",
    )
    assert request.item_ids == ("unit",)
    assert request.context_ids == ()
    assert response_schema_has_no_timestamps(SourceCorrectionResponse)
    with pytest.raises(AppError, match=r"llm\.schema_invalid"):
        response_schema_for(object)
    with pytest.raises(AppError, match=r"llm\.schema_invalid"):
        validate_response_schema({"id": "unit"}, object)


def test_corrected_transcript_is_ordered_and_complete() -> None:
    correction = CorrectedTranscript(
        "transcript-1",
        (
            CorrectedSpan(("word-1",), "one"),
            CorrectedSpan(("word-2",), "two"),
        ),
        ("word-1", "word-2"),
    )
    assert correction.corrected_text_by_word_id == {"word-1": "one", "word-2": "two"}
    assert correction.corrections == correction.spans
    assert correction.word_ids == ("word-1", "word-2")
    assert CorrectedTranscript.from_mapping(correction.to_dict()) == correction
    with pytest.raises(AppError, match=r"transcript\.correction_invalid"):
        CorrectedSpan((), "text")
    with pytest.raises(AppError, match=r"transcript\.correction_invalid"):
        CorrectedSpan(("word",), "not canonical ")
    with pytest.raises(AppError, match=r"transcript\.correction_invalid"):
        CorrectedTranscript("transcript-1", (CorrectedSpan(("word-1",), "one"),), ("word-2",))
    with pytest.raises(AppError, match=r"transcript\.correction_invalid"):
        CorrectedTranscript.from_mapping({})
    with pytest.raises(AppError, match=r"transcript\.correction_invalid"):
        CorrectedTranscript.from_mapping(
            {
                **correction.to_dict(),
                "spans": [{"source_word_ids": "word-1", "corrected_text": "one"}],
            }
        )


def test_chunk_validation_and_counter_failures() -> None:
    class BadCounter:
        def count(self, text: str) -> int:
            del text
            return -1

    with pytest.raises(AppError, match=r"llm\.chunk_config_invalid"):
        ChunkingConfig(max_items=0)
    with pytest.raises(AppError, match=r"llm\.chunk_config_invalid"):
        ChunkingConfig(context_before_items=-1)
    with pytest.raises(AppError, match=r"llm\.chunk_config_invalid"):
        ChunkingConfig(max_audio_context_duration_ms=0)
    with pytest.raises(AppError, match=r"llm\.chunk_item_invalid"):
        ChunkItem("unit", "not canonical ")
    with pytest.raises(AppError, match=r"llm\.chunk_item_invalid"):
        ChunkItem("unit", "text", 2, 1)
    with pytest.raises(AppError, match=r"llm\.token_count_invalid"):
        ChunkPlanner(BadCounter()).plan((ChunkItem("unit", "text"),))
    with pytest.raises(AppError, match=r"llm\.chunk_items_invalid"):
        ChunkPlanner(BadCounter()).plan((ChunkItem("unit", "text"), ChunkItem("unit", "text")))


def test_validation_language_and_id_failure_paths(tmp_path: Path) -> None:
    response = FastTranslationResponse("unit", "source", "translated")
    with pytest.raises(AppError, match=r"llm\.extra_id"):
        validate_responses((response,), ())
    with pytest.raises(AppError, match=r"llm\.response_invalid"):
        validate_responses(({"id": "unit", "translated_text": "ok", "other": "x"},), ("unit",))
    with pytest.raises(AppError, match=r"llm\.empty_text"):
        validate_response({"id": "unit", "translated_text": ""}, ("unit",))
    assert script_heuristic("مرحبا") == "arabic"
    assert script_heuristic("Привет") == "cyrillic"
    assert script_heuristic("こんにちは") == "kana"
    assert script_heuristic("안녕") == "hangul"
    assert not is_obvious_wrong_language("hello", "de")
    assert is_obvious_wrong_language("مرحبا", "en")
    assert not is_obvious_wrong_language("123", "zh-CN")
    loader = PromptLoader(tmp_path)
    with pytest.raises(AppError, match=r"prompt\.not_found"):
        loader.load("missing", "v1")
    (tmp_path / "empty.v1.md").write_text(" ", encoding="utf-8")
    with pytest.raises(AppError, match=r"prompt\.invalid"):
        loader.load("empty", "v1")
