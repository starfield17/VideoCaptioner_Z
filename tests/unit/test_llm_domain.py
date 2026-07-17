from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from captioner.core.domain.errors import AppError
from captioner.core.domain.llm import (
    FastTranslationResponse,
    LLMItem,
    LLMRequest,
    QualityTranslationResponse,
    ReviewResponse,
    SourceCorrectionResponse,
    TerminologyResponse,
    encode_llm_request,
    response_schema_for,
)
from captioner.core.domain.result import JsonValue
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
        TerminologyResponse("unit", ({"source_term": "source", "target_term": "target"},)),
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
    )
    assert request.item_ids == ("unit",)
    assert request.context_ids == ()
    assert response_schema_has_no_timestamps(SourceCorrectionResponse)
    with pytest.raises(AppError, match=r"llm\.schema_invalid"):
        response_schema_for(object)
    with pytest.raises(AppError, match=r"llm\.schema_invalid"):
        validate_response_schema({"id": "unit"}, object)


def test_request_serialization_sends_prompt_once_and_freezes_dynamic_context() -> None:
    prompt = "Use the supplied source text."
    request = LLMRequest(
        "translate_quality",
        (LLMItem("unit", "source"),),
        context=(LLMItem("nearby", "nearby source"),),
        source_language="en",
        target_language="de",
        prompt_id="translate_quality",
        prompt_version="v1",
        prompt_content_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        prompt_content=prompt,
        context_payload={"terminology": [{"source_term": "source", "target_term": "Quelle"}]},
    )

    serialized = json.loads(
        encode_llm_request(request, "unit-model", 0.1, QualityTranslationResponse)
    )
    messages = serialized["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == prompt
    user_payload = json.loads(messages[1]["content"])
    assert "prompt_content" not in user_payload
    assert user_payload["prompt_content_sha256"] == request.prompt_content_sha256
    assert user_payload["context_payload"]["terminology"][0]["target_term"] == "Quelle"

    with pytest.raises(AppError, match=r"llm\.request_invalid"):
        LLMRequest(
            "translate_quality",
            (LLMItem("unit", "source"),),
            context_payload={"start_ms": 10},
        )
    with pytest.raises(AppError, match=r"llm\.request_invalid"):
        LLMRequest(
            "translate_quality",
            (LLMItem("unit", "source"),),
            context_payload={"authorization": "unit-test-key"},
        )
    payloads = cast(
        tuple[object, ...],
        (
            {"unknown": []},
            {"terminology": [{"source_term": "source"}]},
            {"anomalies": [{"cue_id": "cue-1", "reasons": []}]},
            {"nearby_cues": [{"cue_id": "cue-1", "source_text": "source"}]},
        ),
    )
    for payload in payloads:
        typed_payload = cast(dict[str, JsonValue], payload)
        with pytest.raises(AppError, match=r"llm\.request_invalid"):
            LLMRequest(
                "translate_quality",
                (LLMItem("unit", "source"),),
                context_payload=typed_payload,
            )
    dynamic_context = cast(
        dict[str, JsonValue],
        {
            "terminology": [{"source_term": "source", "target_term": "Quelle"}],
            "anomalies": [{"cue_id": "cue-1", "reasons": ["wrong_language"]}],
            "nearby_cues": [
                {
                    "cue_id": "cue-2",
                    "source_text": "nearby",
                    "translated_text": "nahe",
                }
            ],
        },
    )
    dynamic_request = LLMRequest(
        "review",
        (LLMItem("cue-1", "source"),),
        context_payload=dynamic_context,
    )
    assert dynamic_request.to_dict()["context_payload"] == dynamic_context
    invalid_dynamic_context = (
        {"terminology": ["bad"]},
        {"terminology": [{"source_term": 1, "target_term": "Quelle"}]},
        {"anomalies": "bad"},
        {"anomalies": [{"cue_id": "cue-1", "reasons": "bad"}]},
        {"anomalies": [{"cue_id": "cue-1", "reasons": [1]}]},
        {"anomalies": [{"cue_id": "cue-1", "reasons": ["bad "]}]},
        {"nearby_cues": [{"cue_id": "cue-1", "source_text": "source", "translated_text": None}]},
    )
    for payload in cast(tuple[object, ...], invalid_dynamic_context):
        with pytest.raises(AppError, match=r"llm\.request_invalid"):
            LLMRequest(
                "review",
                (LLMItem("cue-1", "source"),),
                context_payload=cast(dict[str, JsonValue], payload),
            )


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


@pytest.mark.parametrize(
    ("prompt_id", "prompt_version"),
    [
        ("../prompt", "v1"),
        ("/absolute", "v1"),
        ("C:\\prompt", "v1"),
        ("prompt/name", "v1"),
        ("prompt", "../v1"),
        ("prompt", "v1/extra"),
    ],
)
def test_prompt_loader_rejects_path_like_identity_components(
    tmp_path: Path,
    prompt_id: str,
    prompt_version: str,
) -> None:
    with pytest.raises(AppError, match=r"prompt\.(invalid|identity_invalid)"):
        PromptLoader(tmp_path).load(prompt_id, prompt_version)


def test_validation_handles_sparse_terms_and_all_script_boundaries() -> None:
    empty_terms = TerminologyResponse("unit", ())
    assert validate_response(empty_terms, ("unit",)) == empty_terms
    validated_mapping = validate_response(
        {"id": "unit", "terms": [{"source_term": "10", "target_term": "10"}]},
        ("unit",),
        source_texts={"unit": "10"},
    )
    assert cast(dict[str, object], validated_mapping)["id"] == "unit"
    with pytest.raises(AppError, match=r"llm\.response_invalid"):
        validate_responses(({"id": "unit", "terms": "bad"},), ("unit",))
    with pytest.raises(AppError, match=r"llm\.response_invalid"):
        validate_responses(({"id": "unit", "terms": [{"source_term": 1}]},), ("unit",))
    with pytest.raises(AppError, match=r"llm\.response_invalid"):
        validate_responses((object(),), ("unit",))

    assert script_heuristic("") == "other"
    assert script_heuristic("hello 世界") == "mixed"
    assert is_obvious_wrong_language("こんにちは", "ja-JP") is False
    assert is_obvious_wrong_language("hello", "ja-JP") is True
    assert is_obvious_wrong_language("안녕", "ko-KR") is False
    assert is_obvious_wrong_language("hello", "ko-KR") is True
    assert is_obvious_wrong_language("مرحبا", "ar-SA") is False
    assert is_obvious_wrong_language("hello", "ar-SA") is True
    assert is_obvious_wrong_language("Привет", "ru-RU") is False
    assert is_obvious_wrong_language("hello", "ru-RU") is True
    assert is_obvious_wrong_language("hello", "xx-XX") is False


def test_provider_schema_name_is_stable() -> None:
    from captioner.core.domain.llm import LLMTaskKind, provider_response_schema_name

    expected = {
        LLMTaskKind.CORRECT_SOURCE.value: "captioner_correct_source_batch_v1",
        LLMTaskKind.TRANSLATE_FAST.value: "captioner_translate_fast_batch_v1",
        LLMTaskKind.TRANSLATE_QUALITY.value: "captioner_translate_quality_batch_v1",
        LLMTaskKind.REVIEW.value: "captioner_review_batch_v1",
        LLMTaskKind.TERMINOLOGY.value: "captioner_terminology_batch_v1",
        LLMTaskKind.REPAIR_STRUCTURED.value: "captioner_repair_structured_batch_v1",
    }
    for task_kind, name in expected.items():
        assert provider_response_schema_name(task_kind) == name
        assert provider_response_schema_name(task_kind) == provider_response_schema_name(task_kind)


def test_provider_schema_name_matches_allowed_pattern() -> None:
    import re

    from captioner.core.domain.llm import provider_response_schema_name

    for task_kind in (
        "correct_source",
        "translate_fast",
        "translate_quality",
        "review",
        "terminology",
        "repair_structured",
    ):
        name = provider_response_schema_name(task_kind)
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
        assert "<locals>" not in name
        assert "." not in name
        assert " " not in name


def test_provider_schema_name_is_at_most_64_characters() -> None:
    from captioner.core.domain.llm import provider_response_schema_name

    for task_kind in (
        "correct_source",
        "translate_fast",
        "translate_quality",
        "review",
        "terminology",
        "repair_structured",
    ):
        assert len(provider_response_schema_name(task_kind)) <= 64


def test_dynamic_batch_class_qualname_never_reaches_wire() -> None:
    import re

    from captioner.core.domain.llm import response_batch_schema

    prompt = "p"
    request = LLMRequest(
        "translate_fast",
        (LLMItem("unit", "source"),),
        prompt_id="translate_fast",
        prompt_version="v1",
        prompt_content_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        prompt_content=prompt,
    )
    batch = response_batch_schema(FastTranslationResponse)
    assert "<locals>" in batch.__qualname__
    body = encode_llm_request(request, "model", 0.1, batch)
    payload = json.loads(body)
    schema_name = payload["response_format"]["json_schema"]["name"]
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", schema_name)
    assert "<locals>" not in schema_name
    assert schema_name == "captioner_translate_fast_batch_v1"


def test_cache_and_wire_use_same_schema_identity() -> None:
    from captioner.core.domain.llm import provider_response_schema_name, response_batch_schema
    from captioner.core.domain.llm_cache import build_llm_cache_key_for_request

    prompt = "p"
    request = LLMRequest(
        "correct_source",
        (LLMItem("unit", "source"),),
        prompt_id="correct_source",
        prompt_version="v1",
        prompt_content_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        prompt_content=prompt,
    )
    batch = response_batch_schema(SourceCorrectionResponse)
    wire_name = json.loads(encode_llm_request(request, "m", 0.1, batch))["response_format"][
        "json_schema"
    ]["name"]
    key = build_llm_cache_key_for_request(
        request,
        provider_kind="openai-compatible",
        provider_identity="default",
        base_url_identity="https://example/v1",
        model="m",
        temperature=0.1,
        profile="fast",
        chunk_config={"max_items": 1},
        response_schema_version=1,
        response_schema=batch,
    )
    assert wire_name == provider_response_schema_name("correct_source")
    schema_meta = key.payload["response_schema"]
    assert isinstance(schema_meta, Mapping)
    assert cast(Mapping[str, object], schema_meta)["name"] == wire_name


def test_repair_request_uses_valid_schema_name() -> None:
    import re

    from captioner.core.application.structured_llm_service import structured_repair_request
    from captioner.core.domain.llm import response_batch_schema

    prompt = "p"
    repair = "repair prompt longer than original"
    request = LLMRequest(
        "translate_quality",
        (LLMItem("unit", "source"),),
        prompt_id="translate_quality",
        prompt_version="v1",
        prompt_content_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        prompt_content=prompt,
        repair_prompt_id="repair_structured",
        repair_prompt_version="v1",
        repair_prompt_content_sha256=hashlib.sha256(repair.encode()).hexdigest(),
        repair_prompt_content=repair,
    )
    repaired = structured_repair_request(
        request,
        repair_prompt_id="repair_structured",
        repair_prompt_version="v1",
        repair_prompt_content_sha256=hashlib.sha256(repair.encode()).hexdigest(),
        repair_prompt_content=repair,
    )
    batch = response_batch_schema(QualityTranslationResponse)
    name = json.loads(encode_llm_request(repaired, "m", 0.1, batch))["response_format"][
        "json_schema"
    ]["name"]
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
    assert name == "captioner_repair_structured_batch_v1"
    with pytest.raises(AppError, match=r"llm\.schema_name_invalid"):
        from captioner.core.domain.llm import provider_response_schema_name

        provider_response_schema_name("not_a_real_task")
