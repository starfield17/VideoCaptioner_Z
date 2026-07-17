from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.llm import (
    FastTranslationResponse,
    LLMItem,
    LLMRequest,
    QualityTranslationResponse,
    ReviewResponse,
    SourceCorrectionResponse,
    TerminologyResponse,
    response_schema_for,
)
from captioner.core.ports.llm import LLMClient
from captioner.infrastructure.prompts import PromptLoader


def test_llm_client_port_is_provider_neutral() -> None:
    signature = inspect.signature(LLMClient.generate_structured)
    assert tuple(signature.parameters) == ("self", "request", "response_schema", "context")
    assert str(signature.return_annotation) == "~T"
    assert ExecutionContext.__module__.startswith("captioner.core.domain")


@pytest.mark.parametrize(
    "response_schema, expected_fields",
    [
        (SourceCorrectionResponse, {"id", "corrected_source"}),
        (FastTranslationResponse, {"id", "corrected_source", "translated_text"}),
        (QualityTranslationResponse, {"id", "translated_text"}),
        (ReviewResponse, {"id", "translated_text"}),
        (TerminologyResponse, {"id", "source_term", "target_term"}),
    ],
)
def test_structured_schemas_are_exact_and_have_no_timing_fields(
    response_schema: type[object], expected_fields: set[str]
) -> None:
    schema = response_schema_for(response_schema)
    assert set(cast(dict[str, object], schema["properties"])) == expected_fields
    assert schema["additionalProperties"] is False

    def assert_no_timing(value: object) -> None:
        if isinstance(value, dict):
            typed = cast(Mapping[str, object], value)
            assert not {key.lower() for key in typed} & {
                "start_ms",
                "end_ms",
                "timestamp",
                "duration",
                "duration_ms",
            }
            for nested in typed.values():
                assert_no_timing(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for nested in cast(Sequence[object], value):
                assert_no_timing(nested)

    assert_no_timing(schema)


def test_response_json_rejects_duplicate_and_unknown_fields() -> None:
    with pytest.raises(AppError, match=r"llm\.response_invalid"):
        SourceCorrectionResponse.from_json('{"id":"unit-1","id":"unit-2","corrected_source":"ok"}')
    with pytest.raises(AppError, match=r"llm\.response_invalid"):
        SourceCorrectionResponse.from_mapping(
            {"id": "unit-1", "corrected_source": "ok", "start_ms": 1}
        )


def test_request_contains_context_but_no_timing_or_word_mapping() -> None:
    request = LLMRequest(
        "translate_fast",
        (LLMItem("unit-1", "Hello"),),
        (LLMItem("unit-0", "Earlier"),),
    )
    encoded = json.dumps(request.to_dict(), ensure_ascii=False)
    assert "timestamp" not in encoded
    assert "start_ms" not in encoded
    assert "source_word_ids" not in encoded


def test_prompt_identity_is_versioned_and_hashed() -> None:
    prompt = PromptLoader(Path("resources/prompts")).load("correct_source", "v1")
    assert prompt.prompt_id == "correct_source"
    assert prompt.prompt_version == "v1"
    assert len(prompt.content_sha256) == 64
    assert prompt.content
