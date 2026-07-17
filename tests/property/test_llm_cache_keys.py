from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from captioner.core.domain.llm import FastTranslationResponse, LLMItem, LLMRequest
from captioner.core.domain.llm_cache import (
    LLMCacheKey,
    build_llm_cache_key,
    build_llm_cache_key_for_request,
)
from captioner.core.domain.result import JsonValue
from captioner.infrastructure.prompts import PromptLoader


def _key(
    *,
    model: str = "model-a",
    target_language: str | None = "de",
    prompt_version: str = "v1",
    api_marker: str = "ignored",
) -> str:
    del api_marker
    return build_llm_cache_key(
        task_kind="translate_fast",
        provider_kind="openai-compatible",
        provider_identity="default",
        base_url_identity="https://provider.example/v1",
        model=model,
        temperature=0.1,
        source_language="en",
        target_language=target_language,
        profile="fast",
        prompt_id="translate_fast",
        prompt_version=prompt_version,
        prompt_content_sha256="a" * 64,
        items=(LLMItem("item-1", "1,000"), LLMItem("item-2", "20%")),
        context=(LLMItem("context-1", "nearby"),),
        chunk_config={
            "max_items": 2,
            "max_input_tokens": 100,
            "context_before_items": 1,
            "context_after_items": 1,
            "max_audio_context_duration_ms": 5000,
        },
    ).digest


@given(st.sampled_from(["model-a", "model-b", "model-c"]))
def test_same_descriptor_is_deterministic(model: str) -> None:
    assert _key(model=model) == _key(model=model)


def test_result_configuration_changes_key_and_api_key_does_not_exist_in_payload() -> None:
    base = _key()
    assert base != _key(model="model-b")
    assert base != _key(target_language="fr")
    assert base != _key(prompt_version="v2")
    assert "unit-test-key" not in _key(api_marker="unit-test-key")


def test_cache_key_is_derived_from_final_request_semantics() -> None:
    loader = PromptLoader(Path("resources/prompts"))
    prompt = loader.load("translate_fast", "v1")

    def request(context_payload: Mapping[str, JsonValue]) -> LLMRequest:
        return LLMRequest(
            "translate_fast",
            (LLMItem("item-1", "source"),),
            (LLMItem("context-1", "nearby"),),
            source_language="en",
            target_language="de",
            prompt_id=prompt.prompt_id,
            prompt_version=prompt.prompt_version,
            prompt_content_sha256=prompt.content_sha256,
            prompt_content=prompt.content,
            context_payload=context_payload,
        )

    def key(value: LLMRequest) -> LLMCacheKey:
        return build_llm_cache_key_for_request(
            value,
            provider_kind="openai-compatible",
            provider_identity="default",
            base_url_identity="https://provider.example/v1",
            model="unit-model",
            temperature=0.1,
            profile="fast",
            chunk_config={"max_items": 1, "max_input_tokens": 100},
            response_schema_version=1,
            response_schema=FastTranslationResponse,
        )

    first = key(request({"terminology": [{"source_term": "one", "target_term": "eins"}]}))
    changed_payload = key(request({"terminology": [{"source_term": "two", "target_term": "zwei"}]}))
    changed_context = key(
        LLMRequest(
            "translate_fast",
            (LLMItem("item-1", "source"),),
            (LLMItem("context-2", "nearby"),),
            source_language="en",
            target_language="de",
            prompt_id=prompt.prompt_id,
            prompt_version=prompt.prompt_version,
            prompt_content_sha256=prompt.content_sha256,
            prompt_content=prompt.content,
            context_payload={"terminology": [{"source_term": "one", "target_term": "eins"}]},
        )
    )
    assert first.digest != changed_payload.digest
    assert first.digest != changed_context.digest
    assert "api_key" not in repr(first.payload)
