from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from captioner.core.domain.llm import LLMItem
from captioner.core.domain.llm_cache import build_llm_cache_key


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
        prompt_content_sha256="content-hash",
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
