from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from captioner.adapters.llm.scripted import ScriptedLLMAdapter
from captioner.adapters.persistence.filesystem_llm_cache import FilesystemLLMCache
from captioner.core.application.llm_chunk_executor import (
    LLMChunkExecutionConfig,
    LLMChunkExecutor,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.llm import (
    FastTranslationResponse,
    LLMItem,
    LLMRequest,
    StructuredResponseBatch,
    encode_llm_request,
    response_batch_schema,
)
from captioner.core.policies.llm_chunking import ChunkingConfig, ChunkItem, ChunkPlanner
from captioner.infrastructure.prompts import PromptLoader


@dataclass(frozen=True, slots=True)
class FakeCounter:
    def count(self, text: str) -> int:
        return len(text.split())


@dataclass(frozen=True, slots=True)
class ByteCounter:
    def count(self, text: str) -> int:
        return len(text.encode("utf-8"))


def _budget_request(
    prompt_id: str,
    prompt_version: str,
    prompt_hash: str,
    prompt_content: str,
    repair_id: str,
    repair_version: str,
    repair_hash: str,
    repair_content: str,
    items: tuple[LLMItem, ...],
    context: tuple[LLMItem, ...] = (),
) -> LLMRequest:
    return LLMRequest(
        "translate_fast",
        items,
        context,
        source_language="en",
        target_language="de",
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        prompt_content_sha256=prompt_hash,
        prompt_content=prompt_content,
        repair_prompt_id=repair_id,
        repair_prompt_version=repair_version,
        repair_prompt_content_sha256=repair_hash,
        repair_prompt_content=repair_content,
    )


def _budget_config(max_input_tokens: int) -> LLMChunkExecutionConfig:
    loader = PromptLoader(Path("resources/prompts"))
    prompt = loader.load("translate_fast", "v1")
    repair = loader.load("repair_structured", "v1")
    return LLMChunkExecutionConfig(
        task_kind="translate_fast",
        provider_kind="openai-compatible",
        provider_identity="default",
        base_url_identity="https://provider.example/v1",
        model="unit-model",
        temperature=0.1,
        source_language="en",
        target_language="de",
        profile="fast",
        prompt_id=prompt.prompt_id,
        prompt_version=prompt.prompt_version,
        prompt_content_sha256=prompt.content_sha256,
        prompt_content=prompt.content,
        chunking=ChunkingConfig(
            max_items=4,
            max_input_tokens=max_input_tokens,
            context_before_items=1,
            context_after_items=1,
        ),
        repair_prompt_id=repair.prompt_id,
        repair_prompt_version=repair.prompt_version,
        repair_prompt_content_sha256=repair.content_sha256,
        repair_prompt_content=repair.content,
    )


def _translation_response(
    request: object,
    response_schema: type[object],
    context: ExecutionContext,
) -> object:
    del context
    typed_request = cast(LLMRequest, request)
    parser = cast(type[StructuredResponseBatch], response_schema).from_mapping
    return parser(
        [
            {
                "id": item.id,
                "corrected_source": item.source,
                "translated_text": item.source,
            }
            for item in typed_request.items
        ]
    )


@given(st.lists(st.integers(min_value=1, max_value=4), min_size=1, max_size=30))
def test_chunk_planner_is_forward_only_and_repeatable(widths: list[int]) -> None:
    items = tuple(
        ChunkItem(f"unit-{index}", " ".join("x" for _ in range(width)))
        for index, width in enumerate(widths)
    )
    config = ChunkingConfig(
        max_items=4,
        max_input_tokens=12,
        context_before_items=2,
        context_after_items=2,
    )
    planner = ChunkPlanner(FakeCounter(), config)
    first = planner.plan(items)
    second = planner.plan(items)
    assert first == second
    assert tuple(item_id for chunk in first for item_id in chunk.item_ids) == tuple(
        item.id for item in items
    )
    for chunk in first:
        assert not set(chunk.item_ids) & set(chunk.context_ids)
        assert len(chunk.items) <= config.max_items
        assert (
            sum(FakeCounter().count(item.text) for item in chunk.items) <= config.max_input_tokens
        )


def test_context_budget_and_audio_budget_are_both_enforced() -> None:
    items = tuple(
        ChunkItem(f"unit-{index}", "word", index * 1000, index * 1000 + 100) for index in range(5)
    )
    planner = ChunkPlanner(
        FakeCounter(),
        ChunkingConfig(
            max_items=2,
            max_input_tokens=4,
            context_before_items=2,
            context_after_items=2,
            max_audio_context_duration_ms=2200,
        ),
    )
    chunks = planner.plan(items)
    assert tuple(item.id for chunk in chunks for item in chunk.items) == tuple(
        item.id for item in items
    )
    for chunk in chunks:
        window = (*chunk.context, *chunk.items)
        assert max(item.end_ms for item in window) - min(item.start_ms for item in window) <= 2200


def test_single_item_over_budget_has_structured_error() -> None:
    from pytest import raises

    from captioner.core.domain.errors import AppError
    from captioner.core.policies.llm_chunking import plan_chunks

    with raises(AppError, match=r"llm\.item_too_large"):
        plan_chunks(
            (ChunkItem("too-large", "one two three"),),
            ChunkingConfig(max_input_tokens=2),
            FakeCounter(),
        )


def test_complete_encoded_request_fits_budget_and_prompt_is_sent_once(tmp_path: Path) -> None:
    loader = PromptLoader(Path("resources/prompts"))
    prompt = loader.load("translate_fast", "v1")
    repair = loader.load("repair_structured", "v1")
    item = LLMItem("item-0", "source 0")
    request = _budget_request(
        prompt.prompt_id,
        prompt.prompt_version,
        prompt.content_sha256,
        prompt.content,
        repair.prompt_id,
        repair.prompt_version,
        repair.content_sha256,
        repair.content,
        (item,),
    )
    batch_schema = response_batch_schema(FastTranslationResponse)
    encoded = encode_llm_request(request, "unit-model", 0.1, batch_schema)
    body = json.loads(encoded)
    assert body["messages"][0]["content"] == prompt.content
    assert "prompt_content" not in request.to_dict()
    assert prompt.content not in body["messages"][1]["content"]

    adapter = ScriptedLLMAdapter(structured_responses=(_translation_response,) * 3)
    config = _budget_config(len(encoded))
    executor = LLMChunkExecutor(
        adapter,
        FilesystemLLMCache(tmp_path),
        ChunkPlanner(ByteCounter(), config.chunking),
        config,
    )
    result = asyncio.run(
        executor.execute(
            tuple(ChunkItem(f"item-{index}", f"source {index}") for index in range(3)),
            FastTranslationResponse,
        )
    )
    assert [response.id for response in result] == ["item-0", "item-1", "item-2"]
    assert all(
        len(
            encode_llm_request(
                call,
                config.model,
                config.temperature,
                response_batch_schema(FastTranslationResponse),
            )
        )
        <= config.chunking.max_input_tokens
        for call in adapter.structured_calls
    )
    assert all(call.context_ids == () for call in adapter.structured_calls)


def test_single_item_complete_request_over_budget_fails_before_adapter(tmp_path: Path) -> None:
    loader = PromptLoader(Path("resources/prompts"))
    prompt = loader.load("translate_fast", "v1")
    repair = loader.load("repair_structured", "v1")
    request = _budget_request(
        prompt.prompt_id,
        prompt.prompt_version,
        prompt.content_sha256,
        prompt.content,
        repair.prompt_id,
        repair.prompt_version,
        repair.content_sha256,
        repair.content,
        (LLMItem("item-0", "source"),),
    )
    batch_schema = response_batch_schema(FastTranslationResponse)
    budget = len(encode_llm_request(request, "unit-model", 0.1, batch_schema)) - 1
    adapter = ScriptedLLMAdapter(structured_responses=(_translation_response,))
    config = _budget_config(budget)
    executor = LLMChunkExecutor(
        adapter,
        FilesystemLLMCache(tmp_path),
        ChunkPlanner(ByteCounter(), config.chunking),
        config,
    )
    with pytest.raises(AppError, match=r"llm\.item_too_large"):
        asyncio.run(executor.execute((ChunkItem("item-0", "source"),), FastTranslationResponse))
    assert adapter.structured_calls == []
