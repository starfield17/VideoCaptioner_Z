from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import pytest

from captioner.adapters.llm.fake import ScriptedCancellation, ScriptedJSON
from captioner.adapters.llm.scripted import ScriptedLLMAdapter
from captioner.adapters.persistence.filesystem_llm_cache import FilesystemLLMCache
from captioner.core.application.llm_chunk_executor import (
    LLMChunkExecutionConfig,
    LLMChunkExecutor,
)
from captioner.core.application.structured_llm_service import StructuredLLMService
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.llm import (
    LLMRequest,
    SourceCorrectionResponse,
    StructuredResponseBatch,
)
from captioner.core.policies.llm_chunking import ChunkingConfig, ChunkItem, ChunkPlanner


class FakeCounter:
    def count(self, text: str) -> int:
        return len(text.split())


def _items(count: int = 4) -> tuple[ChunkItem, ...]:
    return tuple(ChunkItem(f"item-{index}", f"source {index}") for index in range(count))


def _config(*, max_items: int = 2, context_after_items: int = 0) -> LLMChunkExecutionConfig:
    return LLMChunkExecutionConfig(
        task_kind="correct_source",
        provider_kind="openai-compatible",
        provider_identity="default",
        base_url_identity="https://provider.example/v1",
        model="unit-model",
        temperature=0.1,
        source_language="en",
        target_language=None,
        profile="quality",
        prompt_id="correct_source",
        prompt_version="v1",
        prompt_content_sha256="content-hash",
        chunking=ChunkingConfig(
            max_items=max_items,
            max_input_tokens=100,
            context_before_items=1,
            context_after_items=context_after_items,
        ),
    )


def _payload(items: tuple[ChunkItem, ...]) -> ScriptedJSON:
    return ScriptedJSON(
        json.dumps(
            [
                {"id": item.id, "corrected_source": item.text.replace("source", "corrected")}
                for item in items
            ]
        )
    )


def _dynamic_success(
    request: LLMRequest,
    response_schema: type[object],
    context: ExecutionContext,
) -> object:
    del context
    parser = cast(type[StructuredResponseBatch], response_schema).from_mapping
    return parser(
        [
            {"id": item.id, "corrected_source": item.source.replace("source", "corrected")}
            for item in request.items
        ]
    )


def _executor(
    cache: FilesystemLLMCache,
    adapter: ScriptedLLMAdapter,
    config: LLMChunkExecutionConfig,
) -> LLMChunkExecutor:
    return LLMChunkExecutor(adapter, cache, ChunkPlanner(FakeCounter(), config.chunking), config)


def test_resume_requests_only_uncached_chunks(tmp_path: Path) -> None:
    cache = FilesystemLLMCache(tmp_path)
    items = _items()
    first_adapter = ScriptedLLMAdapter(
        structured_responses=[_payload(items[:2]), _payload(items[2:])]
    )
    result = asyncio.run(
        _executor(cache, first_adapter, _config()).execute(items, SourceCorrectionResponse)
    )
    assert [response.id for response in result] == [item.id for item in items]
    assert len(first_adapter.structured_calls) == 2

    resumed_adapter = ScriptedLLMAdapter()
    resumed = asyncio.run(
        _executor(cache, resumed_adapter, _config()).execute(items, SourceCorrectionResponse)
    )
    assert [response.id for response in resumed] == [item.id for item in items]
    assert resumed_adapter.structured_calls == []

    cached_paths = list(cache.root.rglob("*.json"))
    assert len(cached_paths) == 2
    cached_paths[0].write_text("not json", encoding="utf-8")
    miss_adapter = ScriptedLLMAdapter(structured_responses=[_dynamic_success])
    asyncio.run(_executor(cache, miss_adapter, _config()).execute(items, SourceCorrectionResponse))
    assert len(miss_adapter.structured_calls) == 1


def test_id_mismatch_shrinks_current_chunk_deterministically(tmp_path: Path) -> None:
    cache = FilesystemLLMCache(tmp_path)
    items = _items()
    adapter = ScriptedLLMAdapter(
        structured_responses=[
            AppError("llm.missing_id"),
            _payload(items[:2]),
            _payload(items[2:]),
        ]
    )
    result = asyncio.run(
        _executor(cache, adapter, _config(max_items=4, context_after_items=1)).execute(
            items, SourceCorrectionResponse
        )
    )
    assert [response.id for response in result] == [item.id for item in items]
    assert [tuple(call.item_ids) for call in adapter.structured_calls] == [
        tuple(item.id for item in items),
        tuple(item.id for item in items[:2]),
        tuple(item.id for item in items[2:]),
    ]
    assert adapter.structured_calls[1].context_ids == ("item-2",)
    assert adapter.structured_calls[2].context_ids == ("item-1",)


def test_retryable_error_retries_only_current_chunk(tmp_path: Path) -> None:
    cache = FilesystemLLMCache(tmp_path)
    items = _items(2)
    adapter = ScriptedLLMAdapter(
        structured_responses=[AppError("llm.rate_limited", retryable=True), _payload(items)]
    )
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    service = StructuredLLMService(adapter, sleep=sleep)
    executor = LLMChunkExecutor(service, cache, ChunkPlanner(FakeCounter()), _config())
    asyncio.run(executor.execute(items, SourceCorrectionResponse))
    assert delays == [1.0]
    assert len(adapter.structured_calls) == 2


def test_cancellation_leaves_no_partial_cache(tmp_path: Path) -> None:
    cache = FilesystemLLMCache(tmp_path)
    adapter = ScriptedLLMAdapter(structured_responses=[ScriptedCancellation()])
    with pytest.raises(AppError, match=r"operation\.cancelled"):
        asyncio.run(
            _executor(cache, adapter, _config()).execute(
                _items(2), SourceCorrectionResponse, ExecutionContext()
            )
        )
    assert not cache.root.exists() or not list(cache.root.rglob("*.tmp"))


def test_validation_failure_gets_one_repair_before_cache_write(tmp_path: Path) -> None:
    cache = FilesystemLLMCache(tmp_path)
    item = ChunkItem("item-0", "source 10")
    invalid = ScriptedJSON('[{"id":"item-0","corrected_source":"corrected"}]')
    repaired = ScriptedJSON('[{"id":"item-0","corrected_source":"corrected 10"}]')
    adapter = ScriptedLLMAdapter(structured_responses=(invalid, repaired))

    result = asyncio.run(
        _executor(cache, adapter, _config(max_items=1)).execute(
            (item,),
            SourceCorrectionResponse,
        )
    )

    assert result[0].corrected_source == "corrected 10"
    assert [call.task_kind for call in adapter.structured_calls] == [
        "correct_source",
        "repair_structured",
    ]
    assert len(list(cache.root.rglob("*.json"))) == 1


def test_failed_validation_repair_is_not_cached(tmp_path: Path) -> None:
    cache = FilesystemLLMCache(tmp_path)
    item = ChunkItem("item-0", "source 10")
    invalid = ScriptedJSON('[{"id":"item-0","corrected_source":"corrected"}]')
    adapter = ScriptedLLMAdapter(structured_responses=(invalid, invalid))

    with pytest.raises(AppError, match=r"llm\.protected_token_lost"):
        asyncio.run(
            _executor(cache, adapter, _config(max_items=1)).execute(
                (item,),
                SourceCorrectionResponse,
            )
        )

    assert [call.task_kind for call in adapter.structured_calls] == [
        "correct_source",
        "repair_structured",
    ]
    assert not list(cache.root.rglob("*.json"))
