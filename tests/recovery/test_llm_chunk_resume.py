from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
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
from captioner.core.domain.errors import AppError, LLMStructuredDecodeError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.llm import (
    LLMRequest,
    SourceCorrectionResponse,
    StructuredResponseBatch,
    encode_llm_request,
    response_batch_schema,
)
from captioner.core.domain.llm_cache import LLMCacheKey
from captioner.core.policies.llm_chunking import ChunkingConfig, ChunkItem, ChunkPlanner, LLMChunk
from captioner.infrastructure.prompts import PromptLoader


class FakeCounter:
    def count(self, text: str) -> int:
        return len(text.split())


def _items(count: int = 4) -> tuple[ChunkItem, ...]:
    return tuple(ChunkItem(f"item-{index}", f"source {index}") for index in range(count))


def _config(*, max_items: int = 2, context_after_items: int = 0) -> LLMChunkExecutionConfig:
    repair = PromptLoader(Path("resources/prompts")).load("repair_structured", "v2")
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
        prompt_content_sha256="a" * 64,
        chunking=ChunkingConfig(
            max_items=max_items,
            max_input_tokens=100,
            context_before_items=1,
            context_after_items=context_after_items,
        ),
        repair_prompt_id=repair.prompt_id,
        repair_prompt_version=repair.prompt_version,
        repair_prompt_content_sha256=repair.content_sha256,
        repair_prompt_content=repair.content,
    )


def _payload(items: tuple[ChunkItem, ...]) -> ScriptedJSON:
    return ScriptedJSON(
        json.dumps(
            {
                "responses": [
                    {
                        "id": item.id,
                        "corrected_source": item.text.replace("source", "corrected"),
                    }
                    for item in items
                ]
            }
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
        {
            "responses": [
                {"id": item.id, "corrected_source": item.source.replace("source", "corrected")}
                for item in request.items
            ]
        }
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


def test_multi_item_truncation_splits_once_per_chunk_node(tmp_path: Path) -> None:
    cache = FilesystemLLMCache(tmp_path)
    items = _items()
    adapter = ScriptedLLMAdapter(
        structured_responses=[
            AppError("llm.output_truncated"),
            _dynamic_success,
            _dynamic_success,
        ]
    )
    result = asyncio.run(
        _executor(cache, adapter, _config(max_items=4)).execute(items, SourceCorrectionResponse)
    )
    assert [response.id for response in result] == [item.id for item in items]
    assert [tuple(call.item_ids) for call in adapter.structured_calls] == [
        tuple(item.id for item in items),
        tuple(item.id for item in items[:2]),
        tuple(item.id for item in items[2:]),
    ]
    assert len(list(cache.root.rglob("*.json"))) == 2


def test_single_item_truncation_fails_without_repair_or_cache(tmp_path: Path) -> None:
    cache = FilesystemLLMCache(tmp_path)
    adapter = ScriptedLLMAdapter(structured_responses=[AppError("llm.output_truncated")])
    with pytest.raises(AppError, match=r"llm\.output_truncated"):
        asyncio.run(
            _executor(cache, adapter, _config(max_items=1)).execute(
                (ChunkItem("item-0", "source"),), SourceCorrectionResponse
            )
        )
    assert len(adapter.structured_calls) == 1
    assert not list(cache.root.rglob("*.json"))


def test_refusal_fails_without_repair_or_cache(tmp_path: Path) -> None:
    cache = FilesystemLLMCache(tmp_path)
    adapter = ScriptedLLMAdapter(structured_responses=[AppError("llm.refused")])
    with pytest.raises(AppError, match=r"llm\.refused"):
        asyncio.run(
            _executor(cache, adapter, _config(max_items=1)).execute(
                (ChunkItem("item-0", "source"),), SourceCorrectionResponse
            )
        )
    assert len(adapter.structured_calls) == 1
    assert not list(cache.root.rglob("*.json"))


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
    invalid = ScriptedJSON('{"responses":[{"id":"item-0","corrected_source":"corrected"}]}')
    repaired = ScriptedJSON('{"responses":[{"id":"item-0","corrected_source":"corrected 10"}]}')
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
        "correct_source",
    ]
    repair = adapter.structured_calls[1]
    assert repair.repair_context is not None
    assert repair.repair_context.original_task_kind == "correct_source"
    assert repair.repair_context.invalid_response == (
        '{"responses":[{"corrected_source":"corrected","id":"item-0"}]}'
    )
    assert repair.repair_context.diagnostics[0].code == "llm.protected_token_lost"
    assert repair.repair_context.diagnostics[0].item_id == "item-0"
    assert repair.repair_context.diagnostics[0].field == "corrected_source"
    wire = json.loads(
        encode_llm_request(
            repair, "unit-model", 0.1, response_batch_schema(SourceCorrectionResponse)
        )
    )
    assert [message["role"] for message in wire["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert wire["messages"][0]["content"] == (
        repair.prompt_content or "Return only the requested JSON object."
    )
    assert json.loads(wire["messages"][1]["content"]) == repair.original_wire_envelope()
    assert wire["messages"][2]["content"] == repair.repair_context.invalid_response
    assert "llm.protected_token_lost" in wire["messages"][3]["content"]
    assert wire["response_format"]["json_schema"]["name"] == ("captioner_correct_source_batch_v2")
    assert len(list(cache.root.rglob("*.json"))) == 1


def test_failed_validation_repair_is_not_cached(tmp_path: Path) -> None:
    cache = FilesystemLLMCache(tmp_path)
    item = ChunkItem("item-0", "source 10")
    invalid = ScriptedJSON('{"responses":[{"id":"item-0","corrected_source":"corrected"}]}')
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
        "correct_source",
    ]
    assert not list(cache.root.rglob("*.json"))


def test_semantically_invalid_cache_entry_is_removed_before_re_request(tmp_path: Path) -> None:
    cache = FilesystemLLMCache(tmp_path)
    item = ChunkItem("item-0", "source 10")
    config = _config(max_items=1)
    executor = _executor(cache, ScriptedLLMAdapter(), config)
    chunk = ChunkPlanner(FakeCounter(), config.chunking).plan((item,))[0]
    default_request = cast(
        Callable[[LLMChunk], LLMRequest], getattr(executor, "_default_" + "request")
    )
    request = default_request(chunk)
    batch_schema = response_batch_schema(SourceCorrectionResponse)
    invalid = batch_schema.from_mapping(
        {"responses": [{"id": "item-0", "corrected_source": "corrected"}]}
    )
    cache_key = cast(
        Callable[[LLMRequest, type[StructuredResponseBatch]], LLMCacheKey],
        getattr(executor, "_cache_" + "key"),
    )
    key = cache_key(request, batch_schema)
    cache.put(key, invalid, batch_schema)
    adapter = ScriptedLLMAdapter(structured_responses=[AppError("llm.request_rejected")])
    with pytest.raises(AppError, match=r"llm\.request_rejected"):
        asyncio.run(_executor(cache, adapter, config).execute((item,), SourceCorrectionResponse))
    assert adapter.structured_calls
    assert not cache.path_for(key).exists()


def test_aggregate_failure_removes_new_chunk_keys_without_repair(tmp_path: Path) -> None:
    cache = FilesystemLLMCache(tmp_path)
    adapter = ScriptedLLMAdapter(structured_responses=(_dynamic_success, _dynamic_success))

    def fail_aggregate(_responses: tuple[object, ...]) -> tuple[object, ...]:
        raise AppError("llm.terminology_conflict", {"item_id": "item-0"})

    with pytest.raises(AppError, match=r"llm\.terminology_conflict"):
        asyncio.run(
            _executor(cache, adapter, _config()).execute(
                _items(), SourceCorrectionResponse, aggregate_validator=fail_aggregate
            )
        )
    assert len(adapter.structured_calls) == 2
    assert not list(cache.root.rglob("*.json"))


def test_repair_over_budget_is_not_sent(tmp_path: Path) -> None:
    """Original may fit while a longer repair prompt exceeds the budget."""
    from captioner.core.application.llm_chunk_executor import (
        LLMChunkExecutionConfig,
        LLMChunkExecutor,
    )
    from captioner.core.domain.llm import SourceCorrectionResponse
    from captioner.core.policies.llm_chunking import ChunkItem, ChunkPlanner

    class CountingClient:
        def __init__(self) -> None:
            self.calls = 0

        async def generate_structured[T](
            self,
            request: LLMRequest,
            response_schema: type[T],
            context: ExecutionContext,
        ) -> T:
            del request, response_schema, context
            self.calls += 1
            # Force a validation failure so the executor attempts repair.
            raise LLMStructuredDecodeError('{"responses":[{"id":"item-1","corrected_source":""}]}')

    class CharCounter:
        def count(self, text: str) -> int:
            return len(text)

    short_prompt = "short"
    long_repair = "R" * 500
    client = CountingClient()
    cache = FilesystemLLMCache(tmp_path / "cache")
    config = LLMChunkExecutionConfig(
        task_kind="correct_source",
        provider_kind="openai-compatible",
        provider_identity="default",
        base_url_identity="https://example/v1",
        model="m",
        temperature=0.1,
        source_language="en",
        target_language="zh-CN",
        profile="quality",
        prompt_id="correct_source",
        prompt_version="v1",
        prompt_content_sha256=__import__("hashlib").sha256(short_prompt.encode()).hexdigest(),
        prompt_content=short_prompt,
        # Original serialized request is under this budget; the longer repair prompt is not.
        chunking=ChunkingConfig(max_items=1, max_input_tokens=1100),
        repair_prompt_id="repair_structured",
        repair_prompt_version="v1",
        repair_prompt_content_sha256=__import__("hashlib").sha256(long_repair.encode()).hexdigest(),
        repair_prompt_content=long_repair,
    )
    from captioner.core.ports.llm import LLMClient

    executor = LLMChunkExecutor(
        cast(LLMClient, client),
        cache,
        ChunkPlanner(CharCounter()),
        config,
    )
    with pytest.raises(AppError, match=r"llm\.repair_budget_exceeded"):
        asyncio.run(
            executor.execute(
                (ChunkItem("item-1", "source"),),
                SourceCorrectionResponse,
            )
        )
    assert client.calls == 1


def test_over_budget_request_never_reaches_transport(tmp_path: Path) -> None:
    from captioner.core.application.llm_chunk_executor import (
        LLMChunkExecutionConfig,
        LLMChunkExecutor,
    )
    from captioner.core.domain.llm import SourceCorrectionResponse
    from captioner.core.policies.llm_chunking import ChunkItem, ChunkPlanner
    from captioner.core.ports.llm import LLMClient

    class CountingClient:
        def __init__(self) -> None:
            self.calls = 0

        async def generate_structured[T](
            self,
            request: LLMRequest,
            response_schema: type[T],
            context: ExecutionContext,
        ) -> T:
            del request, response_schema, context
            self.calls += 1
            raise AssertionError("transport_called")

    class CharCounter:
        def count(self, text: str) -> int:
            return len(text)

    prompt = "p" * 50
    client = CountingClient()
    cache = FilesystemLLMCache(tmp_path / "cache")
    config = LLMChunkExecutionConfig(
        task_kind="correct_source",
        provider_kind="openai-compatible",
        provider_identity="default",
        base_url_identity="https://example/v1",
        model="m",
        temperature=0.1,
        source_language="en",
        target_language=None,
        profile="quality",
        prompt_id="correct_source",
        prompt_version="v1",
        prompt_content_sha256=__import__("hashlib").sha256(prompt.encode()).hexdigest(),
        prompt_content=prompt,
        chunking=ChunkingConfig(max_items=1, max_input_tokens=10),
    )
    executor = LLMChunkExecutor(
        cast(LLMClient, client),
        cache,
        ChunkPlanner(CharCounter()),
        config,
    )
    with pytest.raises(AppError, match=r"llm\.item_too_large"):
        asyncio.run(
            executor.execute(
                (ChunkItem("item-1", "source text that is already large"),),
                SourceCorrectionResponse,
            )
        )
    assert client.calls == 0
