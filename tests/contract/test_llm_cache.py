from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from captioner.adapters.persistence.filesystem_llm_cache import FilesystemLLMCache
from captioner.core.domain.llm import (
    LLMItem,
    QualityTranslationResponse,
    SourceCorrectionResponse,
    response_batch_schema,
)
from captioner.core.domain.llm_cache import LLMCacheKey, build_llm_cache_key


def _key() -> LLMCacheKey:
    return build_llm_cache_key(
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
        items=(LLMItem("item-1", "source"),),
        chunk_config={"max_items": 1},
    )


def test_cache_round_trip_is_schema_bound_and_atomic(tmp_path: Path) -> None:
    cache = FilesystemLLMCache(tmp_path)
    key = _key()
    assert cache.get(key, SourceCorrectionResponse) is None
    response = SourceCorrectionResponse("item-1", "corrected")
    cache.put(key, response)
    assert cache.get(key, SourceCorrectionResponse) == response
    path = cache.path_for(key)
    assert path == tmp_path / "llm" / "sha256" / path.parent.name / f"{path.stem}.json"
    assert not list(path.parent.glob("*.tmp"))

    path.write_text('{"schema_version": 999}', encoding="utf-8")
    assert cache.get(key, SourceCorrectionResponse) is None


def test_corrupt_duplicate_and_mismatched_entries_are_misses(tmp_path: Path) -> None:
    cache = FilesystemLLMCache(tmp_path)
    key = _key()
    response = SourceCorrectionResponse("item-1", "corrected")
    cache.put(key, response)
    path = cache.path_for(key)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["cache_key"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert cache.get(key, SourceCorrectionResponse) is None
    path.write_text(
        '{"schema_version":1,"schema_version":1,"cache_key":"x","key_payload":{},"response":{}}',
        encoding="utf-8",
    )
    assert cache.get(key, SourceCorrectionResponse) is None


def test_cache_supports_batch_response_schema(tmp_path: Path) -> None:
    cache = FilesystemLLMCache(tmp_path)
    key = build_llm_cache_key(
        task_kind="translate_quality",
        provider_kind="openai-compatible",
        provider_identity="default",
        base_url_identity="https://provider.example/v1",
        model="unit-model",
        temperature=0.1,
        source_language="en",
        target_language="de",
        profile="quality",
        prompt_id="translate_quality",
        prompt_version="v1",
        prompt_content_sha256="content-hash",
        items=(LLMItem("item-1", "source"), LLMItem("item-2", "other")),
        chunk_config={"max_items": 2},
    )
    schema = response_batch_schema(QualityTranslationResponse)
    value = schema.from_mapping(
        [
            {"id": "item-1", "translated_text": "eins"},
            {"id": "item-2", "translated_text": "zwei"},
        ]
    )
    cache.put(key, value, schema)
    loaded = cache.get(key, schema)
    assert loaded is not None
    responses = cast(tuple[QualityTranslationResponse, ...], loaded.responses)
    assert [item.id for item in responses] == ["item-1", "item-2"]
