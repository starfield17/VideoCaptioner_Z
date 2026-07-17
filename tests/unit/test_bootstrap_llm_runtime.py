from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from captioner.adapters.llm.http_transport import HTTPResponse, HTTPTimeout
from captioner.adapters.llm.token_counter import resolve_tokenizer_id
from captioner.bootstrap import build_llm_runtime, create_llm_job_snapshot
from captioner.core.domain.errors import AppError
from captioner.core.domain.llm import LLMItem, LLMRequest, SourceCorrectionResponse
from captioner.core.domain.llm_cache import build_llm_cache_key_for_request
from captioner.core.domain.result import FrozenJsonValue, JsonValue, thaw_json_value
from captioner.core.domain.stage import PipelineProfile
from captioner.infrastructure.app_paths import AppPaths, resolve_app_paths
from captioner.infrastructure.config import write_llm_config


class NoopTransport:
    def __init__(self) -> None:
        self.calls = 0

    async def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        content: bytes,
        timeout: HTTPTimeout,
        max_response_bytes: int,
    ) -> HTTPResponse:
        del method, url, headers, content, timeout, max_response_bytes
        self.calls += 1
        return HTTPResponse(200, {}, b"{}")

    async def close(self) -> None:
        return None


def _paths(tmp_path: Path) -> AppPaths:
    return resolve_app_paths(
        base_dir=tmp_path,
        resource_root_override=Path("resources").resolve(),
    )


def _write_provider(paths: AppPaths, **values: object) -> None:
    defaults: dict[str, object] = {
        "kind": "openai-compatible",
        "base_url": "https://provider.example/v1",
        "api_key": "unit-test-key-a",
        "model": "unit-model-a",
        "max_concurrency": 4,
        "request_timeout_sec": 30,
        "max_retries": 2,
        "temperature": 0.1,
        "tokenizer": "cl100k_base",
    }
    defaults.update(values)
    write_llm_config(
        paths.config_dir / "llm.toml",
        """
[providers.default]
kind = "{kind}"
base_url = "{base_url}"
api_key = "{api_key}"
model = "{model}"
max_concurrency = {max_concurrency}
request_timeout_sec = {request_timeout_sec}
max_retries = {max_retries}
temperature = {temperature}
tokenizer = "{tokenizer}"
""".format(**defaults),
    )


def test_composition_root_creates_one_shared_semaphore(tmp_path: Path) -> None:
    write_llm_config(
        tmp_path / "config" / "llm.toml",
        """
[providers.default]
kind = "openai-compatible"
base_url = "https://provider.example/v1"
api_key = "unit-test-key"
model = "unit-model"
max_concurrency = 4
""",
    )
    paths = resolve_app_paths(base_dir=tmp_path, resource_root_override=tmp_path)
    runtime = build_llm_runtime(paths=paths, transport=NoopTransport())
    try:
        assert runtime.semaphore is runtime.client.semaphore
        assert runtime.service.client is runtime.client
        assert runtime.provider.max_concurrency == 4
        assert "unit-test-key" not in repr(runtime)
    finally:
        asyncio.run(runtime.close())


def test_api_key_rotation_is_allowed_without_changing_public_snapshot(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write_provider(paths)
    snapshot = create_llm_job_snapshot(
        target_language="zh-CN",
        provider_profile="default",
        source_language="en",
        paths=paths,
        pipeline_profile=PipelineProfile.FAST,
    )
    _write_provider(paths, api_key="unit-test-key-b")
    transport = NoopTransport()
    runtime = build_llm_runtime(
        paths=paths,
        transport=transport,
        expected_snapshot=snapshot,
    )
    try:
        assert runtime.provider.api_key == "unit-test-key-b"
        assert transport.calls == 0
        assert "unit-test-key-a" not in repr(runtime)
        assert "unit-test-key-b" not in repr(runtime)
    finally:
        asyncio.run(runtime.close())


def test_quality_snapshot_contains_exact_resource_backed_prompt_set(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_provider(paths)
    snapshot = create_llm_job_snapshot(
        target_language="zh-CN",
        provider_profile="default",
        source_language="en",
        paths=paths,
        pipeline_profile=PipelineProfile.QUALITY,
    )
    raw = cast(dict[str, JsonValue], thaw_json_value(snapshot))
    prompts = cast(dict[str, JsonValue], raw["prompts"])
    assert set(prompts) == {
        "terminology",
        "correct_source",
        "translate_quality",
        "review_anomalies",
        "repair_structured",
    }
    assert cast(dict[str, JsonValue], prompts["terminology"])["prompt_version"] == "v2"
    assert all(
        len(cast(str, cast(dict[str, JsonValue], prompt)["content_sha256"])) == 64
        for prompt in prompts.values()
    )


def test_deterministic_snapshot_creation_is_rejected_before_provider_use(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_provider(paths)
    with pytest.raises(AppError, match=r"llm\.config_invalid"):
        create_llm_job_snapshot(
            target_language="zh-CN",
            provider_profile="default",
            source_language="en",
            paths=paths,
            pipeline_profile=PipelineProfile.DETERMINISTIC,
        )


def test_malformed_runtime_snapshot_fails_before_transport(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_provider(paths)
    transport = NoopTransport()
    malformed = {"model": "unit-model-a"}
    with pytest.raises(AppError, match=r"llm\.provider_snapshot_mismatch"):
        build_llm_runtime(
            paths=paths,
            transport=transport,
            expected_snapshot=cast(Mapping[str, FrozenJsonValue], malformed),
        )
    assert transport.calls == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "other-compatible"),
        ("base_url", "https://other.example/v1"),
        ("model", "unit-model-b"),
        ("max_concurrency", 8),
        ("request_timeout_sec", 60),
        ("max_retries", 4),
        ("temperature", 0.7),
        ("tokenizer", "o200k_base"),
    ],
)
def test_public_provider_drift_fails_before_runtime_creation(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    paths = _paths(tmp_path)
    _write_provider(paths)
    snapshot = create_llm_job_snapshot(
        target_language="zh-CN",
        provider_profile="default",
        source_language="en",
        paths=paths,
        pipeline_profile=PipelineProfile.FAST,
    )
    expected_snapshot = snapshot
    if field == "kind":
        expected_snapshot = dict(snapshot)
        expected_snapshot["kind"] = "other-compatible"
    else:
        _write_provider(paths, **{field: value})
    transport = NoopTransport()
    with pytest.raises(AppError) as raised:
        build_llm_runtime(paths=paths, transport=transport, expected_snapshot=expected_snapshot)
    assert raised.value.code == "llm.provider_snapshot_mismatch"
    assert raised.value.params["fields"] == (field,)
    assert transport.calls == 0
    assert str(value) not in str(raised.value)


def test_production_counter_uses_configured_tokenizer(tmp_path: Path) -> None:
    from captioner.adapters.llm.token_counter import ModelTokenCounter, resolve_tokenizer_id

    paths = _paths(tmp_path)
    _write_provider(paths, tokenizer="cl100k_base")
    runtime = build_llm_runtime(paths=paths, transport=NoopTransport())
    try:
        tokenizer_id = resolve_tokenizer_id(runtime.provider.tokenizer, runtime.provider.model)
        counter = ModelTokenCounter(tokenizer_id)
        # Same character length, different token counts under cl100k_base.
        ascii_like = "aaaaaaaaaa"
        cjk = "你好你好你好你好你好"
        assert len(ascii_like) == len(cjk)
        assert counter.count(ascii_like) != counter.count(cjk)
        assert counter.tokenizer_id == "cl100k_base"
    finally:
        asyncio.run(runtime.close())


def test_unknown_tokenizer_fails_closed(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    with pytest.raises(AppError, match=r"llm\.config_invalid|llm\.tokenizer_unknown"):
        _write_provider(paths, tokenizer="not_a_real_encoding")
        build_llm_runtime(paths=paths, transport=NoopTransport())


def test_tokenizer_is_bound_to_provider_snapshot(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_provider(paths, tokenizer="o200k_base")
    snapshot = create_llm_job_snapshot(
        target_language="zh-CN",
        provider_profile="default",
        source_language="en",
        paths=paths,
        pipeline_profile=PipelineProfile.FAST,
    )
    assert snapshot["tokenizer"] == "o200k_base"


def test_tokenizer_changes_cache_key() -> None:
    prompt = "p"
    request = LLMRequest(
        "correct_source",
        (LLMItem("i", "s"),),
        prompt_id="correct_source",
        prompt_version="v1",
        prompt_content_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        prompt_content=prompt,
    )
    a = build_llm_cache_key_for_request(
        request,
        tokenizer="cl100k_base",
        provider_kind="openai-compatible",
        provider_identity="default",
        base_url_identity="https://example/v1",
        model="m",
        temperature=0.1,
        profile="fast",
        chunk_config={"max_items": 1},
        response_schema_version=1,
        response_schema=SourceCorrectionResponse,
    )
    b = build_llm_cache_key_for_request(
        request,
        tokenizer="o200k_base",
        provider_kind="openai-compatible",
        provider_identity="default",
        base_url_identity="https://example/v1",
        model="m",
        temperature=0.1,
        profile="fast",
        chunk_config={"max_items": 1},
        response_schema_version=1,
        response_schema=SourceCorrectionResponse,
    )
    assert a.digest != b.digest


def test_auto_tokenizer_maps_known_models_and_rejects_unknown() -> None:
    assert resolve_tokenizer_id("auto", "gpt-4o") == "o200k_base"
    assert resolve_tokenizer_id("auto", "gpt-4-turbo") == "cl100k_base"
    assert resolve_tokenizer_id("cl100k_base", "anything") == "cl100k_base"
    with pytest.raises(AppError, match=r"llm\.tokenizer_unknown"):
        resolve_tokenizer_id("auto", "totally-unknown-model-xyz")
    with pytest.raises(AppError, match=r"llm\.tokenizer_unknown"):
        resolve_tokenizer_id("not_real", "gpt-4o")
