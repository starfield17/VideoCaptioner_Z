from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from captioner.adapters.llm.http_transport import HTTPResponse, HTTPTimeout
from captioner.adapters.pipeline.stages import (
    CorrectSourceStage,
    QualityTranslateStage,
    ReviewStage,
)
from captioner.bootstrap import (
    build_durable_service,
    build_llm_runtime,
    create_llm_job_snapshot,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.result import FrozenJsonValue
from captioner.core.domain.stage import PipelineProfile, StageName
from captioner.infrastructure.app_paths import AppPaths, resolve_app_paths
from captioner.infrastructure.config import write_llm_config


class NoopTransport:
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
        return HTTPResponse(200, {}, b"{}")

    async def close(self) -> None:
        return None


def _paths(tmp_path: Path) -> AppPaths:
    return resolve_app_paths(
        base_dir=tmp_path,
        resource_root_override=Path("resources").resolve(),
    )


def _write_config(paths: AppPaths) -> None:
    write_llm_config(
        paths.config_dir / "llm.toml",
        """
[providers.default]
kind = "openai-compatible"
base_url = "https://provider.example/v1"
api_key = "unit-test-key"
model = "unit-model"
max_concurrency = 2
request_timeout_sec = 30
max_retries = 2
temperature = 0.1
""",
    )


def test_fast_snapshot_is_redacted_and_bootstrap_registers_translate_stage(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write_config(paths)
    snapshot = create_llm_job_snapshot(
        target_language="zh-CN",
        provider_profile="default",
        source_language="en",
        paths=paths,
    )
    assert snapshot["target_language"] == "zh-CN"
    prompts = snapshot["prompts"]
    assert isinstance(prompts, Mapping)
    assert set(prompts) == {"translate_fast", "repair_structured"}
    assert "prompt_id" not in snapshot
    assert "api_key" not in repr(snapshot)

    runtime = build_llm_runtime(paths=paths, transport=NoopTransport())
    bundle = build_durable_service(
        "batch-fast",
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        paths=paths,
        pipeline_profile=PipelineProfile.FAST,
        llm=snapshot,
        llm_runtime=runtime,
    )
    try:
        assert StageName.TRANSLATE in bundle.service.runners
    finally:
        asyncio.run(bundle.close())


def test_quality_profile_registers_correction_translation_and_review_stages(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write_config(paths)
    snapshot = create_llm_job_snapshot(
        target_language="zh-CN",
        provider_profile="default",
        source_language="en",
        paths=paths,
        pipeline_profile=PipelineProfile.QUALITY,
    )
    runtime = build_llm_runtime(paths=paths, transport=NoopTransport())
    bundle = build_durable_service(
        "batch-quality",
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        paths=paths,
        pipeline_profile=PipelineProfile.QUALITY,
        llm=snapshot,
        llm_runtime=runtime,
    )
    try:
        assert StageName.CORRECT_SOURCE in bundle.service.runners
        assert StageName.TRANSLATE in bundle.service.runners
        assert StageName.REVIEW in bundle.service.runners
        correction = bundle.service.runners[StageName.CORRECT_SOURCE]
        translation = bundle.service.runners[StageName.TRANSLATE]
        review = bundle.service.runners[StageName.REVIEW]
        assert isinstance(correction, CorrectSourceStage)
        assert isinstance(translation, QualityTranslateStage)
        assert isinstance(review, ReviewStage)
        assert correction.client is runtime.service
        assert translation.client is runtime.service
        assert review.client is runtime.service
        prompts = snapshot["prompts"]
        assert isinstance(prompts, Mapping)
        assert set(prompts) == {
            "terminology",
            "correct_source",
            "translate_quality",
            "review_anomalies",
            "repair_structured",
        }
        assert "api_key" not in repr(snapshot)
    finally:
        asyncio.run(bundle.close())


def test_quality_bootstrap_rejects_changed_prompt_content(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_config(paths)
    snapshot = create_llm_job_snapshot(
        target_language="zh-CN",
        provider_profile="default",
        source_language="en",
        paths=paths,
        pipeline_profile=PipelineProfile.QUALITY,
    )
    raw_prompts = snapshot["prompts"]
    assert isinstance(raw_prompts, Mapping)
    prompts = dict(cast(Mapping[str, object], raw_prompts))
    terminology = prompts["terminology"]
    assert isinstance(terminology, Mapping)
    changed = dict(cast(Mapping[str, object], terminology))
    changed["content_sha256"] = "0" * 64
    prompts["terminology"] = changed
    tampered = dict(snapshot)
    tampered["prompts"] = cast(FrozenJsonValue, prompts)
    runtime = build_llm_runtime(paths=paths, transport=NoopTransport())
    try:
        with pytest.raises(AppError, match=r"prompt\.identity_mismatch"):
            build_durable_service(
                "batch-quality-tampered",
                model_ref="tiny",
                device="cpu",
                compute_type="int8",
                language="en",
                paths=paths,
                pipeline_profile=PipelineProfile.QUALITY,
                llm=tampered,
                llm_runtime=runtime,
            )
    finally:
        asyncio.run(runtime.close())
