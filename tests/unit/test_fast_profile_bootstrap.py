from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest

from captioner.adapters.llm.http_transport import HTTPResponse, HTTPTimeout
from captioner.bootstrap import (
    build_durable_service,
    build_llm_runtime,
    create_llm_job_snapshot,
)
from captioner.core.domain.errors import AppError
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
    assert snapshot["prompt_id"] == "translate_fast"
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


def test_quality_profile_is_structured_unavailable_error(tmp_path: Path) -> None:
    with pytest.raises(AppError, match=r"llm\.profile_unavailable"):
        build_durable_service(
            "batch-quality",
            model_ref="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            paths=_paths(tmp_path),
            pipeline_profile=PipelineProfile.QUALITY,
        )
