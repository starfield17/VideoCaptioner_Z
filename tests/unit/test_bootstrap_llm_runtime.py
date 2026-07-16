from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

from captioner.adapters.llm.http_transport import HTTPResponse, HTTPTimeout
from captioner.bootstrap import build_llm_runtime
from captioner.infrastructure.app_paths import resolve_app_paths
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
