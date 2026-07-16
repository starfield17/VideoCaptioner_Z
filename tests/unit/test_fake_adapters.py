from __future__ import annotations

import asyncio
from typing import cast

import pytest

from captioner.adapters.aligners.fake import FakeAlignerAdapter
from captioner.adapters.asr.fake import FakeASRAdapter
from captioner.adapters.llm.fake import FakeLLMAdapter
from captioner.adapters.llm.scripted import ScriptedLLMAdapter
from captioner.adapters.media.fake import FakeMediaAdapter
from captioner.core.domain.errors import AppError
from captioner.core.domain.result import FrozenJsonValue, JsonValue, thaw_json_value
from captioner.core.ports import CapabilityProbe


def test_fake_adapters_report_configured_success() -> None:
    async def probe_all() -> list[CapabilityProbe]:
        adapters = [
            FakeASRAdapter(details={"kind": "asr"}),
            FakeAlignerAdapter(details={"kind": "aligner"}),
            FakeLLMAdapter(details={"kind": "llm"}),
            FakeMediaAdapter(details={"kind": "media"}),
        ]
        return [await adapter.probe() for adapter in adapters]

    probes = asyncio.run(probe_all())
    assert all(probe.available for probe in probes)
    assert [probe.details["kind"] for probe in probes] == ["asr", "aligner", "llm", "media"]


def test_capability_probe_details_are_recursively_immutable() -> None:
    nested_items = ["before"]
    details = cast(dict[str, JsonValue], {"nested": {"items": nested_items}})
    probe = asyncio.run(FakeASRAdapter(details=details).probe())
    nested_items.append("after")
    assert thaw_json_value(cast(FrozenJsonValue, probe.details)) == {
        "nested": {"items": ["before"]}
    }
    with pytest.raises(TypeError):
        probe.details["nested"]["items"] = []  # type: ignore[index]


def test_fake_failure_and_delay() -> None:
    error = AppError("fake.unavailable", retryable=True)

    async def probe() -> None:
        with pytest.raises(AppError) as raised:
            await FakeASRAdapter(delay_seconds=0.001, failure=error).probe()
        assert raised.value is error

    asyncio.run(probe())
    with pytest.raises(ValueError):
        asyncio.run(FakeMediaAdapter(delay_seconds=-1).probe())


def test_scripted_llm_returns_results_and_errors_in_order() -> None:
    error = AppError("scripted.failure")
    adapter = ScriptedLLMAdapter([CapabilityProbe(True, {"step": 1}), error])

    async def probe() -> None:
        assert (await adapter.probe()).details["step"] == 1
        with pytest.raises(AppError, match=r"scripted\.failure"):
            await adapter.probe()
        with pytest.raises(AppError, match="script_exhausted"):
            await adapter.probe()

    asyncio.run(probe())
