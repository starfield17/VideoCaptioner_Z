from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.recovery.support import config, service

from captioner.adapters.testing.fault_injector import InjectedCrash, ScriptedFaultInjector
from captioner.core.domain.stage import StageName


def test_transcribe_interruption_preserves_upstream_execution_counts(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts, ScriptedFaultInjector("transcribe", "mid_execute"))
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    with pytest.raises(InjectedCrash):
        asyncio.run(current.run(projection))
    result = asyncio.run(service(tmp_path, counts).resume())
    assert counts[StageName.INSPECT] == counts[StageName.NORMALIZE] == 1
    assert counts[StageName.TRANSCRIBE] == 2
    assert result.job("job-000001").stage(StageName.TRANSCRIBE).attempt == 2
