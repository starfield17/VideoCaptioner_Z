from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.recovery.support import config, service

from captioner.adapters.testing.fault_injector import InjectedCrash, ScriptedFaultInjector
from captioner.core.domain.stage import StageName


def test_publish_commit_crash_does_not_repeat_publication(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts, ScriptedFaultInjector("publish", "after_journal_commit"))
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    with pytest.raises(InjectedCrash):
        asyncio.run(current.run(projection))
    asyncio.run(service(tmp_path, counts).resume())
    assert counts[StageName.PUBLISH] == 1
