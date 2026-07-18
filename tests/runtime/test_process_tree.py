from __future__ import annotations

import asyncio
import signal

import pytest

from captioner.adapters.runtime import process_tree


class _Process:
    pid = 1234
    returncode: int | None = None

    def __init__(self, *, terminate_after: int) -> None:
        self.calls = 0
        self.terminate_after = terminate_after

    async def wait(self) -> int:
        self.calls += 1
        if self.calls < self.terminate_after:
            await asyncio.sleep(1)
        self.returncode = 0
        return 0


def test_posix_termination_escalates_from_term_to_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _Process(terminate_after=2)
    signals: list[tuple[int, signal.Signals]] = []

    def record(pid: int, value: signal.Signals) -> None:
        signals.append((pid, value))

    monkeypatch.setattr(process_tree.os, "name", "posix")
    monkeypatch.setattr(process_tree, "_signal_group", record)

    asyncio.run(
        process_tree.terminate_process_tree(
            process,
            grace_timeout=0.01,
            kill_timeout=0.01,
        )
    )

    assert len(signals) == 2


def test_process_group_already_gone_is_not_unknown_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _Process(terminate_after=1)
    monkeypatch.setattr(process_tree.os, "name", "posix")

    def missing(_pid: int, _value: signal.Signals) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(process_tree, "_signal_group", missing)
    asyncio.run(process_tree.terminate_process_tree(process, grace_timeout=0.01))
