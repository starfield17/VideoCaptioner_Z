from __future__ import annotations

import asyncio
import os
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


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group behavior")
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


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group behavior")
def test_process_group_already_gone_is_not_unknown_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _Process(terminate_after=1)
    monkeypatch.setattr(process_tree.os, "name", "posix")

    def missing(_pid: int, _value: signal.Signals) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(process_tree, "_signal_group", missing)
    asyncio.run(process_tree.terminate_process_tree(process, grace_timeout=0.01))


def test_windows_termination_uses_tree_then_force(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _Process(terminate_after=2)
    calls: list[tuple[int, bool]] = []

    def record_taskkill(pid: int, *, force: bool) -> None:
        calls.append((pid, force))

    monkeypatch.setattr(process_tree.os, "name", "nt")
    monkeypatch.setattr(process_tree, "_taskkill", record_taskkill)

    asyncio.run(
        process_tree.terminate_process_tree(
            process,
            grace_timeout=0.01,
            kill_timeout=0.01,
        )
    )

    assert calls == [(1234, False), (1234, True)]
