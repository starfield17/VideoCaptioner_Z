from __future__ import annotations

from pathlib import Path

import pytest

from captioner.adapters.persistence.batch_lease import BatchLease
from captioner.core.domain.errors import AppError


def _lease(path: Path, token: str, pid: int, alive: bool) -> BatchLease:
    return BatchLease(path, token, pid, "host-a", "2026-01-01T00:00:00Z", lambda _pid: alive)


def test_exclusive_acquisition_and_active_rejection(tmp_path: Path) -> None:
    path = tmp_path / "lease.json"
    owner = _lease(path, "token-a", 10, True)
    owner.acquire()
    with pytest.raises(AppError, match=r"batch\.busy"):
        _lease(path, "token-b", 11, True).acquire()
    owner.release()
    assert not path.exists()


def test_stale_same_host_lease_is_reclaimed(tmp_path: Path) -> None:
    path = tmp_path / "lease.json"
    stale = _lease(path, "token-a", 10, False)
    stale.acquire()
    replacement = _lease(path, "token-b", 11, False)
    assert replacement.acquire().token == "token-b"


def test_release_is_token_safe(tmp_path: Path) -> None:
    path = tmp_path / "lease.json"
    lease = _lease(path, "token-a", 10, True)
    lease.acquire()
    path.write_text(path.read_text().replace("token-a", "token-b"), encoding="utf-8")
    with pytest.raises(AppError, match="token_mismatch"):
        lease.release()
