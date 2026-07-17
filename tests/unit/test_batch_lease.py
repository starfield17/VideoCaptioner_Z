from __future__ import annotations

import json
from pathlib import Path

import pytest

from captioner.adapters.persistence.batch_lease import BatchLease, inspect_batch_lease
from captioner.core.domain.errors import AppError


def _lease(path: Path, token: str, pid: int, alive: bool) -> BatchLease:
    return BatchLease(path, token, pid, "host-a", "2026-01-01T00:00:00Z", lambda _pid: alive)


def _write_owner(
    path: Path,
    *,
    token: str = "token-a",
    pid: int = 10,
    hostname: str = "host-a",
    created_timestamp: str = "2026-01-01T00:00:00Z",
) -> bytes:
    payload = (
        json.dumps(
            {
                "token": token,
                "pid": pid,
                "hostname": hostname,
                "created_timestamp": created_timestamp,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    encoded = payload.encode("utf-8")
    path.write_bytes(encoded)
    return encoded


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


def test_inspect_batch_lease_classifies_states_without_mutation(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert inspect_batch_lease(missing, hostname="host-a") == "missing"

    non_file = tmp_path / "lease-as-dir"
    non_file.mkdir()
    assert inspect_batch_lease(non_file, hostname="host-a") == "invalid"

    local = tmp_path / "local.json"
    local_bytes = _write_owner(local, pid=10, hostname="host-a")
    assert (
        inspect_batch_lease(local, hostname="host-a", pid_is_alive=lambda _pid: True)
        == "active_local"
    )
    assert local.read_bytes() == local_bytes

    remote = tmp_path / "remote.json"
    remote_bytes = _write_owner(remote, pid=10, hostname="other-host")
    assert (
        inspect_batch_lease(remote, hostname="host-a", pid_is_alive=lambda _pid: True)
        == "active_remote"
    )
    assert remote.read_bytes() == remote_bytes

    stale = tmp_path / "stale.json"
    stale_bytes = _write_owner(stale, pid=11, hostname="host-a")
    assert inspect_batch_lease(stale, hostname="host-a", pid_is_alive=lambda _pid: False) == "stale"
    assert stale.read_bytes() == stale_bytes

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    invalid_bytes = invalid.read_bytes()
    assert inspect_batch_lease(invalid, hostname="host-a") == "invalid"
    assert invalid.read_bytes() == invalid_bytes


def test_inspect_permission_error_treats_pid_as_alive(tmp_path: Path) -> None:
    path = tmp_path / "lease.json"
    _write_owner(path, pid=42, hostname="host-a")

    def raise_permission(_pid: int) -> bool:
        raise PermissionError("denied")

    assert (
        inspect_batch_lease(path, hostname="host-a", pid_is_alive=raise_permission)
        == "active_local"
    )
