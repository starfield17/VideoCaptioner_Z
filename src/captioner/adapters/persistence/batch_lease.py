"""Single-host local-filesystem Batch writer lease."""

from __future__ import annotations

import json
import os
import socket
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from filelock import FileLock, Timeout

from captioner.core.domain.errors import AppError


@dataclass(frozen=True, slots=True)
class LeaseOwner:
    token: str
    pid: int
    hostname: str
    created_timestamp: str


@dataclass(slots=True)
class BatchLease:
    path: Path
    token: str
    pid: int
    hostname: str
    created_timestamp: str
    pid_is_alive: Callable[[int], bool]
    _acquired: bool = False

    @property
    def _guard_path(self) -> Path:
        return self.path.with_name(f".{self.path.name}.guard")

    def acquire(self) -> LeaseOwner:
        owner = LeaseOwner(self.token, self.pid, self.hostname, self.created_timestamp)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(self._guard_path, timeout=10):
                if self.path.exists():
                    existing = self.read_owner()
                    if existing.hostname != self.hostname:
                        raise AppError("batch.busy", {"reason": "remote_host"})
                    if self.pid_is_alive(existing.pid):
                        raise AppError("batch.busy", {"reason": "active_pid"})
                    self._reclaim_stale(existing)
                _atomic_write(self.path, _lease_bytes(owner))
                self._acquired = True
        except Timeout as exc:
            raise AppError("batch.busy", {"reason": "lease_guard"}) from exc
        except OSError as exc:
            raise AppError("batch.lease_failed") from exc
        return owner

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            with FileLock(self._guard_path, timeout=10):
                existing = self.read_owner()
                if existing.token != self.token:
                    raise AppError("batch.lease_failed", {"reason": "token_mismatch"})
                self.path.unlink()
                _fsync_directory(self.path.parent)
        except OSError as exc:
            raise AppError("batch.lease_failed", {"reason": "release"}) from exc
        self._acquired = False

    def read_owner(self) -> LeaseOwner:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AppError("batch.lease_invalid") from exc
        if not isinstance(value, dict):
            raise AppError("batch.lease_invalid")
        raw = cast(dict[str, object], value)
        if set(raw) != {
            "token",
            "pid",
            "hostname",
            "created_timestamp",
        }:
            raise AppError("batch.lease_invalid")
        token, pid = raw["token"], raw["pid"]
        hostname, created = raw["hostname"], raw["created_timestamp"]
        if (
            not isinstance(token, str)
            or not token
            or not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid < 1
            or not isinstance(hostname, str)
            or not hostname
            or not isinstance(created, str)
            or not created
        ):
            raise AppError("batch.lease_invalid")
        return LeaseOwner(token, pid, hostname, created)

    def _reclaim_stale(self, expected: LeaseOwner) -> None:
        current = self.read_owner()
        if current != expected:
            raise AppError("batch.busy", {"reason": "lease_changed"})
        try:
            self.path.unlink()
        except OSError as exc:
            raise AppError("batch.lease_failed", {"reason": "reclaim"}) from exc


def default_batch_lease(path: Path, *, token: str, created_timestamp: str) -> BatchLease:
    return BatchLease(
        path,
        token,
        os.getpid(),
        socket.gethostname(),
        created_timestamp,
        _pid_is_alive,
    )


def _lease_bytes(owner: LeaseOwner) -> bytes:
    return (
        json.dumps(
            {
                "token": owner.token,
                "pid": owner.pid,
                "hostname": owner.hostname,
                "created_timestamp": owner.created_timestamp,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _atomic_write(path: Path, data: bytes) -> None:
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
