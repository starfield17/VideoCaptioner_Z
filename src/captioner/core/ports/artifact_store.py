"""Staged atomic artifact storage boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class StagedArtifact(Protocol):
    """One single-use, fsynced artifact waiting for an atomic commit."""

    @property
    def key(self) -> str: ...

    @property
    def target_path(self) -> Path: ...

    @property
    def committed(self) -> bool: ...

    def commit(self, *, overwrite: bool) -> Path: ...

    def discard(self) -> None: ...


class ArtifactStorePort(Protocol):
    @property
    def root(self) -> Path:
        """Return the output root owned by this store."""
        ...

    def stage_bytes(self, key: str, data: bytes) -> StagedArtifact:
        """Stage fsynced bytes under a relative logical key."""
        ...

    def write_bytes(self, key: str, data: bytes, *, overwrite: bool = False) -> Path:
        """Stage and atomically commit bytes under a relative logical key."""
        ...

    def read_bytes(self, key: str) -> bytes:
        """Read bytes from a relative logical key."""
        ...

    def exists(self, key: str) -> bool:
        """Return whether a relative logical key exists."""
        ...

    def delete(self, key: str) -> None:
        """Delete one committed artifact during current-run rollback."""
        ...
