"""Atomic byte-oriented artifact storage boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ArtifactStorePort(Protocol):
    @property
    def root(self) -> Path:
        """Return the output root owned by this store."""
        ...

    def write_bytes(self, key: str, data: bytes, *, overwrite: bool = False) -> Path:
        """Atomically store bytes under a relative logical key."""
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
