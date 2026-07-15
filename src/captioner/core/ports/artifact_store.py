"""Small byte-oriented artifact storage boundary."""

from typing import Protocol


class ArtifactStorePort(Protocol):
    def write_bytes(self, key: str, data: bytes) -> None:
        """Store bytes under a logical key."""
        ...

    def read_bytes(self, key: str) -> bytes:
        """Read bytes from a logical key."""
        ...

    def exists(self, key: str) -> bool:
        """Return whether a logical key exists."""
        ...
