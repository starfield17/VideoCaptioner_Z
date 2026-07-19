"""I/O boundary for Runtime archive verification and extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from captioner.core.domain.runtime import RuntimeManifest


class RuntimeArchive(Protocol):
    def sha256_file(self, path: Path) -> str:
        """Return the SHA-256 digest of a local archive."""
        ...

    def extract(self, archive_path: Path, destination: Path, manifest: RuntimeManifest) -> None:
        """Safely extract and verify a Runtime archive."""
        ...


__all__ = ["RuntimeArchive"]
