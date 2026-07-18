"""Boundary for resolving local or HTTPS Runtime package descriptors."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from captioner.core.domain.runtime_package import RuntimePackageDescriptor


class RuntimePackageSource(Protocol):
    """Resolve a descriptor and expose its archive as a local file."""

    def resolve(self, reference: str | Path, destination: Path) -> RuntimePackageDescriptor: ...


RuntimePackageSourcePort = RuntimePackageSource

__all__ = ["RuntimePackageSource", "RuntimePackageSourcePort"]
