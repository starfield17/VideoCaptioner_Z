"""Rebuildable Manifest projection boundary."""

from typing import Protocol

from captioner.core.domain.batch import BatchProjection


class ManifestStorePort(Protocol):
    def read(self) -> dict[str, object] | None: ...
    def write(self, projection: BatchProjection) -> None: ...
    def reconcile(self, projection: BatchProjection) -> str: ...
