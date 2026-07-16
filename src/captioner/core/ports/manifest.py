"""Rebuildable Manifest projection boundary."""

from typing import Literal, Protocol

from captioner.core.domain.batch import BatchProjection

ManifestStatus = Literal["current", "missing", "stale", "ahead", "projection_mismatch", "invalid"]


class ManifestStorePort(Protocol):
    def read(self) -> dict[str, object] | None: ...
    def write(self, projection: BatchProjection) -> None: ...
    def inspect(self, projection: BatchProjection) -> ManifestStatus: ...

    def reconcile(self, projection: BatchProjection) -> ManifestStatus: ...
