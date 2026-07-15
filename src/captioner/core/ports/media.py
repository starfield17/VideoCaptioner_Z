"""Media inspection and audio normalization ports."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.media import AudioArtifact, MediaAsset
from captioner.core.ports import CapabilityProbe


class MediaInspector(Protocol):
    async def inspect(self, source_path: Path, context: ExecutionContext) -> MediaAsset: ...


class AudioNormalizer(Protocol):
    async def normalize(
        self, asset: MediaAsset, workspace: Path, context: ExecutionContext
    ) -> AudioArtifact: ...


class MediaPort(Protocol):
    async def probe(self) -> CapabilityProbe:
        """Report whether a media implementation is available."""
        ...
