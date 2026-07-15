"""Media port without FFmpeg behavior."""

from typing import Protocol

from captioner.core.ports import CapabilityProbe


class MediaPort(Protocol):
    async def probe(self) -> CapabilityProbe:
        """Report whether a media implementation is available."""
        ...
