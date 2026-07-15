"""ASR port without a Phase 0 request/response model."""

from typing import Protocol

from captioner.core.ports import CapabilityProbe


class ASRPort(Protocol):
    async def probe(self) -> CapabilityProbe:
        """Report whether an ASR implementation is available."""
        ...
