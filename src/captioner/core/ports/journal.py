"""Reserved journal probe boundary."""

from typing import Protocol

from captioner.core.ports import CapabilityProbe


class JournalPort(Protocol):
    async def probe(self) -> CapabilityProbe:
        """Report whether a journal is available."""
        ...
