"""Runtime capability boundary."""

from typing import Protocol

from captioner.core.ports import CapabilityProbe


class RuntimePort(Protocol):
    async def probe(self) -> CapabilityProbe:
        """Report whether a runtime is available."""
        ...
