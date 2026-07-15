"""Reserved job-store probe boundary."""

from typing import Protocol

from captioner.core.ports import CapabilityProbe


class JobStorePort(Protocol):
    async def probe(self) -> CapabilityProbe:
        """Report whether a job store is available."""
        ...
