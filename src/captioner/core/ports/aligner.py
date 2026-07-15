"""Alignment port without a Phase 0 domain model."""

from typing import Protocol

from captioner.core.ports import CapabilityProbe


class AlignerPort(Protocol):
    async def probe(self) -> CapabilityProbe:
        """Report whether an aligner implementation is available."""
        ...
