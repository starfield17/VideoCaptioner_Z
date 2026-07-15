"""Dependency-free aligner fake."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from captioner.adapters._probe import empty_details, probe_result
from captioner.core.domain.errors import AppError
from captioner.core.domain.result import JsonValue
from captioner.core.ports import CapabilityProbe


@dataclass(slots=True)
class FakeAlignerAdapter:
    available: bool = True
    details: Mapping[str, JsonValue] = field(default_factory=empty_details)
    delay_seconds: float = 0.0
    failure: AppError | None = None

    async def probe(self) -> CapabilityProbe:
        return await probe_result(
            available=self.available,
            details=self.details,
            delay_seconds=self.delay_seconds,
            failure=self.failure,
        )
