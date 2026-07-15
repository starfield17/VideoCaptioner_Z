"""Shared behavior for dependency-free capability fakes."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from captioner.core.domain.errors import AppError
from captioner.core.domain.result import JsonValue
from captioner.core.ports import CapabilityProbe


def empty_details() -> dict[str, JsonValue]:
    """Create a typed empty capability-details mapping."""
    return {}


async def probe_result(
    *,
    available: bool,
    details: Mapping[str, JsonValue],
    delay_seconds: float,
    failure: AppError | None,
) -> CapabilityProbe:
    """Return a configured probe after an optional async delay."""
    if delay_seconds < 0:
        raise ValueError
    if delay_seconds:
        await asyncio.sleep(delay_seconds)
    if failure is not None:
        raise failure
    return CapabilityProbe(available=available, details=details)
