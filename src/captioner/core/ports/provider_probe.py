"""Port for redacted provider connectivity probes."""

from __future__ import annotations

from typing import Protocol

from captioner.core.application.configuration import ProviderConnectionResult
from captioner.core.ports.configuration_store import ProviderRuntimeProbeSettings


class ProviderProbePort(Protocol):
    def test(
        self,
        settings: ProviderRuntimeProbeSettings,
    ) -> ProviderConnectionResult: ...


__all__ = ["ProviderProbePort"]
