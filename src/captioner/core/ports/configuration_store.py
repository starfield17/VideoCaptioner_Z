"""Port for durable application configuration storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from captioner.core.application.configuration import (
    ConfigurationSnapshot,
    ExecutionPreset,
    GlobalSettings,
    ProviderSettingsUpdate,
)


@dataclass(frozen=True, slots=True, repr=False)
class ProviderRuntimeProbeSettings:
    base_url: str
    api_key: str = field(repr=False)
    timeout_sec: float

    def __repr__(self) -> str:
        return "ProviderRuntimeProbeSettings(<redacted>)"


class ConfigurationStorePort(Protocol):
    def load_snapshot(self) -> ConfigurationSnapshot: ...

    def save_global(self, settings: GlobalSettings) -> None: ...

    def save_provider(self, update: ProviderSettingsUpdate) -> None: ...

    def save_user_preset(self, preset: ExecutionPreset) -> None: ...

    def delete_user_preset(self, name: str) -> None: ...

    def resolve_provider_for_test(
        self,
        update: ProviderSettingsUpdate,
    ) -> ProviderRuntimeProbeSettings: ...


__all__ = ["ConfigurationStorePort", "ProviderRuntimeProbeSettings"]
