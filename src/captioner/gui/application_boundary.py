"""GUI consumer Protocol for Application-owned Queue and configuration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from captioner.core.application.configuration import (
    ConfigurationSnapshot,
    ExecutionPreset,
    GlobalSettings,
    ProviderConnectionResult,
    ProviderSettingsUpdate,
)
from captioner.core.application.input_selection import InputPreview, InputSelectionRequest
from captioner.core.application.queue_projection import QueueSnapshot


class GuiApplicationBoundary(Protocol):
    def get_queue_snapshot(self) -> QueueSnapshot: ...

    def refresh_queue(self) -> QueueSnapshot: ...

    def subscribe_queue(
        self,
        callback: Callable[[QueueSnapshot], None],
    ) -> Callable[[], None]: ...

    def preview_inputs(self, request: InputSelectionRequest) -> InputPreview: ...

    def load_configuration(self) -> ConfigurationSnapshot: ...

    def save_global_settings(self, settings: GlobalSettings) -> ConfigurationSnapshot: ...

    def save_provider_settings(
        self,
        update: ProviderSettingsUpdate,
    ) -> ConfigurationSnapshot: ...

    def save_user_preset(self, preset: ExecutionPreset) -> ConfigurationSnapshot: ...

    def delete_user_preset(self, name: str) -> ConfigurationSnapshot: ...

    def test_provider_connection(
        self,
        update: ProviderSettingsUpdate,
    ) -> ProviderConnectionResult: ...


__all__ = ["GuiApplicationBoundary"]
