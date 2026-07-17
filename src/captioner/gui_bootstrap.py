"""Lightweight GUI composition root without ASR/LLM/Torch SDK imports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from captioner.adapters.llm.http_provider_probe import HTTPProviderProbe
from captioner.adapters.persistence.filesystem_batch_catalog import FilesystemBatchCatalog
from captioner.adapters.persistence.filesystem_input_discovery import (
    FilesystemInputDiscovery,
)
from captioner.adapters.persistence.toml_configuration_store import (
    SETTINGS_FILENAME,
    TomlConfigurationStore,
    load_startup_locale_from_settings,
)
from captioner.core.application.configuration import (
    ConfigurationService,
    ConfigurationSnapshot,
    ExecutionPreset,
    GlobalSettings,
    ProviderConnectionResult,
    ProviderSettingsUpdate,
)
from captioner.core.application.input_selection import InputPreview, InputSelectionRequest
from captioner.core.application.queue_projection import QueueProjectionService, QueueSnapshot
from captioner.core.ports.input_discovery import InputDiscoveryPort
from captioner.gui.application_boundary import GuiApplicationBoundary
from captioner.infrastructure.app_paths import (
    AppPaths,
    ensure_runtime_layout,
    resolve_app_paths,
)


@dataclass(slots=True)
class GuiApplicationService:
    """Concrete Application boundary used by the GUI worker thread."""

    queue: QueueProjectionService
    input_discovery: InputDiscoveryPort
    configuration: ConfigurationService

    def get_queue_snapshot(self) -> QueueSnapshot:
        return self.queue.get_queue_snapshot()

    def refresh_queue(self) -> QueueSnapshot:
        return self.queue.refresh_queue()

    def subscribe_queue(
        self,
        callback: Callable[[QueueSnapshot], None],
    ) -> Callable[[], None]:
        return self.queue.subscribe_queue(callback)

    def preview_inputs(self, request: InputSelectionRequest) -> InputPreview:
        return self.input_discovery.preview(request)

    def load_configuration(self) -> ConfigurationSnapshot:
        return self.configuration.load()

    def save_global_settings(self, settings: GlobalSettings) -> ConfigurationSnapshot:
        return self.configuration.save_global(settings)

    def save_provider_settings(
        self,
        update: ProviderSettingsUpdate,
    ) -> ConfigurationSnapshot:
        return self.configuration.save_provider(update)

    def save_user_preset(self, preset: ExecutionPreset) -> ConfigurationSnapshot:
        return self.configuration.save_user_preset(preset)

    def delete_user_preset(self, name: str) -> ConfigurationSnapshot:
        return self.configuration.delete_user_preset(name)

    def test_provider_connection(
        self,
        update: ProviderSettingsUpdate,
    ) -> ProviderConnectionResult:
        return self.configuration.test_provider(update)


def build_gui_application_boundary(
    *,
    paths: AppPaths | None = None,
    recent_terminal_limit: int = 100,
) -> GuiApplicationBoundary:
    """Compose the Queue + configuration boundary used by the GUI runner."""
    application_paths = resolve_app_paths() if paths is None else paths
    ensure_runtime_layout(application_paths)
    catalog = FilesystemBatchCatalog(application_paths.batches_dir)
    queue = QueueProjectionService(
        catalog=catalog,
        recent_terminal_limit=recent_terminal_limit,
    )
    store = TomlConfigurationStore(application_paths.config_dir)
    configuration = ConfigurationService(
        store=store,
        provider_probe=HTTPProviderProbe(),
    )
    return GuiApplicationService(
        queue=queue,
        input_discovery=FilesystemInputDiscovery(),
        configuration=configuration,
    )


def load_startup_locale(
    *,
    paths: AppPaths,
    explicit_locale: str | None,
) -> tuple[str, str | None]:
    """Resolve GUI startup locale; explicit CLI wins over persisted settings."""
    if explicit_locale is not None and explicit_locale.strip():
        return explicit_locale.strip(), None
    settings_path = Path(paths.config_dir) / SETTINGS_FILENAME
    return load_startup_locale_from_settings(settings_path)


__all__ = [
    "GuiApplicationService",
    "build_gui_application_boundary",
    "load_startup_locale",
]
