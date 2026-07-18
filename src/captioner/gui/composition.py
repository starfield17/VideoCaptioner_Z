"""GUI composition root for Queue, Create, Settings, and operations controllers."""

from __future__ import annotations

from dataclasses import dataclass

from captioner.gui.application_boundary import GuiApplicationBoundary
from captioner.gui.application_runner import ApplicationRunnerBridge
from captioner.gui.batch_controller import BatchController
from captioner.gui.create_controller import CreateController
from captioner.gui.job_operations_controller import JobOperationsController
from captioner.gui.queue_table_model import QueueTableModel
from captioner.gui.recovery_controller import RecoveryController
from captioner.gui.settings_controller import SettingsController
from captioner.gui_bootstrap import build_gui_application_boundary
from captioner.i18n.service import I18nService
from captioner.infrastructure.app_paths import AppPaths


@dataclass(frozen=True, slots=True)
class GuiControllers:
    queue: BatchController
    create: CreateController
    settings: SettingsController
    operations: JobOperationsController
    recovery: RecoveryController


def build_gui_controllers(
    service: I18nService,
    *,
    paths: AppPaths,
    recent_terminal_limit: int = 100,
    refresh_interval_ms: int = 1000,
    startup_issue: str | None = None,
) -> GuiControllers:
    """Compose shared runner and presentation controllers."""

    def boundary_factory() -> GuiApplicationBoundary:
        return build_gui_application_boundary(
            paths=paths,
            recent_terminal_limit=recent_terminal_limit,
        )

    runner = ApplicationRunnerBridge(boundary_factory)
    model = QueueTableModel(service)
    queue = BatchController(
        model,
        runner,
        refresh_interval_ms=refresh_interval_ms,
    )
    create = CreateController(runner)
    settings = SettingsController(runner, startup_issue=startup_issue)
    operations = JobOperationsController(runner)
    recovery = RecoveryController(runner)

    settings.configuration_changed.connect(create.set_configuration)

    def _refresh_queue(_payload: object = None) -> None:
        queue.refresh()

    def _scan_recovery(_payload: object = None) -> None:
        recovery.scan()

    create.batch_submitted.connect(_refresh_queue)
    operations.command_succeeded.connect(_refresh_queue)
    operations.refresh_requested.connect(queue.refresh)
    queue.snapshot_changed.connect(_scan_recovery)

    return GuiControllers(
        queue=queue,
        create=create,
        settings=settings,
        operations=operations,
        recovery=recovery,
    )


def build_batch_controller(
    service: I18nService,
    *,
    paths: AppPaths,
    recent_terminal_limit: int = 100,
    refresh_interval_ms: int = 1000,
) -> BatchController:
    """Compatibility helper: Queue controller from the shared composition."""
    return build_gui_controllers(
        service,
        paths=paths,
        recent_terminal_limit=recent_terminal_limit,
        refresh_interval_ms=refresh_interval_ms,
    ).queue


__all__ = ["GuiControllers", "build_batch_controller", "build_gui_controllers"]
