"""GUI consumer Protocol for Application-owned Queue and configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from captioner.core.application.batch_commands import (
    BatchActionRequest,
    BatchCommandAck,
    CancelLocalWorkRequest,
    ExecutionCompletion,
    JobActionRequest,
    LocalExecutionSnapshot,
    SubmitBatchRequest,
)
from captioner.core.application.configuration import (
    ConfigurationSnapshot,
    ExecutionPreset,
    GlobalSettings,
    ProviderConnectionResult,
    ProviderSettingsUpdate,
)
from captioner.core.application.input_selection import InputPreview, InputSelectionRequest
from captioner.core.application.job_detail import JobDetailRequest, JobDetailSnapshot
from captioner.core.application.queue_projection import QueueSnapshot
from captioner.core.application.recovery import RecoveryRequest, RecoverySnapshot


@dataclass(frozen=True, slots=True)
class ExecutionPoll:
    state: LocalExecutionSnapshot
    completions: tuple[ExecutionCompletion, ...]


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

    def submit_batch(self, request: SubmitBatchRequest) -> BatchCommandAck: ...

    def perform_batch_action(self, request: BatchActionRequest) -> BatchCommandAck: ...

    def perform_job_action(self, request: JobActionRequest) -> BatchCommandAck: ...

    def cancel_local_work(self, request: CancelLocalWorkRequest) -> BatchCommandAck: ...

    def load_job_detail(self, request: JobDetailRequest) -> JobDetailSnapshot: ...

    def scan_recovery(self, request: RecoveryRequest) -> RecoverySnapshot: ...

    def poll_execution(self) -> ExecutionPoll: ...

    def shutdown(self) -> None: ...


__all__ = ["ExecutionPoll", "GuiApplicationBoundary"]
