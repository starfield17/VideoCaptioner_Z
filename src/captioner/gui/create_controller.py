"""Main-thread controller for Create page input preview and draft validation."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from captioner.core.application.configuration import ConfigurationSnapshot, ExecutionPreset
from captioner.core.application.input_selection import (
    BatchDraft,
    InputPreview,
    InputSelectionRequest,
    OutputCollisionPolicy,
    build_batch_draft,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.stage import PipelineProfile
from captioner.gui.application_runner import ApplicationRunnerBridge, RunnerFailure


class CreateController(QObject):
    """Coordinates input selection preview; never creates durable Batches."""

    entries_changed = Signal(object)
    preview_changed = Signal(object)
    configuration_changed = Signal(object)
    draft_changed = Signal(object)
    busy_changed = Signal(bool)
    failure_changed = Signal(object)

    def __init__(
        self,
        runner: ApplicationRunnerBridge,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._entries: tuple[str, ...] = ()
        self._recursive = True
        self._preview: InputPreview | None = None
        self._configuration: ConfigurationSnapshot | None = None
        self._selected_preset: str | None = None
        self._draft: BatchDraft | None = None
        self._preview_busy = False
        self._preview_queued = False
        self._preview_generation = 0
        self._active_generation = 0
        self._validation_error: str | None = None
        self._last_failure: RunnerFailure | None = None

        self._runner.input_preview_ready.connect(self._on_preview)
        self._runner.input_failure.connect(self._on_failure)

    @property
    def entries(self) -> tuple[str, ...]:
        return self._entries

    @property
    def recursive(self) -> bool:
        return self._recursive

    @property
    def preview(self) -> InputPreview | None:
        return self._preview

    @property
    def configuration(self) -> ConfigurationSnapshot | None:
        return self._configuration

    @property
    def selected_preset(self) -> str | None:
        return self._selected_preset

    @property
    def draft(self) -> BatchDraft | None:
        return self._draft

    @property
    def busy(self) -> bool:
        return self._preview_busy

    @property
    def validation_error(self) -> str | None:
        return self._validation_error

    @property
    def last_failure(self) -> RunnerFailure | None:
        return self._last_failure

    def set_entries(self, entries: tuple[str, ...]) -> None:
        self._entries = tuple(entries)
        self._invalidate_draft()
        self.entries_changed.emit(self._entries)
        self._request_preview()

    def append_entries(self, entries: tuple[str, ...]) -> None:
        if not entries:
            return
        self.set_entries(self._entries + tuple(entries))

    def remove_entry(self, index: int) -> None:
        if index < 0 or index >= len(self._entries):
            return
        remaining = list(self._entries)
        del remaining[index]
        self.set_entries(tuple(remaining))

    def clear_entries(self) -> None:
        self.set_entries(())

    def set_recursive(self, recursive: bool) -> None:
        if self._recursive is recursive:
            return
        self._recursive = bool(recursive)
        self._invalidate_draft()
        self._request_preview()

    def set_configuration(self, snapshot: ConfigurationSnapshot) -> None:
        self._configuration = snapshot
        if self._selected_preset is None:
            self._selected_preset = snapshot.global_settings.default_preset_name
        self.configuration_changed.emit(snapshot)

    def select_preset(self, name: str) -> ExecutionPreset | None:
        self._selected_preset = name
        if self._configuration is None:
            return None
        for preset in self._configuration.presets:
            if preset.name == name:
                return preset
        return None

    def save_user_preset(self, preset: ExecutionPreset) -> None:
        """Forward user-preset persistence through the shared Application runner."""
        self._runner.request_preset_save(preset)

    def delete_user_preset(self, name: str) -> None:
        self._runner.request_preset_delete(name)

    def validate_draft(
        self,
        *,
        output_root: str,
        preset_name: str,
        pipeline_profile: PipelineProfile | str,
        model_ref: str,
        device: str,
        compute_type: str,
        source_language: str | None,
        target_language: str | None,
        provider_profile: str,
        ffmpeg_bin: str,
        ffprobe_bin: str,
        collision_policy: OutputCollisionPolicy | str,
    ) -> BatchDraft | None:
        preview = self._preview
        if preview is None or preview.empty:
            self._validation_error = "batch.draft_invalid"
            self._draft = None
            self.draft_changed.emit(None)
            self.failure_changed.emit(RunnerFailure(code="batch.draft_invalid", retryable=False))
            return None
        try:
            draft = build_batch_draft(
                preview,
                output_root=output_root,
                preset_name=preset_name,
                pipeline_profile=pipeline_profile,
                model_ref=model_ref,
                device=device,
                compute_type=compute_type,
                source_language=source_language,
                target_language=target_language,
                provider_profile=provider_profile,
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
                collision_policy=collision_policy,
            )
        except AppError as exc:
            self._validation_error = exc.code
            self._draft = None
            self.draft_changed.emit(None)
            self.failure_changed.emit(RunnerFailure(code=exc.code, retryable=False))
            return None
        self._validation_error = None
        self._draft = draft
        self._last_failure = None
        self.draft_changed.emit(draft)
        self.failure_changed.emit(None)
        return draft

    def _request_preview(self) -> None:
        self._preview_generation += 1
        self._active_generation = self._preview_generation
        if not self._entries:
            self._preview = InputPreview(accepted_paths=(), rejected=())
            self._preview_busy = False
            self._preview_queued = False
            self.preview_changed.emit(self._preview)
            self.busy_changed.emit(False)
            return
        if self._preview_busy:
            # Coalesce: keep at most one follow-up for the latest entries.
            self._preview_queued = True
            return
        self._dispatch_preview()

    def _dispatch_preview(self) -> None:
        self._preview_busy = True
        self._preview_queued = False
        self._pending_generation = self._active_generation
        self.busy_changed.emit(True)
        request = InputSelectionRequest(
            entries=self._entries,
            recursive=self._recursive,
        )
        self._runner.request_input_preview(request)

    def _on_preview(self, preview: object) -> None:
        if not isinstance(preview, InputPreview):
            return
        generation = getattr(self, "_pending_generation", self._active_generation)
        if generation != self._active_generation:
            if self._preview_queued:
                self._dispatch_preview()
            else:
                self._preview_busy = False
                self.busy_changed.emit(False)
            return
        self._preview = preview
        self.preview_changed.emit(preview)
        self._last_failure = None
        self.failure_changed.emit(None)
        if self._preview_queued:
            self._dispatch_preview()
            return
        self._preview_busy = False
        self.busy_changed.emit(False)

    def _on_failure(self, failure: object) -> None:
        if not isinstance(failure, RunnerFailure):
            failure = RunnerFailure(code="gui.application_bridge_failed", retryable=False)
        generation = getattr(self, "_pending_generation", self._active_generation)
        if generation != self._active_generation:
            if self._preview_queued:
                self._dispatch_preview()
            else:
                self._preview_busy = False
                self.busy_changed.emit(False)
            return
        self._last_failure = failure
        self.failure_changed.emit(failure)
        if self._preview_queued:
            self._dispatch_preview()
            return
        self._preview_busy = False
        self.busy_changed.emit(False)

    def _invalidate_draft(self) -> None:
        if self._draft is None and self._validation_error is None:
            return
        self._draft = None
        self._validation_error = None
        self.draft_changed.emit(None)


__all__ = ["CreateController"]
