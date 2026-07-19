"""Backend protocol used only inside the isolated Worker."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Event

ProgressCallback = Callable[[str], None]


class Backend:
    def transcribe(
        self,
        *,
        audio_path: Path,
        model_directory: Path,
        language: str | None,
        task: str,
        initial_prompt: str | None,
        options: Mapping[str, object],
        cancelled: Event,
        progress: ProgressCallback,
        model_identity: Mapping[str, object],
        runtime_info: Mapping[str, object],
    ) -> dict[str, object]:
        raise NotImplementedError

    def doctor_import(self) -> bool:
        raise NotImplementedError
