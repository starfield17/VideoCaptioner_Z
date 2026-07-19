"""Backend-neutral ASR port."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.media import AudioArtifact
from captioner.core.domain.result import JsonValue
from captioner.core.domain.transcript import Transcript
from captioner.core.ports import CapabilityProbe


@dataclass(frozen=True, slots=True)
class ASRCapabilities:
    word_timestamps: bool
    segment_timestamps: bool
    language_detection: bool
    native_long_audio: bool
    internal_batching: bool
    supported_languages: frozenset[str] | None
    supported_devices: frozenset[str]


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    audio: AudioArtifact
    language: str | None
    job_id: str | None = None
    stage_attempt_id: str | None = None
    attempt_workspace: Path | None = None
    task: str = "transcribe"
    word_timestamps: bool = True
    initial_prompt: str | None = None
    backend_options: Mapping[str, JsonValue] | None = None


class ASREngine(Protocol):
    @property
    def engine_id(self) -> str: ...

    @property
    def capabilities(self) -> ASRCapabilities: ...

    async def transcribe(
        self, request: TranscriptionRequest, context: ExecutionContext
    ) -> Transcript: ...


class ASRPort(Protocol):
    async def probe(self) -> CapabilityProbe:
        """Report whether an ASR implementation is available."""
        ...
