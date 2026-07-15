"""Dependency-free scripted ASR adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from captioner.adapters._probe import empty_details, probe_result
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.result import JsonValue
from captioner.core.domain.transcript import Transcript
from captioner.core.ports import CapabilityProbe
from captioner.core.ports.asr import ASRCapabilities, TranscriptionRequest


@dataclass(slots=True)
class FakeASRAdapter:
    available: bool = True
    details: Mapping[str, JsonValue] = field(default_factory=empty_details)
    delay_seconds: float = 0.0
    failure: AppError | None = None
    scripted_transcriptions: Sequence[Transcript | AppError] = field(default_factory=tuple)
    transcription_result: Transcript | None = None
    transcription_failure: AppError | None = None
    _index: int = field(default=0, init=False)

    @property
    def engine_id(self) -> str:
        return "fake-asr"

    @property
    def capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(
            word_timestamps=True,
            segment_timestamps=True,
            language_detection=True,
            native_long_audio=True,
            internal_batching=False,
            supported_languages=None,
            supported_devices=frozenset({"cpu"}),
        )

    async def probe(self) -> CapabilityProbe:
        return await probe_result(
            available=self.available,
            details=self.details,
            delay_seconds=self.delay_seconds,
            failure=self.failure,
        )

    async def transcribe(
        self, request: TranscriptionRequest, context: ExecutionContext
    ) -> Transcript:
        del request
        context.raise_if_cancelled()
        if self.delay_seconds < 0:
            raise ValueError
        remaining = self.delay_seconds
        while remaining > 0:
            context.raise_if_cancelled()
            interval = min(remaining, 0.02)
            await asyncio.sleep(interval)
            remaining -= interval
        context.raise_if_cancelled()
        if self.transcription_failure is not None:
            raise self.transcription_failure
        if self._index < len(self.scripted_transcriptions):
            result = self.scripted_transcriptions[self._index]
            self._index += 1
            if isinstance(result, AppError):
                raise result
            return result
        if self.transcription_result is not None:
            return self.transcription_result
        raise AppError("asr.empty_transcript")
