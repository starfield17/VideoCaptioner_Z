"""Core boundary for a future JSONL Runtime Worker client."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

from captioner.core.domain.runtime import RuntimeInstallation
from captioner.core.domain.worker_protocol import (
    CancelResult,
    DoctorRequest,
    DoctorResponse,
    HandshakeRequest,
    ModelLoadRequest,
    ModelLoadResponse,
    ShutdownResult,
    TranscribeRequest,
    WorkerEvent,
    WorkerHandshake,
)


class WorkerClient(Protocol):
    """One Worker session; implementations must serialize active requests."""

    async def start(
        self,
        runtime: RuntimeInstallation,
        workspace: Path,
        request: HandshakeRequest,
    ) -> WorkerHandshake: ...

    def transcribe(self, request: TranscribeRequest) -> AsyncIterator[WorkerEvent]: ...

    async def cancel(self, request_id: str) -> CancelResult: ...

    async def doctor(self, request: DoctorRequest) -> DoctorResponse: ...

    async def load_model(self, request: ModelLoadRequest) -> ModelLoadResponse: ...

    async def shutdown(self) -> ShutdownResult: ...


WorkerClientPort = WorkerClient

__all__ = ["WorkerClient", "WorkerClientPort"]
