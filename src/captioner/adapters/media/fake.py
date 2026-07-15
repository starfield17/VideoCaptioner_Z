"""Dependency-free media capability fake."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from captioner.adapters._probe import empty_details, probe_result
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.media import AudioArtifact, MediaAsset
from captioner.core.domain.result import JsonValue
from captioner.core.ports import CapabilityProbe


@dataclass(slots=True)
class FakeMediaAdapter:
    available: bool = True
    details: Mapping[str, JsonValue] = field(default_factory=empty_details)
    delay_seconds: float = 0.0
    failure: AppError | None = None
    inspected_asset: MediaAsset | None = None
    normalized_audio: AudioArtifact | None = None

    async def probe(self) -> CapabilityProbe:
        return await probe_result(
            available=self.available,
            details=self.details,
            delay_seconds=self.delay_seconds,
            failure=self.failure,
        )

    async def inspect(self, source_path: Path, context: ExecutionContext) -> MediaAsset:
        context.raise_if_cancelled()
        if self.failure is not None:
            raise self.failure
        if self.inspected_asset is None:
            raise AppError("media.input_missing", {"path": str(source_path)})
        return self.inspected_asset

    async def normalize(
        self, asset: MediaAsset, workspace: Path, context: ExecutionContext
    ) -> AudioArtifact:
        del asset, workspace
        context.raise_if_cancelled()
        if self.failure is not None:
            raise self.failure
        if self.normalized_audio is None:
            raise AppError("media.normalized_audio_missing")
        return self.normalized_audio
