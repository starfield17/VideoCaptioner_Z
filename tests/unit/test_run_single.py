from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.support import make_audio, make_media, make_transcript

from captioner.adapters.asr.fake import FakeASRAdapter
from captioner.adapters.exporters.srt import serialize_bytes as serialize_srt
from captioner.adapters.exporters.transcript_json import serialize_bytes as serialize_transcript
from captioner.adapters.persistence.local_artifact_store import LocalArtifactStore
from captioner.core.application.run_single import (
    RunSingleRequest,
    RunSingleService,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.media import AudioArtifact, MediaAsset
from captioner.core.domain.transcript import Transcript
from captioner.core.ports.artifact_store import ArtifactStorePort
from captioner.core.ports.asr import TranscriptionRequest


@dataclass
class StubInspector:
    asset: MediaAsset

    async def inspect(self, source_path: Path, context: ExecutionContext) -> MediaAsset:
        del source_path
        context.raise_if_cancelled()
        return self.asset


@dataclass
class StubNormalizer:
    audio: AudioArtifact

    async def normalize(
        self, asset: MediaAsset, workspace: Path, context: ExecutionContext
    ) -> AudioArtifact:
        del asset, workspace
        context.raise_if_cancelled()
        return self.audio


class FailingStore:
    def __init__(self, root: Path) -> None:
        self.delegate = LocalArtifactStore(root)

    @property
    def root(self) -> Path:
        return self.delegate.root

    def write_bytes(self, key: str, data: bytes, *, overwrite: bool = False) -> Path:
        if key.endswith(".srt"):
            raise AppError("output.write_failed", {"key": key})
        return self.delegate.write_bytes(key, data, overwrite=overwrite)

    def read_bytes(self, key: str) -> bytes:
        return self.delegate.read_bytes(key)

    def exists(self, key: str) -> bool:
        return self.delegate.exists(key)

    def delete(self, key: str) -> None:
        self.delegate.delete(key)


def _service(
    source: Path,
    output: Path,
    asr: FakeASRAdapter,
    *,
    store_factory: Callable[[Path], ArtifactStorePort] = LocalArtifactStore,
) -> RunSingleService:
    asset = make_media(source)
    audio = make_audio(source)
    return RunSingleService(
        inspector=StubInspector(asset),
        normalizer=StubNormalizer(audio),
        asr_engine=asr,
        artifact_store_factory=store_factory,
        transcript_serializer=serialize_transcript,
        subtitle_serializer=serialize_srt,
        temp_root=output.parent / "temp",
    )


def test_run_single_commits_transcript_then_srt(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "media sample.wav"
        source.write_bytes(b"input")
        output = tmp_path / "output"
        service = _service(
            source,
            output,
            FakeASRAdapter(transcription_result=make_transcript()),
        )
        result = await service.run(RunSingleRequest(source, output, "en", False))
        assert result.word_count == 2
        assert result.subtitle_path.read_text(encoding="utf-8").endswith("\n")
        assert result.transcript_path.is_file()

    asyncio.run(scenario())


def test_failure_before_export_and_commit_roll_back_srt_and_transcript(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "media.wav"
        source.write_bytes(b"input")
        output = tmp_path / "output"
        failing_asr = FakeASRAdapter(transcription_failure=AppError("asr.transcription_failed"))
        service = _service(source, output, failing_asr)
        with pytest.raises(AppError, match="transcription_failed"):
            await service.run(RunSingleRequest(source, output, "en", False))
        assert not list(output.glob("*.srt"))
        assert not list(output.glob("*.transcript.json"))

        output.mkdir(parents=True, exist_ok=True)
        service = _service(
            source,
            output,
            FakeASRAdapter(transcription_result=make_transcript()),
            store_factory=FailingStore,
        )
        with pytest.raises(AppError, match="write_failed"):
            await service.run(RunSingleRequest(source, output, "en", False))
        assert not (output / "media.transcript.json").exists()
        assert not (output / "media.srt").exists()

    asyncio.run(scenario())


def test_cancellation_does_not_commit_final_srt(tmp_path: Path) -> None:
    class CancellingASR(FakeASRAdapter):
        async def transcribe(
            self, request: TranscriptionRequest, context: ExecutionContext
        ) -> Transcript:
            result = await super().transcribe(request, context)
            context.cancel()
            return result

    async def scenario() -> None:
        source = tmp_path / "media.wav"
        source.write_bytes(b"input")
        output = tmp_path / "output"
        service = _service(source, output, CancellingASR(transcription_result=make_transcript()))
        with pytest.raises(AppError, match=r"operation\.cancelled"):
            await service.run(RunSingleRequest(source, output, "en", False))
        assert not (output / "media.srt").exists()

    asyncio.run(scenario())
