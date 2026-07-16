from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.support import make_audio, make_media, make_transcript

import captioner.adapters.persistence.local_artifact_store as local_store_module
from captioner.adapters.asr.fake import FakeASRAdapter
from captioner.adapters.exporters.srt import serialize_bytes as serialize_srt
from captioner.adapters.exporters.transcript_json import serialize_bytes as serialize_transcript
from captioner.adapters.persistence.local_artifact_store import LocalArtifactStore
from captioner.adapters.subtitles.ass import serialize_bytes as serialize_ass
from captioner.adapters.subtitles.json_track import serialize as serialize_track_json
from captioner.adapters.subtitles.webvtt import serialize_bytes as serialize_webvtt
from captioner.core.application.run_single import (
    RunSingleRequest,
    RunSingleService,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.media import AudioArtifact, MediaAsset
from captioner.core.domain.transcript import Transcript
from captioner.core.ports.artifact_store import ArtifactStorePort, StagedArtifact
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


class HookedStagedArtifact:
    def __init__(self, delegate: StagedArtifact, owner: HookedStore) -> None:
        self.delegate = delegate
        self.owner = owner

    @property
    def key(self) -> str:
        return self.delegate.key

    @property
    def target_path(self) -> Path:
        return self.delegate.target_path

    @property
    def committed(self) -> bool:
        return self.delegate.committed

    def commit(self, *, overwrite: bool) -> Path:
        is_transcript = self.key.endswith(".transcript.json")
        is_srt = self.key.endswith(".srt")
        if is_transcript and self.owner.hook == "before-transcript":
            self.owner.context.cancel()
            raise AppError("operation.cancelled")
        if is_srt and self.owner.hook == "before-srt":
            self.owner.context.cancel()
            raise AppError("operation.cancelled")
        if is_srt and self.owner.hook == "srt-failure":
            raise AppError("output.write_failed", {"key": self.key})
        if is_transcript and self.owner.hook == "keyboard-before-transcript":
            raise KeyboardInterrupt
        if is_transcript and self.owner.hook == "cancelled-before-transcript":
            raise asyncio.CancelledError
        path = self.delegate.commit(overwrite=overwrite)
        if is_transcript and self.owner.hook == "after-transcript":
            self.owner.context.cancel()
        if is_srt and self.owner.hook in {"after-srt", "after-both"}:
            self.owner.context.cancel()
        if is_srt and self.owner.hook == "cancelled-srt":
            raise asyncio.CancelledError
        if is_transcript and self.owner.hook == "keyboard-transcript":
            raise KeyboardInterrupt
        if is_srt and self.owner.hook == "keyboard-srt":
            raise KeyboardInterrupt
        return path

    def discard(self) -> None:
        self.owner.discard_attempts.append(self.key)
        if (
            self.owner.fail_cleanup_once
            and not self.owner.cleanup_failure_injected
            and self.key.endswith(".srt")
        ):
            self.owner.cleanup_failure_injected = True
            raise AppError("output.cleanup_failed")
        self.delegate.discard()
        if self.owner.hook == "discard-cancel":
            self.owner.context.cancel()


class HookedStore:
    def __init__(
        self,
        root: Path,
        *,
        context: ExecutionContext | None = None,
        hook: str = "",
        interrupt_during_staging: bool = False,
        fail_rollback: bool = False,
        fail_cleanup_once: bool = False,
    ) -> None:
        self.delegate = LocalArtifactStore(root)
        self.context = ExecutionContext() if context is None else context
        self.hook = hook
        self.interrupt_during_staging = interrupt_during_staging
        self.fail_rollback = fail_rollback
        self.fail_cleanup_once = fail_cleanup_once
        self.cleanup_failure_injected = False
        self.discard_attempts: list[str] = []
        self.staged_artifacts: list[HookedStagedArtifact] = []
        self._stage_count = 0

    @property
    def root(self) -> Path:
        return self.delegate.root

    def stage_bytes(self, key: str, data: bytes) -> StagedArtifact:
        self._stage_count += 1
        staged = self.delegate.stage_bytes(key, data)
        if self.interrupt_during_staging and self._stage_count == 1:
            staged.discard()
            raise KeyboardInterrupt
        hooked = HookedStagedArtifact(staged, self)
        self.staged_artifacts.append(hooked)
        return hooked

    def write_bytes(self, key: str, data: bytes, *, overwrite: bool = False) -> Path:
        if self.fail_rollback:
            raise AppError("output.restore_failed", {"key": key})
        return self.delegate.write_bytes(key, data, overwrite=overwrite)

    def read_bytes(self, key: str) -> bytes:
        return self.delegate.read_bytes(key)

    def exists(self, key: str) -> bool:
        return self.delegate.exists(key)

    def delete(self, key: str) -> None:
        if self.fail_rollback:
            raise AppError("output.delete_failed", {"key": key})
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


def test_run_single_can_publish_all_phase3_subtitle_formats(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "media.wav"
        source.write_bytes(b"input")
        output = tmp_path / "output"
        service = _service(
            source,
            output,
            FakeASRAdapter(transcription_result=make_transcript()),
        )
        service.subtitle_json_serializer = serialize_track_json
        service.webvtt_serializer = serialize_webvtt
        service.ass_serializer = serialize_ass
        result = await service.run(RunSingleRequest(source, output, "en", False))
        assert result.subtitle_json_path is not None and result.subtitle_json_path.is_file()
        assert result.vtt_path is not None and result.vtt_path.is_file()
        assert result.ass_path is not None and result.ass_path.is_file()
        assert sorted(path.name for path in output.iterdir()) == [
            "media.ass",
            "media.srt",
            "media.subtitle.json",
            "media.transcript.json",
            "media.vtt",
        ]

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
            store_factory=lambda root: HookedStore(root, hook="srt-failure"),
        )
        with pytest.raises(AppError, match="write_failed"):
            await service.run(RunSingleRequest(source, output, "en", False))
        assert not (output / "media.transcript.json").exists()
        assert not (output / "media.srt").exists()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "hook",
    [
        "before-transcript",
        "after-transcript",
        "before-srt",
        "after-srt",
        "after-both",
        "discard-cancel",
    ],
)
def test_commit_boundary_cancellation_rolls_back_new_outputs(tmp_path: Path, hook: str) -> None:
    async def scenario() -> None:
        source = tmp_path / "media.wav"
        source.write_bytes(b"input")
        output = tmp_path / "output"
        context = ExecutionContext()
        service = _service(
            source,
            output,
            FakeASRAdapter(transcription_result=make_transcript()),
            store_factory=lambda root: HookedStore(root, context=context, hook=hook),
        )
        with pytest.raises(AppError, match=r"operation\.cancelled"):
            await service.run(RunSingleRequest(source, output, "en", False), context=context)
        assert not list(output.glob("*.srt"))
        assert not list(output.glob("*.transcript.json"))
        assert not list(output.rglob(".*.tmp"))

    asyncio.run(scenario())


@pytest.mark.parametrize("hook", ["after-transcript", "after-srt"])
def test_commit_boundary_cancellation_restores_overwritten_outputs(
    tmp_path: Path, hook: str
) -> None:
    async def scenario() -> None:
        source = tmp_path / "media.wav"
        source.write_bytes(b"input")
        output = tmp_path / "output"
        output.mkdir()
        old_transcript = b"old transcript bytes"
        old_srt = b"old srt bytes\n"
        (output / "media.transcript.json").write_bytes(old_transcript)
        (output / "media.srt").write_bytes(old_srt)
        context = ExecutionContext()
        service = _service(
            source,
            output,
            FakeASRAdapter(transcription_result=make_transcript()),
            store_factory=lambda root: HookedStore(root, context=context, hook=hook),
        )
        with pytest.raises(AppError, match=r"operation\.cancelled"):
            await service.run(RunSingleRequest(source, output, "en", True), context=context)
        assert (output / "media.transcript.json").read_bytes() == old_transcript
        assert (output / "media.srt").read_bytes() == old_srt
        assert not list(output.rglob(".*.tmp"))

    asyncio.run(scenario())


@pytest.mark.parametrize("hook", ["keyboard-transcript", "keyboard-srt"])
def test_keyboard_interrupt_during_commit_rolls_back_outputs(tmp_path: Path, hook: str) -> None:
    async def scenario() -> None:
        source = tmp_path / "media.wav"
        source.write_bytes(b"input")
        output = tmp_path / "output"
        context = ExecutionContext()
        service = _service(
            source,
            output,
            FakeASRAdapter(transcription_result=make_transcript()),
            store_factory=lambda root: HookedStore(root, context=context, hook=hook),
        )
        with pytest.raises(KeyboardInterrupt):
            await service.run(RunSingleRequest(source, output, "en", False), context=context)
        assert not list(output.glob("*.transcript.json"))
        assert not list(output.glob("*.srt"))
        assert not list(output.rglob(".*.tmp"))

    asyncio.run(scenario())


def test_keyboard_interrupt_during_staging_cleans_staging_files(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "media.wav"
        source.write_bytes(b"input")
        output = tmp_path / "output"
        service = _service(
            source,
            output,
            FakeASRAdapter(transcription_result=make_transcript()),
            store_factory=lambda root: HookedStore(root, interrupt_during_staging=True),
        )
        with pytest.raises(KeyboardInterrupt):
            await service.run(RunSingleRequest(source, output, "en", False))
        assert not list(output.rglob(".*.tmp"))

    asyncio.run(scenario())


def _interrupt_replace_for_target(monkeypatch: pytest.MonkeyPatch, target_name: str) -> None:
    real_replace = os.replace
    interrupted = False

    def replace_then_interrupt(source: Path, target: Path) -> None:
        nonlocal interrupted
        real_replace(source, target)
        if not interrupted and target.name == target_name:
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(local_store_module.os, "replace", replace_then_interrupt)


@pytest.mark.parametrize("overwrite", [False, True], ids=["new", "overwrite"])
def test_transcript_replace_interruption_rolls_back_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, overwrite: bool
) -> None:
    async def scenario() -> None:
        source = tmp_path / "media.wav"
        source.write_bytes(b"input")
        output = tmp_path / "output"
        old_transcript = b"old transcript bytes"
        old_srt = b"old srt bytes\n"
        if overwrite:
            output.mkdir()
            (output / "media.transcript.json").write_bytes(old_transcript)
            (output / "media.srt").write_bytes(old_srt)
        service = _service(
            source,
            output,
            FakeASRAdapter(transcription_result=make_transcript()),
        )
        _interrupt_replace_for_target(monkeypatch, "media.transcript.json")

        with pytest.raises(KeyboardInterrupt):
            await service.run(RunSingleRequest(source, output, "en", overwrite))

        if overwrite:
            assert (output / "media.transcript.json").read_bytes() == old_transcript
            assert (output / "media.srt").read_bytes() == old_srt
        else:
            assert not (output / "media.transcript.json").exists()
            assert not (output / "media.srt").exists()
        assert not list(output.rglob(".*.tmp"))

    asyncio.run(scenario())


@pytest.mark.parametrize("overwrite", [False, True], ids=["new", "overwrite"])
def test_srt_replace_interruption_rolls_back_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, overwrite: bool
) -> None:
    async def scenario() -> None:
        source = tmp_path / "media.wav"
        source.write_bytes(b"input")
        output = tmp_path / "output"
        old_transcript = b"old transcript bytes"
        old_srt = b"old srt bytes\n"
        if overwrite:
            output.mkdir()
            (output / "media.transcript.json").write_bytes(old_transcript)
            (output / "media.srt").write_bytes(old_srt)
        service = _service(
            source,
            output,
            FakeASRAdapter(transcription_result=make_transcript()),
        )
        _interrupt_replace_for_target(monkeypatch, "media.srt")

        with pytest.raises(KeyboardInterrupt):
            await service.run(RunSingleRequest(source, output, "en", overwrite))

        if overwrite:
            assert (output / "media.transcript.json").read_bytes() == old_transcript
            assert (output / "media.srt").read_bytes() == old_srt
        else:
            assert not (output / "media.transcript.json").exists()
            assert not (output / "media.srt").exists()
        assert not list(output.rglob(".*.tmp"))

    asyncio.run(scenario())


def test_discard_all_attempts_every_staged_artifact(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "media.wav"
        source.write_bytes(b"input")
        output = tmp_path / "output"
        stores: list[HookedStore] = []

        def make_store(root: Path) -> ArtifactStorePort:
            store = HookedStore(root, hook="before-transcript", fail_cleanup_once=True)
            stores.append(store)
            return store

        service = _service(
            source,
            output,
            FakeASRAdapter(transcription_result=make_transcript()),
            store_factory=make_store,
        )
        with pytest.raises(AppError, match=r"output\.rollback_failed"):
            await service.run(RunSingleRequest(source, output, "en", False))

        store = stores[0]
        assert store.discard_attempts[:2] == ["media.srt", "media.transcript.json"]
        for artifact in store.staged_artifacts:
            artifact.discard()
        assert not list(output.rglob(".*.tmp"))

    asyncio.run(scenario())


def test_cleanup_failure_without_original_error_is_reported(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "media.wav"
        source.write_bytes(b"input")
        output = tmp_path / "output"
        service = _service(
            source,
            output,
            FakeASRAdapter(transcription_result=make_transcript()),
            store_factory=lambda root: HookedStore(root, fail_cleanup_once=True),
        )

        with pytest.raises(AppError, match=r"output\.cleanup_failed"):
            await service.run(RunSingleRequest(source, output, "en", False))

        assert not list(output.glob("*.transcript.json"))
        assert not list(output.glob("*.srt"))
        assert not list(output.rglob(".*.tmp"))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("hook", "expected_type", "expected_code"),
    [
        ("before-transcript", AppError, "operation.cancelled"),
        ("keyboard-before-transcript", KeyboardInterrupt, None),
        ("cancelled-before-transcript", asyncio.CancelledError, None),
        ("srt-failure", AppError, "output.write_failed"),
    ],
)
def test_cleanup_failure_preserves_original_exception(
    tmp_path: Path,
    hook: str,
    expected_type: type[BaseException],
    expected_code: str | None,
) -> None:
    async def scenario() -> None:
        source = tmp_path / "media.wav"
        source.write_bytes(b"input")
        output = tmp_path / "output"
        stores: list[HookedStore] = []

        def make_store(root: Path) -> ArtifactStorePort:
            store = HookedStore(root, hook=hook, fail_cleanup_once=True)
            stores.append(store)
            return store

        service = _service(
            source,
            output,
            FakeASRAdapter(transcription_result=make_transcript()),
            store_factory=make_store,
        )
        with pytest.raises(AppError, match=r"output\.rollback_failed") as raised:
            await service.run(RunSingleRequest(source, output, "en", False))

        cause = raised.value.__cause__
        assert isinstance(cause, expected_type)
        if expected_code is not None:
            assert isinstance(cause, AppError)
            assert cause.code == expected_code
        assert raised.value.params["reason"] == "output.cleanup_failed"
        store = stores[0]
        for artifact in store.staged_artifacts:
            artifact.discard()
        assert not list(output.rglob(".*.tmp"))

    asyncio.run(scenario())


def test_asyncio_cancellation_during_srt_commit_rolls_back_outputs(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "media.wav"
        source.write_bytes(b"input")
        output = tmp_path / "output"
        service = _service(
            source,
            output,
            FakeASRAdapter(transcription_result=make_transcript()),
            store_factory=lambda root: HookedStore(root, hook="cancelled-srt"),
        )
        with pytest.raises(asyncio.CancelledError):
            await service.run(RunSingleRequest(source, output, "en", False))
        assert not list(output.glob("*.transcript.json"))
        assert not list(output.glob("*.srt"))
        assert not list(output.rglob(".*.tmp"))

    asyncio.run(scenario())


def test_rollback_failure_preserves_original_failure_as_cause(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "media.wav"
        source.write_bytes(b"input")
        output = tmp_path / "output"
        output.mkdir()
        (output / "media.transcript.json").write_bytes(b"old transcript")
        (output / "media.srt").write_bytes(b"old srt")
        context = ExecutionContext()
        service = _service(
            source,
            output,
            FakeASRAdapter(transcription_result=make_transcript()),
            store_factory=lambda root: HookedStore(
                root, context=context, hook="after-transcript", fail_rollback=True
            ),
        )
        with pytest.raises(AppError, match=r"output\.rollback_failed") as raised:
            await service.run(RunSingleRequest(source, output, "en", True), context=context)
        assert isinstance(raised.value.__cause__, AppError)
        assert raised.value.__cause__.code == "operation.cancelled"

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
