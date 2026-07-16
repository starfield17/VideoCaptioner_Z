from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.support import make_audio, make_media, make_transcript

from captioner.adapters.persistence.content_addressed_artifact_store import (
    ContentAddressedArtifactStore,
)
from captioner.adapters.persistence.domain_codecs import (
    encode_audio,
    encode_media,
    encode_track,
    encode_transcript,
)
from captioner.adapters.pipeline.stages import (
    ExportStage,
    InspectStage,
    NormalizeStage,
    PublishStage,
    SegmentStage,
    TranscribeStage,
)
from captioner.core.domain.artifact import ArtifactRef
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.job import JobConfig
from captioner.core.domain.media import AudioArtifact, MediaAsset
from captioner.core.domain.stage import STAGE_PLAN
from captioner.core.domain.transcript import Transcript
from captioner.core.policies.simple_segmentation import SimpleSegmentationConfig, segment_transcript
from captioner.core.ports.asr import ASRCapabilities, TranscriptionRequest
from captioner.core.ports.stage_runner import StageExecutionContext, StageExecutionRequest


@dataclass(slots=True)
class Inspector:
    asset: MediaAsset

    async def inspect(self, source_path: Path, context: ExecutionContext) -> MediaAsset:
        del source_path
        context.raise_if_cancelled()
        return self.asset


@dataclass(slots=True)
class Normalizer:
    async def normalize(
        self, asset: MediaAsset, workspace: Path, context: ExecutionContext
    ) -> AudioArtifact:
        del asset
        context.raise_if_cancelled()
        path = workspace / "normalized.wav"
        path.write_bytes(b"wav")
        return make_audio(path)


@dataclass(slots=True)
class Engine:
    transcript: Transcript

    @property
    def engine_id(self) -> str:
        return "fake"

    @property
    def capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(True, True, True, True, False, None, frozenset({"cpu"}))

    async def transcribe(
        self, request: TranscriptionRequest, context: ExecutionContext
    ) -> Transcript:
        del request
        context.raise_if_cancelled()
        return self.transcript


def _config(tmp_path: Path) -> JobConfig:
    return JobConfig(
        "tiny",
        "faster-whisper:tiny",
        "cpu",
        "int8",
        "en",
        True,
        "ffmpeg",
        "ffprobe",
        {"rate": 16000},
        {"limit": 84},
        str((tmp_path / "output").resolve()),
        False,
        {stage.value: "1" for stage in STAGE_PLAN},
    )


def _request(tmp_path: Path, refs: tuple[ArtifactRef, ...] = ()) -> StageExecutionRequest:
    return StageExecutionRequest(
        "batch-a", "job-000001", (tmp_path / "input.wav").resolve(), _config(tmp_path), tuple(refs)
    )


def _context(tmp_path: Path) -> StageExecutionContext:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return StageExecutionContext(ExecutionContext(), workspace)


def _put(store: ContentAddressedArtifactStore, data: bytes, name: str):
    return store.put_bytes(
        data, kind=name, media_type="application/octet-stream", logical_name=name
    )


def test_all_stage_runners_execute_with_durable_inputs(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    (tmp_path / "output").mkdir()
    media = make_media(tmp_path / "input.wav")
    transcript = make_transcript()
    inspect = asyncio.run(
        InspectStage(Inspector(media)).execute(_request(tmp_path), _context(tmp_path))
    )
    assert inspect[0].logical_name == "media.json"
    media_ref = _put(store, encode_media(media), "media.json")
    normalized = asyncio.run(
        NormalizeStage(Normalizer(), store).execute(
            _request(tmp_path, (media_ref,)), _context(tmp_path)
        )
    )
    assert {item.logical_name for item in normalized} == {"normalized.wav", "normalized-audio.json"}
    wav_ref = _put(store, b"wav", "normalized.wav")
    audio_ref = _put(store, encode_audio(make_audio(tmp_path / "a.wav")), "normalized-audio.json")
    transcribed = asyncio.run(
        TranscribeStage(Engine(transcript), store).execute(
            _request(tmp_path, (wav_ref, audio_ref)), _context(tmp_path)
        )
    )
    assert transcribed[0].data == encode_transcript(transcript)
    transcript_ref = _put(store, encode_transcript(transcript), "transcript.json")
    segmented = asyncio.run(
        SegmentStage(store, SimpleSegmentationConfig()).execute(
            _request(tmp_path, (transcript_ref,)), _context(tmp_path)
        )
    )
    assert segmented[0].logical_name == "subtitle-track.json"
    track = segment_transcript(transcript)
    track_ref = _put(store, encode_track(track), "subtitle-track.json")
    exported = asyncio.run(
        ExportStage(store).execute(
            _request(tmp_path, (transcript_ref, track_ref)), _context(tmp_path)
        )
    )
    final_refs = tuple(_put(store, item.data or b"", item.logical_name) for item in exported)
    published = asyncio.run(
        PublishStage(store).execute(_request(tmp_path, final_refs), _context(tmp_path))
    )
    assert published[0].logical_name == "publication-receipt.json"
    assert (tmp_path / "output" / "input.srt").is_file()
    repeated = asyncio.run(
        PublishStage(store).execute(_request(tmp_path, final_refs), _context(tmp_path))
    )
    assert repeated[0].data == published[0].data


@pytest.mark.parametrize("target_name", ["input.transcript.json", "input.srt"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_publish_replace_then_interrupt_rolls_back_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    overwrite: bool,
) -> None:
    from dataclasses import replace

    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    output = tmp_path / "output"
    output.mkdir()
    if overwrite:
        (output / "input.transcript.json").write_bytes(b"old transcript")
        (output / "input.srt").write_bytes(b"old srt")
    current_config = replace(_config(tmp_path), overwrite=overwrite)
    request = StageExecutionRequest(
        "batch-a",
        "job-000001",
        (tmp_path / "input.wav").resolve(),
        current_config,
        (
            _put(store, b"new transcript", "final-transcript.json"),
            _put(store, b"new srt", "final-subtitle.srt"),
        ),
    )
    real_replace = os.replace
    interrupted = False

    def replace_then_interrupt(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        nonlocal interrupted
        real_replace(source, target)
        if not interrupted and Path(os.fsdecode(target)).name == target_name:
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", replace_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(PublishStage(store).execute(request, _context(tmp_path)))
    if overwrite:
        assert (output / "input.transcript.json").read_bytes() == b"old transcript"
        assert (output / "input.srt").read_bytes() == b"old srt"
    else:
        assert not (output / "input.transcript.json").exists()
        assert not (output / "input.srt").exists()
    assert not list(output.glob("*.tmp"))
