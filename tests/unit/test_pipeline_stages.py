from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.support import make_audio, make_media, make_transcript

import captioner.adapters.pipeline.stages as stages_module
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
from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.domain.job import JobConfig
from captioner.core.domain.media import AudioArtifact, MediaAsset
from captioner.core.domain.stage import STAGE_PLAN
from captioner.core.domain.subtitle import SubtitleCue, SubtitleTrack, derive_subtitle_track_id
from captioner.core.domain.transcript import Transcript
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
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


def _context(tmp_path: Path, hook: Callable[[str], None] | None = None) -> StageExecutionContext:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return StageExecutionContext(ExecutionContext(checkpoint_hook=hook), workspace)


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
    assert {item.logical_name for item in exported} == {
        "final-transcript.json",
        "final-subtitle.json",
        "final-subtitle.srt",
        "final-subtitle.vtt",
        "final-subtitle.ass",
    }
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


def test_segment_mid_execute_occurs_during_segmentation(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    transcript = make_transcript(("one ", "two ", "three"))
    transcript_ref = _put(store, encode_transcript(transcript), "transcript.json")
    checkpoints: list[str] = []
    result = asyncio.run(
        SegmentStage(
            store,
            SimpleSegmentationConfig(max_text_units=5),
        ).execute(
            _request(tmp_path, (transcript_ref,)),
            _context(tmp_path, checkpoints.append),
        )
    )
    assert result[0].data is not None
    assert checkpoints == ["mid_execute"]


def test_export_mid_execute_occurs_between_representations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    transcript = make_transcript()
    transcript_ref = _put(store, encode_transcript(transcript), "transcript.json")
    track_ref = _put(store, encode_track(segment_transcript(transcript)), "subtitle-track.json")
    order: list[str] = []
    real_srt = stages_module.serialize_srt

    def serialize_with_observation(track: object) -> bytes:
        order.append("srt")
        return real_srt(track)  # type: ignore[arg-type]  # test observation wrapper

    monkeypatch.setattr(stages_module, "serialize_srt", serialize_with_observation)
    asyncio.run(
        ExportStage(store).execute(
            _request(tmp_path, (transcript_ref, track_ref)),
            _context(tmp_path, lambda point: order.append(point)),
        )
    )
    assert order == ["mid_execute", "srt"]


def test_export_stage_rejects_reordered_word_assignment(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    transcript = make_transcript(("one ", "two ", "three"))
    config = SegmentationPolicyConfig()
    cues = (
        SubtitleCue("cue-000001", 0, 500, ("word-000002",), "two", None, ("two",)),
        SubtitleCue("cue-000002", 600, 1_100, ("word-000001",), "one", None, ("one",)),
        SubtitleCue("cue-000003", 1_200, 1_700, ("word-000003",), "three", None, ("three",)),
    )
    track = SubtitleTrack(
        derive_subtitle_track_id(transcript.id, transcript.language, cues, config.to_mapping()),
        transcript.id,
        transcript.language,
        cues,
        0,
        config.signature,
    )
    transcript_ref = _put(store, encode_transcript(transcript), "transcript.json")
    track_ref = _put(store, encode_track(track), "subtitle-track.json")
    with pytest.raises(AppError, match=r"subtitle\.validation_failed"):
        asyncio.run(
            ExportStage(store, config).execute(
                _request(tmp_path, (transcript_ref, track_ref)), _context(tmp_path)
            )
        )


def test_publish_mid_execute_occurs_after_first_target_commit(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    output = tmp_path / "output"
    output.mkdir()
    refs = (
        _put(store, b"transcript", "final-transcript.json"),
        _put(store, b"srt", "final-subtitle.srt"),
    )
    checkpoints: list[str] = []
    result = asyncio.run(
        PublishStage(store, version="publish-v1").execute(
            _request(tmp_path, refs), _context(tmp_path, checkpoints.append)
        )
    )
    assert result[0].data is not None
    assert checkpoints == ["mid_execute"]


@pytest.mark.parametrize(
    "target_name",
    [
        "input.transcript.json",
        "input.subtitle.json",
        "input.srt",
        "input.vtt",
        "input.ass",
    ],
)
@pytest.mark.parametrize("overwrite", [False, True])
def test_publish_replace_then_interrupt_rolls_back_five_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    overwrite: bool,
) -> None:
    from dataclasses import replace

    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    output = tmp_path / "output"
    output.mkdir()
    old_bytes = {
        "input.transcript.json": b"old transcript",
        "input.subtitle.json": b"old subtitle",
        "input.srt": b"old srt",
        "input.vtt": b"old vtt",
        "input.ass": b"old ass",
    }
    if overwrite:
        for name, data in old_bytes.items():
            (output / name).write_bytes(data)
    current_config = replace(_config(tmp_path), overwrite=overwrite)
    request = StageExecutionRequest(
        "batch-a",
        "job-000001",
        (tmp_path / "input.wav").resolve(),
        current_config,
        (
            _put(store, b"new transcript", "final-transcript.json"),
            _put(store, b"new subtitle", "final-subtitle.json"),
            _put(store, b"new srt", "final-subtitle.srt"),
            _put(store, b"new vtt", "final-subtitle.vtt"),
            _put(store, b"new ass", "final-subtitle.ass"),
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
    for name, data in old_bytes.items():
        if overwrite:
            assert (output / name).read_bytes() == data
        else:
            assert not (output / name).exists()
    assert not list(output.glob("*.tmp"))
