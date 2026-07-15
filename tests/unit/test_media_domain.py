from __future__ import annotations

from pathlib import Path

import pytest
from tests.support import HASH

from captioner.core.domain.errors import AppError
from captioner.core.domain.media import AudioArtifact, MediaAsset
from captioner.core.domain.result import JsonValue


def test_media_asset_and_audio_artifact_are_immutable_and_read_only(tmp_path: Path) -> None:
    asset = MediaAsset(
        id="media-1",
        source_path=(tmp_path / "input.wav").resolve(),
        content_hash=HASH,
        duration_ms=1_000,
        audio_stream_index=0,
        container="wav",
        metadata={"language": "en"},
    )
    assert asset.source_path.is_absolute()
    with pytest.raises(TypeError):
        asset.metadata["new"] = "value"  # type: ignore[index]
    audio = AudioArtifact(
        artifact_id="audio-1",
        path=(tmp_path / "normalized.wav").resolve(),
        sha256=HASH,
        sample_rate=16_000,
        channels=1,
        duration_ms=1_000,
        codec="pcm_s16le",
    )
    assert audio.codec == "pcm_s16le"


@pytest.mark.parametrize(
    ("field", "value"),
    [("duration", 0), ("duration", -1), ("hash", "A" * 64), ("stream", -1)],
)
def test_media_domain_rejects_invalid_values(tmp_path: Path, field: str, value: int | str) -> None:
    kwargs: dict[str, str | Path | int | dict[str, JsonValue]] = {
        "id": "media-1",
        "source_path": (tmp_path / "input.wav").resolve(),
        "content_hash": HASH,
        "duration_ms": 1_000,
        "audio_stream_index": 0,
        "container": "wav",
        "metadata": {},
    }
    if field == "duration":
        kwargs["duration_ms"] = value
    elif field == "hash":
        kwargs["content_hash"] = value
    else:
        kwargs["audio_stream_index"] = value
    with pytest.raises(AppError):
        MediaAsset(**kwargs)  # type: ignore[arg-type]


def test_audio_artifact_rejects_non_phase1_format(tmp_path: Path) -> None:
    with pytest.raises(AppError, match="normalized_audio_invalid"):
        AudioArtifact(
            artifact_id="audio-1",
            path=(tmp_path / "normalized.wav").resolve(),
            sha256=HASH,
            sample_rate=44_100,
            channels=2,
            duration_ms=1_000,
            codec="mp3",
        )


def test_media_domain_rejects_non_integer_milliseconds(tmp_path: Path) -> None:
    with pytest.raises(AppError, match="integer"):
        MediaAsset(
            id="media-1",
            source_path=(tmp_path / "input.wav").resolve(),
            content_hash=HASH,
            duration_ms=1.5,  # type: ignore[arg-type]
            audio_stream_index=0,
            container="wav",
            metadata={},
        )
