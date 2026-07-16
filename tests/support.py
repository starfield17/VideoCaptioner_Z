from __future__ import annotations

from pathlib import Path

from captioner.core.domain.media import AudioArtifact, MediaAsset
from captioner.core.domain.result import JsonValue
from captioner.core.domain.transcript import Transcript, TranscriptSegment, WordToken
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig

HASH = "a" * 64
POLICY_SIGNATURE = SegmentationPolicyConfig().signature


def make_media(path: Path, *, duration_ms: int = 2_000, audio_stream_index: int = 1) -> MediaAsset:
    return MediaAsset(
        id="media-test",
        source_path=path.resolve(),
        content_hash=HASH,
        duration_ms=duration_ms,
        audio_stream_index=audio_stream_index,
        container="wav",
        metadata={"fixture": True},
    )


def make_audio(path: Path) -> AudioArtifact:
    return AudioArtifact(
        artifact_id="audio-test",
        path=path.resolve(),
        sha256=HASH,
        sample_rate=16_000,
        channels=1,
        duration_ms=1_000,
        codec="pcm_s16le",
    )


def make_transcript(
    texts: tuple[str, ...] = ("hello ", "world"),
    *,
    language: str = "en",
    metadata: dict[str, JsonValue] | None = None,
) -> Transcript:
    words: list[WordToken] = []
    cursor = 0
    for number, text in enumerate(texts, start=1):
        start = cursor
        end = start + 500
        words.append(
            WordToken(
                id=f"word-{number:06d}",
                text=text,
                start_ms=start,
                end_ms=end,
                confidence=0.9,
            )
        )
        cursor = end + 100
    segment = TranscriptSegment(
        id="segment-000001",
        word_ids=tuple(word.id for word in words),
        raw_text="".join(texts).strip(),
        start_ms=words[0].start_ms,
        end_ms=words[-1].end_ms,
        confidence=None,
    )
    return Transcript(
        id="transcript-test",
        language=language,
        words=tuple(words),
        segments=(segment,),
        engine_id="fake-asr",
        model_id="test-model",
        metadata={} if metadata is None else metadata,
    )
