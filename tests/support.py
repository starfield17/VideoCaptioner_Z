from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from captioner.core.domain.media import AudioArtifact, MediaAsset
from captioner.core.domain.result import FrozenJsonValue, JsonValue, freeze_json_value
from captioner.core.domain.stage import PipelineProfile
from captioner.core.domain.transcript import Transcript, TranscriptSegment, WordToken
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig

HASH = "a" * 64
POLICY_SIGNATURE = SegmentationPolicyConfig().signature


def llm_snapshot(
    profile: PipelineProfile,
    *,
    target_language: str = "zh-CN",
    source_language: str | None = "en",
    chunk: dict[str, int | None] | None = None,
) -> Mapping[str, FrozenJsonValue]:
    """Build a complete credential-free snapshot for deterministic test doubles."""
    prompts: dict[str, dict[str, str]] = {}
    prompt_ids = {
        PipelineProfile.FAST: ("translate_fast", "repair_structured"),
        PipelineProfile.QUALITY: (
            "terminology",
            "correct_source",
            "translate_quality",
            "review_anomalies",
            "repair_structured",
        ),
    }.get(profile, ())
    for index, prompt_id in enumerate(prompt_ids, start=1):
        prompts[prompt_id] = {
            "prompt_id": prompt_id,
            "prompt_version": "v2" if prompt_id == "terminology" else "v1",
            "content_sha256": f"{index:x}" * 64,
        }
    snapshot: dict[str, object] = {
        "snapshot_schema_version": 1,
        "kind": "openai-compatible",
        "provider_profile": "default",
        "base_url": "https://provider.example/v1",
        "model": "unit-test-model",
        "max_concurrency": 2,
        "request_timeout_sec": 30.0,
        "max_retries": 2,
        "temperature": 0.1,
        "tokenizer": "cl100k_base",
        "profile": profile.value,
        "source_language": source_language,
        "target_language": target_language,
        "chunk": chunk
        or {
            "max_items": 32,
            "max_input_tokens": 4096,
            "context_before_items": 1,
            "context_after_items": 1,
            "max_audio_context_duration_ms": 120_000,
        },
        "prompts": prompts,
        "response_schema_version": 1,
    }
    return cast(Mapping[str, FrozenJsonValue], freeze_json_value(snapshot))


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
