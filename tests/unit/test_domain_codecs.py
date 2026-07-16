from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.support import make_audio, make_media, make_transcript

from captioner.adapters.persistence.domain_codecs import (
    decode_audio,
    decode_json,
    decode_media,
    decode_publication_receipt,
    decode_track,
    decode_transcript,
    encode_audio,
    encode_json,
    encode_media,
    encode_publication_receipt,
    encode_track,
    encode_transcript,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.publication import PublicationReceipt, PublishedTarget
from captioner.core.policies.simple_segmentation import segment_transcript


def test_all_domain_codecs_round_trip(tmp_path: Path) -> None:
    media = make_media(tmp_path / "input.wav")
    audio = make_audio(tmp_path / "normalized.wav")
    transcript = make_transcript(("hello ", "世界"), metadata={"nested": [1, {"ok": True}]})
    track = segment_transcript(transcript)
    assert decode_media(encode_media(media)) == media
    assert decode_audio(encode_audio(audio), path=str(audio.path)) == audio
    assert decode_transcript(encode_transcript(transcript)) == transcript
    assert decode_track(encode_track(track)) == track
    assert encode_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'
    receipt = PublicationReceipt(
        "generation",
        (PublishedTarget(str((tmp_path / "output.srt").resolve()), "a" * 64, 12, "output.srt"),),
    )
    assert decode_publication_receipt(encode_publication_receipt(receipt)) == receipt


def test_publication_receipt_codec_rejects_unknown_fields(tmp_path: Path) -> None:
    receipt = PublicationReceipt(
        "generation",
        (PublishedTarget(str((tmp_path / "output.srt").resolve()), "a" * 64, 12, "output.srt"),),
    )
    document = json.loads(encode_publication_receipt(receipt))
    document["unknown"] = True
    with pytest.raises(AppError, match=r"artifact\.codec_invalid"):
        decode_publication_receipt(encode_json(document))


@pytest.mark.parametrize("data", [b"[]", b"not-json", b"\xff"])
def test_decode_json_rejects_invalid_root_or_encoding(data: bytes) -> None:
    with pytest.raises(AppError, match=r"artifact\.codec_invalid"):
        decode_json(data)


def test_media_codec_rejects_unknown_fields(tmp_path: Path) -> None:
    document = json.loads(encode_media(make_media(tmp_path / "a.wav")))
    document["media"]["unknown"] = True
    with pytest.raises(AppError, match=r"artifact\.codec_invalid"):
        decode_media(encode_json(document))


def test_transcript_and_track_codecs_reject_unknown_fields() -> None:
    transcript = make_transcript()
    transcript_document = json.loads(encode_transcript(transcript))
    transcript_document["transcript"]["unknown"] = True
    with pytest.raises(AppError, match=r"artifact\.codec_invalid"):
        decode_transcript(encode_json(transcript_document))
    track_document = json.loads(encode_track(segment_transcript(transcript)))
    track_document["subtitle_track"]["unknown"] = True
    with pytest.raises(AppError, match=r"artifact\.codec_invalid"):
        decode_track(encode_json(track_document))


def test_transcript_codec_rejects_malformed_collections() -> None:
    document = json.loads(encode_transcript(make_transcript()))
    document["transcript"]["words"] = ["invalid"]
    with pytest.raises(AppError, match=r"artifact\.codec_invalid"):
        decode_transcript(encode_json(document))
