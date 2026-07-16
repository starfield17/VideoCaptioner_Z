from __future__ import annotations

import json

import pytest
from tests.support import POLICY_SIGNATURE, make_transcript

from captioner.adapters.persistence.domain_codecs import encode_json
from captioner.adapters.subtitles.json_track import parse, serialize
from captioner.core.domain.errors import AppError
from captioner.core.domain.subtitle import SubtitleCue, SubtitleTrack
from captioner.core.policies.simple_segmentation import segment_transcript


def test_subtitle_json_round_trip_is_exact_canonical_utf8() -> None:
    track = SubtitleTrack(
        "track-1",
        "transcript-1",
        "en",
        (SubtitleCue("cue-000001", 0, 100, ("word-1",), "世界", None, ("世界",)),),
        0,
        POLICY_SIGNATURE,
    )
    data = serialize(track)
    assert data.endswith(b"\n")
    assert json.loads(data)["schema_version"] == 2
    assert parse(data) == track
    assert serialize(track) == data


def test_subtitle_json_rejects_missing_and_unknown_fields() -> None:
    track = SubtitleTrack(
        "track-1",
        "transcript-1",
        "en",
        (SubtitleCue("cue-000001", 0, 100, ("word-1",), "one", None, ("one",)),),
        0,
        POLICY_SIGNATURE,
    )
    document = json.loads(serialize(track))
    document["subtitle_track"].pop("policy_signature")
    with pytest.raises(AppError, match=r"artifact\.codec_invalid"):
        parse(encode_json(document))
    document = json.loads(serialize(track))
    document["subtitle_track"]["unknown"] = True
    with pytest.raises(AppError, match=r"artifact\.codec_invalid"):
        parse(encode_json(document))


def test_subtitle_json_rejects_unknown_cue_fields() -> None:
    track = SubtitleTrack(
        "track-1",
        "transcript-1",
        "en",
        (SubtitleCue("cue-000001", 0, 100, ("word-1",), "one", None, ("one",)),),
        0,
        POLICY_SIGNATURE,
    )
    document = json.loads(serialize(track))
    document["subtitle_track"]["cues"][0]["unknown"] = True
    with pytest.raises(AppError, match=r"artifact\.codec_invalid"):
        parse(encode_json(document))


def test_schema2_codec_rejects_empty_policy_signature() -> None:
    track = SubtitleTrack(
        "track-1",
        "transcript-1",
        "en",
        (SubtitleCue("cue-000001", 0, 100, ("word-1",), "one", None, ("one",)),),
        0,
        POLICY_SIGNATURE,
    )
    document = json.loads(serialize(track))
    document["subtitle_track"]["policy_signature"] = ""
    with pytest.raises(AppError, match=r"artifact\.codec_invalid"):
        parse(encode_json(document))


def test_schema1_track_remains_readable_with_explicit_legacy_identity() -> None:
    document = json.loads(serialize(segment_transcript(make_transcript(("one",)))))
    document["schema_version"] = 1
    document["subtitle_track"].pop("policy_signature")
    decoded = parse(encode_json(document))
    assert decoded.policy_signature == "legacy-policy-unknown"


@pytest.mark.parametrize(
    "document",
    [
        b'{"schema_version":2,"subtitle_track":{"id":"track","cues":NaN}}',
        b'{"schema_version":2,"schema_version":2,"subtitle_track":{}}',
    ],
)
def test_subtitle_json_rejects_non_finite_or_duplicate_json(document: bytes) -> None:
    with pytest.raises(AppError, match=r"artifact\.codec_invalid"):
        parse(document)
