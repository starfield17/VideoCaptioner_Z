from __future__ import annotations

import json

import pytest

from captioner.adapters.persistence.domain_codecs import encode_json
from captioner.adapters.subtitles.json_track import parse, serialize
from captioner.core.domain.errors import AppError
from captioner.core.domain.subtitle import SubtitleCue, SubtitleTrack


def test_subtitle_json_round_trip_is_exact_canonical_utf8() -> None:
    track = SubtitleTrack(
        "track-1",
        "transcript-1",
        "en",
        (SubtitleCue("cue-000001", 0, 100, ("word-1",), "世界", None, ("世界",)),),
        0,
        "policy-test",
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
        "policy-test",
    )
    document = json.loads(serialize(track))
    document["subtitle_track"].pop("policy_signature")
    with pytest.raises(AppError, match=r"artifact\.codec_invalid"):
        parse(encode_json(document))
    document = json.loads(serialize(track))
    document["subtitle_track"]["unknown"] = True
    with pytest.raises(AppError, match=r"artifact\.codec_invalid"):
        parse(encode_json(document))
