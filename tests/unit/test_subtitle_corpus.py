from __future__ import annotations

from pathlib import Path

import pytest
import scripts.run_subtitle_corpus as corpus

from captioner.core.policies.segmentation_config import SegmentationPolicyConfig


def test_corpus_runner_requires_an_actual_json_track_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Path("tests/fixtures/transcripts/short_words.json")

    monkeypatch.setattr(corpus, "decode_track", lambda _data: object())
    with pytest.raises(corpus.CorpusError):
        corpus._run_fixture(fixture, SegmentationPolicyConfig())
