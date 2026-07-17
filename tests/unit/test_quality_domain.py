from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
from tests.support import make_transcript

from captioner.adapters.persistence.domain_codecs import (
    decode_corrected_transcript,
    decode_terminology,
    encode_corrected_transcript,
    encode_json,
    encode_terminology,
)
from captioner.core.application.anomaly_review import build_reviewed_track, review_report
from captioner.core.application.source_correction import (
    build_corrected_transcript,
    build_terminology_units,
    merge_terminology,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.llm import (
    ReviewResponse,
    SourceCorrectionResponse,
    TerminologyResponse,
)
from captioner.core.domain.subtitle import SubtitleCue, SubtitleTrack, derive_subtitle_track_id
from captioner.core.domain.terminology import Terminology, TerminologyEntry
from captioner.core.domain.transcript import Transcript
from captioner.core.policies.line_breaking import break_lines
from captioner.core.policies.llm_anomalies import (
    AnomalyChunkPlanner,
    detect_anomalies,
)
from captioner.core.policies.llm_chunking import ChunkingConfig, ChunkItem
from captioner.core.policies.segmentation import segment_transcript_dp
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig


class CharacterCounter:
    def count(self, text: str) -> int:
        return len(text)


def _policy() -> SegmentationPolicyConfig:
    return SegmentationPolicyConfig(hard_gap_ms=100, preferred_gap_ms=100)


def _translated_track(
    texts: tuple[str, ...],
) -> tuple[Transcript, SubtitleTrack, SegmentationPolicyConfig]:
    transcript = make_transcript(("hello 10 ", "world"))
    policy = _policy()
    source = segment_transcript_dp(transcript, policy)
    cues = tuple(
        SubtitleCue(
            cue.id,
            cue.start_ms,
            cue.end_ms,
            cue.source_word_ids,
            cue.source_text,
            translated,
            break_lines(translated, policy),
        )
        for cue, translated in zip(source.cues, texts, strict=True)
    )
    return (
        transcript,
        SubtitleTrack(
            derive_subtitle_track_id(transcript.id, "zh-CN", cues, policy.to_mapping()),
            transcript.id,
            "zh-CN",
            cues,
            1,
            policy.signature,
        ),
        policy,
    )


def _terminology() -> Terminology:
    return Terminology(
        "transcript-test",
        "en",
        "zh-CN",
        (
            TerminologyEntry("hello 10", "hello 10", "你好 10", ("word-000001",)),
            TerminologyEntry("world", "world", "世界", ("word-000002",)),
        ),
    )


def test_terminology_merge_is_stable_round_trips_and_rejects_conflicts() -> None:
    transcript = make_transcript(("Alpha ", "Alpha"))
    units = build_terminology_units(transcript)
    responses = (
        TerminologyResponse(
            units[0].id,
            ({"source_term": "Alpha", "target_term": "阿尔法"},),
        ),
    )

    terminology = merge_terminology(transcript, "en", "zh-CN", responses, units)

    assert len(terminology.entries) == 1
    assert terminology.entries[0].source_word_ids == ("word-000001", "word-000002")
    assert decode_terminology(encode_terminology(terminology)) == terminology

    conflicting = (
        TerminologyResponse(
            units[0].id,
            (
                {"source_term": "Alpha", "target_term": "阿尔法"},
                {"source_term": "Alpha", "target_term": "字母甲"},
            ),
        ),
    )
    with pytest.raises(AppError, match=r"llm\.terminology_conflict"):
        merge_terminology(transcript, "en", "zh-CN", conflicting, units)
    with pytest.raises(AppError, match=r"llm\.terminology_units_invalid"):
        merge_terminology(
            transcript,
            "en",
            "zh-CN",
            (TerminologyResponse("word-000001", ()),),
            units,
        )


def test_terminology_rejects_numeric_loss_and_malformed_artifacts() -> None:
    transcript = make_transcript(("10%",))
    with pytest.raises(AppError, match=r"llm\.protected_token_lost"):
        merge_terminology(
            transcript,
            "en",
            "zh-CN",
            (
                TerminologyResponse(
                    "term-unit-000001",
                    ({"source_term": "10%", "target_term": "十"},),
                ),
            ),
        )


def test_sparse_terminology_allows_non_terms_and_uses_token_boundaries() -> None:
    transcript = make_transcript(("the ", "artist"))
    units = build_terminology_units(transcript)
    empty = merge_terminology(
        transcript,
        "en",
        "zh-CN",
        (TerminologyResponse(units[0].id, ()),),
        units,
    )
    assert empty.entries == ()

    with pytest.raises(AppError, match=r"llm\.terminology_invalid"):
        merge_terminology(
            transcript,
            "en",
            "zh-CN",
            (
                TerminologyResponse(
                    units[0].id,
                    ({"source_term": "art", "target_term": "艺术"},),
                ),
            ),
            units,
        )
    with pytest.raises(AppError, match=r"llm\.terminology_invalid"):
        TerminologyEntry("Alpha", "wrong", "阿尔法", ("word-000001",))
    with pytest.raises(AppError, match=r"llm\.terminology_invalid"):
        decode_terminology(
            encode_json(
                {
                    "schema_version": 1,
                    "terminology": {
                        "schema_version": 1,
                        "transcript_id": "transcript-test",
                        "source_language": "en",
                        "target_language": "zh-CN",
                        "entries": "invalid",
                    },
                }
            )
        )


def test_multiword_term_maps_to_consecutive_application_owned_word_ids() -> None:
    transcript = make_transcript(("New ", "York", "is ", "large"))
    units = build_terminology_units(transcript)
    terminology = merge_terminology(
        transcript,
        "en",
        "de",
        (
            TerminologyResponse(
                units[0].id,
                ({"source_term": "New York", "target_term": "New York"},),
            ),
        ),
        units,
    )

    assert terminology.entries[0].source_word_ids == ("word-000001", "word-000002")
    with pytest.raises(AppError, match=r"llm\.terminology_invalid"):
        merge_terminology(
            transcript,
            "en",
            "de",
            (
                TerminologyResponse(
                    units[0].id,
                    ({"source_term": "Yorkshire", "target_term": "Yorkshire"},),
                ),
            ),
            units,
        )


def test_corrected_transcript_round_trip_preserves_original_and_rejects_bad_units() -> None:
    transcript = make_transcript(("hello 10 ", "world"))
    responses = (
        SourceCorrectionResponse("word-000001", "hullo 10"),
        SourceCorrectionResponse("word-000002", "world!"),
    )

    corrected = build_corrected_transcript(transcript, responses)

    assert transcript.words[0].text == "hello 10 "
    assert corrected.corrected_text_by_word_id == {
        "word-000001": "hullo 10",
        "word-000002": "world!",
    }
    assert decode_corrected_transcript(encode_corrected_transcript(corrected)) == corrected

    with pytest.raises(AppError, match=r"llm\.correction_units_invalid"):
        build_corrected_transcript(transcript, responses[:1])
    with pytest.raises(AppError, match=r"llm\.response_invalid"):
        build_corrected_transcript(transcript, (object(), object()))
    with pytest.raises(AppError, match=r"llm\.protected_token_lost"):
        build_corrected_transcript(
            transcript,
            (
                SourceCorrectionResponse("word-000001", "hullo"),
                responses[1],
            ),
        )


def test_corrected_segmentation_changes_only_text_and_rejects_incomplete_mapping() -> None:
    transcript = make_transcript(("hello ", "world"))
    policy = _policy()
    source = segment_transcript_dp(transcript, policy)
    corrected = segment_transcript_dp(
        transcript,
        policy,
        corrected_text_by_word_id={
            "word-000001": "hullo",
            "word-000002": "world!",
        },
    )

    assert [cue.source_text for cue in corrected.cues] == ["hullo", "world!"]
    assert [cue.source_word_ids for cue in corrected.cues] == [
        cue.source_word_ids for cue in source.cues
    ]
    assert [(cue.start_ms, cue.end_ms) for cue in corrected.cues] == [
        (cue.start_ms, cue.end_ms) for cue in source.cues
    ]
    with pytest.raises(AppError, match=r"subtitle\.segmentation_failed"):
        segment_transcript_dp(
            transcript,
            policy,
            corrected_text_by_word_id={"word-000001": "hullo"},
        )


def test_anomaly_detection_selects_language_number_and_terminology_failures() -> None:
    transcript, track, policy = _translated_track(("hello", "世界"))

    anomalies = detect_anomalies(track, transcript, "zh-CN", policy, _terminology())

    assert [anomaly.cue_id for anomaly in anomalies] == ["cue-000001"]
    assert {"wrong_language", "protected_token_loss", "terminology_inconsistent"} <= set(
        anomalies[0].reasons
    )
    assert (
        detect_anomalies(
            _translated_track(("你好 10", "世界"))[1],
            transcript,
            "zh-CN",
            policy,
            _terminology(),
        )
        == ()
    )


def test_review_preserves_mapping_and_rejects_residual_anomalies() -> None:
    transcript, track, policy = _translated_track(("hello", "世界"))
    anomalies = detect_anomalies(track, transcript, "zh-CN", policy, _terminology())

    reviewed = build_reviewed_track(
        track,
        transcript,
        "zh-CN",
        policy,
        anomalies,
        (ReviewResponse("cue-000001", "你好 10"),),
        _terminology(),
    )

    assert reviewed.revision == 2
    assert [(cue.id, cue.start_ms, cue.end_ms, cue.source_word_ids) for cue in reviewed.cues] == [
        (cue.id, cue.start_ms, cue.end_ms, cue.source_word_ids) for cue in track.cues
    ]
    assert review_report(track, anomalies)["input_track_id"] == track.id

    with pytest.raises(AppError, match=r"subtitle\.validation_failed"):
        build_reviewed_track(
            track,
            transcript,
            "zh-CN",
            policy,
            anomalies,
            (ReviewResponse("cue-000001", "问候 10"),),
            _terminology(),
        )
    with pytest.raises(AppError, match=r"llm\.id_mismatch"):
        build_reviewed_track(track, transcript, "zh-CN", policy, anomalies, ())
    with pytest.raises(AppError, match=r"llm\.duplicate_id"):
        build_reviewed_track(
            track,
            transcript,
            "zh-CN",
            policy,
            anomalies,
            (
                ReviewResponse("cue-000001", "你好 10"),
                ReviewResponse("cue-000001", "你好 10"),
            ),
        )


def test_anomaly_chunk_planner_keeps_neighbors_context_only_and_trims_budget() -> None:
    all_items = (
        ChunkItem("cue-000001", "aa", 0, 100),
        ChunkItem("cue-000002", "bb", 100, 200),
        ChunkItem("cue-000003", "cc", 200, 300),
    )
    planner = AnomalyChunkPlanner(CharacterCounter(), all_items)
    config = ChunkingConfig(
        max_items=2,
        max_input_tokens=5,
        context_before_items=1,
        context_after_items=1,
        max_audio_context_duration_ms=250,
    )

    chunk = planner.plan((all_items[1],), config)[0]

    assert chunk.item_ids == ("cue-000002",)
    assert chunk.context_ids == ("cue-000003",)
    assert planner.plan_range((all_items[1],), 0, 1, config).item_ids == ("cue-000002",)


def test_subtitle_cue_allows_empty_anomaly_but_rejects_non_string_translation() -> None:
    cue = SubtitleCue(
        "cue-000001",
        0,
        800,
        ("word-000001",),
        "hello",
        "",
        ("placeholder",),
    )
    assert cue.translated_text == ""
    with pytest.raises(AppError, match=r"subtitle\.invalid"):
        replace(cue, translated_text=cast(str, 1))
