"""Application assembly for anomaly-only subtitle review."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from captioner.core.domain.errors import AppError
from captioner.core.domain.llm import ReviewResponse
from captioner.core.domain.subtitle import SubtitleCue, SubtitleTrack, derive_subtitle_track_id
from captioner.core.domain.subtitle_validation import validate_translated_track
from captioner.core.domain.terminology import Terminology
from captioner.core.domain.transcript import Transcript
from captioner.core.policies.line_breaking import break_lines
from captioner.core.policies.llm_anomalies import SubtitleAnomaly, detect_anomalies
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig


def build_reviewed_track(
    track: SubtitleTrack,
    transcript: Transcript,
    target_language: str,
    config: SegmentationPolicyConfig,
    anomalies: Sequence[SubtitleAnomaly],
    responses: Sequence[object],
    terminology: Terminology | None = None,
    corrected_text_by_word_id: Mapping[str, str] | None = None,
) -> SubtitleTrack:
    """Copy reviewed text onto the existing cue/timestamp/mapping structure."""
    anomaly_ids = tuple(anomaly.cue_id for anomaly in anomalies)
    response_by_id: dict[str, ReviewResponse] = {}
    for response in responses:
        if not isinstance(response, ReviewResponse):
            raise AppError("llm.response_invalid", {"reason": "review_type"})
        if response.id in response_by_id:
            raise AppError("llm.duplicate_id", {"id": response.id})
        response_by_id[response.id] = response
    if set(response_by_id) != set(anomaly_ids):
        raise AppError("llm.id_mismatch", {"reason": "review_ids"})
    cues: list[SubtitleCue] = []
    for cue in track.cues:
        translated = (
            response_by_id[cue.id].translated_text
            if cue.id in response_by_id
            else cue.translated_text
        )
        if translated is None:
            raise AppError("subtitle.translated_text_missing", {"cue_id": cue.id})
        lines = break_lines(translated, config)
        if not lines:
            raise AppError("subtitle.translated_text_missing", {"cue_id": cue.id})
        cues.append(
            SubtitleCue(
                cue.id,
                cue.start_ms,
                cue.end_ms,
                cue.source_word_ids,
                cue.source_text,
                translated,
                lines,
            )
        )
    revision = max(1, track.revision + (1 if anomalies else 0))
    reviewed = SubtitleTrack(
        derive_subtitle_track_id(
            transcript.id,
            target_language,
            cues,
            config.to_mapping(),
        ),
        transcript.id,
        target_language,
        tuple(cues),
        revision,
        config.signature,
    )
    report = validate_translated_track(
        reviewed,
        transcript,
        config,
        target_language,
        corrected_text_by_word_id,
    )
    if not report.is_valid:
        first = next(issue for issue in report.issues if issue.severity.value == "error")
        raise AppError("subtitle.validation_failed", {"reason": first.code})
    remaining = detect_anomalies(
        reviewed,
        transcript,
        target_language,
        config,
        terminology,
    )
    if remaining:
        raise AppError(
            "subtitle.validation_failed",
            {"reason": remaining[0].reasons[0], "cue_id": remaining[0].cue_id},
        )
    return reviewed


def review_report(
    track: SubtitleTrack,
    anomalies: Sequence[SubtitleAnomaly],
    terminology: Terminology | None = None,
) -> dict[str, object]:
    """Return a stable report even when no review request was needed."""
    del terminology
    return {
        "schema_version": 1,
        "source_track_id": track.id,
        "anomaly_count": len(anomalies),
        "anomalies": [
            {"cue_id": anomaly.cue_id, "reasons": list(anomaly.reasons)} for anomaly in anomalies
        ],
        "reviewed": True,
        "llm_called": bool(anomalies),
    }
