"""Pure validation of a rendered subtitle track against its Transcript."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from captioner.core.domain.subtitle import SubtitleTrack, derive_subtitle_track_id
from captioner.core.domain.transcript import Transcript
from captioner.core.policies.line_breaking import join_rendered_lines
from captioner.core.policies.protected_spans import protected_break_cost
from captioner.core.policies.reading_speed import reading_speed
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.unicode_metrics import measure_text, normalize_text


class ValidationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    severity: ValidationSeverity
    cue_id: str | None = None
    word_id: str | None = None
    actual: int | str | None = None
    limit: int | str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity is ValidationSeverity.ERROR for issue in self.issues)


def validate_subtitle_track(
    track: SubtitleTrack,
    transcript: Transcript,
    config: SegmentationPolicyConfig,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    words = {word.id: word for word in transcript.words}
    assigned: list[str] = []
    previous_end = -1
    for cue_number, cue in enumerate(track.cues, start=1):
        if cue.id != f"cue-{cue_number:06d}":
            issues.append(
                ValidationIssue(
                    "subtitle.cue_order_invalid",
                    ValidationSeverity.ERROR,
                    cue.id,
                    actual=cue.id,
                    limit=f"cue-{cue_number:06d}",
                )
            )
        if cue.start_ms < 0 or cue.end_ms <= cue.start_ms:
            issues.append(
                ValidationIssue("subtitle.cue_time_invalid", ValidationSeverity.ERROR, cue.id)
            )
        if cue.start_ms < previous_end:
            issues.append(ValidationIssue("subtitle.cue_overlap", ValidationSeverity.ERROR, cue.id))
        previous_end = max(previous_end, cue.end_ms)
        if len(cue.lines) > config.max_lines:
            issues.append(
                ValidationIssue(
                    "subtitle.line_count_exceeded",
                    ValidationSeverity.ERROR,
                    cue.id,
                    actual=len(cue.lines),
                    limit=config.max_lines,
                )
            )
        for line in cue.lines:
            metrics = measure_text(line)
            if metrics.display_columns > config.max_line_width:
                issues.append(
                    ValidationIssue(
                        "subtitle.line_width_exceeded",
                        ValidationSeverity.WARNING,
                        cue.id,
                        actual=metrics.display_columns,
                        limit=config.max_line_width,
                    )
                )
            if any(ord(character) < 32 and character not in "\t" for character in line):
                issues.append(
                    ValidationIssue("subtitle.control_character", ValidationSeverity.ERROR, cue.id)
                )
        if measure_text(cue.source_text).display_columns > config.max_cue_width:
            issues.append(
                ValidationIssue(
                    "subtitle.line_width_exceeded",
                    ValidationSeverity.WARNING,
                    cue.id,
                    actual=measure_text(cue.source_text).display_columns,
                    limit=config.max_cue_width,
                )
            )
        speed = reading_speed(
            cue.source_text,
            cue.end_ms - cue.start_ms,
            target_cps_milli=config.target_cps_milli,
            max_cps_milli=config.max_cps_milli,
        )
        if speed.status == "error":
            issues.append(
                ValidationIssue(
                    "subtitle.cps_exceeded",
                    ValidationSeverity.WARNING,
                    cue.id,
                    actual=speed.cps_milli,
                    limit=config.max_cps_milli,
                )
            )
        elif speed.status == "warning":
            issues.append(
                ValidationIssue(
                    "subtitle.cps_warning",
                    ValidationSeverity.WARNING,
                    cue.id,
                    actual=speed.cps_milli,
                    limit=config.target_cps_milli,
                )
            )
        normalized_source = normalize_text(cue.source_text)
        if join_rendered_lines(cue.lines) != normalized_source:
            issues.append(
                ValidationIssue("subtitle.text_mismatch", ValidationSeverity.ERROR, cue.id)
            )
        for index, word_id in enumerate(cue.source_word_ids):
            if word_id not in words:
                issues.append(
                    ValidationIssue(
                        "subtitle.word_unknown", ValidationSeverity.ERROR, cue.id, word_id
                    )
                )
            elif word_id in assigned:
                issues.append(
                    ValidationIssue(
                        "subtitle.word_duplicated", ValidationSeverity.ERROR, cue.id, word_id
                    )
                )
            else:
                assigned.append(word_id)
            if index > 0 and cue.source_word_ids[index - 1] == word_id:
                issues.append(
                    ValidationIssue(
                        "subtitle.word_duplicated", ValidationSeverity.ERROR, cue.id, word_id
                    )
                )
        known_ids = tuple(word_id for word_id in cue.source_word_ids if word_id in words)
        expected_text = normalize_text("".join(words[word_id].text for word_id in known_ids))
        if known_ids == cue.source_word_ids and expected_text != normalized_source:
            issues.append(
                ValidationIssue("subtitle.text_mismatch", ValidationSeverity.ERROR, cue.id)
            )
        if len(cue.lines) > 1:
            line_boundary = len(normalize_text(cue.lines[0]))
            if protected_break_cost(normalized_source, line_boundary):
                issues.append(
                    ValidationIssue(
                        "subtitle.protected_span_broken", ValidationSeverity.WARNING, cue.id
                    )
                )
    expected = [word.id for word in transcript.words]
    for word_id in expected:
        if word_id not in assigned:
            issues.append(
                ValidationIssue("subtitle.word_missing", ValidationSeverity.ERROR, word_id=word_id)
            )
    if track.source_transcript_id != transcript.id:
        issues.append(
            ValidationIssue(
                "subtitle.track_id_invalid",
                ValidationSeverity.ERROR,
                actual=track.source_transcript_id,
                limit=transcript.id,
            )
        )
    if track.policy_signature:
        expected_id = derive_subtitle_track_id(
            transcript.id,
            transcript.language,
            track.cues,
            config.to_mapping(),
        )
        if track.id != expected_id:
            issues.append(
                ValidationIssue(
                    "subtitle.track_id_invalid",
                    ValidationSeverity.ERROR,
                    actual=track.id,
                    limit=expected_id,
                )
            )
    return ValidationReport(tuple(issues))
