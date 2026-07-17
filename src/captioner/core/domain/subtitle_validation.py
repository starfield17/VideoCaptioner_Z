"""Pure validation of source and translated subtitle tracks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from captioner.core.domain.subtitle import SubtitleTrack, derive_subtitle_track_id
from captioner.core.domain.transcript import Transcript
from captioner.core.policies.line_breaking import break_lines, join_rendered_lines
from captioner.core.policies.llm_validation import is_obvious_wrong_language
from captioner.core.policies.protected_spans import protected_break_cost, protected_tokens_preserved
from captioner.core.policies.reading_speed import reading_speed
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.unicode_metrics import join_token_texts, measure_text, normalize_text


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
    target_language: str | None = None,
    corrected_text_by_word_id: Mapping[str, str] | None = None,
) -> ValidationReport:
    """Validate the appropriate source or translated-track contract."""
    if _is_translated(track):
        return validate_translated_track(
            track, transcript, config, target_language, corrected_text_by_word_id
        )
    return validate_source_track(track, transcript, config, corrected_text_by_word_id)


def validate_source_mapping(
    track: SubtitleTrack,
    transcript: Transcript,
    config: SegmentationPolicyConfig,
    corrected_text_by_word_id: Mapping[str, str] | None = None,
) -> ValidationReport:
    """Validate timestamps, cue IDs, and complete source Word assignment."""
    return _validate(
        track,
        transcript,
        config,
        translated=False,
        target_language=None,
        corrected_text_by_word_id=corrected_text_by_word_id,
        check_display=True,
    )


def validate_source_track(
    track: SubtitleTrack,
    transcript: Transcript,
    config: SegmentationPolicyConfig,
    corrected_text_by_word_id: Mapping[str, str] | None = None,
) -> ValidationReport:
    """Validate a deterministic source-language track."""
    return _validate(
        track,
        transcript,
        config,
        translated=False,
        target_language=None,
        corrected_text_by_word_id=corrected_text_by_word_id,
        check_display=True,
    )


def validate_translated_track(
    track: SubtitleTrack,
    transcript: Transcript,
    config: SegmentationPolicyConfig,
    target_language: str | None = None,
    corrected_text_by_word_id: Mapping[str, str] | None = None,
) -> ValidationReport:
    """Validate display text while preserving the original source mapping."""
    expected_language = target_language or track.language
    return _validate(
        track,
        transcript,
        config,
        translated=True,
        target_language=expected_language,
        corrected_text_by_word_id=corrected_text_by_word_id,
        check_display=True,
    )


def validate_translated_mapping(
    track: SubtitleTrack,
    transcript: Transcript,
    config: SegmentationPolicyConfig,
    target_language: str | None = None,
    corrected_text_by_word_id: Mapping[str, str] | None = None,
) -> ValidationReport:
    """Validate translated cue identity and mapping before anomaly review."""
    expected_language = target_language or track.language
    return _validate(
        track,
        transcript,
        config,
        translated=True,
        target_language=expected_language,
        corrected_text_by_word_id=corrected_text_by_word_id,
        check_display=False,
    )


def _validate(
    track: SubtitleTrack,
    transcript: Transcript,
    config: SegmentationPolicyConfig,
    *,
    translated: bool,
    target_language: str | None,
    corrected_text_by_word_id: Mapping[str, str] | None,
    check_display: bool,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    words = {word.id: word for word in transcript.words}
    canonical_words = tuple(
        sorted(transcript.words, key=lambda word: (word.start_ms, word.end_ms, word.id))
    )
    canonical_indexes = {word.id: index for index, word in enumerate(canonical_words)}
    canonical_ids = tuple(word.id for word in canonical_words)
    source_texts = (
        {word.id: word.text for word in canonical_words}
        if corrected_text_by_word_id is None
        else dict(corrected_text_by_word_id)
    )
    if corrected_text_by_word_id is not None:
        missing_source_texts = set(canonical_ids) - set(source_texts)
        extra_source_texts = set(source_texts) - set(canonical_ids)
        if missing_source_texts or extra_source_texts:
            issues.append(
                ValidationIssue(
                    "subtitle.corrected_mapping_invalid",
                    ValidationSeverity.ERROR,
                    actual="|".join(sorted(extra_source_texts)),
                    limit="|".join(sorted(missing_source_texts)),
                )
            )
    full_source = join_token_texts(source_texts.get(word.id, word.text) for word in canonical_words)
    assigned: list[str] = []
    previous_end = -1

    if not track.cues:
        issues.append(ValidationIssue("subtitle.cue_empty", ValidationSeverity.ERROR))
    if translated:
        if track.revision < 1:
            issues.append(
                ValidationIssue(
                    "subtitle.revision_invalid",
                    ValidationSeverity.ERROR,
                    actual=track.revision,
                    limit=1,
                )
            )
        if target_language is None or track.language != target_language:
            issues.append(
                ValidationIssue(
                    "subtitle.target_language_mismatch",
                    ValidationSeverity.ERROR,
                    actual=track.language,
                    limit=target_language,
                )
            )
    else:
        if track.revision != 0:
            issues.append(
                ValidationIssue(
                    "subtitle.revision_invalid",
                    ValidationSeverity.ERROR,
                    actual=track.revision,
                    limit=0,
                )
            )
        if track.language != transcript.language:
            issues.append(
                ValidationIssue(
                    "subtitle.language_mismatch",
                    ValidationSeverity.ERROR,
                    actual=track.language,
                    limit=transcript.language,
                )
            )
    if track.policy_signature != config.signature:
        issues.append(
            ValidationIssue(
                "subtitle.policy_signature_invalid",
                ValidationSeverity.ERROR,
                actual=track.policy_signature,
                limit=config.signature,
            )
        )

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

        display_text = cue.source_text if not translated else cue.translated_text
        if display_text is None:
            issues.append(
                ValidationIssue(
                    "subtitle.translated_text_missing", ValidationSeverity.ERROR, cue.id
                )
            )
            display_text = ""
        else:
            canonical_display = normalize_text(display_text)
            if canonical_display != display_text:
                issues.append(
                    ValidationIssue("subtitle.text_not_canonical", ValidationSeverity.ERROR, cue.id)
                )
        if measure_text(display_text).display_columns > config.max_cue_width:
            issues.append(
                ValidationIssue(
                    "subtitle.line_width_exceeded",
                    ValidationSeverity.WARNING,
                    cue.id,
                    actual=measure_text(display_text).display_columns,
                    limit=config.max_cue_width,
                )
            )
        speed = reading_speed(
            display_text,
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
        original_source = join_token_texts(
            words[word_id].text for word_id in cue.source_word_ids if word_id in words
        )
        protected_source = original_source or normalized_source
        if translated:
            if cue.translated_text is not None and join_rendered_lines(cue.lines) != normalize_text(
                cue.translated_text
            ):
                issues.append(
                    ValidationIssue(
                        "subtitle.translated_text_mismatch",
                        ValidationSeverity.ERROR,
                        cue.id,
                    )
                )
            if cue.translated_text is not None and cue.lines != break_lines(
                cue.translated_text, config
            ):
                issues.append(
                    ValidationIssue("subtitle.lines_not_derived", ValidationSeverity.ERROR, cue.id)
                )
            if (
                check_display
                and target_language is not None
                and cue.translated_text is not None
                and is_obvious_wrong_language(cue.translated_text, target_language)
            ):
                issues.append(
                    ValidationIssue(
                        "subtitle.language_mismatch",
                        ValidationSeverity.ERROR,
                        cue.id,
                        actual=target_language,
                    )
                )
            if (
                cue.translated_text is not None
                and check_display
                and (
                    not protected_tokens_preserved(protected_source, cue.translated_text)
                    or not protected_tokens_preserved(protected_source, normalized_source)
                )
            ):
                issues.append(
                    ValidationIssue(
                        "subtitle.protected_token_lost",
                        ValidationSeverity.ERROR,
                        cue.id,
                    )
                )
        elif join_rendered_lines(cue.lines) != normalized_source:
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
        known_indexes = tuple(canonical_indexes[word_id] for word_id in known_ids)
        if known_indexes and known_indexes != tuple(
            range(known_indexes[0], known_indexes[0] + len(known_indexes))
        ):
            issues.append(
                ValidationIssue(
                    "subtitle.word_span_noncontiguous",
                    ValidationSeverity.ERROR,
                    cue.id,
                    actual=known_ids[0],
                )
            )
        if (
            known_ids == cue.source_word_ids
            and known_indexes
            and (not translated or corrected_text_by_word_id is not None)
        ):
            span_start = known_indexes[0]
            span_end = known_indexes[-1] + 1
            expected_text = join_token_texts(
                source_texts.get(word.id, word.text)
                for word in canonical_words[span_start:span_end]
            )
            if expected_text != normalized_source:
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
        last_index = canonical_indexes.get(cue.source_word_ids[-1])
        if last_index is not None and last_index < len(canonical_words) - 1:
            boundary_text = normalize_text(
                "".join(
                    source_texts.get(word.id, word.text)
                    for word in canonical_words[: last_index + 1]
                )
            )
            if protected_break_cost(full_source, len(boundary_text)):
                issues.append(
                    ValidationIssue(
                        "subtitle.protected_span_broken", ValidationSeverity.WARNING, cue.id
                    )
                )

    if tuple(assigned) != canonical_ids:
        issues.append(
            ValidationIssue(
                "subtitle.word_order_invalid",
                ValidationSeverity.ERROR,
                actual="|".join(assigned),
                limit="|".join(canonical_ids),
            )
        )
    for word_id in canonical_ids:
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
    expected_id = derive_subtitle_track_id(
        transcript.id,
        track.language,
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


def _is_translated(track: SubtitleTrack) -> bool:
    return track.revision > 0 or any(cue.translated_text is not None for cue in track.cues)
