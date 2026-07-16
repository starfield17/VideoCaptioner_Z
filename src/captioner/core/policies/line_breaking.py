"""Deterministic two-line display-width balancing."""

from __future__ import annotations

from collections.abc import Sequence

from captioner.core.policies.protected_spans import (
    protected_break_cost,
    punctuation_attachment_cost,
)
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig
from captioner.core.policies.unicode_metrics import (
    grapheme_clusters,
    measure_text,
    normalize_text,
)


def break_lines(text: str, config: SegmentationPolicyConfig) -> tuple[str, ...]:
    value = normalize_text(text)
    if not value:
        return ()
    if config.max_lines == 1:
        return (value,)
    if measure_text(value).display_columns <= config.max_line_width:
        return (value,)
    clusters = grapheme_clusters(value)
    candidates = _candidate_indices(clusters)
    if not candidates:
        return (value,)
    scored: list[tuple[tuple[int, ...], tuple[str, str]]] = []
    for index in candidates:
        left = "".join(clusters[:index]).strip()
        right = "".join(clusters[index:]).strip()
        if not left or not right:
            continue
        left_width = measure_text(left).display_columns
        right_width = measure_text(right).display_columns
        overflow = max(0, left_width - config.max_line_width) + max(
            0, right_width - config.max_line_width
        )
        boundary = len("".join(clusters[:index]))
        protected = protected_break_cost(value, boundary)
        punctuation = punctuation_attachment_cost(value, boundary)
        orphan = int(
            min(measure_text(left).reading_characters, measure_text(right).reading_characters) <= 1
        )
        cost = (
            overflow * config.overflow_penalty,
            protected * config.protected_break_penalty,
            punctuation * config.punctuation_bonus,
            orphan * config.punctuation_bonus,
            max(left_width, right_width),
            abs(left_width - right_width),
            index,
        )
        scored.append((cost, (left, right)))
    if not scored:
        return _fallback_lines(clusters, config.max_line_width)
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def join_rendered_lines(lines: Sequence[str]) -> str:
    """Join canonical lines without inventing spaces in CJK text."""
    if not lines:
        return ""
    result = str(lines[0]).strip()
    for raw_line in lines[1:]:
        line = str(raw_line).strip()
        if not line:
            continue
        if _needs_separator(result[-1:], line[:1]):
            result += " "
        result += line
    return normalize_text(result)


def _candidate_indices(clusters: Sequence[str]) -> tuple[int, ...]:
    indices: list[int] = []
    for index in range(1, len(clusters)):
        left = clusters[index - 1]
        right = clusters[index]
        if (
            left.isspace()
            or right.isspace()
            or (_cjk(left) and _cjk(right))
            or (_cjk(left) != _cjk(right) and not left.isspace() and not right.isspace())
        ):
            indices.append(index)
    return tuple(indices)


def _fallback_lines(clusters: Sequence[str], max_width: int) -> tuple[str, ...]:
    del max_width
    # An indivisible token has no safe semantic break.  Keep the complete
    # grapheme sequence and let validation report the bounded width warning.
    return ("".join(clusters),)


def _cjk(cluster: str) -> bool:
    return any(
        "\u2e80" <= character <= "\u9fff" or "\u3040" <= character <= "\u30ff"
        for character in cluster
    )


def _needs_separator(left: str, right: str) -> bool:
    if not left or not right or _cjk(left) or _cjk(right):
        return False
    return right[0] not in ",.!?;:)]}%\u3001\u3002\uff0c\uff01\uff1f\uff1b\uff1a"
