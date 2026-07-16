"""Unicode-normalized grapheme and display-width metrics."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import regex
from wcwidth import wcswidth

from captioner.core.domain.errors import AppError


@dataclass(frozen=True, slots=True)
class TextMetrics:
    graphemes: int
    reading_characters: int
    display_columns: int


def normalize_text(text: str) -> str:
    """Return canonical rendered text without changing its semantic content."""
    value = unicodedata.normalize("NFC", text)
    value = value.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    for character in value:
        category = unicodedata.category(character)
        if category == "Cc":
            raise AppError("subtitle.control_character", {"reason": "control"})
    value = regex.sub(r"\s+", " ", value)
    return value.strip()


def grapheme_clusters(text: str) -> tuple[str, ...]:
    value = normalize_text(text)
    return tuple(regex.findall(r"\X", value))


def measure_text(text: str) -> TextMetrics:
    clusters = grapheme_clusters(text)
    visible = tuple(cluster for cluster in clusters if _visible(cluster))
    columns = sum(_cluster_width(cluster) for cluster in visible)
    return TextMetrics(len(clusters), len(visible), columns)


def _visible(cluster: str) -> bool:
    if not cluster or cluster.isspace():
        return False
    return any(unicodedata.category(character) not in {"Mn", "Me", "Cf"} for character in cluster)


def _cluster_width(cluster: str) -> int:
    width = wcswidth(cluster)
    if width < 0:
        raise AppError("subtitle.control_character", {"reason": "width"})
    return width
