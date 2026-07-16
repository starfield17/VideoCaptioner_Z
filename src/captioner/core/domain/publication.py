"""Immutable publication receipt for verified user-facing outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from captioner.core.domain.errors import AppError

_SHA256 = re.compile(r"[0-9a-f]{64}")
PUBLICATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PublishedTarget:
    path: str
    sha256: str
    size_bytes: int
    logical_name: str

    def __post_init__(self) -> None:
        target = Path(self.path)
        if (
            not target.is_absolute()
            or str(target.resolve()) != self.path
            or _SHA256.fullmatch(self.sha256) is None
            or self.size_bytes < 0
            or not self.logical_name.strip()
        ):
            raise AppError("output.publication_invalid", {"reason": "target"})


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    output_generation: str
    targets: tuple[PublishedTarget, ...]
    schema_version: int = PUBLICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PUBLICATION_SCHEMA_VERSION or not self.output_generation.strip():
            raise AppError("output.publication_invalid", {"reason": "schema"})
        if not self.targets:
            raise AppError("output.publication_invalid", {"reason": "targets"})
        names = [target.logical_name for target in self.targets]
        paths = [target.path for target in self.targets]
        if len(set(names)) != len(names) or len(set(paths)) != len(paths):
            raise AppError("output.publication_invalid", {"reason": "duplicate_target"})
