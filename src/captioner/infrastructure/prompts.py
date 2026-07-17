"""Versioned prompt resource loading with content identity."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from captioner.core.domain.errors import AppError


@dataclass(frozen=True, slots=True)
class PromptIdentity:
    prompt_id: str
    prompt_version: str
    content_sha256: str
    content: str

    def __post_init__(self) -> None:
        _validate_component(self.prompt_id, "prompt_id")
        _validate_component(self.prompt_version, "prompt_version")
        content_sha256 = cast(object, self.content_sha256)
        if (
            not isinstance(content_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", content_sha256) is None
        ):
            raise AppError("prompt.identity_invalid", {"field": "content_sha256"})
        content = cast(object, self.content)
        if not isinstance(content, str) or not content.strip():
            raise AppError("prompt.invalid", {"reason": "empty"})
        if hashlib.sha256(self.content.encode("utf-8")).hexdigest() != self.content_sha256:
            raise AppError("prompt.identity_mismatch", {"prompt_id": self.prompt_id})

    def to_dict(self) -> dict[str, str]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "content_sha256": self.content_sha256,
            "content": self.content,
        }


Prompt = PromptIdentity


class PromptLoader:
    def __init__(self, resource_dir: Path) -> None:
        self._resource_dir = resource_dir

    def load(self, prompt_id: str, prompt_version: str = "v1") -> PromptIdentity:
        normalized_id, normalized_version = _normalize_identity(prompt_id, prompt_version)
        if not normalized_id or not normalized_version:
            raise AppError("prompt.invalid", {"reason": "identity"})
        path = self._resource_dir / f"{normalized_id}.{normalized_version}.md"
        resource_dir = self._resource_dir.expanduser().resolve()
        path = path.resolve()
        if path.parent != resource_dir:
            raise AppError("prompt.invalid", {"reason": "path"})
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise AppError(
                "prompt.not_found",
                {"prompt_id": normalized_id, "prompt_version": normalized_version},
            ) from exc
        if not content.strip():
            raise AppError("prompt.invalid", {"reason": "empty"})
        return PromptIdentity(
            normalized_id,
            normalized_version,
            hashlib.sha256(content.encode("utf-8")).hexdigest(),
            content,
        )


def load_prompt(resource_dir: Path, prompt_id: str, prompt_version: str = "v1") -> PromptIdentity:
    return PromptLoader(resource_dir).load(prompt_id, prompt_version)


def _normalize_identity(prompt_id: str, prompt_version: str) -> tuple[str, str]:
    raw_id = cast(object, prompt_id)
    raw_version = cast(object, prompt_version)
    if not isinstance(raw_id, str) or not isinstance(raw_version, str):
        return "", ""
    normalized_id = raw_id.strip()
    normalized_version = raw_version.strip()
    if normalized_id.endswith(".md"):
        normalized_id = normalized_id[:-3]
    if normalized_version == "v1" and "." in normalized_id:
        candidate_id, candidate_version = normalized_id.rsplit(".", maxsplit=1)
        if candidate_version.startswith("v"):
            normalized_id, normalized_version = candidate_id, candidate_version
    if not _valid_component(normalized_id) or not _valid_component(normalized_version):
        return "", ""
    return normalized_id, normalized_version


def _valid_component(value: str) -> bool:
    raw_value = cast(object, value)
    return (
        isinstance(raw_value, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", raw_value) is not None
        and ".." not in raw_value
        and not any(ord(character) < 32 or ord(character) == 127 for character in raw_value)
    )


def _validate_component(value: str, field: str) -> None:
    if not _valid_component(value):
        raise AppError("prompt.identity_invalid", {"field": field})
