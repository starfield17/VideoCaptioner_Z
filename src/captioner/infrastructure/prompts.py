"""Versioned prompt resource loading with content identity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from captioner.core.domain.errors import AppError


@dataclass(frozen=True, slots=True)
class PromptIdentity:
    prompt_id: str
    prompt_version: str
    content_sha256: str
    content: str

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
    normalized_id = prompt_id.strip()
    normalized_version = prompt_version.strip()
    if normalized_id.endswith(".md"):
        normalized_id = normalized_id[:-3]
    if normalized_version == "v1" and "." in normalized_id:
        candidate_id, candidate_version = normalized_id.rsplit(".", maxsplit=1)
        if candidate_version.startswith("v"):
            normalized_id, normalized_version = candidate_id, candidate_version
    if (
        not normalized_id
        or not normalized_version
        or any(character in normalized_id for character in "/\\")
        or any(character in normalized_version for character in "/\\")
    ):
        return "", ""
    return normalized_id, normalized_version
