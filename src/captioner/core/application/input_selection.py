"""Application DTOs for media input discovery and in-memory Batch drafts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from captioner.core.domain.errors import AppError
from captioner.core.domain.stage import PipelineProfile

SUPPORTED_MEDIA_EXTENSIONS = frozenset(
    {
        ".aac",
        ".aiff",
        ".alac",
        ".avi",
        ".flac",
        ".m4a",
        ".m4v",
        ".mka",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".ogg",
        ".opus",
        ".ts",
        ".wav",
        ".webm",
        ".wma",
    }
)

OutputCollisionPolicy = Literal[
    "unique_subdir",
    "fail",
    "overwrite",
]

_OUTPUT_COLLISION_POLICIES = frozenset({"unique_subdir", "fail", "overwrite"})
_DEVICES = frozenset({"auto", "cpu", "cuda"})


@dataclass(frozen=True, slots=True)
class InputSelectionRequest:
    entries: tuple[str, ...]
    recursive: bool = True
    maximum_results: int = 10_000

    def __post_init__(self) -> None:
        if not self.entries:
            raise AppError("input.request_invalid", {"field": "entries"})
        if any(not entry.strip() for entry in self.entries):
            raise AppError("input.request_invalid", {"field": "entries"})
        if type(self.maximum_results) is not int or self.maximum_results < 1:
            raise AppError("input.request_invalid", {"field": "maximum_results"})


@dataclass(frozen=True, slots=True)
class InputRejection:
    path: str
    code: Literal[
        "input.not_found",
        "input.unsupported",
        "input.unreadable",
        "input.directory_unreadable",
        "input.result_limit",
    ]


@dataclass(frozen=True, slots=True)
class InputPreview:
    accepted_paths: tuple[str, ...]
    rejected: tuple[InputRejection, ...]

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_paths)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def empty(self) -> bool:
        return not self.accepted_paths


@dataclass(frozen=True, slots=True)
class BatchDraft:
    input_paths: tuple[str, ...]
    output_root: str
    preset_name: str
    pipeline_profile: PipelineProfile
    model_ref: str
    device: Literal["auto", "cpu", "cuda"]
    compute_type: str
    source_language: str | None
    target_language: str | None
    provider_profile: str
    ffmpeg_bin: str
    ffprobe_bin: str
    collision_policy: OutputCollisionPolicy

    def __post_init__(self) -> None:
        if not self.input_paths:
            raise AppError("batch.draft_invalid", {"field": "input_paths"})
        if any(not path.strip() for path in self.input_paths):
            raise AppError("batch.draft_invalid", {"field": "input_paths"})
        if not self.output_root.strip():
            raise AppError("batch.draft_invalid", {"field": "output_root"})
        if not self.preset_name.strip():
            raise AppError("batch.draft_invalid", {"field": "preset_name"})
        profile = PipelineProfile(self.pipeline_profile)
        if not self.model_ref.strip():
            raise AppError("batch.draft_invalid", {"field": "model_ref"})
        device = str(self.device)
        if device not in _DEVICES:
            raise AppError("batch.draft_invalid", {"field": "device"})
        if not self.compute_type.strip():
            raise AppError("batch.draft_invalid", {"field": "compute_type"})
        if self.source_language is not None and not self.source_language.strip():
            raise AppError("batch.draft_invalid", {"field": "source_language"})
        if profile is PipelineProfile.DETERMINISTIC:
            if self.target_language is not None:
                raise AppError("batch.draft_invalid", {"field": "target_language"})
        elif self.target_language is None or not self.target_language.strip():
            raise AppError("batch.draft_invalid", {"field": "target_language"})
        if not self.provider_profile.strip():
            raise AppError("batch.draft_invalid", {"field": "provider_profile"})
        if not self.ffmpeg_bin.strip():
            raise AppError("batch.draft_invalid", {"field": "ffmpeg_bin"})
        if not self.ffprobe_bin.strip():
            raise AppError("batch.draft_invalid", {"field": "ffprobe_bin"})
        if self.collision_policy not in _OUTPUT_COLLISION_POLICIES:
            raise AppError("batch.draft_invalid", {"field": "collision_policy"})
        object.__setattr__(self, "pipeline_profile", profile)
        object.__setattr__(self, "device", device)  # type: ignore[arg-type]
        object.__setattr__(self, "output_root", self.output_root.strip())
        object.__setattr__(self, "preset_name", self.preset_name.strip())
        object.__setattr__(self, "model_ref", self.model_ref.strip())
        object.__setattr__(self, "compute_type", self.compute_type.strip())
        object.__setattr__(
            self,
            "source_language",
            None if self.source_language is None else self.source_language.strip(),
        )
        object.__setattr__(
            self,
            "target_language",
            None if self.target_language is None else self.target_language.strip(),
        )
        object.__setattr__(self, "provider_profile", self.provider_profile.strip())
        object.__setattr__(self, "ffmpeg_bin", self.ffmpeg_bin.strip())
        object.__setattr__(self, "ffprobe_bin", self.ffprobe_bin.strip())


def build_batch_draft(
    preview: InputPreview,
    *,
    output_root: str,
    preset_name: str,
    pipeline_profile: PipelineProfile | str,
    model_ref: str,
    device: str,
    compute_type: str,
    source_language: str | None,
    target_language: str | None,
    provider_profile: str,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    collision_policy: OutputCollisionPolicy | str,
) -> BatchDraft:
    """Build an immutable draft from accepted preview paths only."""
    if preview.empty:
        raise AppError("batch.draft_invalid", {"field": "input_paths"})
    profile = PipelineProfile(pipeline_profile)
    resolved_target = None if profile is PipelineProfile.DETERMINISTIC else target_language
    return BatchDraft(
        input_paths=preview.accepted_paths,
        output_root=output_root,
        preset_name=preset_name,
        pipeline_profile=profile,
        model_ref=model_ref,
        device=device,  # type: ignore[arg-type]
        compute_type=compute_type,
        source_language=source_language,
        target_language=resolved_target,
        provider_profile=provider_profile,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        collision_policy=collision_policy,  # type: ignore[arg-type]
    )


__all__ = [
    "SUPPORTED_MEDIA_EXTENSIONS",
    "BatchDraft",
    "InputPreview",
    "InputRejection",
    "InputSelectionRequest",
    "OutputCollisionPolicy",
    "build_batch_draft",
]
