"""Unit tests for input selection DTOs and batch draft validation."""

from __future__ import annotations

import pytest

from captioner.core.application.input_selection import (
    BatchDraft,
    InputPreview,
    InputRejection,
    InputSelectionRequest,
    build_batch_draft,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.stage import PipelineProfile


def test_request_validation_rejects_empty_and_bad_limits() -> None:
    with pytest.raises(AppError, match=r"input\.request_invalid"):
        InputSelectionRequest(entries=())
    with pytest.raises(AppError, match=r"input\.request_invalid"):
        InputSelectionRequest(entries=(" ",))
    with pytest.raises(AppError, match=r"input\.request_invalid"):
        InputSelectionRequest(entries=("/a.wav",), maximum_results=0)


def test_preview_counts_and_duplicates() -> None:
    preview = InputPreview(
        accepted_paths=("/a.wav", "/a.wav", "/b.mp4"),
        rejected=(InputRejection(path="/c.txt", code="input.unsupported"),),
    )
    assert preview.accepted_count == 3
    assert preview.rejected_count == 1
    assert preview.empty is False
    assert preview.accepted_paths.count("/a.wav") == 2


def test_batch_draft_retains_duplicates_and_has_no_api_key() -> None:
    draft = BatchDraft(
        input_paths=("/a.wav", "/a.wav"),
        output_root="/out",
        preset_name="deterministic",
        pipeline_profile=PipelineProfile.DETERMINISTIC,
        model_ref="tiny",
        device="auto",
        compute_type="default",
        source_language=None,
        target_language=None,
        provider_profile="default",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        collision_policy="unique_subdir",
    )
    assert draft.input_paths == ("/a.wav", "/a.wav")
    assert not hasattr(draft, "api_key")
    assert "api_key" not in draft.__dataclass_fields__


def test_deterministic_rejects_target_language() -> None:
    with pytest.raises(AppError, match=r"batch\.draft_invalid"):
        BatchDraft(
            input_paths=("/a.wav",),
            output_root="/out",
            preset_name="deterministic",
            pipeline_profile=PipelineProfile.DETERMINISTIC,
            model_ref="tiny",
            device="auto",
            compute_type="default",
            source_language=None,
            target_language="zh-CN",
            provider_profile="default",
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            collision_policy="unique_subdir",
        )


def test_fast_requires_target_language() -> None:
    with pytest.raises(AppError, match=r"batch\.draft_invalid"):
        BatchDraft(
            input_paths=("/a.wav",),
            output_root="/out",
            preset_name="fast",
            pipeline_profile=PipelineProfile.FAST,
            model_ref="tiny",
            device="auto",
            compute_type="default",
            source_language=None,
            target_language=None,
            provider_profile="default",
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            collision_policy="unique_subdir",
        )


def test_invalid_output_policy() -> None:
    with pytest.raises(AppError, match=r"batch\.draft_invalid"):
        BatchDraft(
            input_paths=("/a.wav",),
            output_root="/out",
            preset_name="deterministic",
            pipeline_profile=PipelineProfile.DETERMINISTIC,
            model_ref="tiny",
            device="auto",
            compute_type="default",
            source_language=None,
            target_language=None,
            provider_profile="default",
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            collision_policy="explode",  # type: ignore[arg-type]
        )


def test_build_batch_draft_uses_accepted_paths_only() -> None:
    preview = InputPreview(
        accepted_paths=("/a.wav", "/b.mp4"),
        rejected=(InputRejection(path="/c.txt", code="input.unsupported"),),
    )
    draft = build_batch_draft(
        preview,
        output_root="/out",
        preset_name="fast",
        pipeline_profile="fast",
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        source_language=None,
        target_language="zh-CN",
        provider_profile="default",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        collision_policy="fail",
    )
    assert draft.input_paths == ("/a.wav", "/b.mp4")
    assert draft.pipeline_profile is PipelineProfile.FAST


def test_build_batch_draft_empty_preview_fails() -> None:
    with pytest.raises(AppError, match=r"batch\.draft_invalid"):
        build_batch_draft(
            InputPreview((), ()),
            output_root="/out",
            preset_name="deterministic",
            pipeline_profile="deterministic",
            model_ref="tiny",
            device="auto",
            compute_type="default",
            source_language=None,
            target_language=None,
            provider_profile="default",
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            collision_policy="unique_subdir",
        )


def test_dtos_are_frozen() -> None:
    preview = InputPreview(("/a.wav",), ())
    with pytest.raises(AttributeError):
        preview.accepted_paths = ()  # type: ignore[misc]
