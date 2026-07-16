"""Single-input run command boundary."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from captioner.bootstrap import build_run_service
from captioner.core.application.run_single import (
    RunSingleRequest,
    RunSingleResult,
    RunSingleService,
)
from captioner.core.domain.execution import ExecutionContext
from captioner.infrastructure.app_paths import AppPaths


@dataclass(frozen=True, slots=True)
class RunOptions:
    input_path: Path
    output_dir: Path
    model_ref: str
    device: str
    compute_type: str
    language: str | None
    ffmpeg_bin: str
    ffprobe_bin: str
    overwrite: bool


def execute(
    options: RunOptions,
    *,
    paths: AppPaths | None = None,
    service: RunSingleService | None = None,
    context: ExecutionContext | None = None,
) -> RunSingleResult:
    """Run one input synchronously at the CLI boundary."""
    selected_service = (
        build_run_service(
            model_ref=options.model_ref,
            device=options.device,
            compute_type=options.compute_type,
            language=options.language,
            ffmpeg_bin=options.ffmpeg_bin,
            ffprobe_bin=options.ffprobe_bin,
            paths=paths,
        )
        if service is None
        else service
    )
    request = RunSingleRequest(
        input_path=options.input_path,
        output_dir=options.output_dir,
        language=options.language,
        overwrite=options.overwrite,
    )
    return asyncio.run(selected_service.run(request, context=context))
