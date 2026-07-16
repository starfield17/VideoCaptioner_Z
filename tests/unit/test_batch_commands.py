from __future__ import annotations

from pathlib import Path

import pytest

from captioner.bootstrap import build_durable_service, create_job_config, load_batch_config
from captioner.cli.commands import batch
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobState
from captioner.core.domain.stage import StageName
from captioner.infrastructure.app_paths import resolve_app_paths


def test_job_config_composition_uses_stable_model_identity(tmp_path: Path) -> None:
    config = create_job_config(
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        ffmpeg_bin="ffmpeg-custom",
        ffprobe_bin="ffprobe-custom",
        output_dir=tmp_path / "output",
        overwrite=True,
    )
    assert config.model_identity == "faster-whisper:tiny"
    assert config.output_dir == str((tmp_path / "output").resolve())
    assert config.overwrite is True


def test_failed_batch_remains_statusable_resumable_and_cancellable(tmp_path: Path) -> None:
    paths = resolve_app_paths(base_dir=tmp_path / "runtime")
    options = batch.BatchRunOptions(
        inputs=(tmp_path / "missing.wav",),
        output_dir=tmp_path / "output",
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        overwrite=False,
    )
    with pytest.raises(AppError, match=r"media\.input_missing"):
        batch.run(options, paths=paths)
    batch_dirs = list(paths.batches_dir.glob("batch-*"))
    assert len(batch_dirs) == 1
    batch_id = batch_dirs[0].name
    projection = batch.status(batch_id, paths=paths)
    assert projection.jobs[0].state is JobState.FAILED
    assert load_batch_config(batch_id, paths=paths) == projection.jobs[0].config
    with pytest.raises(AppError, match=r"media\.input_missing"):
        batch.resume(batch_id, paths=paths)
    marker = batch.cancel(batch_id, "job-000001", paths=paths)
    assert marker.is_file()
    with pytest.raises(AppError, match=r"retry\.stage_invalid"):
        batch.retry(batch_id, "job-000001", StageName.INSPECT, paths=paths)


def test_projection_payload_has_no_workspace_paths(tmp_path: Path) -> None:
    paths = resolve_app_paths(base_dir=tmp_path / "runtime")
    with pytest.raises(AppError):
        load_batch_config("batch-missing", paths=paths)


def test_fault_injection_requires_explicit_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = resolve_app_paths(base_dir=tmp_path / "runtime")
    monkeypatch.setenv("CAPTIONER_FAULT_POINT", "transcribe:after_journal_commit")
    with pytest.raises(AppError, match=r"fault_injection\.disabled"):
        build_durable_service(
            "batch-a",
            model_ref="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            paths=paths,
        )
    monkeypatch.setenv("CAPTIONER_ENABLE_FAULT_INJECTION", "1")
    monkeypatch.setenv("CAPTIONER_FAULT_POINT", "invalid")
    with pytest.raises(AppError, match=r"fault_injection\.invalid"):
        build_durable_service(
            "batch-a",
            model_ref="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
            paths=paths,
        )
    monkeypatch.setenv("CAPTIONER_FAULT_POINT", "transcribe:after_journal_commit")
    bundle = build_durable_service(
        "batch-a",
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        paths=paths,
    )
    assert bundle.service.executor.fault_injector.__class__.__name__ == "ScriptedFaultInjector"
