from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import cast

import pytest

from captioner.adapters.persistence.jsonl_journal import JsonlJournal
from captioner.bootstrap import build_durable_service, create_job_config, load_batch_config
from captioner.cli.commands import batch
from captioner.core.application.durable_pipeline import DurablePipelineService
from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobConfig, JobProjection, JobState
from captioner.core.domain.stage import StageName
from captioner.infrastructure.app_paths import resolve_app_paths

batch_private = import_module("captioner.cli.commands.batch")


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
    resumed = batch.resume(batch_id, paths=paths)
    assert resumed.jobs[0].state is JobState.FAILED
    with pytest.raises(AppError, match=r"batch\.cancel_invalid"):
        batch.cancel(batch_id, "job-000001", paths=paths)
    with pytest.raises(AppError, match=r"media\.input_missing"):
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


def test_batch_helpers_cover_override_and_collision_policies(tmp_path: Path) -> None:
    apply_overrides = cast(
        Callable[[JobConfig, batch.ResumeOverrides], JobConfig],
        batch_private._apply_overrides,
    )
    earliest_change = cast(
        Callable[[JobConfig, JobConfig], StageName], batch_private._earliest_change
    )
    validate_collisions = cast(
        Callable[[tuple[Path, ...], Path], None],
        batch_private._validate_output_collisions,
    )
    base = create_job_config(
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        output_dir=tmp_path / "output",
        overwrite=False,
    )
    selected = apply_overrides(
        base,
        batch.ResumeOverrides(model_ref="small", device="cuda", compute_type="float16"),
    )
    assert selected.model_ref == "small"
    assert selected.device == "cuda"
    unchanged_model = apply_overrides(
        base, batch.ResumeOverrides(language="zh-CN", output_dir=tmp_path / "other")
    )
    assert unchanged_model.model_identity == base.model_identity
    assert unchanged_model.language == "zh-CN"
    assert earliest_change(base, replace(base, device="cuda")) is StageName.TRANSCRIBE
    assert earliest_change(base, replace(base, segmentation={"limit": 42})) is StageName.SEGMENT
    assert (
        earliest_change(base, replace(base, output_dir=str(tmp_path / "other")))
        is StageName.PUBLISH
    )
    with pytest.raises(AppError, match=r"batch\.output_collision"):
        validate_collisions(
            (tmp_path / "one" / "news.wav", tmp_path / "two" / "news.mp4"),
            tmp_path / "output",
        )


def test_resume_output_creates_new_directory_and_rejects_file(tmp_path: Path) -> None:
    prepare = cast(Callable[[Path], Path], batch_private._prepare_output_directory)
    missing = tmp_path / "new" / "nested-output"

    assert prepare(missing) == missing.resolve()
    assert missing.is_dir()
    existing = tmp_path / "existing"
    existing.mkdir()
    assert prepare(existing) == existing.resolve()
    regular_file = tmp_path / "output-file"
    regular_file.write_bytes(b"not a directory")

    with pytest.raises(AppError, match=r"output\.directory_failed"):
        prepare(regular_file)


def test_resume_output_creates_directory_before_config_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = resolve_app_paths(base_dir=tmp_path / "runtime")
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"source")
    initial_output = tmp_path / "initial-output"
    initial = create_job_config(
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        output_dir=initial_output,
        overwrite=False,
    )
    bundle = build_durable_service(
        "batch-a",
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        paths=paths,
    )
    bundle.service.create("batch-a", (("job-000001", input_path, initial),))
    new_output = tmp_path / "created" / "output"
    observed_event_counts: list[int] = []
    real_prepare = batch_private._prepare_output_directory
    journal_path = bundle.batch_dir / "journal.jsonl"

    def observe_prepare(path: Path) -> Path:
        observed_event_counts.append(len(JsonlJournal(journal_path).read_snapshot().events))
        return real_prepare(path)

    monkeypatch.setattr(batch_private, "_prepare_output_directory", observe_prepare)

    async def stop_resume(self: DurablePipelineService) -> BatchProjection:
        del self
        raise AppError("test.resume_stop")

    monkeypatch.setattr(DurablePipelineService, "resume", stop_resume)
    with pytest.raises(AppError):
        batch.resume(
            "batch-a",
            paths=paths,
            overrides=batch.ResumeOverrides(output_dir=new_output),
        )

    assert new_output.is_dir()
    assert observed_event_counts == [2]
    assert sum(
        event.type == "batch.config_updated"
        for event in JsonlJournal(journal_path).read_snapshot().events
    ) == 1


def test_resume_output_failure_leaves_journal_unchanged(tmp_path: Path) -> None:
    paths = resolve_app_paths(base_dir=tmp_path / "runtime")
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"source")
    initial = create_job_config(
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        output_dir=tmp_path / "initial-output",
        overwrite=False,
    )
    bundle = build_durable_service(
        "batch-a",
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        paths=paths,
    )
    bundle.service.create("batch-a", (("job-000001", input_path, initial),))
    journal_path = bundle.batch_dir / "journal.jsonl"
    before = journal_path.read_bytes()
    regular_file = tmp_path / "not-a-directory"
    regular_file.write_bytes(b"file")

    with pytest.raises(AppError, match=r"output\.directory_failed"):
        batch.resume(
            "batch-a",
            paths=paths,
            overrides=batch.ResumeOverrides(output_dir=regular_file),
        )

    assert journal_path.read_bytes() == before


def test_projection_payload_reports_stale_execution_and_cancel_markers(tmp_path: Path) -> None:
    paths = resolve_app_paths(base_dir=tmp_path / "runtime")
    job_config = create_job_config(
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        output_dir=tmp_path / "output",
        overwrite=False,
    )
    job = replace(
        JobProjection("job-000001", str((tmp_path / "input.wav").resolve()), job_config),
        state=JobState.RUNNING,
    )
    projection = BatchProjection("batch-a", (job,))
    batch_dir = paths.batches_dir / "batch-a"
    (batch_dir / "control").mkdir(parents=True)
    (batch_dir / "control" / "cancel-job-000001").write_text("cancel\n", encoding="utf-8")
    payload = batch.projection_payload(projection, paths=paths)
    assert payload["cancel_requested"] is True
    assert payload["state"] == "interrupted"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("{", True),
        (json.dumps({"pid": os.getpid(), "hostname": "other"}), False),
        (json.dumps({"pid": os.getpid(), "hostname": ""}), True),
    ],
)
def test_lease_staleness_classification(tmp_path: Path, value: str | None, expected: bool) -> None:
    lease_is_stale = cast(Callable[[Path], bool], batch_private._lease_is_stale)
    path = tmp_path / "lease.json"
    if value is not None:
        path.write_text(value, encoding="utf-8")
    assert lease_is_stale(path) is expected
