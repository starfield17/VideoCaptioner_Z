"""Unit tests for LocalBatchGateway durable creation and markers."""

from __future__ import annotations

from pathlib import Path

import pytest

from captioner.adapters.pipeline.local_batch_gateway import LocalBatchGateway
from captioner.core.application.input_selection import BatchDraft
from captioner.core.domain.errors import AppError
from captioner.core.domain.stage import PipelineProfile
from captioner.infrastructure.app_paths import AppPaths


def _paths(tmp_path: Path) -> AppPaths:
    root = tmp_path / "resources"
    for name in ("i18n", "prompts", "runtime", "tokenizers"):
        (root / name).mkdir(parents=True)
    (root / "i18n" / "en.json").write_text("{}", encoding="utf-8")
    return AppPaths(
        app_name="Captioner",
        resource_root=root,
        i18n_resource_dir=root / "i18n",
        prompt_resource_dir=root / "prompts",
        runtime_manifest_resource_dir=root / "runtime",
        config_dir=tmp_path / "config",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "log",
        temp_dir=tmp_path / "temp",
    )


def _draft(tmp_path: Path, *names: str, policy: str = "unique_subdir") -> BatchDraft:
    paths: list[str] = []
    for name in names:
        media = tmp_path / name
        media.write_bytes(b"audio")
        paths.append(str(media))
    collision = policy if policy in {"unique_subdir", "fail", "overwrite"} else "unique_subdir"
    return BatchDraft(
        input_paths=tuple(paths),
        output_root=str(tmp_path / "out"),
        preset_name="deterministic",
        pipeline_profile=PipelineProfile.DETERMINISTIC,
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        source_language="en",
        target_language=None,
        provider_profile="default",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        collision_policy=collision,  # type: ignore[arg-type]
    )


def test_create_batch_unique_subdir_and_job_ids(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    gateway = LocalBatchGateway(paths)
    draft = _draft(tmp_path, "a.wav", "a.wav")
    created = gateway.create_batch(draft)
    assert created.job_ids == ("job-000001", "job-000002")
    batch_dir = paths.batches_dir / created.batch_id
    assert (batch_dir / "journal.jsonl").is_file()
    assert (batch_dir / "manifest.json").is_file()
    # No execution during creation.
    assert not (batch_dir / "lease.json").exists()


def test_fail_policy_rejects_inter_job_collision(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    gateway = LocalBatchGateway(paths)
    draft = _draft(tmp_path, "clip.wav", "clip.wav", policy="fail")
    with pytest.raises(AppError, match=r"batch\.output_collision"):
        gateway.create_batch(draft)


def test_pause_and_cancel_markers(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    gateway = LocalBatchGateway(paths)
    created = gateway.create_batch(_draft(tmp_path, "a.wav"))
    batch_dir = paths.batches_dir / created.batch_id
    gateway.request_pause(created.batch_id, execution_scheduled=False)
    assert (batch_dir / "control" / "pause-batch").is_file()
    gateway.request_cancel(
        created.batch_id,
        job_id=None,
        execution_scheduled=False,
    )
    assert not (batch_dir / "control" / "pause-batch").exists()
    # Finalized cancel should mark jobs cancelled.
    from captioner.adapters.persistence.jsonl_journal import JsonlJournal
    from captioner.core.domain.journal import replay

    projection = replay(JsonlJournal(batch_dir / "journal.jsonl").read_snapshot().events)
    assert all(job.state.value == "cancelled" for job in projection.jobs)


def test_gui_bootstrap_import_is_light() -> None:
    import subprocess
    import sys

    script = (
        "import captioner.gui_bootstrap as g; "
        "import sys; "
        "banned={'faster_whisper','torch','transformers','openai'}; "
        "loaded=banned.intersection(sys.modules); "
        "assert not loaded, loaded"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_run_again_creates_new_batch(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    gateway = LocalBatchGateway(paths)
    created = gateway.create_batch(_draft(tmp_path, "a.wav"))
    # Mark original job terminal via cancel finalize.
    gateway.request_cancel(created.batch_id, job_id=None, execution_scheduled=False)
    from captioner.adapters.persistence.jsonl_journal import JsonlJournal
    from captioner.core.domain.job import JobState
    from captioner.core.domain.journal import replay

    projection = replay(
        JsonlJournal(paths.batches_dir / created.batch_id / "journal.jsonl").read_snapshot().events
    )
    # Force terminal succeeded path for run-again by rewriting is heavy; use cancelled terminal.
    assert projection.jobs[0].state is JobState.CANCELLED
    again = gateway.create_run_again(created.batch_id, "job-000001")
    assert again.batch_id != created.batch_id
    assert again.job_ids == ("job-000001",)
    new_dir = paths.batches_dir / again.batch_id
    assert (new_dir / "journal.jsonl").is_file()
    # Original batch remains.
    assert (paths.batches_dir / created.batch_id / "journal.jsonl").is_file()


def test_run_again_missing_input_blocked(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    gateway = LocalBatchGateway(paths)
    media = tmp_path / "gone.wav"
    media.write_bytes(b"audio")
    created = gateway.create_batch(
        BatchDraft(
            input_paths=(str(media),),
            output_root=str(tmp_path / "out"),
            preset_name="deterministic",
            pipeline_profile=PipelineProfile.DETERMINISTIC,
            model_ref="tiny",
            device="cpu",
            compute_type="int8",
            source_language="en",
            target_language=None,
            provider_profile="default",
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            collision_policy="unique_subdir",
        )
    )
    gateway.request_cancel(created.batch_id, job_id=None, execution_scheduled=False)
    media.unlink()
    with pytest.raises(AppError, match=r"recovery\.input_missing"):
        gateway.create_run_again(created.batch_id, "job-000001")


def test_job_detail_and_recovery_sources(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    gateway = LocalBatchGateway(paths)
    created = gateway.create_batch(_draft(tmp_path, "a.wav"))
    detail = gateway.read_job_detail_source(created.batch_id, "job-000001")
    assert detail.batch_id == created.batch_id
    assert detail.job_id == "job-000001"
    assert detail.input_exists is True
    assert detail.events
    sources = gateway.read_recovery_sources()
    assert any(source.batch_id == created.batch_id for source in sources)
    gateway.close_shared_runtime()


def test_overwrite_policy_rejects_inter_job_collision(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    gateway = LocalBatchGateway(paths)
    draft = _draft(tmp_path, "same.wav", "same.wav", policy="overwrite")
    with pytest.raises(AppError, match=r"batch\.output_collision"):
        gateway.create_batch(draft)


def test_fail_policy_rejects_existing_outputs(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    gateway = LocalBatchGateway(paths)
    out = tmp_path / "out"
    out.mkdir()
    # Create one input and pre-existing target.
    media = tmp_path / "clip.wav"
    media.write_bytes(b"audio")
    (out / "clip.srt").write_text("x", encoding="utf-8")
    draft = BatchDraft(
        input_paths=(str(media),),
        output_root=str(out),
        preset_name="deterministic",
        pipeline_profile=PipelineProfile.DETERMINISTIC,
        model_ref="tiny",
        device="cpu",
        compute_type="int8",
        source_language="en",
        target_language=None,
        provider_profile="default",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        collision_policy="fail",
    )
    with pytest.raises(AppError, match=r"batch\.output_exists"):
        gateway.create_batch(draft)


def test_execute_and_resume_with_monkeypatched_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    gateway = LocalBatchGateway(paths)
    created = gateway.create_batch(_draft(tmp_path, "a.wav"))
    calls: list[str] = []

    class FakeLease:
        def acquire(self) -> None:
            calls.append("acquire")

        def release(self) -> None:
            calls.append("release")

    class FakeService:
        async def run(self, projection: object) -> object:
            calls.append("run")
            return projection

        async def resume(self) -> object:
            calls.append("resume")
            return object()

        async def retry(self, job_id: str, stage: object) -> object:
            calls.append(f"retry:{job_id}")
            return object()

        def acknowledge_cancel_requests(
            self, projection: object, *, active_job_id: str | None
        ) -> object:
            calls.append("ack")
            return projection

    class FakeBundle:
        def __init__(self) -> None:
            self.service = FakeService()
            self.batch_dir = paths.batches_dir / created.batch_id
            self.runtime = None

        async def close(self) -> None:
            calls.append("close")

    def _lease(_batch_dir: Path) -> FakeLease:
        return FakeLease()

    def _bundle(*_args: object, **_kwargs: object) -> FakeBundle:
        return FakeBundle()

    monkeypatch.setattr("captioner.bootstrap.create_batch_lease", _lease)
    monkeypatch.setattr("captioner.bootstrap.build_durable_service", _bundle)
    gateway.execute_created_batch(created.batch_id)
    assert "run" in calls
    assert "acquire" in calls
    assert "release" in calls
    gateway.request_pause(created.batch_id, execution_scheduled=True)
    gateway.resume_batch(created.batch_id)
    assert "resume" in calls
    gateway.close_shared_runtime()


def test_cancel_job_marker_when_scheduled(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    gateway = LocalBatchGateway(paths)
    created = gateway.create_batch(_draft(tmp_path, "a.wav"))
    gateway.request_cancel(
        created.batch_id,
        job_id="job-000001",
        execution_scheduled=True,
    )
    marker = paths.batches_dir / created.batch_id / "control" / "cancel-job-000001"
    assert marker.is_file()


def test_close_shared_runtime_with_fake_runtime(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    gateway = LocalBatchGateway(paths)
    closed: list[str] = []

    class FakeRuntime:
        async def close(self) -> None:
            closed.append("closed")

    object.__setattr__(gateway, "_shared_runtime", FakeRuntime())
    object.__setattr__(gateway, "_shared_runtime_snapshot", {"provider_profile": "default"})
    gateway.close_shared_runtime()
    assert closed == ["closed"]
    assert object.__getattribute__(gateway, "_shared_runtime") is None


def test_terminal_cancel_rejected(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    gateway = LocalBatchGateway(paths)
    created = gateway.create_batch(_draft(tmp_path, "a.wav"))
    gateway.request_cancel(created.batch_id, job_id=None, execution_scheduled=False)
    with pytest.raises(AppError, match=r"batch\.cancel_invalid"):
        gateway.request_cancel(created.batch_id, job_id=None, execution_scheduled=False)


def test_pause_terminal_rejected(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    gateway = LocalBatchGateway(paths)
    created = gateway.create_batch(_draft(tmp_path, "a.wav"))
    gateway.request_cancel(created.batch_id, job_id=None, execution_scheduled=False)
    with pytest.raises(AppError, match=r"batch\.pause_invalid"):
        gateway.request_pause(created.batch_id, execution_scheduled=False)


def test_missing_input_on_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    gateway = LocalBatchGateway(paths)
    media = tmp_path / "gone.wav"
    media.write_bytes(b"x")
    created = gateway.create_batch(
        BatchDraft(
            input_paths=(str(media),),
            output_root=str(tmp_path / "out"),
            preset_name="deterministic",
            pipeline_profile=PipelineProfile.DETERMINISTIC,
            model_ref="tiny",
            device="cpu",
            compute_type="int8",
            source_language="en",
            target_language=None,
            provider_profile="default",
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            collision_policy="unique_subdir",
        )
    )
    media.unlink()
    with pytest.raises(AppError, match=r"recovery\.input_missing"):
        gateway.resume_batch(created.batch_id)
