"""Unit tests for the read-only filesystem Batch catalog adapter."""

from __future__ import annotations

import json
from pathlib import Path

from captioner.adapters.persistence.filesystem_batch_catalog import FilesystemBatchCatalog
from captioner.adapters.persistence.json_manifest_store import JsonManifestStore
from captioner.adapters.persistence.jsonl_journal import JsonlJournal
from captioner.core.domain.job import JobConfig
from captioner.core.domain.journal import JournalEvent, replay
from captioner.core.domain.stage import PipelineProfile, stage_plan_for


def _config(tmp_path: Path) -> JobConfig:
    plan = stage_plan_for(PipelineProfile.DETERMINISTIC)
    return JobConfig(
        model_ref="tiny",
        model_identity="faster-whisper:tiny",
        device="cpu",
        compute_type="int8",
        language="en",
        vad_filter=True,
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        normalization={"sample_rate": 16000, "channels": 1},
        segmentation={"max_duration_ms": 7000},
        output_dir=str((tmp_path / "output").resolve()),
        overwrite=False,
        stage_versions={stage.value: "1" for stage in plan},
    )


def _event(
    seq: int,
    event_type: str,
    payload: dict[str, object],
    *,
    batch_id: str = "batch-a",
    timestamp_utc: str = "2026-01-01T00:00:00+00:00",
) -> JournalEvent:
    return JournalEvent(
        seq=seq,
        event_id=f"event-{seq:06d}",
        timestamp_utc=timestamp_utc,
        batch_id=batch_id,
        type=event_type,
        payload=payload,  # type: ignore[arg-type]
    )


def _write_batch(
    batches_dir: Path,
    batch_id: str,
    *,
    job_ids: tuple[str, ...] = ("job-000001",),
    timestamp_utc: str = "2026-01-01T00:00:00+00:00",
    tmp_path: Path | None = None,
) -> Path:
    root = batches_dir / batch_id
    root.mkdir(parents=True, exist_ok=True)
    config_root = tmp_path if tmp_path is not None else batches_dir.parent
    journal = JsonlJournal(root / "journal.jsonl")
    events = [_event(1, "batch.created", {}, batch_id=batch_id, timestamp_utc=timestamp_utc)]
    for index, job_id in enumerate(job_ids):
        events.append(
            _event(
                index + 2,
                "job.created",
                {
                    "job_id": job_id,
                    "input_path": str((config_root / f"{job_id}.wav").resolve()),
                    "config": _config(config_root).to_dict(),
                },
                batch_id=batch_id,
                timestamp_utc=timestamp_utc,
            )
        )
    journal.append_many(events)
    return root


def _write_lease(
    path: Path,
    *,
    token: str = "token-a",
    pid: int = 10,
    hostname: str = "host-a",
    created_timestamp: str = "2026-01-01T00:00:00Z",
) -> None:
    path.write_text(
        json.dumps(
            {
                "token": token,
                "pid": pid,
                "hostname": hostname,
                "created_timestamp": created_timestamp,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def test_missing_root_returns_empty_and_does_not_create(tmp_path: Path) -> None:
    missing = tmp_path / "batches"
    snapshot = FilesystemBatchCatalog(missing).read_snapshot()
    assert snapshot.batches == ()
    assert snapshot.issues == ()
    assert not missing.exists()


def test_valid_batch(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    root = _write_batch(
        batches,
        "batch-a",
        job_ids=("job-000001", "job-000002"),
        timestamp_utc="2026-02-01T12:00:00+00:00",
        tmp_path=tmp_path,
    )
    projection = replay(JsonlJournal(root / "journal.jsonl").read_snapshot().events)
    JsonManifestStore(root / "manifest.json").write(projection)
    snapshot = FilesystemBatchCatalog(batches).read_snapshot()
    assert len(snapshot.batches) == 1
    assert snapshot.issues == ()
    entry = snapshot.batches[0]
    assert entry.batch_id == "batch-a"
    assert entry.created_at_utc == "2026-02-01T12:00:00+00:00"
    assert entry.projection == projection
    assert entry.journal_tail_status == "clean"
    assert entry.manifest_status == "current"
    assert [job.job_id for job in entry.projection.jobs] == ["job-000001", "job-000002"]


def test_deterministic_directory_order(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    for batch_id in ("batch-c", "batch-a", "batch-b"):
        _write_batch(batches, batch_id, tmp_path=tmp_path)
    snapshot = FilesystemBatchCatalog(batches).read_snapshot()
    assert [entry.batch_id for entry in snapshot.batches] == [
        "batch-a",
        "batch-b",
        "batch-c",
    ]


def test_corrupt_batch_isolation(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    _write_batch(batches, "batch-good", tmp_path=tmp_path)
    bad = batches / "batch-bad"
    bad.mkdir(parents=True)
    (bad / "journal.jsonl").write_text("{not-json\n", encoding="utf-8")
    snapshot = FilesystemBatchCatalog(batches).read_snapshot()
    assert [entry.batch_id for entry in snapshot.batches] == ["batch-good"]
    assert len(snapshot.issues) == 1
    assert snapshot.issues[0].batch_name == "batch-bad"
    assert snapshot.issues[0].code == "journal.corrupt"
    assert "not-json" not in snapshot.issues[0].code


def test_incomplete_journal_tail_is_preserved(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    root = _write_batch(batches, "batch-a", tmp_path=tmp_path)
    journal_path = root / "journal.jsonl"
    before = journal_path.read_bytes()
    journal_path.write_bytes(before + b'{"seq":99,"partial"')
    after_write = journal_path.read_bytes()
    snapshot = FilesystemBatchCatalog(batches).read_snapshot()
    assert len(snapshot.batches) == 1
    entry = snapshot.batches[0]
    assert entry.journal_tail_status == "incomplete"
    assert len(entry.projection.jobs) == 1
    assert journal_path.read_bytes() == after_write


def test_identity_mismatch_is_issue(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    root = batches / "batch-dir"
    root.mkdir(parents=True)
    journal = JsonlJournal(root / "journal.jsonl")
    journal.append(_event(1, "batch.created", {}, batch_id="batch-other"))
    snapshot = FilesystemBatchCatalog(batches).read_snapshot()
    assert snapshot.batches == ()
    assert len(snapshot.issues) == 1
    assert snapshot.issues[0].batch_name == "batch-dir"
    assert snapshot.issues[0].code == "queue.batch_identity_mismatch"


def test_invalid_directory_name_is_issue(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    (batches / "bad name").mkdir(parents=True)
    _write_batch(batches, "batch-good", tmp_path=tmp_path)
    snapshot = FilesystemBatchCatalog(batches).read_snapshot()
    assert [entry.batch_id for entry in snapshot.batches] == ["batch-good"]
    issue = next(item for item in snapshot.issues if item.batch_name == "bad name")
    assert issue.code == "queue.batch_name_invalid"


def test_manifest_states(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    missing_root = _write_batch(batches, "batch-missing", tmp_path=tmp_path)
    current_root = _write_batch(batches, "batch-current", tmp_path=tmp_path)
    stale_root = _write_batch(batches, "batch-stale", tmp_path=tmp_path)
    invalid_root = _write_batch(batches, "batch-invalid", tmp_path=tmp_path)

    current_projection = replay(JsonlJournal(current_root / "journal.jsonl").read_snapshot().events)
    JsonManifestStore(current_root / "manifest.json").write(current_projection)

    stale_projection = replay(JsonlJournal(stale_root / "journal.jsonl").read_snapshot().events)
    JsonManifestStore(stale_root / "manifest.json").write(stale_projection)
    stale_document = json.loads((stale_root / "manifest.json").read_text(encoding="utf-8"))
    stale_document["last_event_seq"] = 0
    (stale_root / "manifest.json").write_text(
        json.dumps(stale_document, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    (invalid_root / "manifest.json").write_text("{not-json", encoding="utf-8")
    assert not (missing_root / "manifest.json").exists()

    snapshot = FilesystemBatchCatalog(batches).read_snapshot()
    by_id = {entry.batch_id: entry.manifest_status for entry in snapshot.batches}
    assert by_id["batch-missing"] == "missing"
    assert by_id["batch-current"] == "current"
    assert by_id["batch-stale"] == "stale"
    assert by_id["batch-invalid"] == "invalid"


def test_cancel_markers(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    root = _write_batch(
        batches,
        "batch-a",
        job_ids=("job-000001", "job-000002"),
        tmp_path=tmp_path,
    )
    control = root / "control"
    control.mkdir()
    (control / "cancel-batch").write_text("1\n", encoding="utf-8")
    (control / "cancel-job-000002").write_text("1\n", encoding="utf-8")
    (control / ".cancel-job-000001.tmp").write_text("1\n", encoding="utf-8")
    (control / "cancel-unknown-job").write_text("1\n", encoding="utf-8")
    (control / "random-file").write_text("1\n", encoding="utf-8")
    snapshot = FilesystemBatchCatalog(batches).read_snapshot()
    entry = snapshot.batches[0]
    assert entry.batch_cancel_requested is True
    assert entry.job_cancel_requests == frozenset({"job-000002"})


def test_lease_states(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    _write_batch(batches, "batch-missing", tmp_path=tmp_path)
    local_root = _write_batch(batches, "batch-local", tmp_path=tmp_path)
    remote_root = _write_batch(batches, "batch-remote", tmp_path=tmp_path)
    stale_root = _write_batch(batches, "batch-stale", tmp_path=tmp_path)
    invalid_root = _write_batch(batches, "batch-invalid", tmp_path=tmp_path)
    _write_lease(local_root / "lease.json", pid=10, hostname="host-a")
    _write_lease(remote_root / "lease.json", pid=10, hostname="other-host")
    _write_lease(stale_root / "lease.json", pid=11, hostname="host-a")
    (invalid_root / "lease.json").write_text("{", encoding="utf-8")

    def pid_is_alive(pid: int) -> bool:
        return pid == 10

    snapshot = FilesystemBatchCatalog(
        batches,
        hostname="host-a",
        pid_is_alive=pid_is_alive,
    ).read_snapshot()
    by_id = {entry.batch_id: entry.lease_state for entry in snapshot.batches}
    assert by_id["batch-missing"] == "missing"
    assert by_id["batch-local"] == "active_local"
    assert by_id["batch-remote"] == "active_remote"
    assert by_id["batch-stale"] == "stale"
    assert by_id["batch-invalid"] == "invalid"


def test_empty_batch_directory_is_issue(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    (batches / "batch-empty").mkdir(parents=True)
    snapshot = FilesystemBatchCatalog(batches).read_snapshot()
    assert snapshot.batches == ()
    assert snapshot.issues[0].batch_name == "batch-empty"
    assert snapshot.issues[0].code == "queue.batch_empty"


def test_catalog_does_not_mutate_journal_or_manifest(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    root = _write_batch(batches, "batch-a", tmp_path=tmp_path)
    journal_before = (root / "journal.jsonl").read_bytes()
    manifest_path = root / "manifest.json"
    assert not manifest_path.exists()
    FilesystemBatchCatalog(batches).read_snapshot()
    assert (root / "journal.jsonl").read_bytes() == journal_before
    assert not manifest_path.exists()
    assert not (root / "lease.json").exists()
