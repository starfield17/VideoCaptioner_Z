from __future__ import annotations

import asyncio
from pathlib import Path

from tests.recovery.support import config, service

from captioner.core.application.durable_pipeline import IntegrityIssue
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobProjection
from captioner.core.domain.stage import StageName, StageProjection


def test_cli_status_reports_missing_durable_artifact_without_mutation(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    projection = asyncio.run(current.run(projection))
    ref = projection.job("job-000001").stage(StageName.TRANSCRIBE).artifacts[0]
    current.executor.artifact_store.resolve(ref).unlink()
    journal_path = tmp_path / "batch" / "journal.jsonl"
    manifest_path = tmp_path / "batch" / "manifest.json"
    journal_before = journal_path.read_bytes()
    manifest_before = manifest_path.read_bytes()

    result = current.read_status()

    assert result.projection.state.value == "succeeded"
    assert result.integrity == "invalid"
    assert result.integrity_errors == (
        IntegrityIssue(
            "job-000001", "transcribe", "artifact.missing", "transcribe.bin", ref.sha256
        ),
    )
    assert journal_path.read_bytes() == journal_before
    assert manifest_path.read_bytes() == manifest_before


def test_cli_status_reports_corrupt_published_target_without_mutation(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    projection = asyncio.run(current.run(projection))
    output = Path(projection.job("job-000001").config.output_dir)
    output.mkdir(parents=True)
    (output / "input.transcript.json").write_bytes(b"transcript")
    (output / "input.srt").write_bytes(b"subtitle")

    def verify_target(job: JobProjection, stage: StageProjection) -> None:
        del stage
        target = Path(job.config.output_dir) / "input.srt"
        if target.is_symlink() or not target.is_file() or target.read_bytes() != b"subtitle":
            raise AppError("output.publication_invalid", {"logical_name": "input.srt"})

    current.executor.committed_verifier = verify_target
    (output / "input.srt").write_bytes(b"subtitl!")
    journal_path = tmp_path / "batch" / "journal.jsonl"
    manifest_path = tmp_path / "batch" / "manifest.json"
    journal_before = journal_path.read_bytes()
    manifest_before = manifest_path.read_bytes()

    result = current.read_status()

    assert result.integrity == "invalid"
    assert result.integrity_errors[-1].code == "output.publication_invalid"
    assert result.integrity_errors[-1].logical_name == "input.srt"
    assert journal_path.read_bytes() == journal_before
    assert manifest_path.read_bytes() == manifest_before


def test_status_reports_incomplete_tail_without_repairing_it(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    asyncio.run(current.run(projection))
    journal_path = tmp_path / "batch" / "journal.jsonl"
    journal_path.write_bytes(journal_path.read_bytes() + b"{")
    before = journal_path.read_bytes()

    result = current.read_status()

    assert result.journal_tail_status == "incomplete"
    assert journal_path.read_bytes() == before
