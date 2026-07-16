from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.recovery.support import config, service

from captioner.adapters.persistence.domain_codecs import (
    decode_publication_receipt,
    encode_publication_receipt,
)
from captioner.adapters.pipeline.stages import PublishStage, verify_publication
from captioner.adapters.testing.fault_injector import InjectedCrash, ScriptedFaultInjector
from captioner.core.application.durable_pipeline import DurablePipelineService
from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import JobProjection
from captioner.core.domain.publication import PublicationReceipt, PublishedTarget
from captioner.core.domain.stage import StageName, StageProjection
from captioner.core.ports.stage_runner import (
    ProducedArtifact,
    StageExecutionContext,
    StageExecutionRequest,
)


@dataclass(slots=True)
class PublishedExportStage:
    version: str = "export-v1"
    name: StageName = StageName.EXPORT

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        del request
        context.execution.raise_if_cancelled()
        return (
            ProducedArtifact(
                "final-transcript-json",
                "application/json",
                "final-transcript.json",
                data=b"transcript",
            ),
            ProducedArtifact(
                "final-subtitle-srt",
                "application/x-subrip",
                "final-subtitle.srt",
                data=b"subtitle",
            ),
        )


def _configure_publisher(current: DurablePipelineService) -> None:
    current.runners = {
        **current.runners,
        StageName.EXPORT: PublishedExportStage(),
        StageName.PUBLISH: PublishStage(current.executor.artifact_store),
    }

    def verify(job: JobProjection, stage: StageProjection) -> None:
        if stage.name is not StageName.PUBLISH:
            return
        receipt_ref = next(
            ref for ref in stage.artifacts if ref.logical_name == "publication-receipt.json"
        )
        verify_publication(
            current.executor.artifact_store.read_bytes(receipt_ref),
            output_dir=Path(job.config.output_dir),
            input_path=Path(job.input_path),
            export_refs=job.stage(StageName.EXPORT).artifacts,
        )

    current.executor.committed_verifier = verify


def _published_batch(
    tmp_path: Path,
) -> tuple[DurablePipelineService, BatchProjection, dict[StageName, int]]:
    counts: dict[StageName, int] = {}
    output = tmp_path / "output"
    output.mkdir()
    current = service(tmp_path, counts)
    _configure_publisher(current)
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path, output=output)),)
    )
    projection = asyncio.run(current.run(projection))
    return current, projection, counts


def test_publish_commit_crash_does_not_repeat_publication(tmp_path: Path) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts, ScriptedFaultInjector("publish", "after_journal_commit"))
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path)),)
    )
    with pytest.raises(InjectedCrash):
        asyncio.run(current.run(projection))
    asyncio.run(service(tmp_path, counts).resume())
    assert counts[StageName.PUBLISH] == 1


def test_publish_target_corruption_invalidates_publish_only(tmp_path: Path) -> None:
    _, projection, counts = _published_batch(tmp_path)
    target = Path(projection.job("job-000001").config.output_dir) / "input.srt"
    target.write_bytes(b"subtitl")

    recovered = service(tmp_path, counts)
    _configure_publisher(recovered)
    result = asyncio.run(recovered.resume())
    job = result.job("job-000001")

    assert job.state.value == "succeeded"
    assert job.stage(StageName.INSPECT).attempt == 1
    assert job.stage(StageName.EXPORT).attempt == 1
    assert job.stage(StageName.PUBLISH).attempt == 2
    assert target.read_bytes() == b"subtitle"


@pytest.mark.parametrize("target_name", ["input.transcript.json", "input.srt"])
@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
def test_status_reports_invalid_published_target_without_mutation(
    tmp_path: Path, target_name: str, mutation: str
) -> None:
    current, projection, _ = _published_batch(tmp_path)
    output = Path(projection.job("job-000001").config.output_dir)
    target = output / target_name
    original = target.read_bytes()
    if mutation == "missing":
        target.unlink()
    else:
        target.write_bytes(bytes(byte ^ 1 for byte in original))
    journal_path = tmp_path / "batch" / "journal.jsonl"
    manifest_path = tmp_path / "batch" / "manifest.json"
    journal_before = journal_path.read_bytes()
    manifest_before = manifest_path.read_bytes()

    result = current.read_status()

    assert result.integrity == "invalid"
    assert any(
        issue.code == "output.publication_invalid" and issue.logical_name == target_name
        for issue in result.integrity_errors
    )
    assert journal_path.read_bytes() == journal_before
    assert manifest_path.read_bytes() == manifest_before


@pytest.mark.skipif(os.name == "nt", reason="symlink creation may require elevated Windows rights")
def test_status_reports_symlink_published_target_without_mutation(tmp_path: Path) -> None:
    current, projection, _ = _published_batch(tmp_path)
    output = Path(projection.job("job-000001").config.output_dir)
    target = output / "input.srt"
    replacement = tmp_path / "replacement.srt"
    replacement.write_bytes(b"subtitle")
    target.unlink()
    target.symlink_to(replacement)
    journal_path = tmp_path / "batch" / "journal.jsonl"
    manifest_path = tmp_path / "batch" / "manifest.json"
    journal_before = journal_path.read_bytes()
    manifest_before = manifest_path.read_bytes()

    result = current.read_status()

    assert result.integrity == "invalid"
    assert any(issue.code == "output.publication_invalid" for issue in result.integrity_errors)
    assert journal_path.read_bytes() == journal_before
    assert manifest_path.read_bytes() == manifest_before


@pytest.mark.parametrize("corruption", ["missing", "corrupt"])
def test_status_reports_invalid_publication_receipt_without_mutation(
    tmp_path: Path, corruption: str
) -> None:
    current, projection, _ = _published_batch(tmp_path)
    receipt = next(
        ref
        for ref in projection.job("job-000001").stage(StageName.PUBLISH).artifacts
        if ref.logical_name == "publication-receipt.json"
    )
    receipt_path = current.executor.artifact_store.resolve(receipt)
    if corruption == "missing":
        receipt_path.unlink()
    else:
        receipt_path.write_bytes(b"corrupt")
    journal_path = tmp_path / "batch" / "journal.jsonl"
    manifest_path = tmp_path / "batch" / "manifest.json"
    journal_before = journal_path.read_bytes()
    manifest_before = manifest_path.read_bytes()

    result = current.read_status()

    assert result.integrity == "invalid"
    assert any(
        issue.code == f"artifact.{corruption}" and issue.logical_name == "publication-receipt.json"
        for issue in result.integrity_errors
    )
    assert journal_path.read_bytes() == journal_before
    assert manifest_path.read_bytes() == manifest_before


def test_publication_verifier_rejects_altered_receipt_target_metadata(tmp_path: Path) -> None:
    current, projection, _ = _published_batch(tmp_path)
    receipt_ref = projection.job("job-000001").stage(StageName.PUBLISH).artifacts[0]
    receipt = current.executor.artifact_store.read_bytes(receipt_ref)
    decoded = decode_publication_receipt(receipt)
    altered = PublicationReceipt(
        decoded.output_generation,
        (
            PublishedTarget(
                decoded.targets[0].path,
                "0" * 64,
                decoded.targets[0].size_bytes,
                decoded.targets[0].logical_name,
            ),
            decoded.targets[1],
        ),
    )

    with pytest.raises(AppError, match=r"output\.publication_invalid"):
        verify_publication(
            encode_publication_receipt(altered),
            output_dir=Path(projection.job("job-000001").config.output_dir),
            input_path=Path(projection.job("job-000001").input_path),
            export_refs=projection.job("job-000001").stage(StageName.EXPORT).artifacts,
        )
