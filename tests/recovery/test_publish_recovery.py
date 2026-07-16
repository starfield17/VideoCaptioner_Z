from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.recovery.support import config, service

import captioner.adapters.pipeline.stages as stages_module
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
from captioner.core.domain.stage import STAGE_PLAN, StageName, StageProjection
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


@dataclass(slots=True)
class FullPublishedExportStage:
    counts: dict[StageName, int]
    version: str = "export-v2"
    name: StageName = StageName.EXPORT

    async def execute(
        self, request: StageExecutionRequest, context: StageExecutionContext
    ) -> tuple[ProducedArtifact, ...]:
        del request
        context.execution.raise_if_cancelled()
        self.counts[self.name] = self.counts.get(self.name, 0) + 1
        return tuple(
            ProducedArtifact(
                logical_name,
                "application/octet-stream",
                logical_name,
                data=logical_name.encode(),
            )
            for logical_name in (
                "final-transcript.json",
                "final-subtitle.json",
                "final-subtitle.srt",
                "final-subtitle.vtt",
                "final-subtitle.ass",
            )
        )


def _configure_publisher(current: DurablePipelineService) -> None:
    current.runners = {
        **current.runners,
        StageName.EXPORT: PublishedExportStage(),
        StageName.PUBLISH: PublishStage(current.executor.artifact_store, version="publish-v1"),
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
            publication_version="publish-v1",
        )

    current.executor.committed_verifier = verify


def _configure_full_publisher(
    current: DurablePipelineService, counts: dict[StageName, int]
) -> None:
    current.runners = {
        **current.runners,
        StageName.EXPORT: FullPublishedExportStage(counts),
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


def _full_published_batch(
    tmp_path: Path,
) -> tuple[DurablePipelineService, BatchProjection, dict[StageName, int]]:
    counts: dict[StageName, int] = {}
    output = tmp_path / "output"
    output.mkdir()
    current = service(tmp_path, counts)
    _configure_full_publisher(current, counts)
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


@pytest.mark.parametrize("corruption", ["missing", "corrupt"])
def test_publication_receipt_recovery_reruns_publish_only(tmp_path: Path, corruption: str) -> None:
    current, projection, counts = _published_batch(tmp_path)
    receipt_ref = next(
        ref
        for ref in projection.job("job-000001").stage(StageName.PUBLISH).artifacts
        if ref.logical_name == "publication-receipt.json"
    )
    receipt_path = current.executor.artifact_store.resolve(receipt_ref)
    if corruption == "missing":
        receipt_path.unlink()
    else:
        receipt_path.write_bytes(b"corrupt")

    recovered = service(tmp_path, counts)
    _configure_publisher(recovered)
    result = asyncio.run(recovered.resume())
    job = result.job("job-000001")

    assert job.state.value == "succeeded"
    assert all(
        job.stage(stage).attempt == 1 for stage in StageName if stage is not StageName.PUBLISH
    )
    assert job.stage(StageName.PUBLISH).attempt == 2
    assert all(
        event.type not in {"stage.failed", "job.failed"}
        for event in recovered.journal.read_snapshot().events
    )
    new_receipt = next(
        ref
        for ref in job.stage(StageName.PUBLISH).artifacts
        if ref.logical_name == "publication-receipt.json"
    )
    recovered.executor.verify_artifact(new_receipt)
    for ref in job.stage(StageName.EXPORT).artifacts:
        recovered.executor.verify_artifact(ref)
    if corruption == "corrupt":
        assert receipt_path.read_bytes() != b"corrupt"
    assert recovered.read_status().integrity == "valid"


def test_publication_verifier_wraps_target_disappearance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current, projection, _ = _published_batch(tmp_path)
    receipt_ref = projection.job("job-000001").stage(StageName.PUBLISH).artifacts[0]
    receipt = current.executor.artifact_store.read_bytes(receipt_ref)

    def disappear(path: Path) -> str:
        del path
        raise OSError

    monkeypatch.setattr(stages_module, "_sha256", disappear)
    with pytest.raises(AppError, match=r"output\.publication_invalid"):
        verify_publication(
            receipt,
            output_dir=Path(projection.job("job-000001").config.output_dir),
            input_path=Path(projection.job("job-000001").input_path),
            export_refs=projection.job("job-000001").stage(StageName.EXPORT).artifacts,
            publication_version="publish-v1",
        )


def test_publication_verifier_checks_each_target_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current, projection, _ = _published_batch(tmp_path)
    receipt_ref = projection.job("job-000001").stage(StageName.PUBLISH).artifacts[0]
    receipt = current.executor.artifact_store.read_bytes(receipt_ref)
    calls = 0

    def count_hash(path: Path) -> str:
        nonlocal calls
        calls += 1
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(stages_module, "_sha256", count_hash)
    verify_publication(
        receipt,
        output_dir=Path(projection.job("job-000001").config.output_dir),
        input_path=Path(projection.job("job-000001").input_path),
        export_refs=projection.job("job-000001").stage(StageName.EXPORT).artifacts,
        publication_version="publish-v1",
    )
    assert calls == 2


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
            publication_version="publish-v1",
        )


@pytest.mark.parametrize("mutation", ["omitted", "permuted", "extra"])
def test_publish_v2_receipt_requires_exact_ordered_five_target_set(
    tmp_path: Path, mutation: str
) -> None:
    current, projection, _ = _full_published_batch(tmp_path)
    receipt_ref = next(
        ref
        for ref in projection.job("job-000001").stage(StageName.PUBLISH).artifacts
        if ref.logical_name == "publication-receipt.json"
    )
    decoded = decode_publication_receipt(current.executor.artifact_store.read_bytes(receipt_ref))
    targets = decoded.targets
    if mutation == "omitted":
        altered_targets = targets[:-1]
    elif mutation == "permuted":
        altered_targets = tuple(reversed(targets))
    else:
        extra = Path(projection.job("job-000001").config.output_dir) / "input.extra"
        altered_targets = (*targets, PublishedTarget(str(extra.resolve()), "0" * 64, 0, "extra"))
    altered = PublicationReceipt(decoded.output_generation, altered_targets)

    with pytest.raises(AppError, match=r"output\.publication_invalid"):
        verify_publication(
            encode_publication_receipt(altered),
            output_dir=Path(projection.job("job-000001").config.output_dir),
            input_path=Path(projection.job("job-000001").input_path),
            export_refs=projection.job("job-000001").stage(StageName.EXPORT).artifacts,
            publication_version="publish-v2",
        )


def test_publish_v2_rejects_phase2_pair_even_when_refs_are_valid(tmp_path: Path) -> None:
    current, projection, _ = _published_batch(tmp_path)
    receipt_ref = projection.job("job-000001").stage(StageName.PUBLISH).artifacts[0]
    with pytest.raises(AppError, match=r"output\.publication_invalid"):
        verify_publication(
            current.executor.artifact_store.read_bytes(receipt_ref),
            output_dir=Path(projection.job("job-000001").config.output_dir),
            input_path=Path(projection.job("job-000001").input_path),
            export_refs=projection.job("job-000001").stage(StageName.EXPORT).artifacts,
            publication_version="publish-v2",
        )


@pytest.mark.parametrize("mutation", ["generation", "hash", "size"])
def test_publish_v2_rejects_receipt_metadata_mismatch(tmp_path: Path, mutation: str) -> None:
    current, projection, _ = _full_published_batch(tmp_path)
    receipt_ref = next(
        ref
        for ref in projection.job("job-000001").stage(StageName.PUBLISH).artifacts
        if ref.logical_name == "publication-receipt.json"
    )
    decoded = decode_publication_receipt(current.executor.artifact_store.read_bytes(receipt_ref))
    first = decoded.targets[0]
    if mutation == "generation":
        altered = PublicationReceipt("0" * 64, decoded.targets)
    else:
        altered_target = PublishedTarget(
            first.path,
            "0" * 64 if mutation == "hash" else first.sha256,
            first.size_bytes + 1 if mutation == "size" else first.size_bytes,
            first.logical_name,
        )
        altered = PublicationReceipt(
            decoded.output_generation, (altered_target, *decoded.targets[1:])
        )

    with pytest.raises(AppError, match=r"output\.publication_invalid"):
        verify_publication(
            encode_publication_receipt(altered),
            output_dir=Path(projection.job("job-000001").config.output_dir),
            input_path=Path(projection.job("job-000001").input_path),
            export_refs=projection.job("job-000001").stage(StageName.EXPORT).artifacts,
            publication_version="publish-v2",
        )


def test_publication_receipt_constructor_rejects_duplicate_path_and_name() -> None:
    target = PublishedTarget("/tmp/a.srt", "0" * 64, 0, "a")
    with pytest.raises(AppError, match=r"output\.publication_invalid"):
        PublicationReceipt(
            "generation", (target, PublishedTarget(target.path, target.sha256, 0, "b"))
        )
    with pytest.raises(AppError, match=r"output\.publication_invalid"):
        PublicationReceipt(
            "generation", (target, PublishedTarget("/tmp/b.srt", target.sha256, 0, "a"))
        )


@pytest.mark.parametrize(
    "target_name",
    [
        "input.transcript.json",
        "input.subtitle.json",
        "input.srt",
        "input.vtt",
        "input.ass",
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
def test_five_publication_targets_rerun_publish_only(
    tmp_path: Path, target_name: str, mutation: str
) -> None:
    _current, projection, counts = _full_published_batch(tmp_path)
    target = Path(projection.job("job-000001").config.output_dir) / target_name
    original = target.read_bytes()
    if mutation == "missing":
        target.unlink()
    else:
        target.write_bytes(bytes(byte ^ 1 for byte in original))

    recovered = service(tmp_path, counts)
    _configure_full_publisher(recovered, counts)
    result = asyncio.run(recovered.resume())
    job = result.job("job-000001")

    assert job.state.value == "succeeded"
    assert all(job.stage(stage).attempt == 1 for stage in STAGE_PLAN[:-1])
    assert job.stage(StageName.PUBLISH).attempt == 2
    assert target.read_bytes() == original
    assert recovered.read_status().integrity == "valid"


@pytest.mark.parametrize(
    "logical_name",
    [
        "final-subtitle.json",
        "final-subtitle.srt",
        "final-subtitle.vtt",
        "final-subtitle.ass",
    ],
)
@pytest.mark.parametrize("corruption", ["missing", "corrupt"])
def test_each_phase3_export_artifact_reruns_export_and_publish(
    tmp_path: Path, logical_name: str, corruption: str
) -> None:
    current, projection, counts = _full_published_batch(tmp_path)
    bad_ref = next(
        ref
        for ref in projection.job("job-000001").stage(StageName.EXPORT).artifacts
        if ref.logical_name == logical_name
    )
    bad_path = current.executor.artifact_store.resolve(bad_ref)
    if corruption == "missing":
        bad_path.unlink()
    else:
        bad_path.write_bytes(b"corrupt")

    recovered = service(tmp_path, counts)
    _configure_full_publisher(recovered, counts)
    result = asyncio.run(recovered.resume())
    job = result.job("job-000001")

    assert job.state.value == "succeeded"
    assert job.stage(StageName.EXPORT).attempt == 2
    assert job.stage(StageName.PUBLISH).attempt == 2
    assert all(job.stage(stage).attempt == 1 for stage in STAGE_PLAN[:3])
    recovered.executor.verify_artifact(
        next(
            ref for ref in job.stage(StageName.EXPORT).artifacts if ref.logical_name == logical_name
        )
    )
    if bad_path.exists():
        assert bad_path.read_bytes() != b"corrupt"


@pytest.mark.parametrize("corruption", ["missing", "corrupt"])
def test_phase3_export_artifact_corruption_reruns_export_and_publish_only(
    tmp_path: Path, corruption: str
) -> None:
    counts: dict[StageName, int] = {}
    current = service(tmp_path, counts)
    _configure_full_publisher(current, counts)
    output = tmp_path / "output"
    output.mkdir()
    projection = current.create(
        "batch-a", (("job-000001", tmp_path / "input.wav", config(tmp_path, output=output)),)
    )
    projection = asyncio.run(current.run(projection))
    job = projection.job("job-000001")
    bad_ref = next(
        ref
        for ref in job.stage(StageName.EXPORT).artifacts
        if ref.logical_name == "final-subtitle.vtt"
    )
    bad_path = current.executor.artifact_store.resolve(bad_ref)
    if corruption == "missing":
        bad_path.unlink()
    else:
        bad_path.write_bytes(b"corrupt")

    recovered = service(tmp_path, counts)
    _configure_full_publisher(recovered, counts)
    result = asyncio.run(recovered.resume())
    recovered_job = result.job("job-000001")
    assert recovered_job.state.value == "succeeded"
    assert recovered_job.stage(StageName.EXPORT).attempt == 2
    assert recovered_job.stage(StageName.PUBLISH).attempt == 2
    assert all(
        recovered_job.stage(stage).attempt == 1
        for stage in STAGE_PLAN[: STAGE_PLAN.index(StageName.EXPORT)]
    )
    recovered_vtt = next(
        ref
        for ref in recovered_job.stage(StageName.EXPORT).artifacts
        if ref.logical_name == "final-subtitle.vtt"
    )
    recovered.executor.artifact_store.verify(recovered_vtt)
    if bad_path.exists():
        assert bad_path.read_bytes() != b"corrupt"
    assert recovered.read_status().integrity == "valid"
