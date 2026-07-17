"""Read-only filesystem discovery of durable Batch catalogs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from captioner.adapters.persistence.batch_lease import inspect_batch_lease
from captioner.adapters.persistence.json_manifest_store import JsonManifestStore
from captioner.adapters.persistence.jsonl_journal import JsonlJournal
from captioner.core.domain.errors import AppError
from captioner.core.domain.job import validate_identifier
from captioner.core.domain.journal import replay
from captioner.core.ports.batch_catalog import (
    BatchCatalogEntry,
    BatchCatalogIssue,
    BatchCatalogSnapshot,
    LeaseExecutionState,
)


@dataclass(frozen=True, slots=True)
class FilesystemBatchCatalog:
    batches_dir: Path
    hostname: str | None = None
    pid_is_alive: Callable[[int], bool] | None = None

    def read_snapshot(self) -> BatchCatalogSnapshot:
        root = self.batches_dir
        if not root.exists():
            return BatchCatalogSnapshot((), ())
        if not root.is_dir():
            return BatchCatalogSnapshot(
                (),
                (BatchCatalogIssue(root.name or str(root), "queue.batch_read_failed"),),
            )
        batches: list[BatchCatalogEntry] = []
        issues: list[BatchCatalogIssue] = []
        try:
            children = sorted(
                (
                    child
                    for child in root.iterdir()
                    if child.is_dir() and not child.name.startswith(".")
                ),
                key=lambda path: path.name,
            )
        except OSError:
            return BatchCatalogSnapshot(
                (),
                (BatchCatalogIssue(root.name or str(root), "queue.batch_read_failed"),),
            )
        for child in children:
            try:
                entry = self._read_batch(child)
            except AppError as exc:
                issues.append(BatchCatalogIssue(child.name, _issue_code(exc)))
            except OSError:
                issues.append(BatchCatalogIssue(child.name, "queue.batch_read_failed"))
            else:
                batches.append(entry)
        return BatchCatalogSnapshot(tuple(batches), tuple(issues))

    def _read_batch(self, batch_dir: Path) -> BatchCatalogEntry:
        name = batch_dir.name
        try:
            validate_identifier(name, field="batch_id")
        except AppError as exc:
            raise AppError("queue.batch_name_invalid", {"batch_name": name}) from exc

        journal = JsonlJournal(batch_dir / "journal.jsonl")
        snapshot = journal.read_snapshot()
        if not snapshot.events:
            raise AppError("queue.batch_empty", {"batch_name": name})

        first = snapshot.events[0]
        if first.type != "batch.created" or first.batch_id != name:
            raise AppError("queue.batch_identity_mismatch", {"batch_name": name})

        projection = replay(snapshot.events)
        if projection.batch_id != name:
            raise AppError("queue.batch_identity_mismatch", {"batch_name": name})

        manifest_status = JsonManifestStore(batch_dir / "manifest.json").inspect(projection)
        lease_state = self._inspect_lease(batch_dir / "lease.json")
        batch_cancel_requested, job_cancel_requests = _cancel_markers(
            batch_dir / "control",
            {job.job_id for job in projection.jobs},
        )
        return BatchCatalogEntry(
            batch_id=name,
            created_at_utc=first.timestamp_utc,
            projection=projection,
            journal_tail_status=snapshot.tail_status,
            manifest_status=manifest_status,
            lease_state=lease_state,
            batch_cancel_requested=batch_cancel_requested,
            job_cancel_requests=job_cancel_requests,
        )

    def _inspect_lease(self, path: Path) -> LeaseExecutionState:
        return inspect_batch_lease(
            path,
            hostname=self.hostname,
            pid_is_alive=self.pid_is_alive,
        )


def _cancel_markers(
    control_dir: Path,
    job_ids: set[str],
) -> tuple[bool, frozenset[str]]:
    if not control_dir.is_dir():
        return False, frozenset()
    batch_cancel = False
    job_cancels: set[str] = set()
    try:
        children = list(control_dir.iterdir())
    except OSError as exc:
        raise AppError("queue.batch_read_failed") from exc
    for child in children:
        if not child.is_file() or child.name.startswith("."):
            continue
        if child.name == "cancel-batch":
            batch_cancel = True
            continue
        if not child.name.startswith("cancel-"):
            continue
        job_id = child.name.removeprefix("cancel-")
        if job_id in job_ids:
            job_cancels.add(job_id)
    return batch_cancel, frozenset(job_cancels)


def _issue_code(error: AppError) -> str:
    code = error.code
    if code in {
        "queue.batch_name_invalid",
        "queue.batch_empty",
        "queue.batch_identity_mismatch",
        "queue.batch_read_failed",
        "journal.corrupt",
        "journal.transition_invalid",
        "batch.lease_invalid",
    }:
        return code
    if code == "job.identity_invalid":
        return "queue.batch_name_invalid"
    if code == "journal.empty":
        return "queue.batch_empty"
    if code == "journal.read_failed":
        return "queue.batch_read_failed"
    if code.startswith("manifest."):
        return "manifest.invalid"
    if code.startswith("journal."):
        return "journal.corrupt"
    return "queue.batch_read_failed"


__all__ = ["FilesystemBatchCatalog"]
