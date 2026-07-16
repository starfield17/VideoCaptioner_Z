from __future__ import annotations

from pathlib import Path

from captioner.adapters.persistence.local_artifact_store import LocalArtifactStore
from captioner.core.application.output_transaction import commit_output_set
from captioner.core.domain.execution import ExecutionContext


def test_output_transaction_commits_a_deterministic_five_file_set(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "output")
    store.root.mkdir()
    outputs = tuple((f"file-{index}.txt", f"data-{index}".encode()) for index in range(5))
    paths = commit_output_set(store, outputs, overwrite=False, context=ExecutionContext())
    assert paths == tuple(store.root / key for key, _ in outputs)
    assert tuple(path.read_bytes() for path in paths) == tuple(data for _, data in outputs)
