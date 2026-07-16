from __future__ import annotations

from pathlib import Path

import pytest

from captioner.cli.commands import batch
from captioner.core.domain.errors import AppError
from captioner.infrastructure.app_paths import resolve_app_paths, resolve_safe_child

HOSTILE_IDS = (
    "../../outside",
    "../batch",
    "/tmp/absolute",
    r"C:\outside",
    "C:relative",
    r"\\server\share",
    "job/../../outside",
    r"job\..\outside",
    ".",
    "..",
    "",
    "   ",
    "batch\noutside",
)


@pytest.mark.parametrize("identifier", HOSTILE_IDS)
def test_hostile_identifier_never_resolves_outside_root(tmp_path: Path, identifier: str) -> None:
    root = tmp_path / "batches"
    with pytest.raises(AppError, match=r"job\.identity_invalid"):
        resolve_safe_child(root, identifier, field="batch_id")
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("identifier", HOSTILE_IDS)
def test_status_and_cancel_do_not_touch_external_paths(tmp_path: Path, identifier: str) -> None:
    paths = resolve_app_paths(base_dir=tmp_path / "runtime")
    external = tmp_path / "outside" / "journal.jsonl"
    external.parent.mkdir()
    external.write_bytes(b"unterminated")
    with pytest.raises(AppError):
        batch.status(identifier, paths=paths)
    with pytest.raises(AppError):
        batch.cancel(identifier, None, paths=paths)
    assert external.read_bytes() == b"unterminated"
    assert not list(tmp_path.rglob("cancel-batch"))


def test_valid_generated_identifiers_resolve_directly_below_root(tmp_path: Path) -> None:
    root = tmp_path / "batches"
    child = resolve_safe_child(root, "batch-0123abcd", field="batch_id")
    assert child == root.resolve() / "batch-0123abcd"
