from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from captioner.core.domain.errors import AppError
from captioner.infrastructure.app_paths import resolve_safe_child


@given(
    st.sampled_from(
        ["../outside", "../../outside", "/tmp/outside", r"C:\\outside", r"job\\..\\x", ".."]
    )
)
def test_generated_hostile_ids_are_rejected(identifier: str) -> None:
    with tempfile.TemporaryDirectory() as directory, pytest.raises(AppError):
        resolve_safe_child(Path(directory) / "batches", identifier, field="batch_id")


@given(st.from_regex(r"[A-Za-z0-9][A-Za-z0-9_-]{0,16}", fullmatch=True))
def test_generated_valid_ids_stay_under_root(identifier: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "batches"
        child = resolve_safe_child(root, identifier, field="job_id")
        assert child.parent == root.resolve()
