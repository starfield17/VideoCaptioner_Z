from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from captioner.core.application.worker_result_validation import validate_worker_result
from captioner.core.domain.errors import AppError
from captioner.core.domain.worker_protocol import ResultDescriptor


def _descriptor(data: bytes = b"abc", **kwargs: object) -> ResultDescriptor:
    size = kwargs.get("size_bytes", len(data))
    sha256 = kwargs.get("sha256", hashlib.sha256(data).hexdigest())
    schema_id = kwargs.get("schema_id", "transcript")
    schema_version = kwargs.get("schema_version", 1)
    if not isinstance(size, int) or not isinstance(sha256, str) or not isinstance(schema_id, str):
        raise TypeError
    if not isinstance(schema_version, int):
        raise TypeError
    return ResultDescriptor(
        relative_path=str(kwargs.get("relative_path", "result.json")),
        size_bytes=size,
        sha256=sha256,
        schema_id=schema_id,
        schema_version=schema_version,
    )


def test_valid_relative_result_passes(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_bytes(b"abc")
    assert (
        validate_worker_result(
            _descriptor(), tmp_path, supported_schema_versions={"transcript": {1}}
        )
        == result
    )


@pytest.mark.parametrize(
    "relative_path", ["/result.json", "../result.json", "nested/../../result.json"]
)
def test_result_descriptor_rejects_absolute_and_traversal_paths(relative_path: str) -> None:
    with pytest.raises(AppError, match=r"worker\.result_descriptor_invalid"):
        _descriptor(relative_path=relative_path)


def test_missing_result_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(AppError, match=r"worker\.result_missing"):
        validate_worker_result(
            _descriptor(), tmp_path, supported_schema_versions={"transcript": {1}}
        )


def test_size_mismatch_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_bytes(b"abc")
    with pytest.raises(AppError, match=r"worker\.result_size_mismatch"):
        validate_worker_result(
            _descriptor(size_bytes=4),
            tmp_path,
            supported_schema_versions={"transcript": {1}},
        )


def test_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_bytes(b"abc")
    with pytest.raises(AppError, match=r"worker\.result_hash_mismatch"):
        validate_worker_result(
            _descriptor(sha256="b" * 64),
            tmp_path,
            supported_schema_versions={"transcript": {1}},
        )


def test_schema_mismatch_is_rejected_before_file_commit(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_bytes(b"abc")
    with pytest.raises(AppError, match=r"worker\.result_schema_unsupported"):
        validate_worker_result(
            _descriptor(schema_id="future", schema_version=9),
            tmp_path,
            supported_schema_versions={"transcript": {1}},
        )
