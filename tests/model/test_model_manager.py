from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from captioner.adapters.model.filesystem_local_model_inspector import (
    FilesystemLocalModelInspector,
)
from captioner.adapters.model.filesystem_model_repository import FilesystemModelRepository
from captioner.adapters.model.filesystem_model_validator import FilesystemModelValidator
from captioner.core.application.model_manager import ModelManager
from captioner.core.domain.errors import AppError
from captioner.core.domain.model import (
    ModelSourceCandidate,
    ModelSourceCapabilities,
    ModelSourceReference,
    ModelState,
)
from captioner.core.domain.operation_progress import OperationProgress
from captioner.core.ports.model_source import ProgressCallback


class _Source:
    def capabilities(self) -> ModelSourceCapabilities:
        return ModelSourceCapabilities(search=False, exact_repository=True)

    def search(self, query: str, backend_id: str, limit: int) -> tuple[ModelSourceCandidate, ...]:
        del query, backend_id, limit
        return ()

    def resolve_exact(
        self,
        repository_id: str,
        revision: str | None,
        backend_id: str,
        model_format_hint: str | None = None,
    ) -> ModelSourceReference:
        return ModelSourceReference(
            "huggingface",
            repository_id,
            revision or "a" * 40,
            backend_id,
            model_format_hint,
        )


class _Materializer:
    def __init__(self, source: Path) -> None:
        self.source = source

    def materialize(
        self,
        reference: ModelSourceReference,
        destination: Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> None:
        del reference, progress
        shutil.copytree(self.source, destination)


def _write_ct2_model(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"model_type": "whisper"}), encoding="utf-8")
    (root / "tokenizer.json").write_text(json.dumps({"version": 1}), encoding="utf-8")
    (root / "model.bin").write_bytes(b"ct2 weights")


def _manager(tmp_path: Path, source: Path | None = None) -> ModelManager:
    paths = tmp_path / "data"
    repository = FilesystemModelRepository(paths / "models", staging_dir=paths / "staging")
    validator = FilesystemModelValidator()
    return ModelManager(
        repository=repository,
        inspector=FilesystemLocalModelInspector(validator),
        validator=validator,
        models_dir=paths / "models",
        staging_dir=paths / "staging",
        sources={} if source is None else {"huggingface": _Source()},
        materializers={} if source is None else {"huggingface": _Materializer(source)},
    )


def test_managed_local_import_is_atomic_and_offline(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_ct2_model(source)
    progress: list[OperationProgress] = []
    manager = _manager(tmp_path)

    installed = manager.import_local(source, progress=progress.append)

    assert installed.state is ModelState.INSTALLED
    assert installed.managed is True
    assert installed.model_directory != source.resolve()
    assert installed.model_directory.is_dir()
    assert progress[-1].phase == "completed"
    assert manager.list_models()[0].identity == installed.identity


def test_external_registration_remove_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "external"
    _write_ct2_model(source)
    manager = _manager(tmp_path)

    external = manager.register_external(source, developer_mode=True)
    manager.remove(external.identity)

    assert source.is_dir()
    assert (source / "model.bin").read_bytes() == b"ct2 weights"
    assert manager.list_models() == ()


def test_remote_install_uses_fixed_revision_and_cleans_staging(tmp_path: Path) -> None:
    source = tmp_path / "remote"
    _write_ct2_model(source)
    manager = _manager(tmp_path, source)
    progress: list[OperationProgress] = []

    installed = manager.install_remote(
        "huggingface",
        "org/model",
        "a" * 40,
        "faster-whisper",
        "faster-whisper-ct2",
        progress=progress.append,
    )

    assert installed.state is ModelState.INSTALLED
    assert progress[-1].phase == "completed"
    assert not (tmp_path / "data" / "staging" / "models").exists() or not list(
        (tmp_path / "data" / "staging" / "models").iterdir()
    )


def test_reinstalling_same_remote_identity_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "remote"
    _write_ct2_model(source)
    manager = _manager(tmp_path, source)

    first = manager.install_remote(
        "huggingface", "org/model", "a" * 40, "faster-whisper", "faster-whisper-ct2"
    )
    progress: list[OperationProgress] = []
    second = manager.install_remote(
        "huggingface",
        "org/model",
        "a" * 40,
        "faster-whisper",
        "faster-whisper-ct2",
        progress=progress.append,
    )

    assert second == first
    assert len(manager.list_models()) == 1
    assert progress[-1].phase == "completed"


def test_validation_failure_updates_managed_record_and_recovery_reconstructs_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_ct2_model(source)
    manager = _manager(tmp_path)
    installed = manager.import_local(source)

    (installed.model_directory / "model.bin").write_bytes(b"changed")
    report = manager.validate(installed.identity)
    assert not report.ok
    failed = manager.list_models()[0]
    assert failed.state is ModelState.FAILED
    assert failed.validation_passed is False

    # A complete payload without its record is recoverable after an interrupted
    # metadata write; a corrupt payload is quarantined instead.
    recovery = tmp_path / "recovery-source"
    _write_ct2_model(recovery)
    recovered_model = _manager(tmp_path / "recovery").import_local(recovery)
    record = recovered_model.model_directory.parent / "installation.json"
    record.unlink()
    recovery_manager = _manager(tmp_path / "recovery")
    assert recovered_model.identity in recovery_manager.recover()
    assert recovery_manager.list_models()[0].identity == recovered_model.identity


def test_managed_remove_deletes_only_the_clean_installation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_ct2_model(source)
    manager = _manager(tmp_path)
    installed = manager.import_local(source)

    manager.remove(installed.identity)

    assert not installed.model_directory.parent.exists()
    assert source.is_dir()
    assert manager.list_models() == ()


def test_invalid_offline_model_is_rejected_before_registration(tmp_path: Path) -> None:
    source = tmp_path / "invalid"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    manager = _manager(tmp_path)

    with pytest.raises(AppError, match=r"model\.format_unknown"):
        manager.import_local(source)
    assert manager.list_models() == ()


def test_full_validator_runs_before_managed_registration(tmp_path: Path) -> None:
    source = tmp_path / "empty-weights"
    _write_ct2_model(source)
    (source / "model.bin").write_bytes(b"")
    manager = _manager(tmp_path)

    with pytest.raises(AppError, match=r"model\.model_bin_empty"):
        manager.import_local(source)

    assert manager.list_models() == ()


def test_full_validator_runs_before_external_registration(tmp_path: Path) -> None:
    source = tmp_path / "empty-external"
    _write_ct2_model(source)
    (source / "model.bin").write_bytes(b"")
    manager = _manager(tmp_path)

    with pytest.raises(AppError, match=r"model\.model_bin_empty"):
        manager.register_external(source, developer_mode=True)

    assert manager.list_models() == ()
