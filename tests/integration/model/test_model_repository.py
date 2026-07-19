from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.fakes.phase6_values import model_installation

from captioner.adapters.model.filesystem_local_model_inspector import (
    FilesystemLocalModelInspector,
)
from captioner.adapters.model.filesystem_model_repository import FilesystemModelRepository
from captioner.adapters.model.filesystem_model_validator import FilesystemModelValidator
from captioner.core.application.model_manager import ModelManager
from captioner.core.domain.errors import AppError
from captioner.core.domain.model import ModelState


def test_model_store_can_be_reloaded_from_atomic_records(tmp_path: Path) -> None:
    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "whisper"}), encoding="utf-8")
    (source / "tokenizer.json").write_text(json.dumps({"version": 1}), encoding="utf-8")
    (source / "model.bin").write_bytes(b"weights")
    validator = FilesystemModelValidator()

    repository = FilesystemModelRepository(
        tmp_path / "data" / "models",
        staging_dir=tmp_path / "data" / "staging",
    )
    manager = ModelManager(
        repository=repository,
        inspector=FilesystemLocalModelInspector(validator),
        validator=validator,
        models_dir=tmp_path / "data" / "models",
        staging_dir=tmp_path / "data" / "staging",
    )
    installed = manager.import_local(source)

    reloaded = FilesystemModelRepository(
        tmp_path / "data" / "models", staging_dir=tmp_path / "data" / "staging"
    ).get_by_identity(installed.identity)
    assert reloaded is not None
    assert reloaded.identity == installed.identity
    assert reloaded.model_directory == installed.model_directory


def test_external_identity_record_names_do_not_collide(tmp_path: Path) -> None:
    repository = FilesystemModelRepository(tmp_path / "models")
    first = model_installation(
        source_id="external-path",
        repository_id="foo.bar",
        state=ModelState.EXTERNAL_UNMANAGED,
        managed=False,
        validation_passed=True,
        display_name="first",
    )
    second = model_installation(
        source_id="external-path",
        repository_id="foo_bar",
        state=ModelState.EXTERNAL_UNMANAGED,
        managed=False,
        validation_passed=True,
        display_name="second",
    )

    repository.register_external_model(first)
    repository.register_external_model(second)
    records = sorted((tmp_path / "models" / "external").glob("*.json"))

    assert len(records) == 2
    assert {item.identity.repository_id for item in repository.list_installed_models()} == {
        "foo.bar",
        "foo_bar",
    }
    repository.remove_model(first.identity)
    assert repository.get_by_identity(second.identity) is not None


def test_model_use_lock_blocks_removal_until_released(tmp_path: Path) -> None:
    repository = FilesystemModelRepository(tmp_path / "models")
    model = model_installation(
        source_id="external-path",
        repository_id="external/one",
        state=ModelState.EXTERNAL_UNMANAGED,
        managed=False,
        validation_passed=True,
    )
    repository.register_external_model(model)

    with repository.use_lock(model.identity), pytest.raises(AppError, match=r"model\.in_use"):
        repository.remove_model(model.identity)
    repository.remove_model(model.identity)
    assert repository.get_by_identity(model.identity) is None
