from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from typing import NoReturn, cast

import pytest

captioner = import_module("captioner")


def test_source_tree_version_is_read_from_project_metadata() -> None:
    read_source_version = cast(Callable[[], str | None], captioner._source_tree_version)
    assert read_source_version() == "0.0.0"


def test_version_falls_back_to_source_tree_when_distribution_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str) -> NoReturn:
        raise PackageNotFoundError

    monkeypatch.setattr(captioner, "installed_version", missing)
    assert captioner.get_version() == "0.0.0"
