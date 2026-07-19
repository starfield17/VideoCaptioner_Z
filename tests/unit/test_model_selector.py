from __future__ import annotations

import pytest
from tests.fakes.phase6_values import model_installation

from captioner.core.application.model_selector import select_model
from captioner.core.domain.errors import AppError
from captioner.core.domain.model import ModelState


def test_selector_resolves_repository_and_sha_prefix_without_network() -> None:
    model = model_installation(repository_id="org/model", display_name="large-v3")

    assert select_model("org/model", (model,)) is model
    assert select_model(model.identity.manifest_sha256[:12], (model,)) is model
    assert select_model("large-v3", (model,)) is model


@pytest.mark.parametrize("state", [ModelState.STAGED, ModelState.FAILED])
def test_selector_does_not_expose_non_installable_records(state: ModelState) -> None:
    model = model_installation(state=state, validation_passed=False)

    with pytest.raises(AppError, match=r"model\.not_installed"):
        select_model(model.identity.manifest_sha256, (model,))


def test_selector_rejects_unvalidated_external_record() -> None:
    model = model_installation(
        state=ModelState.EXTERNAL_UNMANAGED,
        managed=False,
        validation_passed=False,
    )

    with pytest.raises(AppError, match=r"model\.not_installed"):
        select_model(model.identity.manifest_sha256, (model,))


def test_selector_reports_ambiguous_display_name() -> None:
    first = model_installation(repository_id="org/one", display_name="same")
    second = model_installation(repository_id="org/two", display_name="same")

    with pytest.raises(AppError, match=r"model\.selector_ambiguous"):
        select_model("same", (first, second))
