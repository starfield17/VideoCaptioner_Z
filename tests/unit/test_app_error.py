from __future__ import annotations

from typing import cast

import pytest

from captioner.core.domain.errors import AppError
from captioner.core.domain.result import JsonValue, Result


def test_empty_code_fails() -> None:
    with pytest.raises(ValueError):
        AppError("  ")


def test_non_serializable_params_fail() -> None:
    params = cast(dict[str, JsonValue], {"bad": object()})
    with pytest.raises(TypeError):
        AppError("test.invalid", params)


def test_to_dict_is_stable_and_preserves_retryable() -> None:
    error = AppError("test.failed", {"b": 2, "a": "one"}, retryable=True)
    assert error.to_dict() == {
        "code": "test.failed",
        "params": {"b": 2, "a": "one"},
        "retryable": True,
    }
    assert str(error) == 'test.failed: {"a": "one", "b": 2}'


def test_cause_is_kept_by_exception_chaining() -> None:
    cause = ValueError("root")
    with pytest.raises(AppError) as raised:
        raise AppError("test.caused") from cause
    assert raised.value.__cause__ is cause


def test_result_success_and_failure() -> None:
    success = Result[str].success("value")
    failure = Result[str].failure(RuntimeError("failed"))
    assert success.ok is True
    assert success.value == "value"
    assert failure.ok is False
    assert isinstance(failure.error, RuntimeError)
