from __future__ import annotations

import pytest

from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import CancellationToken, ExecutionContext


def test_cancellation_token_is_thread_safe_and_structured() -> None:
    token = CancellationToken()
    context = ExecutionContext(token)
    assert not token.is_cancelled
    context.cancel()
    assert token.is_cancelled
    assert context.is_cancelled
    with pytest.raises(AppError, match=r"operation\.cancelled"):
        context.raise_if_cancelled()
