"""Small identifier helper kept independent of domain models."""

from __future__ import annotations

from uuid import uuid4


def new_id(prefix: str = "") -> str:
    """Return a process-independent identifier."""
    return f"{prefix}{uuid4().hex}"
