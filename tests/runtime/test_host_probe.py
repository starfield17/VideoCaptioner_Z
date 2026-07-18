from __future__ import annotations

import pytest

from captioner.adapters.runtime.host_probe import (
    HostProbe,
    normalize_architecture,
    normalize_platform,
)
from captioner.core.domain.errors import AppError


@pytest.mark.parametrize(
    ("raw", "normalized"),
    (("darwin", "macos"), ("win32", "windows"), ("Linux", "linux")),
)
def test_platform_normalization(raw: str, normalized: str) -> None:
    assert normalize_platform(raw) == normalized


@pytest.mark.parametrize(
    ("raw", "normalized"),
    (("aarch64", "arm64"), ("amd64", "x86_64"), ("x86-64", "x86_64")),
)
def test_architecture_normalization(raw: str, normalized: str) -> None:
    assert normalize_architecture(raw) == normalized


def test_unknown_host_values_are_typed_failures() -> None:
    with pytest.raises(AppError, match=r"runtime\.host_unsupported"):
        normalize_platform("plan9")
    with pytest.raises(AppError, match=r"runtime\.host_unsupported"):
        normalize_architecture("mips")


def test_rosetta_is_not_native_arm64() -> None:
    facts = HostProbe(
        system=lambda: "Darwin",
        machine=lambda: "arm64",
        mac_version=lambda: "14.5.0",
        translated_probe=lambda: True,
    ).probe()
    assert facts.platform == "macos"
    assert facts.architecture == "arm64"
    assert not facts.native_architecture
