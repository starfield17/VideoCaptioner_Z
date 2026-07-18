from __future__ import annotations

import pytest
from tests.fakes.phase6_values import model_installation, runtime_installation

from captioner.core.application.runtime_selection import (
    HostFacts,
    select_runtime,
    try_select_runtime,
)
from captioner.core.domain.errors import AppError


def _mlx_runtime():
    return runtime_installation(
        backend_id="mlx-whisper",
        device_kind="metal",
        model_format="mlx-whisper",
        runtime_id="mlx-whisper-metal-macos-arm64",
    )


def _faster_runtime():
    return runtime_installation()


def test_apple_silicon_auto_prefers_mlx_for_mlx_model() -> None:
    model = model_installation(
        backend_id="mlx-whisper",
        model_format="mlx-whisper",
        repository_id="org/mlx-model",
    )
    selection = select_runtime(
        host=HostFacts("macos", "arm64", "14.0", True),
        runtimes=(_faster_runtime(), _mlx_runtime()),
        model=model,
    )
    assert selection.effective_backend_id == "mlx-whisper"
    assert selection.effective_device == "metal"
    assert selection.effective_runtime_identity.runtime_id == "mlx-whisper-metal-macos-arm64"


def test_mlx_model_never_falls_back_to_faster_whisper_cpu() -> None:
    model = model_installation(
        backend_id="mlx-whisper",
        model_format="mlx-whisper",
        repository_id="org/mlx-model",
    )
    result = try_select_runtime(
        host=HostFacts("macos", "arm64", "14.0", True),
        runtimes=(_faster_runtime(),),
        model=model,
    )
    assert not result.ok
    assert result.error_code == "runtime.preflight_failed"


def test_faster_whisper_model_uses_cpu_when_mlx_is_available() -> None:
    model = model_installation()
    selection = select_runtime(
        host=HostFacts("macos", "arm64", "14.0", True),
        runtimes=(_mlx_runtime(), _faster_runtime()),
        model=model,
    )
    assert selection.effective_backend_id == "faster-whisper"
    assert selection.effective_device == "cpu"


@pytest.mark.parametrize(
    "host",
    [
        HostFacts("macos", "arm64", "13.6", True),
        HostFacts("macos", "arm64", "14.0", False),
        HostFacts("windows", "x86_64", "14.0", True),
        HostFacts("linux", "x86_64", "14.0", True),
    ],
)
def test_mlx_is_not_selected_outside_native_supported_host(host: HostFacts) -> None:
    model = model_installation(
        backend_id="mlx-whisper",
        model_format="mlx-whisper",
        repository_id="org/mlx-model",
    )
    result = try_select_runtime(host=host, runtimes=(_mlx_runtime(),), model=model)
    assert not result.ok


def test_explicit_mlx_unavailable_fails_without_fallback() -> None:
    result = try_select_runtime(
        requested_backend_id="mlx-whisper",
        requested_device="metal",
        host=HostFacts("macos", "arm64", "14.0", True),
        runtimes=(_faster_runtime(),),
        model=model_installation(
            backend_id="mlx-whisper",
            model_format="mlx-whisper",
            repository_id="org/mlx-model",
        ),
    )
    assert not result.ok
    with pytest.raises(AppError, match=r"runtime\.preflight_failed"):
        select_runtime(
            requested_backend_id="mlx-whisper",
            requested_device="metal",
            host=HostFacts("macos", "arm64", "14.0", True),
            runtimes=(_faster_runtime(),),
            model=model_installation(
                backend_id="mlx-whisper",
                model_format="mlx-whisper",
                repository_id="org/mlx-model",
            ),
        )


def test_no_compatible_runtime_is_typed_failure() -> None:
    result = try_select_runtime(
        host=HostFacts("macos", "arm64", "14.0", True),
        runtimes=(_faster_runtime(),),
        model=model_installation(),
        requested_device="cuda",
    )
    assert not result.ok
    assert result.reasons == ("no_compatible_runtime",)
