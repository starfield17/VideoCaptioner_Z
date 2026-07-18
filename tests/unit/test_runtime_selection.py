from __future__ import annotations

from typing import cast

import pytest
from tests.fakes.phase6_values import model_installation, runtime_installation

from captioner.core.application.runtime_selection import (
    HostFacts,
    select_runtime,
    try_select_runtime,
)
from captioner.core.domain.errors import AppError
from captioner.core.domain.model import ModelInstallation, ModelState
from captioner.core.domain.runtime import RuntimeState, RuntimeTarget


def _host(
    platform: str = "macos",
    architecture: str = "arm64",
    os_version: str = "14.0",
    native_architecture: bool = True,
) -> HostFacts:
    return HostFacts(platform, architecture, os_version, native_architecture)


def _mlx_runtime(*, state: RuntimeState = RuntimeState.AVAILABLE):
    return runtime_installation(
        backend_id="mlx-whisper",
        device_kind="metal",
        model_format="mlx-whisper",
        runtime_id="mlx-whisper-metal-macos-arm64",
        state=state,
    )


def _faster_runtime(
    *,
    device_kind: str = "cpu",
    platform: str = "macos",
    architecture: str = "arm64",
    state: RuntimeState = RuntimeState.AVAILABLE,
    runtime_id: str | None = None,
    version: str = "1.0.0",
):
    return runtime_installation(
        backend_id="faster-whisper",
        device_kind=device_kind,
        model_format="faster-whisper-ct2",
        platform=platform,
        architecture=architecture,
        runtime_id=runtime_id,
        version=version,
        minimum_os_version="1.0.0",
        state=state,
    )


def _mlx_model():
    return model_installation(
        backend_id="mlx-whisper",
        model_format="mlx-whisper",
        repository_id="org/mlx-model",
    )


def _ct2_model(*, state: ModelState = ModelState.INSTALLED):
    return model_installation(state=state)


def test_apple_silicon_auto_prefers_mlx_for_mlx_model() -> None:
    selection = select_runtime(
        host=_host(),
        active_runtimes=(_faster_runtime(), _mlx_runtime()),
        model=_mlx_model(),
    )
    assert selection.effective_backend_id == "mlx-whisper"
    assert selection.effective_device == "metal"
    assert selection.effective_runtime_identity.runtime_id == "mlx-whisper-metal-macos-arm64"


def test_mlx_model_never_falls_back_to_faster_whisper() -> None:
    result = try_select_runtime(
        host=_host(),
        active_runtimes=(_faster_runtime(),),
        model=_mlx_model(),
    )
    assert not result.ok
    assert result.reasons == ("no_compatible_runtime",)


def test_faster_whisper_model_uses_cpu_when_mlx_is_available() -> None:
    selection = select_runtime(
        host=_host(),
        active_runtimes=(_mlx_runtime(), _faster_runtime()),
        model=_ct2_model(),
    )
    assert selection.effective_backend_id == "faster-whisper"
    assert selection.effective_device == "cpu"


@pytest.mark.parametrize("platform", ["windows", "linux"])
def test_auto_prefers_available_cuda_before_cpu(platform: str) -> None:
    selection = select_runtime(
        host=_host(platform=platform, architecture="x86_64"),
        active_runtimes=(
            _faster_runtime(device_kind="cpu", platform=platform, architecture="x86_64"),
            _faster_runtime(device_kind="cuda", platform=platform, architecture="x86_64"),
        ),
        model=_ct2_model(),
    )
    assert selection.effective_backend_id == "faster-whisper"
    assert selection.effective_device == "cuda"


def test_auto_uses_cpu_when_cuda_is_not_available() -> None:
    selection = select_runtime(
        host=_host(platform="windows", architecture="x86_64"),
        active_runtimes=(
            _faster_runtime(
                device_kind="cuda",
                platform="windows",
                architecture="x86_64",
                state=RuntimeState.INSTALLED,
            ),
            _faster_runtime(device_kind="cpu", platform="windows", architecture="x86_64"),
        ),
        model=_ct2_model(),
    )
    assert selection.effective_device == "cpu"


def test_auto_uses_cpu_when_cuda_runtime_is_installed_but_not_available() -> None:
    selection = select_runtime(
        host=_host(platform="linux", architecture="x86_64"),
        active_runtimes=(
            _faster_runtime(
                device_kind="cuda",
                platform="linux",
                architecture="x86_64",
                state=RuntimeState.INSTALLED,
            ),
            _faster_runtime(device_kind="cpu", platform="linux", architecture="x86_64"),
        ),
        model=_ct2_model(),
    )
    assert selection.effective_device == "cpu"


def test_explicit_cpu_does_not_fallback_to_cuda() -> None:
    selection = select_runtime(
        requested_device="cpu",
        host=_host(platform="windows", architecture="x86_64"),
        active_runtimes=(
            _faster_runtime(device_kind="cuda", platform="windows", architecture="x86_64"),
            _faster_runtime(device_kind="cpu", platform="windows", architecture="x86_64"),
        ),
        model=_ct2_model(),
    )
    assert selection.effective_device == "cpu"


def test_explicit_cuda_unavailable_fails_without_cpu_fallback() -> None:
    result = try_select_runtime(
        requested_device="cuda",
        host=_host(platform="linux", architecture="x86_64"),
        active_runtimes=(
            _faster_runtime(device_kind="cpu", platform="linux", architecture="x86_64"),
        ),
        model=_ct2_model(),
    )
    assert not result.ok
    assert result.reasons == ("no_compatible_runtime",)


def test_mlx_model_never_chooses_cuda_or_cpu() -> None:
    result = try_select_runtime(
        host=_host(platform="windows", architecture="x86_64"),
        active_runtimes=(
            _faster_runtime(device_kind="cuda", platform="windows", architecture="x86_64"),
            _faster_runtime(device_kind="cpu", platform="windows", architecture="x86_64"),
        ),
        model=_mlx_model(),
    )
    assert not result.ok


def test_ct2_model_never_chooses_mlx() -> None:
    result = try_select_runtime(
        host=_host(),
        active_runtimes=(_mlx_runtime(),),
        model=_ct2_model(),
    )
    assert not result.ok


@pytest.mark.parametrize(
    "host",
    [
        _host(os_version="13.6"),
        _host(native_architecture=False),
        _host(platform="windows", architecture="x86_64"),
        _host(platform="linux", architecture="x86_64"),
    ],
)
def test_mlx_is_not_selected_outside_native_supported_host(host: HostFacts) -> None:
    result = try_select_runtime(host=host, active_runtimes=(_mlx_runtime(),), model=_mlx_model())
    assert not result.ok


def test_explicit_mlx_unavailable_fails_without_fallback() -> None:
    result = try_select_runtime(
        requested_backend_id="mlx-whisper",
        requested_device="metal",
        host=_host(),
        active_runtimes=(_faster_runtime(),),
        model=_mlx_model(),
    )
    assert not result.ok
    with pytest.raises(AppError, match=r"runtime\.preflight_failed"):
        select_runtime(
            requested_backend_id="mlx-whisper",
            requested_device="metal",
            host=_host(),
            active_runtimes=(_faster_runtime(),),
            model=_mlx_model(),
        )


def test_no_compatible_runtime_is_typed_failure() -> None:
    result = try_select_runtime(
        host=_host(),
        active_runtimes=(_faster_runtime(),),
        model=_ct2_model(),
        requested_device="cuda",
    )
    assert not result.ok
    assert result.reasons == ("no_compatible_runtime",)


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        (ModelState.STAGED, "model_not_installed"),
        (ModelState.FAILED, "model_failed"),
    ],
)
def test_selector_rejects_non_executable_model_states(state: ModelState, reason: str) -> None:
    result = try_select_runtime(
        host=_host(),
        active_runtimes=(_faster_runtime(),),
        model=_ct2_model(state=state),
    )
    assert not result.ok
    assert result.reasons == (reason,)


def test_selector_requires_external_model_validation() -> None:
    result = try_select_runtime(
        host=_host(),
        active_runtimes=(_faster_runtime(),),
        model=model_installation(
            state=ModelState.EXTERNAL_UNMANAGED,
            managed=False,
            validation_passed=False,
        ),
    )
    assert not result.ok
    assert result.reasons == ("model_not_validated",)


def test_selector_accepts_validated_external_model() -> None:
    selection = select_runtime(
        host=_host(),
        active_runtimes=(_faster_runtime(),),
        model=model_installation(
            state=ModelState.EXTERNAL_UNMANAGED,
            managed=False,
            validation_passed=True,
        ),
    )
    assert selection.effective_backend_id == "faster-whisper"


def test_selector_rejects_bare_manifest_before_effective_selection() -> None:
    result = try_select_runtime(
        host=_host(),
        active_runtimes=(_faster_runtime(),),
        model=cast(ModelInstallation, model_installation().manifest),
    )
    assert not result.ok
    assert result.reasons == ("model_not_installed",)


@pytest.mark.parametrize("state", [ModelState.INSTALLED, ModelState.LOAD_VERIFIED])
def test_invalid_managed_validation_projection_is_rejected(state: ModelState) -> None:
    with pytest.raises(AppError, match=r"model\.installation_invalid"):
        model_installation(state=state, validation_passed=False)


def test_same_active_slot_is_ambiguous_without_version_guessing() -> None:
    result = try_select_runtime(
        host=_host(),
        active_runtimes=(
            _faster_runtime(runtime_id="cpu-v1", version="1.0.0"),
            _faster_runtime(runtime_id="cpu-v2", version="2.0.0"),
        ),
        model=_ct2_model(),
    )
    assert not result.ok
    assert result.error_code == "runtime.active_selection_ambiguous"
    assert result.reasons == ("active_selection_ambiguous",)


def test_runtime_target_key_excludes_minimum_os_version() -> None:
    first = RuntimeTarget("macos", "arm64", "cpu", "13.0")
    second = RuntimeTarget("macos", "arm64", "cpu", "14.0")
    assert first.key == second.key == ("macos", "arm64", "cpu")


@pytest.mark.parametrize(
    "target",
    [
        ("darwin", "arm64", "cpu"),
        ("win32", "x86_64", "cpu"),
        ("linux", "aarch64", "cpu"),
        ("linux", "amd64", "cpu"),
        ("linux", "x86_64", "gpu"),
        ("linux", "x86_64", "mps"),
    ],
)
def test_runtime_target_rejects_unnormalized_values(
    target: tuple[str, str, str],
) -> None:
    with pytest.raises(AppError, match=r"runtime\.target_invalid"):
        RuntimeTarget(*target, "1.0")


@pytest.mark.parametrize(
    "facts",
    [
        ("darwin", "arm64", "14.0", True),
        ("macos", "aarch64", "14.0", True),
        ("windows", "amd64", "14.0", True),
    ],
)
def test_host_facts_reject_unnormalized_values(
    facts: tuple[str, str, str, bool],
) -> None:
    with pytest.raises(AppError, match=r"runtime\.host_facts_invalid"):
        HostFacts(*facts)
