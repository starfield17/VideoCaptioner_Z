from __future__ import annotations

from pathlib import Path

import pytest
from tests.fakes.phase6_values import model_installation, runtime_installation

from captioner.core.application.model_preflight import preflight_new_job
from captioner.core.application.runtime_selection import HostFacts
from captioner.core.domain.asr_job_snapshot import ASRJobSnapshot
from captioner.core.domain.errors import AppError
from captioner.core.domain.model import ModelState, ModelValidationCheck, ModelValidationReport


class _Validator:
    def __init__(self, report: ModelValidationReport) -> None:
        self.report = report

    def validate(self, manifest: object, model_directory: Path) -> ModelValidationReport:
        del manifest, model_directory
        return self.report


def _valid_report() -> ModelValidationReport:
    return ModelValidationReport(True, (ModelValidationCheck("static", True),))


def _host() -> HostFacts:
    return HostFacts("macos", "arm64", "14.0", True)


def test_new_job_preflight_persists_one_effective_cpu_selection() -> None:
    model = model_installation()
    runtime = runtime_installation()

    result = preflight_new_job(
        model_selector="org/model",
        requested_device="auto",
        compute_type="int8",
        host=_host(),
        installed_models=(model,),
        active_runtimes=(runtime,),
        validator=_Validator(_valid_report()),
    )

    assert result.ok
    assert result.snapshot is not None
    assert result.snapshot.effective_runtime_identity == runtime.identity
    assert result.snapshot.effective_device_kind == "cpu"
    assert result.snapshot.requested_device == "auto"


def test_preflight_rejects_missing_runtime_without_installing_one() -> None:
    result = preflight_new_job(
        model_selector="org/model",
        requested_device="auto",
        compute_type="default",
        host=_host(),
        installed_models=(model_installation(),),
        active_runtimes=(),
    )

    assert not result.ok
    assert result.error_code == "runtime.preflight_failed"
    assert result.reasons == ("no_available_runtime",)


def test_external_preflight_revalidates_content_before_selection() -> None:
    external = model_installation(
        source_id="external-path",
        repository_id="external/registration",
        state=ModelState.EXTERNAL_UNMANAGED,
        managed=False,
        validation_passed=True,
    )
    invalid = ModelValidationReport(
        False,
        (
            ModelValidationCheck(
                "hash", False, "model.external_content_changed", "model.external_content_changed"
            ),
        ),
        error_code="model.external_content_changed",
        message_code="model.external_content_changed",
    )

    result = preflight_new_job(
        model_selector="external/registration",
        requested_device="auto",
        compute_type="default",
        host=_host(),
        installed_models=(external,),
        active_runtimes=(runtime_installation(),),
        validator=_Validator(invalid),
    )

    assert not result.ok
    assert result.error_code == "model.external_content_changed"


def test_mlx_preflight_only_accepts_default_compute_type() -> None:
    model = model_installation(
        backend_id="mlx-whisper",
        model_format="mlx-whisper",
        repository_id="org/mlx",
    )
    runtime = runtime_installation(
        backend_id="mlx-whisper",
        device_kind="metal",
        model_format="mlx-whisper",
        runtime_id="mlx-whisper-metal-macos-arm64",
    )

    result = preflight_new_job(
        model_selector="org/mlx",
        requested_device="auto",
        compute_type="int8",
        host=_host(),
        installed_models=(model,),
        active_runtimes=(runtime,),
    )

    assert not result.ok
    assert result.error_code == "asr.compute_type_invalid"


@pytest.mark.parametrize("compute_type", ["", " int8", "float16"])
def test_ct2_preflight_rejects_unsafe_compute_type(compute_type: str) -> None:
    result = preflight_new_job(
        model_selector="org/model",
        requested_device="auto",
        compute_type=compute_type,
        host=_host(),
        installed_models=(model_installation(),),
        active_runtimes=(runtime_installation(),),
    )

    assert not result.ok
    assert result.error_code == "asr.compute_type_invalid"


def test_preflight_requires_the_installed_model_selector() -> None:
    with pytest.raises(AppError, match=r"job\.asr_snapshot_invalid|model"):
        # The assertion is intentionally about construction: an empty selector
        # must never become a durable schema-3 snapshot.
        model = model_installation()
        ASRJobSnapshot(
            1,
            "",
            "auto",
            "faster-whisper",
            runtime_installation().identity,
            model.identity,
            "cpu",
            "default",
        )
