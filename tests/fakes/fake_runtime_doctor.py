"""Configurable Runtime Doctor fake."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from captioner.core.domain.runtime import DoctorReport, RuntimeInstallation


@dataclass(slots=True)
class FakeRuntimeDoctor:
    static_report: DoctorReport
    activation_report: DoctorReport
    static_calls: list[RuntimeInstallation] | None = None
    activation_calls: list[tuple[RuntimeInstallation, Path]] | None = None

    def __post_init__(self) -> None:
        if self.static_calls is None:
            self.static_calls = []
        if self.activation_calls is None:
            self.activation_calls = []

    def static_doctor(self, runtime: RuntimeInstallation) -> DoctorReport:
        assert self.static_calls is not None
        self.static_calls.append(runtime)
        return self.static_report

    def activation_doctor(self, runtime: RuntimeInstallation, workspace: Path) -> DoctorReport:
        assert self.activation_calls is not None
        self.activation_calls.append((runtime, workspace))
        return self.activation_report


__all__ = ["FakeRuntimeDoctor"]
