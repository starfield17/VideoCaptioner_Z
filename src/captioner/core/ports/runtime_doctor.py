"""Static and activation Runtime Doctor boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from captioner.core.domain.runtime import DoctorReport, RuntimeInstallation


class RuntimeDoctor(Protocol):
    def static_doctor(self, runtime: RuntimeInstallation) -> DoctorReport: ...

    def activation_doctor(self, runtime: RuntimeInstallation, workspace: Path) -> DoctorReport: ...


RuntimeDoctorPort = RuntimeDoctor

__all__ = ["RuntimeDoctor", "RuntimeDoctorPort"]
