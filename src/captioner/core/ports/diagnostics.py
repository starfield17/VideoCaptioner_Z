"""Ports for diagnostics environment probes and redacted bundle writing."""

from __future__ import annotations

from typing import Literal, Protocol

from captioner.core.application.diagnostics import (
    DiagnosticExportRequest,
    DiagnosticExportResult,
    DiagnosticsSnapshot,
    RuntimeAvailability,
)


class DiagnosticsEnvironmentPort(Protocol):
    def collect_runtime_availability(
        self,
        *,
        provider_configured: bool,
        credential_source: Literal["config", "environment", "missing"],
    ) -> RuntimeAvailability: ...


class DiagnosticBundleWriterPort(Protocol):
    def write_bundle(
        self,
        request: DiagnosticExportRequest,
        *,
        snapshot: DiagnosticsSnapshot,
    ) -> DiagnosticExportResult: ...


__all__ = ["DiagnosticBundleWriterPort", "DiagnosticsEnvironmentPort"]
