"""Unit tests for LocalDiagnosticsAdapter probes and redacted ZIP export."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

from captioner.adapters.diagnostics.local_diagnostics import LocalDiagnosticsAdapter
from captioner.core.application.diagnostics import (
    DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
    DiagnosticExportRequest,
    DiagnosticsConfigurationSummary,
    DiagnosticsQueueSummary,
    DiagnosticsRecoverySummary,
    DiagnosticsSnapshot,
    RuntimeAvailability,
)
from captioner.core.domain.errors import AppError

SENTINELS = (
    b"sk-diagnostic-secret-123",
    b"https://user:password@example.internal/v1",
    b"PRIVATE_SOURCE_SENTINEL",
    b"PRIVATE_SUBTITLE_SENTINEL",
    b"/private/home/alice/media/input-secret.wav",
    b"C:\\Users\\Alice\\Videos\\input-secret.mp4",
    b"prompt-secret-sentinel",
    b"provider-response-secret",
)
FORBIDDEN_KEYS = (b"api_key", b"authorization", b"bearer", b"password", b"secret_key")


def _snapshot() -> DiagnosticsSnapshot:
    runtime = RuntimeAvailability(
        packaged=False,
        operating_system="Linux",
        architecture="x86_64",
        python_version="3.13.0",
        app_version="0.0.0",
        ffmpeg_available=True,
        ffprobe_available=False,
        asr_runtime_available=False,
        provider_configured=True,
        credential_source="config",
    )
    return DiagnosticsSnapshot(
        schema_version=1,
        request_id="req-bundle-1",
        generated_at_utc="2026-07-18T12:00:00+00:00",
        runtime=runtime,
        queue=DiagnosticsQueueSummary(
            schema_version=1,
            revision=2,
            active_jobs=1,
            terminal_jobs=0,
            omitted_terminal_jobs=0,
            state_counts=(("running", 1),),
            profile_counts=(("fast", 1),),
            stage_counts=(("transcribe", 1),),
            issue_codes=(),
        ),
        configuration=DiagnosticsConfigurationSummary(
            schema_version=1,
            locale="en",
            built_in_preset_count=3,
            user_preset_count=0,
            provider_configured=True,
            credential_source="config",
            issue_codes=(),
        ),
        recovery=DiagnosticsRecoverySummary(
            schema_version=1,
            recoverable_batches=0,
            blocked_batches=0,
            paused_batches=0,
            issue_codes=(),
        ),
    )


def test_runtime_probe_is_lightweight() -> None:
    adapter = LocalDiagnosticsAdapter()
    runtime = adapter.collect_runtime_availability(
        provider_configured=False,
        credential_source="missing",
    )
    assert runtime.operating_system
    assert runtime.architecture
    assert runtime.python_version
    assert runtime.app_version
    assert runtime.provider_configured is False
    assert runtime.credential_source == "missing"
    # Booleans only — no paths.
    assert isinstance(runtime.ffmpeg_available, bool)
    assert isinstance(runtime.ffprobe_available, bool)
    assert isinstance(runtime.asr_runtime_available, bool)
    assert isinstance(runtime.packaged, bool)


def test_bundle_members_canonical_and_hashed(tmp_path: Path) -> None:
    adapter = LocalDiagnosticsAdapter()
    destination = tmp_path / "diag.zip"
    result = adapter.write_bundle(
        DiagnosticExportRequest(
            request_id="req-bundle-1",
            destination=str(destination),
        ),
        snapshot=_snapshot(),
    )
    assert destination.is_file()
    assert result.size_bytes == destination.stat().st_size
    assert result.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    if os.name == "posix":
        mode = stat.S_IMODE(destination.stat().st_mode)
        assert mode == 0o600

    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
        assert names == [
            "manifest.json",
            "application.json",
            "configuration.json",
            "queue.json",
            "recovery.json",
            "capabilities.json",
        ]
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        assert manifest["schema_version"] == DIAGNOSTIC_BUNDLE_SCHEMA_VERSION
        assert [entry["name"] for entry in manifest["files"]] == [
            "application.json",
            "configuration.json",
            "queue.json",
            "recovery.json",
            "capabilities.json",
        ]
        for entry in manifest["files"]:
            payload = archive.read(entry["name"])
            assert entry["size_bytes"] == len(payload)
            assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
            assert payload.endswith(b"\n")
            # Canonical JSON: sorted keys, compact separators.
            parsed = json.loads(payload.decode("utf-8"))
            rebuilt = (
                json.dumps(
                    parsed,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            assert payload == rebuilt


def test_destination_validation(tmp_path: Path) -> None:
    adapter = LocalDiagnosticsAdapter()
    snapshot = _snapshot()
    with pytest.raises(AppError) as missing_suffix:
        adapter.write_bundle(
            DiagnosticExportRequest(request_id="req-a", destination=str(tmp_path / "x.txt")),
            snapshot=snapshot,
        )
    assert missing_suffix.value.code == "diagnostics.destination_invalid"

    missing_parent = tmp_path / "nope" / "out.zip"
    with pytest.raises(AppError) as parent_err:
        adapter.write_bundle(
            DiagnosticExportRequest(request_id="req-b", destination=str(missing_parent)),
            snapshot=snapshot,
        )
    assert parent_err.value.code == "diagnostics.destination_invalid"

    destination = tmp_path / "exists.zip"
    destination.write_bytes(b"old")
    with pytest.raises(AppError) as exists_err:
        adapter.write_bundle(
            DiagnosticExportRequest(request_id="req-c", destination=str(destination)),
            snapshot=snapshot,
        )
    assert exists_err.value.code == "diagnostics.destination_exists"

    result = adapter.write_bundle(
        DiagnosticExportRequest(
            request_id="req-d",
            destination=str(destination),
            overwrite=True,
        ),
        snapshot=snapshot,
    )
    assert result.size_bytes > 0
    assert destination.read_bytes()[:2] == b"PK"

    link = tmp_path / "link.zip"
    target = tmp_path / "target.zip"
    target.write_bytes(b"x")
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unsupported")
    with pytest.raises(AppError) as link_err:
        adapter.write_bundle(
            DiagnosticExportRequest(
                request_id="req-e",
                destination=str(link),
                overwrite=True,
            ),
            snapshot=snapshot,
        )
    assert link_err.value.code == "diagnostics.destination_invalid"


def test_no_sentinel_leakage(tmp_path: Path) -> None:
    adapter = LocalDiagnosticsAdapter()
    destination = tmp_path / "safe.zip"
    result = adapter.write_bundle(
        DiagnosticExportRequest(request_id="req-safe", destination=str(destination)),
        snapshot=_snapshot(),
    )
    raw = destination.read_bytes()
    for sentinel in SENTINELS:
        assert sentinel not in raw
    for key in FORBIDDEN_KEYS:
        assert key not in raw
    with zipfile.ZipFile(destination) as archive:
        for name in archive.namelist():
            body = archive.read(name)
            for sentinel in SENTINELS:
                assert sentinel not in body
            for key in FORBIDDEN_KEYS:
                assert key not in body
            # credential_source field name is public and allowed.
            if name in {"configuration.json", "capabilities.json"}:
                assert b"credential_source" in body
    assert "sk-" not in result.sha256
    rendered = repr(_snapshot())
    # Snapshot has no private paths/secrets by construction.
    assert "input-secret" not in rendered
    assert "PRIVATE_SOURCE" not in rendered
    assert "sk-diagnostic" not in rendered
