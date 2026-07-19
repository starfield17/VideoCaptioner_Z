from __future__ import annotations

import json
from pathlib import Path

import pytest

from captioner.cli import cli_entry
from captioner.cli.commands import batch as batch_command
from captioner.core.domain.asr_job_snapshot import ASRJobSnapshot
from captioner.core.domain.batch import BatchProjection
from captioner.core.domain.errors import AppError
from captioner.core.domain.model import ModelIdentity
from captioner.core.domain.runtime import RuntimeIdentity
from captioner.infrastructure.app_paths import AppPaths


def _projection() -> BatchProjection:
    return BatchProjection("batch-test", last_event_seq=1)


def _snapshot(**kwargs: object) -> ASRJobSnapshot:
    del kwargs
    return ASRJobSnapshot(
        1,
        "model",
        "auto",
        "faster-whisper",
        RuntimeIdentity("runtime", "1.0.0"),
        ModelIdentity(
            "faster-whisper",
            "local-import",
            "local-import/model",
            "1",
            "faster-whisper-ct2",
            "0" * 64,
        ),
        "cpu",
        "default",
    )


def _payload(_projection: BatchProjection, *, paths: AppPaths) -> dict[str, object]:
    del paths
    return {
        "schema_version": 1,
        "batch_id": "batch-test",
        "state": "succeeded",
        "last_event_seq": 1,
        "cancel_requested": False,
        "jobs": [],
    }


def _run_success(_options: batch_command.BatchRunOptions, *, paths: AppPaths) -> BatchProjection:
    del paths
    return _projection()


def test_run_json_output_is_locale_neutral(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(batch_command, "run", _run_success)
    monkeypatch.setattr(batch_command, "projection_payload", _payload)
    monkeypatch.setattr(cli_entry, "create_asr_job_snapshot", _snapshot)
    assert (
        cli_entry.main(
            [
                "run",
                "input.wav",
                "second.wav",
                "--output",
                str(tmp_path),
                "--model",
                "model",
                "--json",
                "--language",
                "en",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["batch_id"] == "batch-test"
    assert payload["jobs"] == []


def test_run_human_output_remains_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(batch_command, "run", _run_success)
    monkeypatch.setattr(batch_command, "projection_payload", _payload)
    monkeypatch.setattr(cli_entry, "create_asr_job_snapshot", _snapshot)
    assert (
        cli_entry.main(
            [
                "--lang",
                "zh-CN",
                "run",
                "input.wav",
                "--output",
                str(tmp_path),
                "--model",
                "model",
            ]
        )
        == 0
    )
    assert "batch_id: batch-test" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (AppError("media.ffprobe_failed"), 3),
        (AppError("asr.runtime_missing"), 4),
        (AppError("output.write_failed"), 5),
    ],
)
def test_run_maps_structured_errors_to_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    error: AppError,
    expected: int,
) -> None:
    def fail(_options: batch_command.BatchRunOptions, *, paths: AppPaths) -> BatchProjection:
        del paths
        raise error

    monkeypatch.setattr(batch_command, "run", fail)
    monkeypatch.setattr(cli_entry, "create_asr_job_snapshot", _snapshot)
    assert (
        cli_entry.main(
            [
                "run",
                "input.wav",
                "--output",
                str(tmp_path),
                "--model",
                "model",
                "--json",
            ]
        )
        == expected
    )
    assert json.loads(capsys.readouterr().err)["code"] == error.code


def test_phase2_command_parsers() -> None:
    parser = cli_entry.build_parser()
    assert parser.parse_args(["status", "batch-a", "--json"]).command == "status"
    assert parser.parse_args(["resume", "batch-a"]).command == "resume"
    assert (
        parser.parse_args(["retry", "batch-a", "--job", "job-000001", "--stage", "segment"]).command
        == "retry"
    )
    assert parser.parse_args(["cancel", "batch-a", "--job", "job-000001"]).command == "cancel"
