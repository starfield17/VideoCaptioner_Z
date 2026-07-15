from __future__ import annotations

import json
from pathlib import Path

import pytest

from captioner.cli import cli_entry
from captioner.cli.commands import run as run_command
from captioner.core.application.run_single import RunSingleResult
from captioner.core.domain.errors import AppError
from captioner.infrastructure.app_paths import AppPaths


def _result(tmp_path: Path) -> RunSingleResult:
    return RunSingleResult(
        media_id="media-test",
        transcript_id="transcript-test",
        transcript_path=tmp_path / "input.transcript.json",
        subtitle_path=tmp_path / "input.srt",
        detected_language="en",
        word_count=2,
        cue_count=1,
    )


def _successful_execute(
    _options: run_command.RunOptions, *, paths: AppPaths, result: RunSingleResult
) -> RunSingleResult:
    del paths
    return result


def test_run_json_output_is_locale_neutral(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _result(tmp_path)

    def execute(options: run_command.RunOptions, *, paths: AppPaths) -> RunSingleResult:
        return _successful_execute(options, paths=paths, result=result)

    monkeypatch.setattr(run_command, "execute", execute)
    assert (
        cli_entry.main(
            ["run", "input.wav", "--output", str(tmp_path), "--json", "--language", "en"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["media_id"] == "media-test"
    assert set(payload) == {
        "media_id",
        "transcript_id",
        "transcript_path",
        "subtitle_path",
        "detected_language",
        "word_count",
        "cue_count",
    }


def test_run_human_output_is_localized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _result(tmp_path)

    def execute(options: run_command.RunOptions, *, paths: AppPaths) -> RunSingleResult:
        return _successful_execute(options, paths=paths, result=result)

    monkeypatch.setattr(run_command, "execute", execute)
    assert cli_entry.main(["--lang", "zh-CN", "run", "input.wav", "--output", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "媒体 ID:" in output
    assert "字幕条数:" in output


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
    def fail(_options: run_command.RunOptions, *, paths: AppPaths) -> RunSingleResult:
        del paths
        raise error

    monkeypatch.setattr(run_command, "execute", fail)
    assert cli_entry.main(["run", "input.wav", "--output", str(tmp_path), "--json"]) == expected
    assert json.loads(capsys.readouterr().err)["code"] == error.code
