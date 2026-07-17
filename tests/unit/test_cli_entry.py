from __future__ import annotations

import json
from pathlib import Path

import pytest

from captioner import __version__
from captioner.cli.cli_entry import build_parser, main
from captioner.cli.commands import batch as batch_command
from captioner.core.domain.batch import BatchProjection
from captioner.infrastructure.app_paths import AppPaths


def test_cli_help_and_parser() -> None:
    assert build_parser().prog == "captioner"
    with pytest.raises(SystemExit) as raised:
        main(["--help"])
    assert raised.value.code == 0


def test_doctor_json_and_locale(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--lang", "zh-CN", "doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == __version__
    assert payload["locale"] == "zh-CN"
    assert payload["catalog_valid"] is True
    assert set(payload) == {
        "version",
        "python_version",
        "platform",
        "resource_root",
        "config_dir",
        "data_dir",
        "cache_dir",
        "log_dir",
        "temp_dir",
        "locale",
        "catalog_valid",
    }


def test_doctor_human_output_uses_locale_labels(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--lang", "en", "doctor"]) == 0
    english = capsys.readouterr().out
    assert "Version:" in english
    assert "Config directory:" in english

    assert main(["--lang", "zh-CN", "doctor"]) == 0
    chinese = capsys.readouterr().out
    assert "版本:" in chinese
    assert "配置目录:" in chinese


def test_doctor_tokenizer_smoke_initializes_both_packaged_encodings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["doctor", "--tokenizer-smoke", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["tokenizers"]) == {"cl100k_base", "o200k_base"}
    assert all(isinstance(value, int) and value > 0 for value in payload["tokenizers"].values())


def test_default_command_and_invalid_locale(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--lang", "en"]) == 0
    assert "Locale: en" in capsys.readouterr().out
    assert main(["--lang", "fr-FR", "doctor"]) == 2
    assert "locale_unsupported" in capsys.readouterr().err


def test_subtitle_corpus_cli_runs_all_fixtures(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["subtitle-corpus", "tests/fixtures/transcripts", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["fixture_count"] == 14
    assert payload["passed"] == 14
    assert payload["failed"] == 0
    assert payload["errors"] == []


def test_run_language_auto_is_explicit_none_and_omitted_is_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    languages: list[str | None] = []

    def capture_run(options: batch_command.BatchRunOptions, *, paths: AppPaths) -> BatchProjection:
        del paths
        languages.append(options.language)
        return BatchProjection("batch-test", last_event_seq=1)

    def payload(_projection: BatchProjection, *, paths: AppPaths) -> dict[str, object]:
        del paths
        return {"batch_id": "batch-test"}

    monkeypatch.setattr(batch_command, "run", capture_run)
    monkeypatch.setattr(batch_command, "projection_payload", payload)
    assert main(["run", "input.wav", "--output", str(tmp_path)]) == 0
    assert main(["run", "input.wav", "--output", str(tmp_path), "--language", "auto"]) == 0
    assert languages == [None, None]


def test_resume_language_states_distinguish_omitted_auto_and_explicit_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[batch_command.ResumeOverrides] = []

    def capture_resume(
        batch_id: str,
        *,
        paths: AppPaths,
        overrides: batch_command.ResumeOverrides,
    ) -> BatchProjection:
        del batch_id, paths
        captured.append(overrides)
        return BatchProjection("batch-test", last_event_seq=1)

    def payload(_projection: BatchProjection, *, paths: AppPaths) -> dict[str, object]:
        del paths
        return {"batch_id": "batch-test"}

    monkeypatch.setattr(batch_command, "resume", capture_resume)
    monkeypatch.setattr(batch_command, "projection_payload", payload)
    assert main(["resume", "batch-test"]) == 0
    assert main(["resume", "batch-test", "--language", "auto"]) == 0
    assert main(["resume", "batch-test", "--language", "ja"]) == 0
    assert not captured[0].has_language_override
    assert captured[1].has_language_override and captured[1].language is None
    assert captured[2].has_language_override and captured[2].language == "ja"


def test_language_detect_is_rejected() -> None:
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(
            ["run", "input.wav", "--output", "output", "--language", "detect"]
        )
    assert raised.value.code == 2
