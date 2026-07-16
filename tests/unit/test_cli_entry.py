from __future__ import annotations

import json

import pytest

from captioner import __version__
from captioner.cli.cli_entry import build_parser, main


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
