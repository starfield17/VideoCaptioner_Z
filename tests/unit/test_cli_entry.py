from __future__ import annotations

import json

import pytest

from captioner.cli.cli_entry import build_parser, main


def test_cli_help_and_parser() -> None:
    assert build_parser().prog == "captioner"
    with pytest.raises(SystemExit) as raised:
        main(["--help"])
    assert raised.value.code == 0


def test_doctor_json_and_locale(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--lang", "zh-CN", "doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["locale"] == "zh-CN"
    assert payload["catalog_valid"] is True


def test_default_command_and_invalid_locale(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--lang", "en"]) == 0
    assert "locale: en" in capsys.readouterr().out
    assert main(["--lang", "fr-FR", "doctor"]) == 2
    assert "locale_unsupported" in capsys.readouterr().err
