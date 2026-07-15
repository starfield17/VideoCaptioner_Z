from __future__ import annotations

from collections.abc import Sequence

import pytest

from captioner import entrypoint


def test_dispatch_uses_first_argument_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    def fake_gui(arguments: Sequence[str]) -> int:
        calls.append(("gui", list(arguments)))
        return 11

    def fake_cli(arguments: Sequence[str]) -> int:
        calls.append(("cli", list(arguments)))
        return 22

    monkeypatch.setattr(entrypoint, "_run_gui", fake_gui)
    monkeypatch.setattr(entrypoint, "_run_cli", fake_cli)

    assert entrypoint.main([]) == 11
    assert entrypoint.main(["--gui", "--smoke-test"]) == 11
    assert entrypoint.main(["--cli", "doctor", "--json"]) == 22
    assert entrypoint.main(["doctor", "--json"]) == 22
    assert entrypoint.main(["doctor", "--gui"]) == 22
    assert calls == [
        ("gui", []),
        ("gui", ["--smoke-test"]),
        ("cli", ["doctor", "--json"]),
        ("cli", ["doctor", "--json"]),
        ("cli", ["doctor", "--gui"]),
    ]


def test_nonfirst_gui_flag_fails_through_cli_parser(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_gui(_arguments: Sequence[str]) -> int:
        pytest.fail("GUI selected")

    monkeypatch.setattr(entrypoint, "_run_gui", fail_gui)
    with pytest.raises(SystemExit) as raised:
        entrypoint.main(["doctor", "--gui"])
    assert raised.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
