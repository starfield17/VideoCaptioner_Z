from __future__ import annotations

from pathlib import Path

from scripts.check_forbidden_patterns import check_source


def test_checker_ignores_comments_and_strings() -> None:
    source = """
text = "except Exception: pass"
# except: pass
# type: ignore[reportUnusedImport] because this is a fixture
"""
    assert check_source(source, Path("src/captioner/cli/example.py")) == []


def test_checker_reports_ast_and_comment_violations() -> None:
    source = """
import openai

try:
    value = 1
except Exception:
    pass

print(value)
write_golden(value)
# type: ignore
# noqa
"""
    violations = check_source(source, Path("src/captioner/core/domain/example.py"))
    messages = {violation.message for violation in violations}
    assert "exception is silently swallowed" in messages
    assert "core code uses print()" in messages
    assert "automatic golden-file update" in messages
    assert "domain imports openai" in messages
    assert "type: ignore needs a rule number" in messages
    assert "noqa needs an explanation" in messages


def test_checker_detects_gui_sdk_and_command_run_imports() -> None:
    gui_violations = check_source(
        "import torch\n",
        Path("src/captioner/gui/bad.py"),
    )
    command_violations = check_source(
        "from captioner.cli.commands.other import run\n",
        Path("src/captioner/cli/commands/bad.py"),
    )
    assert any("GUI imports torch" in violation.message for violation in gui_violations)
    assert any("another command run" in violation.message for violation in command_violations)


def test_checker_detects_shell_float_timestamps_and_model_downloads() -> None:
    source = """
import subprocess
subprocess.run([\"ffmpeg\"], shell=True)
model.from_pretrained(\"tiny\")
"""
    violations = check_source(source, Path("tests/unit/test_bad.py"))
    messages = {violation.message for violation in violations}
    assert "shell=True is forbidden" in messages
    assert "unit test downloads a model" in messages
    domain_violations = check_source(
        "class Word:\n    start_ms: float\n", Path("src/captioner/core/domain/bad.py")
    )
    assert any("timestamp field uses float" in item.message for item in domain_violations)
