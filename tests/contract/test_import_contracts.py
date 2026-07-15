from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("source_module", "forbidden_module"),
    [
        ("captioner.gui", "captioner.cli"),
        ("captioner.cli", "captioner.gui"),
        ("captioner.core.application", "captioner.adapters"),
        ("captioner.core.application", "captioner.cli"),
    ],
)
def test_forbidden_direction_fixtures_fail_import_linter(
    tmp_path: Path, source_module: str, forbidden_module: str
) -> None:
    """Each explicit contract must fail when an isolated fixture violates it."""
    package_root = tmp_path / "captioner"
    source_path = package_root.joinpath(*source_module.split(".")[1:])
    package_root.mkdir()
    package_paths = [package_root]
    for module in {source_module, forbidden_module}:
        parts = module.split(".")[1:]
        package_paths.extend(
            package_root.joinpath(*parts[:index]) for index in range(1, len(parts) + 1)
        )
    for package_path in set(package_paths):
        package_path.mkdir(parents=True, exist_ok=True)
        (package_path / "__init__.py").write_text("", encoding="utf-8")
    (source_path / "bad.py").write_text(
        f"from {forbidden_module} import marker\n",
        encoding="utf-8",
    )
    config = tmp_path / ".importlinter"
    config.write_text(
        "\n".join(
            (
                "[importlinter]",
                "root_package = captioner",
                "include_external_packages = true",
                "",
                "[importlinter:contract:fixture]",
                "name = Fixture forbidden direction",
                "type = forbidden",
                f"source_modules =\n    {source_module}",
                f"forbidden_modules =\n    {forbidden_module}",
                "",
            )
        ),
        encoding="utf-8",
    )
    executable = shutil.which("lint-imports")
    assert executable is not None
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [executable, "--config", str(config), "--no-cache"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout + result.stderr
