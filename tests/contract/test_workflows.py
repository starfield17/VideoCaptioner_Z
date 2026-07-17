from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _workflow(path: str) -> tuple[dict[str, Any], str]:
    workflow_path = ROOT / path
    text = workflow_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict)
    return cast(dict[str, Any], parsed), text


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    # PyYAML's YAML 1.1 loader exposes the GitHub key ``on`` as True.
    value: object = workflow.get("on")
    if value is None:
        value = cast(dict[object, Any], workflow).get(True)
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def test_fast_gate_yaml_and_scope() -> None:
    workflow, text = _workflow(".github/workflows/ci.yml")
    triggers = _triggers(workflow)
    assert workflow["name"] == "Fast Gate"
    assert "pull_request" in triggers
    assert triggers["push"] == {"branches": ["main"]}
    assert workflow["concurrency"]["group"] == (
        "fast-gate-${{ github.event.pull_request.number || github.ref }}"
    )
    jobs = workflow["jobs"]
    assert set(jobs) == {"fast-gate"}
    assert jobs["fast-gate"]["runs-on"] == "ubuntu-24.04"
    assert "strategy" not in jobs["fast-gate"]
    assert "scripts/check.py --fast" in text
    lowered = text.lower()
    for forbidden in ("nuitka", "build_nuitka", "coverage", "ffmpeg", "matrix"):
        assert forbidden not in lowered


def test_release_full_gate_yaml_and_scope() -> None:
    workflow, text = _workflow(".github/workflows/release-full.yml")
    triggers = _triggers(workflow)
    assert workflow["name"] == "Release Full Gate"
    assert "workflow_dispatch" in triggers
    assert triggers["push"] == {"tags": ["v*"]}
    jobs = workflow["jobs"]
    assert {"full-checks", "package-ubuntu", "package-windows", "package-macos"} <= set(jobs)
    assert jobs["package-ubuntu"]["needs"] == "full-checks"
    assert jobs["package-windows"]["needs"] == "full-checks"
    assert jobs["package-macos"]["needs"] == "full-checks"
    assert {jobs[name]["runs-on"] for name in jobs} == {
        "ubuntu-24.04",
        "windows-2022",
        "macos-15",
    }
    assert "scripts/check.py --full" in text
    assert "build_nuitka.py" in text
    assert "actions/upload-artifact@v4" in text
