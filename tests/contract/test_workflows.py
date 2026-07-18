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


def _job_steps(job: object) -> list[dict[str, Any]]:
    assert isinstance(job, dict)
    typed_job = cast(dict[str, Any], job)
    raw_value: object = typed_job.get("steps")
    assert isinstance(raw_value, list)
    raw_steps = cast(list[object], raw_value)
    steps: list[dict[str, Any]] = []
    for raw_step in raw_steps:
        assert isinstance(raw_step, dict)
        steps.append(cast(dict[str, Any], raw_step))
    return steps


def _step(job: object, name: str) -> dict[str, Any]:
    for step in _job_steps(job):
        if step.get("name") == name:
            return step
    raise AssertionError


def _run(step: dict[str, Any]) -> str:
    value = step.get("run", "")
    assert isinstance(value, str)
    return value


def _assert_archive_smoke_order(job: object, platform_name: str) -> None:
    expected = (
        f"Pre-archive {platform_name} compiled smoke",
        f"Archive {platform_name} distribution",
        f"Extract {platform_name} archive",
        f"Post-extraction {platform_name} compiled smoke",
        f"Upload {platform_name} archive",
    )
    names = [str(step.get("name")) for step in _job_steps(job)]
    positions = [names.index(name) for name in expected]
    assert positions == sorted(positions)


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

    ubuntu = jobs["package-ubuntu"]
    windows = jobs["package-windows"]
    macos = jobs["package-macos"]
    _assert_archive_smoke_order(ubuntu, "Ubuntu")
    _assert_archive_smoke_order(windows, "Windows")
    _assert_archive_smoke_order(macos, "macOS")

    assert jobs["package-ubuntu"]["name"] == "ubuntu-24.04-cli-package"
    assert jobs["package-windows"]["name"] == "windows-2022-cli-package"
    assert jobs["package-macos"]["name"] == "macos-15-cli-package"
    assert text.count("--target cli") == 3
    assert "--gui" not in text
    assert "Captioner.app" not in text

    ubuntu_pre = _run(_step(ubuntu, "Pre-archive Ubuntu compiled smoke"))
    ubuntu_archive = _run(_step(ubuntu, "Archive Ubuntu distribution"))
    ubuntu_extract = _run(_step(ubuntu, "Extract Ubuntu archive"))
    ubuntu_post = _run(_step(ubuntu, "Post-extraction Ubuntu compiled smoke"))
    ubuntu_upload = _step(ubuntu, "Upload Ubuntu archive")
    assert "$GITHUB_WORKSPACE/dist/captioner/captioner" in ubuntu_pre
    assert "tar -C dist -czf dist/captioner-cli-linux.tar.gz captioner" in ubuntu_archive
    assert "tar -xzf dist/captioner-cli-linux.tar.gz" in ubuntu_extract
    assert 'test -x "$EXTRACTED"' in ubuntu_post
    assert "--cli" not in ubuntu_pre
    assert "--cli" not in ubuntu_post
    assert ubuntu_upload["with"]["path"] == "dist/captioner-cli-linux.tar.gz"
    assert ubuntu_upload["with"]["name"] == "captioner-cli-ubuntu-${{ github.run_id }}"
    assert ubuntu_upload["with"]["path"] != "dist/captioner"

    windows_pre = _run(_step(windows, "Pre-archive Windows compiled smoke"))
    windows_archive = _run(_step(windows, "Archive Windows distribution"))
    windows_extract = _run(_step(windows, "Extract Windows archive"))
    windows_post = _run(_step(windows, "Post-extraction Windows compiled smoke"))
    windows_upload = _step(windows, "Upload Windows archive")
    assert "dist\\captioner\\captioner.exe" in windows_pre
    assert "Compress-Archive" in windows_archive
    assert "dist\\captioner-cli-windows.zip" in windows_archive
    assert "Expand-Archive" in windows_extract
    assert "Test-Path $Extracted" in windows_post
    assert 'Join-Path $env:CAPTIONER_EXTRACT_ROOT "captioner"' in windows_post
    assert 'Join-Path $ExtractedRoot "THIRD_PARTY_NOTICES.md"' in windows_post
    assert (
        'Join-Path $ExtractedRoot "resources\\tokenizers\\tokenizer-manifest.json"' in windows_post
    )
    assert 'Join-Path $ExtractedRoot "resources\\tokenizers\\cl100k_base.tiktoken"' in windows_post
    assert 'Join-Path $ExtractedRoot "resources\\tokenizers\\o200k_base.tiktoken"' in windows_post
    assert 'Join-Path $env:CAPTIONER_EXTRACT_ROOT "THIRD_PARTY_NOTICES.md"' not in windows_post
    assert "--cli" not in windows_pre
    assert "--cli" not in windows_post
    assert windows_upload["with"]["path"] == "dist/captioner-cli-windows.zip"
    assert windows_upload["with"]["name"] == "captioner-cli-windows-${{ github.run_id }}"

    macos_pre = _run(_step(macos, "Pre-archive macOS compiled smoke"))
    macos_archive = _run(_step(macos, "Archive macOS distribution"))
    macos_extract = _run(_step(macos, "Extract macOS archive"))
    macos_post = _run(_step(macos, "Post-extraction macOS compiled smoke"))
    macos_upload = _step(macos, "Upload macOS archive")
    macos_executable = "$GITHUB_WORKSPACE/dist/captioner/captioner"
    assert macos_executable in macos_pre
    assert "ditto -c -k --keepParent" in macos_archive
    assert "dist/captioner" in macos_archive
    assert "dist/captioner-cli-macos.zip" in macos_archive
    assert "ditto -x -k dist/captioner-cli-macos.zip" in macos_extract
    assert 'test -x "$EXTRACTED"' in macos_post
    assert "$CAPTIONER_EXTRACT_ROOT/captioner/captioner" in macos_post
    assert "--cli" not in macos_pre
    assert "--cli" not in macos_post
    assert macos_upload["with"]["path"] == "dist/captioner-cli-macos.zip"
    assert macos_upload["with"]["name"] == "captioner-cli-macos-${{ github.run_id }}"

    for package_job in (ubuntu, windows, macos):
        upload = next(
            step
            for step in _job_steps(package_job)
            if str(step.get("name", "")).startswith("Upload ")
        )
        assert not str(upload["with"]["path"]).endswith("/captioner")
