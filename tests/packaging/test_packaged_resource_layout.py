from __future__ import annotations

from pathlib import Path

from scripts.build_nuitka import layout_for_platform, stage_artifact, verify_layout

ROOT = Path(__file__).resolve().parents[2]


def test_staged_linux_layout_contains_executable_resources_and_readme(tmp_path: Path) -> None:
    layout = layout_for_platform("linux", target="cli", dist_root=tmp_path / "dist")
    artifact = layout.work_root / "captioner.dist"
    artifact.mkdir(parents=True)
    (artifact / "captioner").write_text("binary", encoding="utf-8")
    # Fake Nuitka output is executable-only; resources come from the project root.
    stage_artifact(layout, artifact, project_root=ROOT)
    verify_layout(layout)
    assert layout.executable_path.is_file()
    assert (layout.resource_root / "i18n" / "en.json").is_file()
    assert (layout.resource_root / "prompts").is_dir()
    assert (layout.resource_root / "runtime").is_dir()
    assert layout.notice_path.is_file()
    # Staging prefers move over copytree, so the work-tree artifact is consumed.
    assert not artifact.exists()


def test_macos_desktop_layout_stages_bundle_resources_from_project(tmp_path: Path) -> None:
    layout = layout_for_platform("macos", target="desktop", dist_root=tmp_path / "dist")
    artifact = layout.work_root / "Captioner.app"
    executable = artifact / "Contents" / "MacOS" / "captioner"
    executable.parent.mkdir(parents=True)
    executable.write_text("binary", encoding="utf-8")
    # Do not pre-create Contents/Resources/resources — wrapper owns staging.
    stage_artifact(layout, artifact, project_root=ROOT)
    verify_layout(layout)
    assert layout.executable_path.is_file()
    assert layout.notice_path.is_file()
    assert (layout.resource_root / "tokenizers" / "tokenizer-manifest.json").is_file()
