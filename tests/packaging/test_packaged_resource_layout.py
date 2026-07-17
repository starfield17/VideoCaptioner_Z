from __future__ import annotations

from pathlib import Path

from scripts.build_nuitka import layout_for_platform, stage_artifact, verify_layout


def test_staged_linux_layout_contains_executable_resources_and_readme(tmp_path: Path) -> None:
    layout = layout_for_platform("linux", dist_root=tmp_path / "dist")
    artifact = layout.work_root / "captioner.dist"
    (artifact / "resources" / "i18n").mkdir(parents=True)
    (artifact / "resources" / "tokenizers").mkdir(parents=True)
    (artifact / "captioner").write_text("binary", encoding="utf-8")
    (artifact / "resources" / "i18n" / "en.json").write_text("{}", encoding="utf-8")
    for filename in (
        "tokenizer-manifest.json",
        "cl100k_base.tiktoken",
        "o200k_base.tiktoken",
    ):
        (artifact / "resources" / "tokenizers" / filename).write_bytes(b"resource")
    (artifact / "README.md").write_text("readme", encoding="utf-8")
    stage_artifact(layout, artifact)
    verify_layout(layout)
    assert layout.executable_path.is_file()
    assert (layout.resource_root / "i18n" / "en.json").is_file()
    # Staging prefers move over copytree, so the work-tree artifact is consumed.
    assert not artifact.exists()
