from __future__ import annotations

from pathlib import Path

from captioner.infrastructure.app_paths import ensure_runtime_layout, resolve_app_paths


def _make_resources(root: Path) -> None:
    for directory in ("i18n", "prompts", "runtime", "tokenizers"):
        (root / directory).mkdir(parents=True)
    (root / "i18n" / "en.json").write_text("{}", encoding="utf-8")
    (root / "tokenizers" / "tokenizer-manifest.json").write_text("{}", encoding="utf-8")
    (root / "tokenizers" / "cl100k_base.tiktoken").write_bytes(b"cl100k")
    (root / "tokenizers" / "o200k_base.tiktoken").write_bytes(b"o200k")


def test_resources_are_read_only_boundary(tmp_path: Path) -> None:
    resource_root = tmp_path / "bundle" / "resources"
    _make_resources(resource_root)
    paths = resolve_app_paths(
        base_dir=tmp_path / "user",
        resource_root_override=resource_root,
        compiled=True,
        executable_path=tmp_path / "bundle" / "captioner",
    )
    ensure_runtime_layout(paths)
    (paths.data_dir / "state.json").write_text("{}", encoding="utf-8")
    assert (paths.data_dir / "state.json").is_file()
    assert not (resource_root / "state.json").exists()
    assert paths.data_dir != resource_root
