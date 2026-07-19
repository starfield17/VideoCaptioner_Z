from pathlib import Path

from scripts.build_runtime import prune_runtime_packages


def test_prune_packages_removes_only_explicit_payload_packages(tmp_path: Path) -> None:
    site_packages = tmp_path / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    for name in (
        "torch",
        "torch-2.13.0.dist-info",
        "networkx",
        "captioner_runtime_worker",
    ):
        path = site_packages / name
        path.mkdir()
        (path / "marker").write_text("payload", encoding="utf-8")

    prune_runtime_packages(tmp_path, ["torch", "networkx"])

    assert not (site_packages / "torch").exists()
    assert not (site_packages / "torch-2.13.0.dist-info").exists()
    assert not (site_packages / "networkx").exists()
    assert (site_packages / "captioner_runtime_worker").exists()
