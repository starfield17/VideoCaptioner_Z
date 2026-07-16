from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from tests.golden.manifest import GoldenManifestError, verify_manifest

ROOT = Path(__file__).parent / "data"
FIXTURES = Path(__file__).parents[1] / "fixtures" / "transcripts"


def _copy_manifest_data(tmp_path: Path) -> Path:
    target = tmp_path / "data"
    shutil.copytree(ROOT, target)
    return target


def test_golden_manifest_matches_exact_file_set_and_hashes() -> None:
    verify_manifest(ROOT, FIXTURES)


def test_golden_manifest_rejects_hash_change(tmp_path: Path) -> None:
    root = _copy_manifest_data(tmp_path)
    path = root / "short_words.srt"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(GoldenManifestError):
        verify_manifest(root, FIXTURES)


def test_golden_manifest_rejects_extra_file(tmp_path: Path) -> None:
    root = _copy_manifest_data(tmp_path)
    (root / "unexpected.srt").write_bytes(b"unexpected")
    with pytest.raises(GoldenManifestError):
        verify_manifest(root, FIXTURES)


@pytest.mark.parametrize("change", ["policy", "exporter"])
def test_golden_manifest_rejects_policy_or_exporter_version_change(
    tmp_path: Path, change: str
) -> None:
    root = _copy_manifest_data(tmp_path)
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if change == "policy":
        document["policy_signature"] = "policy-" + "0" * 64
    else:
        document["exporter_versions"]["ass"] = "ass-v99"
    (root / "manifest.json").write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    with pytest.raises(GoldenManifestError):
        verify_manifest(root, FIXTURES)
