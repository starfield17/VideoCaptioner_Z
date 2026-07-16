from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from scripts.update_subtitle_goldens import EXPORTER_VERSIONS

from captioner.core.policies.segmentation_config import SegmentationPolicyConfig

_MANIFEST_FIELDS = {"schema_version", "policy_signature", "exporter_versions", "golden_sha256"}
_GOLDEN_SUFFIXES = ("track.json", "srt", "vtt", "ass")


class GoldenManifestError(AssertionError):
    pass


def verify_manifest(root: Path, fixtures: Path) -> None:
    fixture_paths = tuple(sorted(fixtures.glob("*.json"), key=lambda path: path.as_posix()))
    fixture_names = [path.stem for path in fixture_paths]
    if len(fixture_names) != len({name.casefold() for name in fixture_names}):
        raise GoldenManifestError
    expected_files = tuple(
        sorted(f"{stem}.{suffix}" for stem in fixture_names for suffix in _GOLDEN_SUFFIXES)
    )
    actual_files = tuple(
        sorted(
            path.name for path in root.iterdir() if path.is_file() and path.name != "manifest.json"
        )
    )
    if actual_files != expected_files:
        raise GoldenManifestError

    try:
        document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GoldenManifestError from exc
    if not isinstance(document, dict):
        raise GoldenManifestError
    document = cast(dict[str, object], document)
    if set(document) != _MANIFEST_FIELDS:
        raise GoldenManifestError
    if document.get("schema_version") != 1:
        raise GoldenManifestError
    if document.get("policy_signature") != SegmentationPolicyConfig().signature:
        raise GoldenManifestError
    if document.get("exporter_versions") != EXPORTER_VERSIONS:
        raise GoldenManifestError
    hashes = document.get("golden_sha256")
    if not isinstance(hashes, dict):
        raise GoldenManifestError
    hashes = cast(dict[str, object], hashes)
    if tuple(sorted(hashes)) != expected_files:
        raise GoldenManifestError
    for name in expected_files:
        if Path(name).as_posix() != name or hashes.get(name) != _sha256(root / name):
            raise GoldenManifestError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
