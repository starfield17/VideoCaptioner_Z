from __future__ import annotations

import json
from pathlib import Path

from tests.fakes.phase6_values import model_manifest

from captioner.adapters.model.filesystem_local_model_inspector import (
    FilesystemLocalModelInspector,
)
from captioner.adapters.model.filesystem_model_validator import FilesystemModelValidator


def _write_ct2(root: Path) -> None:
    root.mkdir(exist_ok=True)
    (root / "config.json").write_text('{"model_type":"whisper"}', encoding="utf-8")
    (root / "tokenizer.json").write_text('{"version":1}', encoding="utf-8")
    (root / "model.bin").write_bytes(b"ct2 weights")


def _write_mlx(root: Path, *, tokenizer_json: bool = True) -> None:
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"whisper"}', encoding="utf-8")
    (root / "weights.npz").write_bytes(b"mlx weights")
    if tokenizer_json:
        (root / "tokenizer.json").write_text('{"version":1}', encoding="utf-8")
    else:
        (root / "vocab.json").write_text("{}", encoding="utf-8")
        (root / "merges.txt").write_text("", encoding="utf-8")


def test_valid_ct2_payload_is_offline_loadable_by_static_contract(tmp_path: Path) -> None:
    root = tmp_path / "ct2"
    _write_ct2(root)
    validator = FilesystemModelValidator()
    inspector = FilesystemLocalModelInspector(validator)

    inspection = inspector.inspect(root)
    manifest = model_manifest(files=inspection.file_inventory)

    assert inspection.detected_backend_id == "faster-whisper"
    assert inspection.detected_model_format == "faster-whisper-ct2"
    assert inspection.validation_passed
    assert validator.validate(manifest, root).ok


def test_ct2_requires_model_bin_and_valid_tokenizer_json(tmp_path: Path) -> None:
    root = tmp_path / "ct2"
    _write_ct2(root)
    validator = FilesystemModelValidator()
    manifest = model_manifest(files=validator.inventory(root))

    (root / "model.bin").unlink()
    missing = validator.validate(manifest, root)
    assert not missing.ok
    assert missing.error_code == "model.extra_or_missing_file"

    _write_ct2(root)
    (root / "tokenizer.json").write_text("not-json", encoding="utf-8")
    invalid = validator.validate(manifest, root)
    assert not invalid.ok
    assert invalid.error_code == "model.file_hash_mismatch"
    assert any(check.error_code == "model.json_invalid" for check in invalid.checks)


def test_mlx_accepts_weights_npz_and_vocab_merges_tokenizer_fallback(tmp_path: Path) -> None:
    root = tmp_path / "mlx"
    _write_mlx(root, tokenizer_json=False)
    validator = FilesystemModelValidator()
    inspector = FilesystemLocalModelInspector(validator)

    inspection = inspector.inspect(root)
    manifest = model_manifest(
        backend_id="mlx-whisper",
        model_format="mlx-whisper",
        repository_id="org/mlx",
        files=inspection.file_inventory,
    )

    assert inspection.validation_passed
    assert validator.validate(manifest, root).ok


def test_inspector_rejects_ambiguous_unknown_and_missing_mlx_tokenizer(tmp_path: Path) -> None:
    ambiguous = tmp_path / "ambiguous"
    _write_ct2(ambiguous)
    (ambiguous / "model.safetensors").write_bytes(b"mlx")
    inspector = FilesystemLocalModelInspector()
    assert inspector.inspect(ambiguous).validation_report.error_code == "model.format_ambiguous"

    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "config.json").write_text("{}", encoding="utf-8")
    assert inspector.inspect(unknown).validation_report.error_code == "model.format_unknown"

    missing_tokenizer = tmp_path / "missing-tokenizer"
    _write_mlx(missing_tokenizer, tokenizer_json=True)
    (missing_tokenizer / "tokenizer.json").unlink()
    assert (
        inspector.inspect(missing_tokenizer).validation_report.error_code
        == "model.mlx_tokenizer_missing"
    )


def test_validator_rejects_symlinks_hidden_files_and_extra_manifest_files(tmp_path: Path) -> None:
    root = tmp_path / "ct2"
    _write_ct2(root)
    validator = FilesystemModelValidator()
    manifest = model_manifest(files=validator.inventory(root))

    (root / "extra.json").write_text("{}", encoding="utf-8")
    assert validator.validate(manifest, root).error_code == "model.extra_or_missing_file"
    (root / "extra.json").unlink()
    (root / ".cache").write_text("cache", encoding="utf-8")
    assert validator.validate(manifest, root).error_code == "model.hidden_file_rejected"
    (root / ".cache").unlink()
    (root / "link").symlink_to(root / "model.bin")
    assert validator.validate(manifest, root).error_code == "model.symlink_rejected"


def test_validator_rejects_non_object_or_oversized_json(tmp_path: Path) -> None:
    root = tmp_path / "ct2"
    _write_ct2(root)
    validator = FilesystemModelValidator(max_json_bytes=8)
    manifest = model_manifest(files=FilesystemModelValidator().inventory(root))
    (root / "config.json").write_text(json.dumps(["too", "large"]), encoding="utf-8")

    report = validator.validate(manifest, root)
    assert not report.ok
    assert any(check.error_code == "model.json_invalid" for check in report.checks)
