from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from captioner.adapters.model.huggingface_materializer import HuggingFaceModelMaterializer
from captioner.adapters.model.huggingface_source import HuggingFaceModelSource
from captioner.adapters.model.modelscope_materializer import ModelScopeModelMaterializer
from captioner.adapters.model.modelscope_source import ModelScopeModelSource
from captioner.core.domain.errors import AppError
from captioner.core.domain.model import ModelSourceReference
from captioner.infrastructure.model_source_config import ModelSourceConfig


class _HfApi:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def list_models(self, **kwargs: object) -> list[dict[str, object]]:
        self.calls.append(kwargs)
        return [{"id": "org/model", "sha": "a" * 40, "description": "safe"}]

    def model_info(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"sha": "b" * 40}


def test_huggingface_candidate_is_not_a_durable_identity() -> None:
    api = _HfApi()
    source = HuggingFaceModelSource(ModelSourceConfig("huggingface"), api=api)

    candidates = source.search("whisper", "faster-whisper", 10)
    reference = source.resolve_exact("org/model", None, "faster-whisper")

    assert candidates[0].revision == "a" * 40
    assert not hasattr(candidates[0], "manifest_sha256")
    assert reference.revision == "b" * 40
    assert api.calls[1]["files_metadata"] is True


def test_huggingface_infers_format_and_maps_provider_errors() -> None:
    class RevisionNotFoundError(Exception):
        pass

    class Api:
        def list_models(self, **kwargs: object) -> list[dict[str, object]]:
            del kwargs
            return [
                {
                    "id": "org/mlx",
                    "siblings": [{"rfilename": "config.json"}, {"rfilename": "weights.npz"}],
                }
            ]

        def model_info(self, **kwargs: object) -> object:
            del kwargs
            raise RevisionNotFoundError

    source = HuggingFaceModelSource(ModelSourceConfig("huggingface"), api=Api())
    assert source.search("mlx", "mlx-whisper", 1)[0].model_format_hint == "mlx-whisper"
    with pytest.raises(AppError, match=r"model\.source_revision_not_found"):
        source.resolve_exact("org/mlx", "bad-revision", "mlx-whisper")


def test_huggingface_materializer_uses_fixed_revision_and_allowlist(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> None:
        calls.append(kwargs)
        destination = kwargs["local_dir"]
        assert isinstance(destination, Path)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text(json.dumps({}), encoding="utf-8")

    materializer = HuggingFaceModelMaterializer(
        ModelSourceConfig("huggingface"),
        snapshot_download=snapshot_download,
    )
    materializer.materialize(
        ModelSourceReference(
            "huggingface", "org/model", "c" * 40, "faster-whisper", "faster-whisper-ct2"
        ),
        tmp_path / "destination",
    )

    assert calls[0]["revision"] == "c" * 40
    allow_patterns = calls[0]["allow_patterns"]
    assert isinstance(allow_patterns, list)
    assert "model.bin" in allow_patterns
    assert "*.py" not in allow_patterns


def test_modelscope_search_is_explicitly_unsupported() -> None:
    source = ModelScopeModelSource(ModelSourceConfig("modelscope"), api=object())
    assert source.capabilities().search is False
    with pytest.raises(AppError, match=r"model\.source_search_unsupported"):
        source.search("whisper", "faster-whisper", 5)


def test_modelscope_requires_concrete_revision() -> None:
    class Api:
        def get_model(self, *args: object, **kwargs: object) -> dict[str, object]:
            del args, kwargs
            return {"revision": "master"}

    source = ModelScopeModelSource(ModelSourceConfig("modelscope"), api=Api())
    with pytest.raises(AppError, match=r"model\.source_revision_unresolved"):
        source.resolve_exact("org/model", "master", "mlx-whisper", "mlx-whisper")


def test_modelscope_exact_resolution_returns_provider_commit() -> None:
    class Api:
        def get_model(self, *args: object, **kwargs: object) -> dict[str, object]:
            assert args == ("org/model",)
            assert kwargs["revision"] == "v1"
            return {"commit_hash": "d" * 40}

    source = ModelScopeModelSource(ModelSourceConfig("modelscope"), api=Api())
    reference = source.resolve_exact("org/model", "v1", "faster-whisper", "faster-whisper-ct2")
    assert reference.revision == "d" * 40


def test_modelscope_materializer_uses_transaction_local_cache(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> None:
        calls.append(kwargs)
        destination = Path(str(kwargs["local_dir"]))
        (destination / "config.json").write_text("{}", encoding="utf-8")
        (destination / ".modelscope-cache" / "metadata").write_text("cache", encoding="utf-8")

    materializer = ModelScopeModelMaterializer(
        ModelSourceConfig("modelscope"), snapshot_download=download
    )
    destination = tmp_path / "destination"
    materializer.materialize(
        ModelSourceReference(
            "modelscope", "org/model", "e" * 40, "faster-whisper", "faster-whisper-ct2"
        ),
        destination,
    )

    assert calls[0]["revision"] == "e" * 40
    assert calls[0]["model_id"] == "org/model"
    assert not (destination / ".modelscope-cache").exists()
    assert "MODELSCOPE_CACHE" not in os.environ
