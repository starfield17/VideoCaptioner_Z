"""Hugging Face fixed-revision materializer."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from contextlib import redirect_stdout
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from captioner.core.domain.errors import AppError
from captioner.core.domain.model import ModelSourceReference
from captioner.core.domain.operation_progress import OperationProgress
from captioner.core.ports.model_source import ModelMaterializer, ProgressCallback
from captioner.infrastructure.model_source_config import ModelSourceConfig

HF_CT2_ALLOW_PATTERNS = (
    "config.json",
    "model.bin",
    "tokenizer.json",
    "preprocessor_config.json",
    "vocabulary.json",
    "vocabulary.txt",
    "vocabulary.*",
    "README.md",
    "LICENSE*",
)
HF_MLX_ALLOW_PATTERNS = (
    "config.json",
    "model.safetensors",
    "weights.safetensors",
    "weights.npz",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "preprocessor_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "README.md",
    "LICENSE*",
)


class _SnapshotDownload(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


class HuggingFaceModelMaterializer(ModelMaterializer):
    def __init__(
        self,
        config: ModelSourceConfig,
        *,
        snapshot_download: Callable[..., object] | None = None,
    ) -> None:
        if config.source_id != "huggingface":
            raise AppError("model.source_config_invalid", {"field": "source_id"})
        self.config = config
        self._snapshot_download = snapshot_download

    def materialize(
        self,
        reference: ModelSourceReference,
        destination: Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> None:
        _ensure_enabled(self.config.enabled)
        if reference.source_id != "huggingface":
            raise AppError("model.source_reference_invalid")
        if not reference.revision:
            raise AppError("model.source_revision_unresolved")
        if destination.expanduser().is_symlink():
            raise AppError("model.symlink_rejected")
        destination = destination.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        _emit(progress, "resolving_source")
        downloader = self._downloader()
        allow_patterns = (
            HF_MLX_ALLOW_PATTERNS
            if reference.model_format_hint == "mlx-whisper"
            else HF_CT2_ALLOW_PATTERNS
        )
        _emit(progress, "downloading")
        try:
            with redirect_stdout(sys.stderr):
                downloader(
                    repo_id=reference.repository_id,
                    revision=reference.revision,
                    local_dir=destination,
                    token=self.config.token,
                    max_workers=self.config.max_workers,
                    allow_patterns=list(allow_patterns),
                    tqdm_class=None,
                    endpoint=self.config.endpoint,
                )
        except TypeError:
            # Older hub versions do not accept ``tqdm_class``.  The fixed
            # revision and local destination remain mandatory in either API.
            try:
                with redirect_stdout(sys.stderr):
                    downloader(
                        repo_id=reference.repository_id,
                        revision=reference.revision,
                        local_dir=destination,
                        token=self.config.token,
                        max_workers=self.config.max_workers,
                        allow_patterns=list(allow_patterns),
                        endpoint=self.config.endpoint,
                    )
            except Exception as exc:
                raise AppError("model.source_materialize_failed", retryable=True) from exc
        except Exception as exc:
            raise AppError("model.source_materialize_failed", retryable=True) from exc
        _remove_sdk_cache(destination)
        _emit(progress, "inspecting")

    def _downloader(self) -> _SnapshotDownload:
        if self._snapshot_download is None:
            try:
                module = import_module("huggingface_hub")
                downloader = cast(_SnapshotDownload, module.__dict__["snapshot_download"])
            except (ImportError, KeyError) as exc:
                raise AppError("model.source_sdk_missing") from exc
            self._snapshot_download = downloader
        return cast(_SnapshotDownload, self._snapshot_download)


def _remove_sdk_cache(destination: Path) -> None:
    cache = destination / ".cache" / "huggingface"
    if cache.is_dir() and not cache.is_symlink():
        shutil.rmtree(cache)
    parent = cache.parent
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()


def _emit(progress: ProgressCallback | None, phase: str) -> None:
    if progress is not None:
        progress(OperationProgress("model", phase, f"model.{phase}", {}))


def _ensure_enabled(enabled: bool) -> None:
    if not enabled:
        raise AppError("model.source_disabled")


__all__ = [
    "HF_CT2_ALLOW_PATTERNS",
    "HF_MLX_ALLOW_PATTERNS",
    "HuggingFaceModelMaterializer",
]
