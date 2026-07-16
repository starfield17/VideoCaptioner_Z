"""Shared cancellation-safe transaction for a pair of user output files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

from captioner.core.domain.errors import AppError
from captioner.core.domain.execution import ExecutionContext
from captioner.core.ports.artifact_store import ArtifactStorePort, StagedArtifact


def commit_output_pair(
    store: ArtifactStorePort,
    outputs: tuple[tuple[str, bytes], tuple[str, bytes]],
    *,
    overwrite: bool,
    context: ExecutionContext,
) -> tuple[Path, Path]:
    previous = {key: store.read_bytes(key) if store.exists(key) else None for key, _ in outputs}
    staged: list[StagedArtifact] = []
    committed: list[str] = []
    try:
        staged.extend(store.stage_bytes(key, data) for key, data in outputs)
        context.raise_if_cancelled()
        paths: list[Path] = []
        for index, artifact in enumerate(staged):
            paths.append(_commit_staged(artifact, overwrite=overwrite, committed=committed))
            if index == 0:
                context.checkpoint("mid_execute")
            context.raise_if_cancelled()
        cleanup_error = _discard_all(staged)
        if cleanup_error is not None:
            _raise_cleanup_error(cleanup_error)
        context.raise_if_cancelled()
        return paths[0], paths[1]
    except BaseException as original:
        _record_committed(staged, committed)
        rollback_error = _rollback(store, committed, previous)
        cleanup_error = _discard_all(staged)
        if rollback_error is not None or cleanup_error is not None:
            reason = cleanup_error.code if cleanup_error is not None else "output.rollback_failed"
            raise AppError("output.rollback_failed", {"reason": reason}) from original
        raise


def _commit_staged(artifact: StagedArtifact, *, overwrite: bool, committed: list[str]) -> Path:
    try:
        path = artifact.commit(overwrite=overwrite)
    except BaseException:
        _record_one(artifact, committed)
        raise
    _record_one(artifact, committed)
    return path


def _record_one(artifact: StagedArtifact, committed: list[str]) -> None:
    if artifact.committed and artifact.key not in committed:
        committed.append(artifact.key)


def _record_committed(staged: list[StagedArtifact], committed: list[str]) -> None:
    for artifact in staged:
        _record_one(artifact, committed)


def _discard_all(staged: list[StagedArtifact]) -> AppError | None:
    first: AppError | None = None
    for artifact in reversed(staged):
        try:
            artifact.discard()
        except AppError as exc:
            if first is None:
                first = exc
    return first


def _raise_cleanup_error(error: AppError) -> NoReturn:
    raise AppError("output.cleanup_failed", {"reason": error.code}) from error


def _rollback(
    store: ArtifactStorePort,
    committed: list[str],
    previous: Mapping[str, bytes | None],
) -> BaseException | None:
    try:
        for key in reversed(committed):
            old = previous[key]
            if old is None:
                store.delete(key)
            else:
                store.write_bytes(key, old, overwrite=True)
    except BaseException as exc:
        return exc
    return None
