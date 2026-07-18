"""Filesystem adapter for lightweight media input discovery."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from captioner.core.application.input_selection import (
    SUPPORTED_MEDIA_EXTENSIONS,
    InputPreview,
    InputRejection,
    InputSelectionRequest,
)


def _empty_paths() -> list[str]:
    return []


def _empty_rejections() -> list[InputRejection]:
    return []


@dataclass
class _PreviewState:
    accepted: list[str] = field(default_factory=_empty_paths)
    rejected: list[InputRejection] = field(default_factory=_empty_rejections)
    maximum_results: int = 10_000
    overflow: bool = False

    def full(self) -> bool:
        return len(self.accepted) >= self.maximum_results


@dataclass(frozen=True, slots=True)
class FilesystemInputDiscovery:
    """Discover supported media paths without opening media contents."""

    def preview(self, request: InputSelectionRequest) -> InputPreview:
        state = _PreviewState(maximum_results=request.maximum_results)

        for entry in request.entries:
            if state.full():
                state.overflow = True
                break
            try:
                path = Path(entry).expanduser()
            except (OSError, RuntimeError, ValueError):
                state.rejected.append(InputRejection(path=entry, code="input.unreadable"))
                continue
            try:
                exists = path.exists()
            except OSError:
                state.rejected.append(InputRejection(path=entry, code="input.unreadable"))
                continue
            if not exists:
                state.rejected.append(InputRejection(path=str(path), code="input.not_found"))
                continue
            try:
                is_dir = path.is_dir()
                is_link = path.is_symlink()
            except OSError:
                state.rejected.append(InputRejection(path=str(path), code="input.unreadable"))
                continue

            if is_dir and is_link:
                state.rejected.append(InputRejection(path=str(path), code="input.unsupported"))
                continue

            if is_dir:
                self._scan_directory(path, recursive=request.recursive, state=state)
                continue

            self._accept_file(path, display_path=str(path), state=state)

        if state.overflow and not any(item.code == "input.result_limit" for item in state.rejected):
            state.rejected.append(InputRejection(path="", code="input.result_limit"))

        return InputPreview(
            accepted_paths=tuple(state.accepted),
            rejected=tuple(state.rejected),
        )

    def _scan_directory(
        self,
        directory: Path,
        *,
        recursive: bool,
        state: _PreviewState,
    ) -> None:
        candidates: list[Path] = []
        try:
            if recursive:
                scan_errors: list[OSError] = []

                def onerror(error: OSError) -> None:
                    scan_errors.append(error)

                for root, dirnames, filenames in os.walk(
                    directory,
                    topdown=True,
                    followlinks=False,
                    onerror=onerror,
                ):
                    root_path = Path(root)
                    dirnames[:] = [name for name in dirnames if not (root_path / name).is_symlink()]
                    for filename in filenames:
                        candidates.append(root_path / filename)
                if scan_errors:
                    state.rejected.append(
                        InputRejection(
                            path=str(directory),
                            code="input.directory_unreadable",
                        )
                    )
            else:
                try:
                    children = list(directory.iterdir())
                except OSError:
                    state.rejected.append(
                        InputRejection(
                            path=str(directory),
                            code="input.directory_unreadable",
                        )
                    )
                    return
                for child in children:
                    try:
                        if child.is_dir():
                            continue
                        candidates.append(child)
                    except OSError:
                        state.rejected.append(
                            InputRejection(path=str(child), code="input.unreadable")
                        )
        except OSError:
            state.rejected.append(
                InputRejection(path=str(directory), code="input.directory_unreadable")
            )
            return

        def sort_key(item: Path) -> str:
            try:
                return item.relative_to(directory).as_posix().casefold()
            except ValueError:
                return item.as_posix().casefold()

        ordered = sorted(candidates, key=sort_key)
        for index, child in enumerate(ordered):
            if state.full():
                if index < len(ordered):
                    state.overflow = True
                return
            self._accept_file(child, display_path=str(child), state=state)
            if state.full() and index + 1 < len(ordered):
                state.overflow = True
                return

    def _accept_file(
        self,
        path: Path,
        *,
        display_path: str,
        state: _PreviewState,
    ) -> None:
        if state.full():
            return
        try:
            if path.is_dir():
                state.rejected.append(InputRejection(path=display_path, code="input.unsupported"))
                return
            if not path.is_file():
                state.rejected.append(InputRejection(path=display_path, code="input.unsupported"))
                return
            suffix = path.suffix.casefold()
            if suffix not in SUPPORTED_MEDIA_EXTENSIONS:
                state.rejected.append(InputRejection(path=display_path, code="input.unsupported"))
                return
            resolved = str(path.resolve())
        except OSError:
            state.rejected.append(InputRejection(path=display_path, code="input.unreadable"))
            return
        if state.full():
            return
        state.accepted.append(resolved)


__all__ = ["FilesystemInputDiscovery"]
