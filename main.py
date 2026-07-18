"""Thin source-tree and Nuitka entry point for Captioner."""

import sys
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"

if SOURCE_ROOT.is_dir():
    source_root_text = str(SOURCE_ROOT)
    if source_root_text not in sys.path:
        sys.path.insert(0, source_root_text)

from captioner.entrypoint import main  # noqa: E402  # source bootstrap must run first
from captioner.infrastructure.app_paths import (  # noqa: E402  # source bootstrap must run first
    CompiledRuntime,
    compiled_runtime_from_descriptor,
)


def _compiled_runtime_override() -> CompiledRuntime | None:
    # Nuitka injects this module-global only in compiled execution.
    try:
        compiled = __compiled__  # type: ignore[name-defined]
    except NameError:
        return None
    return compiled_runtime_from_descriptor(cast(object, compiled))


if __name__ == "__main__":
    raise SystemExit(main(compiled_runtime=_compiled_runtime_override()))
