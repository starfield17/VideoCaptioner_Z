"""Thin source-tree and Nuitka entry point for Captioner."""

from typing import cast

from captioner.entrypoint import main
from captioner.infrastructure.app_paths import (
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
