"""Package-owned top-level CLI/GUI dispatcher."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from captioner.infrastructure.app_paths import CompiledRuntime


def _run_gui(arguments: Sequence[str], compiled_runtime: CompiledRuntime | None = None) -> int:
    """Import the GUI only when the dispatcher selected it."""
    from captioner.gui.gui_entry import main as gui_main

    return gui_main(arguments, compiled_runtime=compiled_runtime)


def _run_cli(arguments: Sequence[str], compiled_runtime: CompiledRuntime | None = None) -> int:
    """Import the CLI selected by the dispatcher."""
    from captioner.cli.cli_entry import main as cli_main

    return cli_main(arguments, compiled_runtime=compiled_runtime)


def main(
    argv: Sequence[str] | None = None,
    *,
    compiled_runtime: CompiledRuntime | None = None,
) -> int:
    """Dispatch using only the first argument."""
    arguments = list(sys.argv[1:] if argv is None else argv)

    def run_gui(gui_arguments: Sequence[str]) -> int:
        if compiled_runtime is None:
            return _run_gui(gui_arguments)
        return _run_gui(gui_arguments, compiled_runtime)

    def run_cli(cli_arguments: Sequence[str]) -> int:
        if compiled_runtime is None:
            return _run_cli(cli_arguments)
        return _run_cli(cli_arguments, compiled_runtime)

    try:
        if not arguments:
            return run_gui([])
        if arguments[0] == "--gui":
            return run_gui(arguments[1:])
        if arguments[0] == "--cli":
            return run_cli(arguments[1:])
        return run_cli(arguments)
    except KeyboardInterrupt:
        return 130
