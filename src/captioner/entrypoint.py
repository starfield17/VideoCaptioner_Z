"""Package-owned top-level CLI/GUI dispatcher."""

from __future__ import annotations

import sys
from collections.abc import Sequence


def _run_gui(arguments: Sequence[str]) -> int:
    """Import the GUI only when the dispatcher selected it."""
    from captioner.gui.gui_entry import main as gui_main

    return gui_main(arguments)


def _run_cli(arguments: Sequence[str]) -> int:
    """Import the CLI selected by the dispatcher."""
    from captioner.cli.cli_entry import main as cli_main

    return cli_main(arguments)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch using only the first argument."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if not arguments:
            return _run_gui([])
        if arguments[0] == "--gui":
            return _run_gui(arguments[1:])
        if arguments[0] == "--cli":
            return _run_cli(arguments[1:])
        return _run_cli(arguments)
    except KeyboardInterrupt:
        return 130
