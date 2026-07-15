"""Thin source-tree entry point for Captioner."""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Select the GUI or CLI without loading application services."""
    arguments = list(sys.argv[1:] if argv is None else argv)

    if not arguments or "--gui" in arguments:
        from captioner.gui.gui_entry import main as gui_main

        return gui_main([argument for argument in arguments if argument != "--gui"])

    from captioner.cli.cli_entry import main as cli_main

    return cli_main([argument for argument in arguments if argument != "--cli"])


if __name__ == "__main__":
    raise SystemExit(main())
