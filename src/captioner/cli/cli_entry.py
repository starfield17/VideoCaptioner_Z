"""Thin argparse CLI entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from captioner import __version__
from captioner.cli.commands import doctor
from captioner.cli.output import doctor_labels, render
from captioner.core.domain.errors import AppError
from captioner.i18n.service import I18nService
from captioner.infrastructure.app_paths import resolve_app_paths


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 0 CLI parser."""
    parser = argparse.ArgumentParser(
        prog="captioner",
        description="Batch subtitle generation tool",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--lang", dest="lang", default="en", help="Locale, for example zh-CN")
    subparsers = parser.add_subparsers(dest="command")
    doctor_parser = subparsers.add_parser("doctor", help="Report Phase 0 diagnostics")
    doctor_parser.add_argument("--json", action="store_true", help="Emit JSON")
    doctor_parser.add_argument("--lang", dest="lang", default=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    try:
        namespace = parser.parse_args(None if argv is None else list(argv))
        if namespace.command not in (None, "doctor"):
            parser.error(f"unknown command: {namespace.command}")
        paths = resolve_app_paths()
        service = I18nService(
            locale=namespace.lang,
            resource_dir=paths.i18n_resource_dir,
            strict=True,
        )
        options = doctor.DoctorOptions(
            locale=namespace.lang,
            as_json=bool(getattr(namespace, "json", False)),
            paths=paths,
        )
        payload = doctor.run(options, service=service)
    except AppError as exc:
        print(render(exc.to_dict(), as_json=True), file=sys.stderr)
        return 2
    else:
        labels = None if options.as_json else doctor_labels(service)
        print(render(payload, as_json=options.as_json, labels=labels))
        return 0
