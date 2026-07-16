"""Thin argparse CLI entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from captioner import __version__
from captioner.cli.commands import doctor
from captioner.cli.commands import run as run_command
from captioner.cli.outcomes import exit_code_for_error
from captioner.cli.output import doctor_labels, render, run_labels
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
    run_parser = subparsers.add_parser("run", help="Transcribe one media file")
    run_parser.add_argument("input", type=Path, help="One input audio or video file")
    run_parser.add_argument("--output", type=Path, required=True, help="Output directory")
    run_parser.add_argument("--model", dest="model_ref", default="tiny")
    run_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run_parser.add_argument("--compute-type", default="default")
    run_parser.add_argument("--language", default=None)
    run_parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    run_parser.add_argument("--ffprobe-bin", default="ffprobe")
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--json", action="store_true", help="Emit JSON")
    run_parser.add_argument("--lang", dest="lang", default=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    service: I18nService | None = None
    as_json = True
    try:
        namespace = parser.parse_args(None if argv is None else list(argv))
        if namespace.command not in (None, "doctor", "run"):
            parser.error(f"unknown command: {namespace.command}")
        paths = resolve_app_paths()
        service = I18nService(
            locale=namespace.lang,
            resource_dir=paths.i18n_resource_dir,
            strict=True,
        )
        as_json = bool(getattr(namespace, "json", False))
        if namespace.command in (None, "doctor"):
            options = doctor.DoctorOptions(locale=namespace.lang, as_json=as_json, paths=paths)
            payload = doctor.run(options, service=service)
            labels = None if as_json else doctor_labels(service)
        else:
            run_options = run_command.RunOptions(
                input_path=namespace.input,
                output_dir=namespace.output,
                model_ref=namespace.model_ref,
                device=namespace.device,
                compute_type=namespace.compute_type,
                language=namespace.language,
                ffmpeg_bin=namespace.ffmpeg_bin,
                ffprobe_bin=namespace.ffprobe_bin,
                overwrite=namespace.overwrite,
            )
            result = run_command.execute(run_options, paths=paths)
            payload = {
                "media_id": result.media_id,
                "transcript_id": result.transcript_id,
                "transcript_path": str(result.transcript_path),
                "subtitle_path": str(result.subtitle_path),
                "detected_language": result.detected_language,
                "word_count": result.word_count,
                "cue_count": result.cue_count,
            }
            labels = None if as_json else run_labels(service)
    except AppError as exc:
        if service is not None and not as_json:
            print(f"{service.translate('cli.error')}: {exc.code}", file=sys.stderr)
        else:
            print(render(exc.to_dict(), as_json=True), file=sys.stderr)
        return exit_code_for_error(exc)
    except KeyboardInterrupt:
        return 130
    else:
        print(render(payload, as_json=as_json, labels=labels))
        return 0
