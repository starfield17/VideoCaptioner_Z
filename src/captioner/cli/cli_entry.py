"""Thin argparse CLI entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from captioner import __version__
from captioner.adapters.subtitles.corpus import run_project_subtitle_corpus
from captioner.cli.commands import batch as batch_command
from captioner.cli.commands import doctor
from captioner.cli.outcomes import exit_code_for_error
from captioner.cli.output import doctor_labels, render
from captioner.core.domain.errors import AppError
from captioner.core.domain.result import JsonValue
from captioner.core.domain.stage import PipelineProfile, StageName
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
    run_parser.add_argument("input", nargs="+", type=Path, help="Input audio or video files")
    run_parser.add_argument("--output", type=Path, required=True, help="Output directory")
    run_parser.add_argument("--model", dest="model_ref", default="tiny")
    run_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run_parser.add_argument("--compute-type", default="default")
    run_parser.add_argument("--language", default=None)
    run_parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    run_parser.add_argument("--ffprobe-bin", default="ffprobe")
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument(
        "--profile",
        choices=tuple(profile.value for profile in PipelineProfile),
        default=PipelineProfile.DETERMINISTIC.value,
    )
    run_parser.add_argument("--target-language", default=None)
    run_parser.add_argument("--llm-provider-profile", default="default")
    run_parser.add_argument("--json", action="store_true", help="Emit JSON")
    run_parser.add_argument("--lang", dest="lang", default=argparse.SUPPRESS)
    corpus_parser = subparsers.add_parser(
        "subtitle-corpus", help="Run deterministic subtitle fixtures without ASR"
    )
    corpus_parser.add_argument("fixture_directory", type=Path)
    corpus_parser.add_argument("--json", action="store_true", help="Emit JSON")
    for command in ("status", "resume"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("batch_id")
        command_parser.add_argument("--json", action="store_true")
        if command == "resume":
            command_parser.add_argument("--model")
            command_parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
            command_parser.add_argument("--compute-type")
            command_parser.add_argument("--language", default=argparse.SUPPRESS)
            command_parser.add_argument("--output", type=Path)
            command_parser.add_argument(
                "--profile",
                choices=tuple(profile.value for profile in PipelineProfile),
            )
            command_parser.add_argument("--target-language")
            command_parser.add_argument("--llm-provider-profile")
    retry_parser = subparsers.add_parser("retry")
    retry_parser.add_argument("batch_id")
    retry_parser.add_argument("--job", required=True)
    retry_parser.add_argument(
        "--stage", required=True, choices=tuple(stage.value for stage in StageName)
    )
    retry_parser.add_argument("--json", action="store_true")
    cancel_parser = subparsers.add_parser("cancel")
    cancel_parser.add_argument("batch_id")
    cancel_parser.add_argument("--job")
    cancel_parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    service: I18nService | None = None
    as_json = True
    try:
        namespace = parser.parse_args(None if argv is None else list(argv))
        if namespace.command not in (
            None,
            "doctor",
            "run",
            "status",
            "resume",
            "retry",
            "cancel",
            "subtitle-corpus",
        ):
            parser.error(f"unknown command: {namespace.command}")
        paths = resolve_app_paths()
        service = I18nService(
            locale=namespace.lang,
            resource_dir=paths.i18n_resource_dir,
            strict=True,
        )
        as_json = bool(getattr(namespace, "json", False))
        if namespace.command == "subtitle-corpus":
            report = run_project_subtitle_corpus(namespace.fixture_directory)
            payload = cast(dict[str, JsonValue], report.to_dict())
            print(render(payload, as_json=as_json))
            return int(report.failed != 0 or bool(report.errors))
        if namespace.command in (None, "doctor"):
            options = doctor.DoctorOptions(locale=namespace.lang, as_json=as_json, paths=paths)
            payload = doctor.run(options, service=service)
            labels = None if as_json else doctor_labels(service)
        elif namespace.command == "run":
            run_options = batch_command.BatchRunOptions(
                inputs=tuple(namespace.input),
                output_dir=namespace.output,
                model_ref=namespace.model_ref,
                device=namespace.device,
                compute_type=namespace.compute_type,
                language=namespace.language,
                ffmpeg_bin=namespace.ffmpeg_bin,
                ffprobe_bin=namespace.ffprobe_bin,
                overwrite=namespace.overwrite,
                pipeline_profile=PipelineProfile(namespace.profile),
                target_language=namespace.target_language,
                llm_provider_profile=namespace.llm_provider_profile,
            )
            projection = batch_command.run(run_options, paths=paths)
            payload = cast(
                dict[str, JsonValue], batch_command.projection_payload(projection, paths=paths)
            )
            labels = None
        elif namespace.command == "status":
            payload = cast(
                dict[str, JsonValue],
                batch_command.projection_payload(
                    batch_command.status(namespace.batch_id, paths=paths), paths=paths
                ),
            )
            labels = None
        elif namespace.command == "resume":
            payload = cast(
                dict[str, JsonValue],
                batch_command.projection_payload(
                    batch_command.resume(
                        namespace.batch_id,
                        paths=paths,
                        overrides=batch_command.ResumeOverrides(
                            model_ref=namespace.model,
                            device=namespace.device,
                            compute_type=namespace.compute_type,
                            language=getattr(namespace, "language", batch_command.LANGUAGE_UNSET),
                            output_dir=namespace.output,
                            pipeline_profile=(
                                None
                                if namespace.profile is None
                                else PipelineProfile(namespace.profile)
                            ),
                            target_language=namespace.target_language,
                            llm_provider_profile=namespace.llm_provider_profile,
                        ),
                    ),
                    paths=paths,
                ),
            )
            labels = None
        elif namespace.command == "retry":
            projection = batch_command.retry(
                namespace.batch_id, namespace.job, StageName(namespace.stage), paths=paths
            )
            payload = cast(
                dict[str, JsonValue], batch_command.projection_payload(projection, paths=paths)
            )
            labels = None
        else:
            marker = batch_command.cancel(namespace.batch_id, namespace.job, paths=paths)
            payload = {
                "schema_version": 1,
                "batch_id": namespace.batch_id,
                "job_id": namespace.job,
                "cancel_requested": True,
                "marker": marker.name,
            }
            labels = None
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
