"""Thin argparse CLI entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from captioner import __version__
from captioner.adapters.subtitles.corpus import run_project_subtitle_corpus
from captioner.bootstrap import create_asr_job_snapshot
from captioner.cli.commands import batch as batch_command
from captioner.cli.commands import doctor
from captioner.cli.commands import model as model_command
from captioner.cli.commands import runtime as runtime_command
from captioner.cli.outcomes import exit_code_for_error
from captioner.cli.output import doctor_labels, render
from captioner.core.domain.errors import AppError
from captioner.core.domain.result import JsonValue
from captioner.core.domain.stage import PipelineProfile, StageName
from captioner.i18n.service import I18nService
from captioner.infrastructure.app_paths import (
    CompiledRuntime,
    ensure_runtime_layout,
    resolve_app_paths,
)


class _SourceLanguageArgumentError(argparse.ArgumentTypeError):
    @classmethod
    def detect(cls) -> _SourceLanguageArgumentError:
        return cls("use --language auto for automatic detection")

    @classmethod
    def empty(cls) -> _SourceLanguageArgumentError:
        return cls("language must not be empty")


def _parse_source_language(value: str) -> str | None:
    """Map the explicit CLI ``auto`` value to the optional source language."""
    normalized = value.strip()
    if normalized.casefold() == "auto":
        return None
    if normalized.casefold() == "detect":
        raise _SourceLanguageArgumentError.detect()
    if not normalized:
        raise _SourceLanguageArgumentError.empty()
    return normalized


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
    doctor_parser.add_argument(
        "--tokenizer-smoke",
        action="store_true",
        help="Initialize both packaged offline tokenizers",
    )
    doctor_parser.add_argument("--lang", dest="lang", default=argparse.SUPPRESS)
    run_parser = subparsers.add_parser("run", help="Transcribe one media file")
    run_parser.add_argument("input", nargs="+", type=Path, help="Input audio or video files")
    run_parser.add_argument("--output", type=Path, required=True, help="Output directory")
    run_parser.add_argument(
        "--model",
        dest="model_ref",
        required=True,
        help="Selector for an installed model",
    )
    run_parser.add_argument("--device", choices=("auto", "cpu", "cuda", "metal"), default="auto")
    run_parser.add_argument("--compute-type", default="default")
    run_parser.add_argument(
        "--language",
        default=None,
        type=_parse_source_language,
        help="Configured source language, or auto for automatic detection",
    )
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
            command_parser.add_argument("--device", choices=("auto", "cpu", "cuda", "metal"))
            command_parser.add_argument("--compute-type")
            command_parser.add_argument(
                "--language",
                default=argparse.SUPPRESS,
                type=_parse_source_language,
                help="Override configured source language, or auto to clear it",
            )
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
    runtime_parser = subparsers.add_parser("runtime", help="Manage isolated ASR Runtimes")
    runtime_subparsers = runtime_parser.add_subparsers(dest="runtime_command", required=True)
    runtime_list = runtime_subparsers.add_parser("list", help="List installed Runtimes")
    runtime_list.add_argument("--json", action="store_true")
    runtime_install = runtime_subparsers.add_parser("install", help="Install a Runtime package")
    runtime_install.add_argument("reference")
    runtime_install.add_argument("--no-activate", action="store_true")
    runtime_install.add_argument("--json", action="store_true")
    runtime_doctor = runtime_subparsers.add_parser("doctor", help="Run Runtime Doctor")
    runtime_doctor.add_argument("runtime_id")
    runtime_doctor.add_argument("--version", dest="runtime_version", required=True)
    runtime_doctor.add_argument("--activation", action="store_true")
    runtime_doctor.add_argument("--json", action="store_true")
    runtime_activate = runtime_subparsers.add_parser("activate", help="Activate a Runtime")
    runtime_activate.add_argument("runtime_id")
    runtime_activate.add_argument("--version", dest="runtime_version", required=True)
    runtime_activate.add_argument("--json", action="store_true")
    runtime_rollback = runtime_subparsers.add_parser("rollback", help="Rollback an active Runtime")
    runtime_rollback.add_argument("--backend", required=True)
    runtime_rollback.add_argument(
        "--platform", choices=("macos", "windows", "linux"), required=True
    )
    runtime_rollback.add_argument("--architecture", choices=("arm64", "x86_64"), required=True)
    runtime_rollback.add_argument("--device", choices=("cpu", "cuda", "metal"), required=True)
    runtime_rollback.add_argument("--json", action="store_true")
    runtime_remove = runtime_subparsers.add_parser("remove", help="Remove a managed Runtime")
    runtime_remove.add_argument("runtime_id")
    runtime_remove.add_argument("--version", dest="runtime_version", required=True)
    runtime_remove.add_argument("--json", action="store_true")
    runtime_external = runtime_subparsers.add_parser(
        "register-external", help="Register a developer-managed Runtime"
    )
    runtime_external.add_argument("--manifest", type=Path, required=True)
    runtime_external.add_argument("--root", type=Path, required=True)
    runtime_external.add_argument("--developer-mode", action="store_true")
    runtime_external.add_argument("--json", action="store_true")

    model_parser = subparsers.add_parser("model", help="Manage local ASR models")
    model_subparsers = model_parser.add_subparsers(dest="model_command", required=True)
    model_list = model_subparsers.add_parser("list", help="List installed models")
    model_list.add_argument("--json", action="store_true")
    model_search = model_subparsers.add_parser("search-hf", help="Search Hugging Face models")
    model_search.add_argument("query")
    model_search.add_argument("--backend", required=True)
    model_search.add_argument("--limit", type=int, default=20)
    model_search.add_argument("--json", action="store_true")
    for source_command, source_name in (
        ("install-hf", "Install a Hugging Face model"),
        ("install-modelscope", "Install a ModelScope model"),
    ):
        source_parser = model_subparsers.add_parser(source_command, help=source_name)
        source_parser.add_argument("repository_id")
        source_parser.add_argument("--revision", required=source_command == "install-modelscope")
        source_parser.add_argument("--backend", required=True)
        source_parser.add_argument("--format", dest="model_format", required=True)
        source_parser.add_argument("--display-name")
        source_parser.add_argument("--verify-load", action="store_true")
        source_parser.add_argument("--runtime-id")
        source_parser.add_argument("--runtime-version")
        source_parser.add_argument(
            "--device", choices=("auto", "cpu", "cuda", "metal"), default="auto"
        )
        source_parser.add_argument("--json", action="store_true")
    for import_command, import_help in (
        ("import", "Import a managed local model"),
        ("register-external", "Register an unmanaged local model"),
    ):
        import_parser = model_subparsers.add_parser(import_command, help=import_help)
        import_parser.add_argument("directory", type=Path)
        import_parser.add_argument("--backend")
        import_parser.add_argument("--format", dest="model_format")
        import_parser.add_argument("--display-name")
        import_parser.add_argument("--verify-load", action="store_true")
        import_parser.add_argument("--runtime-id")
        import_parser.add_argument("--runtime-version")
        import_parser.add_argument(
            "--device", choices=("auto", "cpu", "cuda", "metal"), default="auto"
        )
        if import_command == "register-external":
            import_parser.add_argument("--developer-mode", action="store_true")
        import_parser.add_argument("--json", action="store_true")
    model_validate = model_subparsers.add_parser("validate", help="Validate an installed model")
    model_validate.add_argument("selector")
    model_validate.add_argument("--json", action="store_true")
    model_verify = model_subparsers.add_parser("verify-load", help="Verify a model load")
    model_verify.add_argument("selector")
    model_verify.add_argument("--runtime-id")
    model_verify.add_argument("--runtime-version")
    model_verify.add_argument("--device", choices=("auto", "cpu", "cuda", "metal"), default="auto")
    model_verify.add_argument("--json", action="store_true")
    model_remove = model_subparsers.add_parser("remove", help="Remove an installed model")
    model_remove.add_argument("selector")
    model_remove.add_argument("--json", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    compiled_runtime: CompiledRuntime | None = None,
) -> int:
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
            "runtime",
            "model",
        ):
            parser.error(f"unknown command: {namespace.command}")
        paths = resolve_app_paths(compiled_runtime=compiled_runtime)
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
            options = doctor.DoctorOptions(
                locale=namespace.lang,
                as_json=as_json,
                paths=paths,
                tokenizer_smoke=bool(getattr(namespace, "tokenizer_smoke", False)),
            )
            payload = doctor.run(options, service=service)
            labels = None if as_json else doctor_labels(service)
        elif namespace.command == "run":
            selected_model_ref = namespace.model_ref
            asr_snapshot = create_asr_job_snapshot(
                model_selector=selected_model_ref,
                requested_device=namespace.device,
                compute_type=namespace.compute_type,
                paths=paths,
            )
            run_options = batch_command.BatchRunOptions(
                inputs=tuple(namespace.input),
                output_dir=namespace.output,
                model_ref=selected_model_ref,
                device=namespace.device,
                compute_type=namespace.compute_type,
                language=namespace.language,
                ffmpeg_bin=namespace.ffmpeg_bin,
                ffprobe_bin=namespace.ffprobe_bin,
                overwrite=namespace.overwrite,
                pipeline_profile=PipelineProfile(namespace.profile),
                target_language=namespace.target_language,
                llm_provider_profile=namespace.llm_provider_profile,
                asr_snapshot=asr_snapshot,
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
        elif namespace.command == "runtime":
            ensure_runtime_layout(paths)
            payload = runtime_command.execute(namespace, paths=paths)
            labels = None
        elif namespace.command == "model":
            ensure_runtime_layout(paths)
            payload = model_command.execute(namespace, paths=paths)
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
