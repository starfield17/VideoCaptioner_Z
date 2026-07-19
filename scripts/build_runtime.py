"""Build a relocatable standalone Runtime archive and sidecar descriptor."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

from captioner.adapters.runtime.runtime_archive import (
    build_file_manifest,
    create_deterministic_archive,
    safe_extract_archive,
    sha256_file,
)
from captioner.core.domain.asr_backend import BackendCapability
from captioner.core.domain.runtime import RuntimeIdentity, RuntimeManifest, RuntimeTarget
from captioner.core.domain.runtime_package import RuntimePackageDescriptor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_WORKER_ROOT = PROJECT_ROOT / "runtime_worker"
RUNTIME_PROJECTS_ROOT = PROJECT_ROOT / "runtime_projects"
_MAX_DESCRIPTOR_BYTES = 2 * 1024 * 1024


class RuntimeBuildError(RuntimeError):
    """A safe, user-facing build failure with a stable short reason."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", choices=("faster-whisper-cpu", "mlx-whisper-metal"), required=True
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    _validate_version(options.version)
    _build(options.project, options.version, options.output.expanduser().resolve())
    return 0


def _build(project_name: str, version: str, output: Path) -> None:
    project_root = RUNTIME_PROJECTS_ROOT / project_name
    config = _load_toml(project_root / "runtime-build.toml")
    host_platform = _normalize_platform(platform.system())
    host_architecture = _normalize_architecture(platform.machine())
    supported_platforms = _string_list(config, "supported_platforms")
    supported_architectures = _string_list(config, "supported_architectures")
    if host_platform not in supported_platforms or host_architecture not in supported_architectures:
        raise RuntimeBuildError("target_mismatch")
    if project_name == "mlx-whisper-metal" and (host_platform, host_architecture) != (
        "macos",
        "arm64",
    ):
        raise RuntimeBuildError("mlx_native_macos_arm64_required")
    if project_name == "mlx-whisper-metal" and _rosetta_translated():
        raise RuntimeBuildError("mlx_native_macos_arm64_required")
    python_version = _string(config, "python")
    runtime_id = _string(config, "runtime_id").format(
        platform=host_platform,
        architecture=host_architecture,
    )
    backend_id = _string(config, "backend_id")
    backend_version = _string(config, "backend_version")
    device_kind = _string(config, "device_kind")
    model_format = _string(config, "model_format")
    minimum_os = _string(config, "minimum_os_version")
    archive_filename = (
        f"captioner-runtime-{runtime_id}-{version}-{host_platform}-{host_architecture}.tar.gz"
    )
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"captioner-runtime-{runtime_id}-") as temporary_name:
        build_root = Path(temporary_name)
        payload = build_root / "payload"
        payload.mkdir()
        python_root = payload / "python"
        _run(["uv", "sync", "--locked", "--project", str(project_root)])
        managed_python = _managed_python(python_version)
        _copy_python_distribution(managed_python, python_root)
        interpreter = _runtime_interpreter(python_root, host_platform)
        worker_dist = build_root.parent / f"{build_root.name}-worker-dist"
        requirements_path = build_root.parent / f"{build_root.name}-requirements.txt"
        worker_wheel = _build_worker_wheel(worker_dist)
        _run(
            [
                "uv",
                "export",
                "--project",
                str(project_root),
                "--locked",
                "--no-dev",
                "--no-emit-project",
                "--format",
                "requirements-txt",
                "--output-file",
                str(requirements_path),
            ]
        )
        _run(
            [
                "uv",
                "pip",
                "install",
                "--break-system-packages",
                "--python",
                str(interpreter),
                "--requirement",
                str(requirements_path),
            ]
        )
        requirements_path.unlink(missing_ok=True)
        _run(
            [
                "uv",
                "pip",
                "install",
                "--break-system-packages",
                "--python",
                str(interpreter),
                "--no-deps",
                str(worker_wheel),
            ]
        )
        prune_runtime_packages(
            python_root,
            _optional_string_list(config, "prune_packages"),
        )
        _remove_generated_files(python_root)
        build_info = {
            "schema_version": 1,
            "worker_version": "1.0.0",
            "protocol_version": "1.1",
            "runtime_id": runtime_id,
            "runtime_version": version,
            "backend_id": backend_id,
            "backend_version": backend_version,
            "platform": host_platform,
            "architecture": host_architecture,
            "device_kind": device_kind,
            "supported_model_formats": [model_format],
            "capabilities": [
                "language_detection",
                "runtime_doctor",
                "translation_task",
                "word_timestamps",
            ],
        }
        (payload / "build_info.json").write_text(
            json.dumps(build_info, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        _run([str(interpreter), "-I", "-B", "-c", "import captioner_runtime_worker"])
        shutil.rmtree(worker_dist, ignore_errors=True)
        files = build_file_manifest(build_root)
        identity = RuntimeIdentity(runtime_id, version)
        capability = BackendCapability(
            backend_id=backend_id,
            device_kind=device_kind,
            supported_model_formats=(model_format,),
            word_timestamps=True,
            language_detection=True,
            translation_task=True,
            additional_capabilities=("runtime_doctor",),
        )
        manifest = RuntimeManifest(
            schema_version=1,
            runtime_identity=identity,
            worker_protocol_version="1.1",
            backend_id=backend_id,
            backend_version=backend_version,
            target=RuntimeTarget(
                host_platform,
                host_architecture,
                device_kind,
                minimum_os,
            ),
            capabilities=capability,
            supported_model_formats=(model_format,),
            archive_sha256="0" * 64,
            files=files,
        )
        archive_path = output / archive_filename
        create_deterministic_archive(build_root, archive_path)
        archive_sha256 = sha256_file(archive_path)
        manifest = replace(manifest, archive_sha256=archive_sha256)
        descriptor = RuntimePackageDescriptor(
            package_schema_version=1,
            archive_filename=archive_filename,
            archive_size_bytes=archive_path.stat().st_size,
            runtime_manifest=manifest,
        )
        descriptor_path = output / f"{archive_filename[:-7]}.runtime.json"
        descriptor_bytes = (
            json.dumps(descriptor.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        if len(descriptor_bytes) > _MAX_DESCRIPTOR_BYTES:
            raise RuntimeBuildError("descriptor_too_large")
        descriptor_path.write_bytes(descriptor_bytes)
        with tempfile.TemporaryDirectory(prefix="captioner-runtime-relocation-") as relocated_name:
            safe_extract_archive(archive_path, Path(relocated_name), manifest)
            relocated_interpreter = _runtime_interpreter(
                Path(relocated_name) / "payload" / "python", host_platform
            )
            _run([str(relocated_interpreter), "-I", "-B", "-c", "import captioner_runtime_worker"])


def _build_worker_wheel(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    _run(
        ["uv", "build", "--wheel", "--out-dir", str(output), "--project", str(RUNTIME_WORKER_ROOT)]
    )
    wheels = sorted(output.glob("captioner_runtime_worker-*.whl"))
    if len(wheels) != 1:
        raise RuntimeBuildError("worker_wheel_not_deterministic")
    return wheels[0]


def _managed_python(version: str) -> Path:
    _run(["uv", "python", "install", version, "--managed-python"])
    result = _run(["uv", "python", "find", f"=={version}", "--managed-python"], capture=True)
    value = result.stdout.strip()
    if not value:
        raise RuntimeBuildError("managed_python_missing")
    path = Path(value)
    if not path.is_file():
        raise RuntimeBuildError("managed_python_invalid")
    return path


def _copy_python_distribution(interpreter: Path, destination: Path) -> None:
    if interpreter.parent.name == "bin":
        source_root = interpreter.parent.parent
    else:
        source_root = interpreter.parent
    shutil.copytree(source_root, destination, symlinks=False)
    if interpreter.parent.name == "bin":
        target = destination / "bin" / "python3"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(interpreter, target)
    else:
        target = destination / "python.exe"
        shutil.copy2(interpreter, target)


def _runtime_interpreter(python_root: Path, host_platform: str) -> Path:
    return python_root / ("python.exe" if host_platform == "windows" else "bin/python3")


def _remove_generated_files(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file() and (path.suffix == ".pyc" or path.name == "__pycache__"):
            path.unlink()
        elif path.is_dir() and path.name == "__pycache__":
            shutil.rmtree(path)


def prune_runtime_packages(python_root: Path, package_names: list[str]) -> None:
    """Remove explicitly excluded optional packages from the final payload.

    Some Runtime packages publish conversion or training dependencies that are
    not needed by the isolated inference worker.  The lock still resolves the
    complete upstream dependency set; this step only omits the declared,
    unused packages from the distributable Runtime.
    """
    site_packages_roots = (
        python_root / "lib" / "python3.12" / "site-packages",
        python_root / "Lib" / "site-packages",
    )
    names = set(package_names)
    for site_packages in site_packages_roots:
        if not site_packages.is_dir():
            continue
        for path in site_packages.iterdir():
            if path.name in names or any(
                path.name.startswith(f"{name}-") and path.name.endswith(".dist-info")
                for name in names
            ):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()


def _load_toml(path: Path) -> dict[str, object]:
    import tomllib

    with path.open("rb") as stream:
        return cast(dict[str, object], tomllib.load(stream))


def _string(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise RuntimeBuildError(f"missing_field:{key}")
    return item


def _string_list(value: Mapping[str, object], key: str) -> list[str]:
    item = value.get(key)
    if not isinstance(item, list):
        raise RuntimeBuildError(f"invalid_field:{key}")
    entries = cast(list[object], item)
    if any(not isinstance(entry, str) for entry in entries):
        raise RuntimeBuildError(f"invalid_field:{key}")
    return cast(list[str], entries)


def _optional_string_list(value: Mapping[str, object], key: str) -> list[str]:
    item = value.get(key)
    if item is None:
        return []
    if not isinstance(item, list):
        raise RuntimeBuildError(f"invalid_field:{key}")
    entries = cast(list[object], item)
    if any(not isinstance(entry, str) or not entry.strip() for entry in entries):
        raise RuntimeBuildError(f"invalid_field:{key}")
    return cast(list[str], entries)


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, check=False, text=True, capture_output=capture)
    if result.returncode != 0:
        raise RuntimeBuildError(f"command_failed:{command[0]}")
    return result


def _normalize_platform(value: str) -> str:
    mapping = {"Darwin": "macos", "Windows": "windows", "Linux": "linux"}
    try:
        return mapping[value]
    except KeyError as exc:
        raise RuntimeBuildError("unsupported_host_platform") from exc


def _normalize_architecture(value: str) -> str:
    mapping = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x86_64", "AMD64": "x86_64"}
    try:
        return mapping[value]
    except KeyError as exc:
        raise RuntimeBuildError("unsupported_host_architecture") from exc


def _validate_version(value: str) -> None:
    parts = value.split(".")
    if len(parts) < 3 or any(not item.isdigit() for item in parts):
        raise RuntimeBuildError("version_invalid")


def _rosetta_translated() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        result = subprocess.run(
            ["sysctl", "-in", "sysctl.proc_translated"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.stdout.strip() == "1"


if __name__ == "__main__":
    raise SystemExit(main())
