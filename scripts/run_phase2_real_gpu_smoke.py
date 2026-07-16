"""Download, trim, and run the explicit Phase 2 CUDA smoke test."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CudaSmokeError(RuntimeError):
    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        message = reason if not detail else f"{reason}: {detail}"
        super().__init__(message)

    @classmethod
    def package_missing(cls, name: str) -> CudaSmokeError:
        return cls("CUDA 12 package missing", name)

    @classmethod
    def directory_missing(cls, path: Path) -> CudaSmokeError:
        return cls("CUDA library directory missing", str(path))

    @classmethod
    def dependencies_unresolved(cls, lines: list[str]) -> CudaSmokeError:
        return cls("unresolved CTranslate2 CUDA dependencies", "; ".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("CAPTIONER_REAL_MEDIA_URL"))
    parser.add_argument("--duration", type=int, default=180)
    args = parser.parse_args(None if argv is None else list(argv))
    if not args.url:
        parser.error("--url or CAPTIONER_REAL_MEDIA_URL is required")
    library_dirs = discover_cuda_library_dirs()
    environment = cuda_environment(library_dirs)
    work = ROOT / "build" / "phase2-real"
    work.mkdir(parents=True, exist_ok=True)
    source = work / "source-media"
    audio = work / "english-real.wav"
    output = work / "output"
    _run(
        [
            "wget",
            "--https-only",
            "--tries=3",
            "--timeout=30",
            "--output-document",
            str(source),
            args.url,
        ]
    )
    diagnostics = _cuda_diagnostics(environment, library_dirs)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-t",
            str(args.duration),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio),
        ]
    )
    command = [
        "uv",
        "run",
        "--extra",
        "asr-faster-whisper-cuda12",
        "captioner",
        "run",
        str(audio),
        "--output",
        str(output),
        "--model",
        "small",
        "--device",
        "cuda",
        "--compute-type",
        "float16",
        "--language",
        "en",
        "--json",
    ]
    completed = _run(command, capture=True, environment=environment)
    payload = json.loads(completed.stdout)
    print(
        json.dumps(
            {
                "source_sha256": _sha256(source),
                "trimmed_seconds": args.duration,
                "model": "small",
                "device": "cuda",
                "compute_type": "float16",
                "model_cache": os.environ.get("CAPTIONER_FASTER_WHISPER_CACHE"),
                "cuda_library_dirs": [str(path) for path in library_dirs],
                "cuda_diagnostics": diagnostics,
                "result": payload,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def discover_cuda_library_dirs() -> tuple[Path, ...]:
    """Find CUDA 12 shared libraries installed by the optional extra."""
    locations: list[Path] = []
    for distribution_name, relative in (
        ("nvidia-cublas-cu12", Path("nvidia/cublas/lib")),
        ("nvidia-cudnn-cu12", Path("nvidia/cudnn/lib")),
    ):
        try:
            location = Path(
                str(importlib.metadata.distribution(distribution_name).locate_file(relative))
            )
        except importlib.metadata.PackageNotFoundError as exc:
            raise CudaSmokeError.package_missing(distribution_name) from exc
        if not location.is_dir():
            raise CudaSmokeError.directory_missing(location)
        locations.append(location)
    return tuple(locations)


def cuda_environment(library_dirs: tuple[Path, ...]) -> dict[str, str]:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    prefix = os.pathsep.join(str(path) for path in library_dirs)
    return {**os.environ, "LD_LIBRARY_PATH": prefix + (os.pathsep + existing if existing else "")}


def _cuda_diagnostics(
    environment: dict[str, str], library_dirs: tuple[Path, ...]
) -> dict[str, object]:
    nvidia = subprocess.run(
        ["nvidia-smi"],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    code = (
        "import ctranslate2, importlib.metadata, json, pathlib; "
        "extension = next(pathlib.Path(ctranslate2.__file__).parent.glob('*.so')); "
        "print(json.dumps({'faster_whisper': importlib.metadata.version('faster-whisper'), "
        "'ctranslate2': importlib.metadata.version('ctranslate2'), "
        "'cuda_device_count': ctranslate2.get_cuda_device_count(), "
        "'compute_types': sorted(ctranslate2.get_supported_compute_types('cuda')), "
        "'extension': str(extension.resolve())}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    value = json.loads(result.stdout)
    extension = str(value["extension"])
    ldd = subprocess.run(
        ["ldd", extension], check=False, text=True, capture_output=True, env=environment
    )
    unresolved = [line.strip() for line in ldd.stdout.splitlines() if "not found" in line]
    if unresolved:
        raise CudaSmokeError.dependencies_unresolved(unresolved)
    resolved = {
        name: str((directory / name).resolve())
        for name in ("libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9")
        for directory in library_dirs
        if (directory / name).exists()
    }
    nvidia_summary = nvidia.stdout.strip()
    value["nvidia_smi"] = nvidia_summary
    value["cuda_driver_capability"] = next(
        (
            line.strip()
            for line in nvidia_summary.splitlines()
            if "CUDA Version:" in line or "CUDA UMD Version:" in line
        ),
        None,
    )
    value["nvidia_smi_returncode"] = nvidia.returncode
    value["resolved_libraries"] = resolved
    value["ldd"] = ldd.stdout
    return value


def _run(
    command: list[str], *, capture: bool = False, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, check=True, text=True, capture_output=capture, env=environment
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
