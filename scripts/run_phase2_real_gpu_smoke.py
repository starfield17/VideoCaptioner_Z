"""Download, trim, and run the explicit Phase 2 CUDA smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("CAPTIONER_REAL_MEDIA_URL"))
    parser.add_argument("--duration", type=int, default=180)
    args = parser.parse_args(None if argv is None else list(argv))
    if not args.url:
        parser.error("--url or CAPTIONER_REAL_MEDIA_URL is required")
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
        "asr-faster-whisper",
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
    completed = _run(command, capture=True)
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
                "result": payload,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=capture)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
