# Captioner

Captioner is a durable batch subtitle-generation application. Phase 2 runs one
or more inputs through inspect, normalize, transcribe, segment, export, and
publish Stages. An fsynced Journal is the source of truth; a rebuildable
Manifest and verified content-addressed artifacts support resume and retry.

## Local commands

The supported local interpreter for this checkout is the `Lab` environment at
`/home/hazel/miniconda3/envs/Lab/bin/python` (Python 3.13).

```bash
uv sync --frozen
uv run python scripts/check.py --full
uv run python main.py --cli --help
uv run python main.py --cli run --help
uv run python main.py --cli doctor --json
QT_QPA_PLATFORM=offscreen uv run python main.py --gui --smoke-test
uv run python scripts/build_nuitka.py --clean --version 0.0.0
```

FFmpeg and FFprobe must be available on `PATH` for a real run. Faster Whisper
is optional and is installed separately:

```bash
uv sync --frozen --extra asr-faster-whisper
captioner run input.mp4 second.wav --output build/output \
  --model tiny --device cpu --compute-type int8 --language en --json
python scripts/validate_subtitle.py build/output/input.srt
captioner status batch-... --json
captioner resume batch-... --json
captioner retry batch-... --job job-000001 --stage transcribe --json
captioner cancel batch-... --job job-000001 --json
```

For Linux CUDA 12 systems, install the reproducible optional runtime and run
the guarded manual diagnostic:

```bash
uv sync --frozen --extra asr-faster-whisper-cuda12
export CAPTIONER_REAL_MEDIA_URL="https://example.invalid/direct-public-domain-media"
uv run --no-sync python scripts/run_phase2_real_gpu_smoke.py \
  --url "$CAPTIONER_REAL_MEDIA_URL" --duration 180
```

The script discovers CUDA 12 cuBLAS/cuDNN directories, prepends them to the
child `LD_LIBRARY_PATH`, reports Faster Whisper/CTranslate2 versions, runs
`ldd` diagnostics, and refuses to claim CUDA success with unresolved libraries.
CUDA libraries are not included in the default installation or Nuitka app.

The run writes `<source-stem>.transcript.json` and `<source-stem>.srt` only
after successful transcription, validation, and a staged atomic artifact
transaction. Cancellation or failure rolls back outputs committed by the
current run; `--overwrite` restores the previous bytes when rollback is needed.
Domain JSON metadata is recursively immutable, and public model identities do
not contain machine-specific local model paths.

CLI exit codes are stable: `0` success, `2` usage/configuration, `3`
media/FFmpeg, `4` ASR/runtime/model, `5` output/export, and `130` cancellation.
Phase 1 loads one Faster Whisper model per adapter instance and keeps active
ASR concurrency at one. Unit tests and default CI never download models.

Durable state lives under the platformdirs data directory in `batches/` and
`artifacts/`; workspaces are attempt-scoped and disposable. Built-in resources
are read from `resources/`. User-writable paths are owned
by the operating system's standard application directories through
`platformdirs`.

Batch and Job IDs are validated before durable path construction. Status is a
non-mutating read: it reports incomplete Journal tails and Manifest status but
does not repair or rewrite either file. Resume and retry acquire the Batch
writer lease before repair. All Jobs in one Batch share runtime
ASR/media/segmentation settings; output target collisions are rejected before
`batch.created`. Failed or cancelled Jobs require explicit `retry`, which
appends `job.retry_requested`.
