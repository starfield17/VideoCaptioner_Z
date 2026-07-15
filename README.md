# Captioner

Captioner is a batch subtitle-generation application. Phase 1 provides a
single-input vertical slice: FFprobe inspection, FFmpeg normalization to
16 kHz mono PCM WAV, optional Faster Whisper transcription, deterministic
Transcript JSON, and SRT export.

The application still has no GUI workflow, batch jobs, LLM, translation,
alignment, muxing, runtime installer, or model manager.

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
captioner run input.mp4 --output build/output \
  --model tiny --device cpu --compute-type int8 --language en --json
python scripts/validate_subtitle.py build/output/input.srt
```

The run writes `<source-stem>.transcript.json` and `<source-stem>.srt` only
after successful transcription, validation, and atomic artifact commits. Use
`--overwrite` to replace existing outputs.

CLI exit codes are stable: `0` success, `2` usage/configuration, `3`
media/FFmpeg, `4` ASR/runtime/model, `5` output/export, and `130` cancellation.
Phase 1 loads one Faster Whisper model per adapter instance and keeps active
ASR concurrency at one. Unit tests and default CI never download models.

Built-in resources are read from `resources/`. User-writable paths are owned
by the operating system's standard application directories through
`platformdirs`.
