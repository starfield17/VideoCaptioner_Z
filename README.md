# Captioner

Captioner is a batch subtitle-generation application. This repository currently
contains the Phase 0 engineering skeleton only: package boundaries, empty CLI
and GUI entry points, i18n, runtime paths, ports/fakes, quality gates, and the
initial Nuitka wrapper.

No ASR, media processing, LLM, model, FFmpeg, queue, or pipeline behavior is
implemented in this phase.

## Local commands

The supported local interpreter for this checkout is the `Lab` environment at
`/home/hazel/miniconda3/envs/Lab/bin/python` (Python 3.13).

```bash
uv sync --frozen
uv run python scripts/check.py --full
uv run python main.py --cli --help
uv run python main.py --cli doctor --json
QT_QPA_PLATFORM=offscreen uv run python main.py --gui --smoke-test
uv run python scripts/build_nuitka.py --clean --version 0.0.0
```

Built-in resources are read from `resources/`. User-writable paths are owned
by the operating system's standard application directories through
`platformdirs`.
