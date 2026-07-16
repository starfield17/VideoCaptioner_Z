# Testing and quality gates

`python scripts/check.py --quick` runs formatting, Ruff, Pyright, and unit plus
contract tests. `python scripts/check.py --full` additionally verifies the lock
file, import contracts, i18n catalogs, forbidden patterns, all Phase 0 tests,
and branch coverage with an 85% minimum.

Tests are grouped into `unit`, `property`, `contract`, `integration`, and
`packaging`. Property tests use Hypothesis for locale, domain and segmentation
invariants. Unit tests use fake processes, fake ASR models and local artifact
stores; they do not execute FFmpeg or download models. Output-transaction unit
tests exercise every cancellation/interrupt boundary, overwrite restoration,
staging cleanup, and staged-artifact single-use rule. Integration tests use
the installed FFprobe/FFmpeg and are marked `integration`. The real Faster
Whisper test is marked `slow`, uses the optional extra and a configurable model
cache, and is not part of default PR CI. Packaging tests inspect commands and
layouts without compiling Nuitka; the local build wrapper then performs the
real platform smoke build.

Before submitting a patch, run:

```bash
uv sync --frozen
uv run python scripts/check.py --full
uv run pytest tests/integration/test_ffmpeg_pipeline.py -q -m integration
uv run pytest tests/packaging -q
uv run python main.py --cli --help
uv run python main.py --cli doctor --json
QT_QPA_PLATFORM=offscreen uv run python main.py --gui --smoke-test
uv run python scripts/build_nuitka.py --clean --version 0.0.0
```

For the optional local ASR validation:

```bash
uv sync --frozen --extra asr-faster-whisper
uv run --extra asr-faster-whisper pytest \
  tests/integration/test_faster_whisper_smoke.py -q -m slow
```
