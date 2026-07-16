# Testing and quality gates

`python scripts/check.py --quick` runs formatting, Ruff, Pyright, and unit plus
contract tests. `python scripts/check.py --full` additionally verifies the lock
file, import contracts, i18n catalogs, forbidden patterns, recovery/property/
packaging tests, and branch coverage with an 85% minimum.

Tests are grouped into `unit`, `property`, `contract`, `recovery`, `integration`,
and `packaging`. Recovery covers all six Stages at all six fault points.
Property tests use Hypothesis for locale, domain, segmentation and Journal
transition invariants, including atomic Batch configuration and interrupted
cancellation. Recovery also covers status purity, exact multi-Artifact cleanup,
publication-target corruption and Batch cancellation acknowledgement. Unit
tests use fake processes, fake ASR models and local artifact stores; they do not
execute FFmpeg or download models. Output-transaction unit
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

The reproducible CUDA 12 environment is separate from CPU ASR:

```bash
uv sync --frozen --extra asr-faster-whisper-cuda12
uv run --no-sync python scripts/run_phase2_real_gpu_smoke.py \
  --url "$CAPTIONER_REAL_MEDIA_URL" --duration 180
```

The manual script reports GPU, driver, CUDA loader paths, CTranslate2 device
count, supported compute types, `ldd` output, and clean/recovered hashes.
Normal CI never installs CUDA packages or downloads models.

The manual Small/CUDA media run is intentionally outside CI:

```bash
export CAPTIONER_REAL_MEDIA_URL=https://example.invalid/direct-public-domain-media
export CAPTIONER_FASTER_WHISPER_CACHE="$PWD/build/model-cache"
uv run --extra asr-faster-whisper-cuda12 python scripts/run_phase2_real_gpu_smoke.py
```

Record the source page, direct URL, license, downloaded SHA-256, duration, GPU,
CUDA evidence, runtime, Batch/Job IDs, and clean/recovered comparisons. Do not
claim CUDA success from model selection alone.
