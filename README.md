# Captioner

Captioner is a durable batch subtitle-generation application. The deterministic
profile runs inspect, normalize, transcribe, segment, export, and publish.
Phase 4 adds Fast translation and Quality terminology/correction/translation/
anomaly-review profiles while keeping timestamps, Word mapping, segmentation,
validation, and publication under deterministic application control. An
fsynced Journal is the source of truth; a rebuildable Manifest and verified
content-addressed artifacts support resume and retry.

## Local commands

The supported local interpreter for this checkout is the `Lab` environment at
`/home/hazel/miniconda3/envs/Lab/bin/python` (Python 3.13).

```bash
uv sync --frozen
uv run python scripts/check.py --full
uv run python main.py --cli --help
uv run python main.py --cli run --help
uv run python main.py --cli doctor --json
QT_QPA_PLATFORM=offscreen uv run captioner-gui --lang en --smoke-test
QT_QPA_PLATFORM=offscreen uv run captioner-gui --lang zh-CN --smoke-test
uv run python scripts/build_nuitka.py --clean --version 0.0.0
```

Phase 5 desktop workflow pages: Create, Queue, History, Settings, and
Diagnostics. English and Simplified Chinese catalogs are required. Diagnostics
exports a redacted aggregate ZIP (no credentials, media paths, source text, or
subtitles). Overall branch coverage hard floor is 80%; 85% remains an
engineering target.

Compiled smoke tests must run from a temporary working directory outside the
repository so a broken resource resolver cannot use `resources/` from the
source tree. Packaged GUI smoke also uses isolated temporary home/config
directories and both `--lang en` and `--lang zh-CN`. The Release Full Gate
archives and then extracts the exact tested outputs: Linux
`captioner-linux.tar.gz`, Windows `captioner-windows.zip`, and macOS
`Captioner-macos.zip` containing `Captioner.app`. These are portable artifacts;
the project does not sign or notarize them, create an installer, or publish a
GitHub Release automatically.

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

Fast and Quality use an OpenAI-compatible provider profile from the OS config
directory; credentials are never accepted on the CLI or persisted in Job data:

```bash
captioner run input.mp4 --output build/output --profile fast \
  --target-language zh-CN --llm-provider-profile default --json
captioner run input.mp4 --output build/output --profile quality \
  --target-language zh-CN --llm-provider-profile default --json
```

The durable Job stores the complete redacted public provider snapshot and
profile-specific Prompt identities. Resume must match every public provider
field before creating the HTTP client; only API-key rotation is allowed. All
LLM Stages and Jobs share one provider client and Semaphore, and complete
encoded requests (including Prompt, dynamic context, and response schema) are
checked against the Chunk budget.

See [docs/llm.md](docs/llm.md) for the plaintext `llm.toml` format, structured
schemas, Cache identity, retry classification, recovery, and limitations.

Phase 3's deterministic fixture command runs without ASR, FFmpeg, models or
network access. It performs DP segmentation, Track validation, canonical JSON
decode/re-encode, and SRT/WebVTT/ASS round trips:

```bash
uv run captioner subtitle-corpus tests/fixtures/transcripts --json
./dist/captioner/captioner --cli subtitle-corpus tests/fixtures/transcripts --json
```

The committed subtitle golden manifest is strict: it binds the complete fixture
file set, policy signature, exporter versions and every file SHA-256. Goldens
can be changed only after reviewing the updater's proposed semantic diff and
passing `--accept PHASE3_GOLDENS_REVIEWED` explicitly.

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

The durable Export and Publish Stages write these five deterministic targets:

```text
<source-stem>.transcript.json
<source-stem>.subtitle.json
<source-stem>.srt
<source-stem>.vtt
<source-stem>.ass
```

They are produced only after successful transcription, deterministic cue
segmentation, Track validation, and a staged atomic artifact transaction.
Cancellation or failure rolls back outputs committed by the current run;
`--overwrite` restores the previous bytes when rollback is needed.
Domain JSON metadata is recursively immutable, and public model identities do
not contain machine-specific local model paths.

CLI exit codes are stable: `0` success, `2` usage/configuration, `3`
media/FFmpeg, `4` ASR/runtime/model, `5` output/export, `7` LLM/provider, and
`130` cancellation. One shared application-level Semaphore bounds all LLM
Stages and Jobs. Unit tests and default CI never download models or call real
providers.

Durable state lives under the platformdirs data directory in `batches/` and
`artifacts/`; workspaces are attempt-scoped and disposable. Built-in resources
are read from `resources/`. User-writable paths are owned
by the operating system's standard application directories through
`platformdirs`.

Batch and Job IDs are validated before durable path construction. Status is a
non-mutating read: it reports incomplete Journal tails and Manifest status,
verifies every committed Artifact and publication target, and does not repair
or rewrite either file. Journal-derived state and current output integrity are
separate; a durable `succeeded` state can therefore be reported with
`integrity: invalid`. JSON status returns the document with its integrity
errors instead of converting that result into an exception. Resume and retry
acquire the Batch writer lease before repair. At every complete Journal event
boundary, all Jobs in a Batch share one runtime configuration signature; a
Batch-wide override is one `batch.config_updated` event. Failed or cancelled
Jobs require explicit `retry`, which appends `job.retry_requested`.

## Deterministic subtitle processing

The subtitle flow is:

```text
Transcript → canonical Word order → bounded dynamic programming
→ display-width line breaking → SubtitleTrack validation
→ canonical JSON, SRT, WebVTT and ASS
```

It uses NFC text normalization, Unicode grapheme clusters, `wcwidth` display
columns, exact integer CPS arithmetic, protected numeric spans, punctuation and
silence boundaries, and documented deterministic tie-breaks. It does not call
an LLM, translation model, network service, clock, locale formatter, or
filesystem iteration order.

Reviewed corpus goldens are checked byte-for-byte. Updating them requires the
explicit acknowledgement `PHASE3_GOLDENS_REVIEWED`; normal tests never rewrite
goldens:

```bash
uv run python scripts/run_subtitle_corpus.py tests/fixtures/transcripts
uv run python scripts/update_subtitle_goldens.py \
  --accept PHASE3_GOLDENS_REVIEWED
```
