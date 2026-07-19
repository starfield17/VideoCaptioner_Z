# Runtime Builds

Runtime Workers are separate packages from the Core application. Each Runtime
project has its own `pyproject.toml`, `runtime-build.toml`, and `uv.lock` and
requires the exact managed Python 3.12.9. Build commands use locked
dependencies and package a complete interpreter plus site-packages under:

```text
payload/python/
  bin/python3       # macOS/Linux
  python.exe         # Windows
```

The build script builds the stdlib-only `captioner-runtime-worker` wheel,
installs the Runtime's locked backend dependencies into that interpreter,
writes immutable `build_info.json`, removes caches and bytecode, runs import
and relocation smoke checks, inventories files, and creates a deterministic
`.tar.gz` plus external `.runtime.json` descriptor. Model weights are never
part of a Runtime archive.

```bash
uv run python scripts/build_runtime.py \
  --project faster-whisper-cpu \
  --version 1.0.0 \
  --output dist/runtimes
```

The Faster Whisper project targets macOS arm64, Windows x86_64, and Linux
x86_64 with CPU `int8`. The MLX project can only be built on native macOS
arm64; the build refuses a translated/Rosetta process. No cross-platform
archive is fabricated.

After a local build, the optional smoke command installs the descriptor through
the normal manager and runs Static plus Activation Doctor:

```bash
uv run python scripts/runtime_smoke.py \
  --descriptor dist/runtimes/captioner-runtime-*.runtime.json \
  --doctor
```

“Implemented” means the code and contract tests exist. “Tested in unit/
integration” means the local test suites passed. “Real runtime smoke verified”
is reported only for a build and Doctor run actually performed on the matching
host and dependencies. No release artifact is published by this PR.
