# Architecture

Phase 1 establishes the following dependency direction:

```text
GUI / CLI
   ↓
CLI boundary / composition root
   ↓
Core application and policies
   ↓
Core domain and ports
   ↓
Adapters / infrastructure
```

The presentation layers select and render commands. `core.application.run_single`
orchestrates one input without knowing concrete adapters. `core.domain` contains
immutable, localization-neutral media, transcript, subtitle, execution and
error values. `core.policies` contains deterministic segmentation. `core.ports`
defines media, process, ASR, and artifact boundaries, while `adapters/*` owns
FFprobe, FFmpeg, Faster Whisper, exporters and local persistence. GUI and CLI
are not allowed to import each other, and Core cannot depend on adapters or SDKs.

`bootstrap.py` is the explicit composition root. It creates one process runner,
one FFprobe/FFmpeg pair, one Faster Whisper engine, one local artifact store per
run, and the injected serializers. The application uses a temporary workspace
for normalized audio and never writes it beside the input or into resources.

The vertical slice is:

```text
input → FFprobe → normalized WAV → ASR → Transcript
      → simple segmentation → Transcript JSON + SRT → atomic commit
```

Faster Whisper is dynamically imported only by its adapter and is an optional
extra. The default Core App build can therefore provide `run --help` and a
structured `asr.runtime_missing` error without bundling the ASR runtime.

The long-term runtime direction keeps three concerns separate:

- Core App: business orchestration and policies.
- Runtime: installed/runtime capability management.
- Model: model loading and provider-specific execution.

Phase 1 deliberately has no reusable Stage framework, GUI workflow, batch/job
state, manifest/journal recovery, LLM, translation, forced alignment, muxing,
runtime installer, model manager, or release workflow. Faster Whisper
cancellation is cooperative at process/segment boundaries; hard worker
isolation is deferred to the runtime phase.
