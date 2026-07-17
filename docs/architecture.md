# Architecture

Phase 2 through Phase 4 preserve the following dependency direction:

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
error values. `core.policies` contains Unicode metrics, reading-speed,
protected-span, line-breaking, and deterministic segmentation policies.
`core.ports` defines media, process, ASR, artifact, and subtitle-exporter
boundaries, while `adapters/*` owns FFprobe, FFmpeg, Faster Whisper, subtitle
format codecs and local persistence. GUI and CLI are not allowed to import
each other, and Core cannot depend on adapters or SDKs.

`bootstrap.py` is the explicit composition root. It creates one process runner,
one FFprobe/FFmpeg pair, one Faster Whisper engine, one local artifact store per
run, and the injected serializers. When an LLM profile runs, it also creates one
shared provider runtime, HTTP client, and application-wide Semaphore. Every LLM
Stage and Job receives that same runtime. The application uses a temporary
workspace for normalized audio and never writes it beside the input or into
resources.

The durable Stage plan is selected from the persisted Job profile:

```text
deterministic: inspect → normalize → transcribe → segment → export → publish
fast:          inspect → normalize → transcribe → segment → translate → export → publish
quality:       inspect → normalize → transcribe → correct_source → segment
               → translate → review → export → publish
```

Unused profile Stages do not receive synthetic commits. Schema-v1 Jobs replay as
the original deterministic six-Stage plan.

Segment decodes the Transcript, canonicalizes Words, solves cue boundaries,
breaks lines, validates an immutable `SubtitleTrack`, and writes
`subtitle-track.json`. Export validates the Track again and serializes the
Transcript plus the Track as JSON, SRT, WebVTT, and ASS. Publish stages all
five user-facing targets in deterministic logical-name order. It checks its
`ExecutionContext` before the first commit, between commits, and before
returning success. A failure or cancellation
restores every output committed by the current invocation, including previous
bytes in overwrite mode; this is an in-process transaction, not durable crash
recovery.

Deterministic-profile subtitle processing remains a pure function of the
Transcript, canonical policy configuration, and exporter versions. Fast and
Quality may obtain text from an LLM, but Cue boundaries, timestamps, Word
mapping, line breaking, validation, serialization, and publication remain
application-owned and deterministic.

The Faster Whisper adapter keeps `model_ref` (the SDK loading reference)
separate from `model_identity` (the stable public value). Named models use a
provider-prefixed identity. Local models use a bounded manifest of recognized
identity files and never serialize their absolute directory path.

Faster Whisper is dynamically imported only by its adapter and is an optional
extra. The default Core App build can therefore provide `run --help` and a
structured `asr.runtime_missing` error without bundling the ASR runtime.

The long-term runtime direction keeps three concerns separate:

- Core App: business orchestration and policies.
- Runtime: installed/runtime capability management.
- Model: model loading and provider-specific execution.

Stage runners only produce workspace files or bytes. The generic executor owns
attempt events, CAS import and verification, the authoritative
`stage.committed` event, then Manifest projection. Journal replay is pure and
immutable. Recovery repairs a stale Manifest, records open attempts as
interrupted, verifies artifacts, and invalidates only the affected suffix.

Phase 4 keeps sequential Jobs and one ASR engine per active Batch. LLM Chunks
may execute concurrently through one application-wide provider gate. It has no
GUI workflow, parallel Job scheduler, provider fallback/routing, forced
alignment, distributed locking, artifact GC, runtime installer, muxing, or
release workflow.
