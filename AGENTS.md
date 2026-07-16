# Captioner engineering rules

This file is the repository-local contract for humans and coding agents.

- CLI commands do not call another CLI command's `run()` function.
- GUI code does not directly import ASR, media, model, or LLM SDKs.
- Domain code does not import adapters, CLI, GUI, or external SDKs.
- A future Stage must not mutate its input in place.
- An LLM must never modify timestamps.
- Exceptions must not be silently swallowed.
- `# type: ignore` and `# noqa` require a specific rule and an explanation.
- Unit tests must not call real APIs, real networks, models, or FFmpeg; use
  injected process and model fakes.
- FFprobe and FFmpeg subprocesses must use argument arrays and never a shell.
- Faster Whisper is an optional lazy-loaded adapter; it must not be imported by
  CLI help, GUI startup, Core, or the default Nuitka Core App.
- Domain timestamps are integer milliseconds. SDK float seconds are converted
  once at the adapter boundary.
- The final subtitle outputs are committed only after ASR, deterministic
  segmentation, domain validation, and all five exports succeed. Exporters do
  not mutate domain objects.
- Output artifacts are staged and fsynced before commit; cancellation or failure
  must roll back every current-run commit and restore overwrite targets.
- Domain JSON metadata is recursively frozen; exporters must thaw fresh mutable
  JSON values rather than exposing internal containers.
- Transcript words belong to exactly one segment and referenced words must lie
  within segment time ranges.
- Faster Whisper `model_ref` is an SDK loading reference; only stable
  `model_identity` may enter public Transcript data.
- Non-empty malformed ASR segments must raise a structured error rather than be
  silently discarded.
- Coding agents must not lower strictness, coverage, or lint standards to make CI pass.
- Coding agents must not automatically batch-update golden files.
- Every patch must report the tests run and known limitations.

Phase 2 has a fixed sequential six-Stage durable pipeline. Phase 3 adds only
deterministic subtitle policies, validation, and SRT/WebVTT/ASS/JSON export.
Journal is authoritative, Manifest is rebuildable, and Stage commit cannot
precede durable artifact verification. The repository still has no GUI
workflow, parallel execution, LLM, translation, alignment, distributed
workers, runtime installation, muxing, or release behavior.
