# Captioner engineering rules

This file is the repository-local contract for humans and coding agents.

- CLI commands do not call another CLI command's `run()` function.
- GUI code does not directly import ASR, media, model, or LLM SDKs.
- Domain code does not import adapters, CLI, GUI, or external SDKs.
- A future Stage must not mutate its input in place.
- An LLM must never modify timestamps.
- Exceptions must not be silently swallowed.
- `# type: ignore` and `# noqa` require a specific rule and an explanation.
- Unit tests must not call real APIs, real networks, models, or FFmpeg.
- Coding agents must not lower strictness, coverage, or lint standards to make CI pass.
- Coding agents must not automatically batch-update golden files.
- Every patch must report the tests run and known limitations.

Phase 0 intentionally contains no real subtitle processing, ASR, LLM, model,
FFmpeg, queue, job-state, runtime-installation, or release behavior.
