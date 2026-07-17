# Captioner engineering rules

This file is the repository-local contract for humans and coding agents.

- CLI commands do not call another CLI command's `run()` function.
- GUI code does not directly import ASR, media, model, or LLM SDKs.
- Domain code does not import adapters, CLI, GUI, or external SDKs.
- A Stage must not mutate its input in place.
- An LLM must never modify timestamps or source Word mapping.
- API keys are plaintext credentials in the OS config directory and must never
  enter JobConfig, Journal, Manifest, Artifact, Cache metadata, logs, errors, or
  CLI JSON output.
- Every LLM Stage and Job shares the single application-level provider client
  and Semaphore created by the composition root.
- Prompt content changes require a new prompt version; an existing prompt file
  must not be silently replaced.
- Exceptions must not be silently swallowed.
- `# type: ignore` and `# noqa` require a specific rule and an explanation.
- Unit, contract, and recovery tests must not call real APIs, real networks,
  models, or FFmpeg; use injected process and model fakes.
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

Phase 4 has profile-specific sequential plans. Deterministic keeps the original
six Stages; Fast adds Translate; Quality adds CorrectSource, Translate, and
Review. Journal is authoritative, Manifest is rebuildable, and Stage commit
cannot precede durable artifact verification. The repository still has no GUI
workflow, parallel Job scheduler, provider fallback/routing, alignment,
distributed workers, runtime installation, muxing, or release behavior.

Phase 3 subtitle processing is deterministic and has no legacy greedy runtime
path: all supported segmentation configurations execute the same bounded DP
solver. Current schema-2 SubtitleTracks require a policy signature bound to the
active policy, a verified Track ID, canonical Word order, and contiguous Cue
spans. ASS serialization quantizes the complete ordered track and rejects
timings that cannot be represented within its 10 ms tolerance. The strict
`publish-v2` receipt contains exactly the five transcript/subtitle targets.

LLM responses can provide only stable IDs and text. Application code copies all
Cue IDs, timestamps, and Word IDs, and deterministic policies retain exclusive
ownership of Cue boundaries and line breaking. Only fully validated Chunk
responses enter the LLM Cache. Prompt versions and content hashes both
participate in Cache identity.

The `subtitle-corpus` command is a normal application command that performs
the deterministic subtitle pipeline through source or Nuitka builds without
ASR, FFmpeg, network or model access. Golden manifests are enforced by tests;
golden updates require explicit human-review acknowledgement and must not be
performed automatically by pytest or CI.
