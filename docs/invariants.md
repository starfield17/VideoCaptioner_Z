# Invariants

- CLI commands do not call CLI commands.
- GUI does not own a business state machine.
- Core does not produce localized error sentences.
- Built-in resources are read-only.
- Writable data uses OS-standard `platformdirs` locations.
- External SDKs may appear only behind adapter/runtime boundaries.
- Future ASR concurrency defaults to one.
- Future LLM calls use one global provider concurrency gate.
- LLMs never modify timestamps.
- Future stages must not mutate their input in place.
- FFprobe and FFmpeg never invoke a shell; process arguments remain separate.
- Domain timestamps are integer milliseconds; SDK float seconds are converted at
  the ASR adapter boundary.
- SRT is committed only after ASR, domain validation, segmentation and export
  succeed.
- Exporters never mutate Domain objects.
- The same Transcript and segmentation configuration produce deterministic cue
  IDs, JSON bytes and SRT bytes.
- Phase 1 has no LLM; Faster Whisper is optional and loaded once per engine.
