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
- Domain JSON metadata is recursively immutable and exporters thaw fresh JSON
  containers without exposing internal mappings or tuples.
- Every Transcript word is assigned to exactly one TranscriptSegment, and every
  referenced word lies within its segment time range.
- All five subtitle outputs are committed only after ASR, deterministic
  segmentation, domain validation and export succeed. Every output is staged
  before any output is committed.
- A cancelled one-shot run leaves no newly committed Transcript or subtitle
  output;
  overwrite rollback restores the previous bytes.
- Public model identity is stable and never contains a machine-specific local
  model path.
- Malformed non-empty Faster Whisper segments are never silently discarded;
  blank segments are ignored only when they contain no words.
- Exporters never mutate Domain objects.
- The same Transcript, canonical policy configuration and exporter versions
  produce deterministic cue IDs, Track IDs and all exported bytes. Simple
  segmentation is only a legacy compatibility facade; Phase 3 uses bounded
  dynamic programming.
- Phase 1 has no LLM; Faster Whisper is optional and loaded once per engine.
- Journal is the durable source of truth; Manifest is only a rebuildable projection.
- `stage.committed` is the linearization point and references only verified CAS artifacts.
- Manifest projection never precedes the corresponding Journal commit.
- Abrupt interruption is `interrupted`; cooperative cancellation is `cancelled`, never failed.
- Retry invalidates only the selected Stage and its downstream suffix.
- Replaying identical Journal bytes always yields an identical immutable projection.
- Content-addressed artifact paths derive only from lowercase SHA-256; orphan bytes are allowed.
- Abrupt interruption may leave incomplete workspace or output projection state, but replay and
  Artifact verification either repair it or fail explicitly.
- External Batch and Job IDs are validated before durable path construction.
- Status never repairs Journal or rewrites Manifest; repair requires the writer lease.
- Status verifies committed content-addressed Artifacts and PublicationReceipt targets without
  changing durable state.
- A Journal-derived `succeeded` state does not by itself prove current output integrity.
- At every complete Journal event boundary, all Jobs in a Batch share one runtime configuration
  signature.
- Batch-wide configuration changes are represented by one crash-atomic `batch.config_updated`
  event.
- A corrupt CAS blob is removed only through its validated `ArtifactRef`; a healthy sibling is
  never removed during detection.
- A cooperative cancel marker is removed only after cancellation events and Manifest projection
  are durable; an interrupted Job may transition directly to `cancelled`.
- Workspace cleanup failure after durable cancellation preserves `cancelled` and never creates a
  failure event.
- A Batch uses one common runtime configuration and distinct publication targets.
- Failed and cancelled Jobs require an explicit `job.retry_requested` before retry.
- Publication receipts are strict and reverify the exact five final target files.
- Publication target verification performs one complete regular-file, size, and hash pass; target
  races and I/O failures are exposed as `output.publication_invalid`.

Phase 3 subtitle invariants:

- Every source Word is assigned to exactly one Cue.
- Final Cue timing is ordered, positive and non-overlapping even when source
  Word timestamps overlap.
- Segmentation and export are pure deterministic functions of Transcript,
  canonical policy configuration and exporter versions.
- Exporter execution never modifies `SubtitleTrack`.
- Golden files cannot be modified without an explicit human-review
  acknowledgement.
- No LLM participates in Phase 3 subtitle segmentation, validation, line
  breaking or export.
- All supported segmentation configuration forms execute the same current
  deterministic dynamic-programming policy; no legacy greedy runtime path remains.
- A schema-2 SubtitleTrack is valid only when its policy signature matches the
  active canonical policy configuration.
- Track identity and language are bound to the source Transcript and active policy.
- Flattening Cue Word assignments yields the complete canonical Transcript Word
  order exactly, and every Cue owns one contiguous canonical Word span.
- ASS export never emits overlapping Dialogue events; unrepresentable timing
  sequences fail with `export.ass_unrepresentable`.
- The committed golden manifest is enforced, including its exact file set,
  policy signature, exporter versions and SHA-256 values.
- `publish-v2` receipts contain the exact five published target formats.
- Source and packaged `subtitle-corpus` execution performs actual JSON, SRT,
  WebVTT and ASS round trips without ASR, models or network access.
