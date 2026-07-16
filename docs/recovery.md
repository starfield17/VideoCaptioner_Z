# Recovery

Resume acquires the local Batch lease, repairs only an unterminated Journal tail,
replays all complete events, reconciles the Manifest, records stale open attempts
as `stage.interrupted`, verifies every committed Artifact, invalidates the first
bad Stage and its downstream suffix, removes stale workspaces, and continues.

`status` is non-mutating. It reads complete Journal records, reports an
incomplete final fragment, and inspects Manifest as `current`, `missing`,
`stale`, `ahead`, `projection_mismatch`, or `invalid`. Only a writer holding
the Batch lease may repair Journal tails or rewrite a Manifest. It separately
verifies each committed CAS blob, then verifies the Publish receipt and both
canonical target files (including target hash, size, regular-file and symlink
checks). Journal state remains visible even when `integrity` is `invalid`.

When recovery finds multiple bad Stage artifacts, it first records every bad
`ArtifactRef` and removes/quarantines only those exact hash paths. A healthy
sibling is not a cleanup candidate. A healthy receipt with an invalid target
invalidates Publish only; a missing or corrupt receipt blob invalidates Publish,
removes that receipt blob, and skips receipt/target verification until the new
receipt is committed. Export artifacts and existing targets are retained while
Publish is rerun.

All Jobs in a Batch share one persisted runtime configuration. Output target
collisions are rejected before `batch.created`. Failed or cancelled Jobs are
not resumed automatically; `retry` appends `job.retry_requested` and reopens
only the requested Stage suffix while retaining historical terminal events.

A cooperative cancel marker becomes durable cancellation events when observed.
The owning service appends missing `stage.cancelled`/`job.cancelled` events,
projects the Manifest once, and removes the marker only after that write
succeeds. Job-only cancellation clears only its Job marker and continues other
Jobs. Batch cancellation covers pending, running and interrupted Jobs in one
lease. A process disappearance is interrupted, not cancelled or failed; an
interrupted Job may subsequently become cancelled without rewriting its
historical Stage event. Once cancellation is durably projected, workspace
cleanup failure preserves cancellation and cannot append failure events. Retry
appends invalidation events and never rewrites history. Local leases do not
claim network-filesystem or distributed-worker safety.

Batch-wide resume overrides are committed as one `batch.config_updated` event.
Model/language/device/compute/VAD changes invalidate Transcribe onward;
segmentation changes invalidate Segment onward; output/overwrite changes
invalidate Publish only; FFprobe changes invalidate Inspect; FFmpeg or
normalization changes invalidate Normalize onward. A new `resume --output`
directory is expanded, created and verified while holding the writer lease,
before that event is appended. Creation failure leaves the Journal unchanged.

Manual fault injection is disabled unless `CAPTIONER_ENABLE_FAULT_INJECTION=1`.
With that guard, `CAPTIONER_FAULT_POINT=transcribe:after_journal_commit`
selects one documented checkpoint; normal CLI operation cannot activate it.

Abrupt interruption may leave incomplete workspace or output projection state,
but Journal replay and Artifact verification either repair it or fail explicitly.

Publication target verification performs one regular-file, size, and SHA-256
pass per target; filesystem races and I/O failures are reported as
`output.publication_invalid`. Concrete Stage midpoint boundaries are: Inspect after inspection and before
return; Normalize after the first non-empty normalized-WAV hash chunk;
Transcribe while consuming the first Faster Whisper segment; Segment after the
first accepted cue while more segmentation work remains; Export between JSON
serialization and SRT serialization; and Publish after the first target commit
inside the shared output transaction. No completed Stage returns from a
midpoint checkpoint.
