# Recovery

Resume acquires the local Batch lease, repairs only an unterminated Journal tail,
replays all complete events, reconciles the Manifest, records stale open attempts
as `stage.interrupted`, verifies every committed Artifact, invalidates the first
bad Stage and its downstream suffix, removes stale workspaces, and continues.

A cooperative cancel marker becomes durable cancellation events when observed.
A process disappearance is interrupted, not cancelled or failed. Retry appends
invalidation events and never rewrites history. Local leases do not claim
network-filesystem or distributed-worker safety.

Manual fault injection is disabled unless `CAPTIONER_ENABLE_FAULT_INJECTION=1`.
With that guard, `CAPTIONER_FAULT_POINT=transcribe:after_journal_commit`
selects one documented checkpoint; normal CLI operation cannot activate it.

Abrupt interruption may leave incomplete workspace or output projection state,
but Journal replay and Artifact verification either repair it or fail explicitly.
