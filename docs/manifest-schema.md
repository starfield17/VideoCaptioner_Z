# Manifest schema

Manifest schema version 1 stores Batch ID, last event sequence, deterministic
projection hash, derived Batch state, Jobs, Stages, attempts, cache keys, and
Artifact references. It is written through same-directory temporary bytes,
file fsync, replace, and directory fsync.

Missing or behind Manifests rebuild from Journal. Ahead is rejected. Equal
sequence with a different projection or hash is `manifest.inconsistent`.
Manifest is never used to synthesize Journal history.
