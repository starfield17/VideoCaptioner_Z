# Artifact store

Durable bytes enter `artifacts/.incoming`, are streamed through SHA-256 while
being written, flushed and fsynced, then replaced into
`artifacts/sha256/<prefix>/<digest>`. Existing digest paths must pass size and
hash verification before deduplication. Physical paths are derived and are not
serialized in Journal events.

An artifact stored before a Stage commit is an allowed orphan and does not make
the Stage committed. Missing or corrupt committed artifacts invalidate that
Stage and downstream dependents. Garbage collection and cross-Batch result
indexes are outside Phase 2.
