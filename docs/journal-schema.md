# Journal schema

`journal.jsonl` is UTF-8 canonical JSON, one fsynced event per newline, ordered by
strictly increasing `seq`. Events contain schema version, event ID, UTC timestamp,
Batch ID, type, and bounded payload. Stage events identify Job, Stage, and attempt.
Committed events contain only cache keys and `ArtifactRef` objects.

`batch.config_updated` has one payload for the complete Batch:

```json
{
  "config": {"schema_version": 1, "...": "the complete JobConfig"},
  "earliest_stage": "transcribe"
}
```

Replay applies that event to every Job in memory before the single event is
appended. It replaces the common configuration and invalidates the affected
Stage suffix while preserving attempt numbers. The CLI never emits a sequence
of per-Job configuration events; legacy `job.config_updated` records remain
readable only for older development Journals.

Phase 3 persists the complete segmentation policy in the canonical JobConfig.
Legacy Phase 2 mappings containing only `max_duration_ms`, `max_text_units`,
and `hard_gap_ms` are accepted and filled with deterministic defaults; old
events are not rewritten. Stage versions `segment-v2`, `export-v2`, and
`publish-v2` make an upgraded Batch invalidate the correct suffix. At every
complete event boundary, all Jobs have one runtime configuration signature.

Bytes after the final newline are truncated even if valid JSON. Any malformed,
invalid UTF-8, schema-invalid, missing-sequence, or duplicate complete line is
`journal.corrupt`. An uncertain append is reconciled by exact event ID, sequence,
and content; duplicate appends are never guessed.
