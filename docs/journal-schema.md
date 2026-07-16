# Journal schema

`journal.jsonl` is UTF-8 canonical JSON, one fsynced event per newline, ordered by
strictly increasing `seq`. Events contain schema version, event ID, UTC timestamp,
Batch ID, type, and bounded payload. Stage events identify Job, Stage, and attempt.
Committed events contain only cache keys and `ArtifactRef` objects.

Bytes after the final newline are truncated even if valid JSON. Any malformed,
invalid UTF-8, schema-invalid, missing-sequence, or duplicate complete line is
`journal.corrupt`. An uncertain append is reconciled by exact event ID, sequence,
and content; duplicate appends are never guessed.
