# Runtime Manager

Phase 6.1 stores isolated ASR Runtimes below the application data root:

```text
<data_dir>/runtimes/
├── active.json
├── .manager.lock
├── <runtime-id>/<version>/
│   ├── runtime-package.json
│   ├── runtime-manifest.json
│   ├── installation.json
│   └── payload/
├── external/<safe-record-id>.json
<staging_dir>/runtimes/<transaction-id>/
<downloads_dir>/runtimes/<transaction-id>.part
<log_dir>/runtimes/<runtime-id>/<session-id>.log
```

## Package and install transaction

The `.runtime.json` sidecar is resolved from a local path or an HTTPS URL. Its
`archive_filename`, declared byte size, and nested Runtime Manifest are
required. The Manifest is external to the archive so its archive digest is not
self-referential.

Installation holds the manager lock, materializes a `.part` archive, verifies
the declared size and SHA-256, preflights every tar member, extracts into a
transaction staging directory, verifies the complete file inventory, runs
Static Doctor, and atomically moves the staged directory into its final
identity/version directory. A failure leaves the active pointer unchanged.
An identical already-installed identity is idempotent; a different Manifest
for the same identity is rejected.

Archive extraction rejects absolute paths, traversal, drive paths, links,
special files, duplicate entries, extra files, missing files, and hash/size
mismatches. The Runtime's Python interpreter is inside `payload/python`; Core
does not use the system Python, shell activation, or an implicit `PATH` Python.

## Activation, rollback, and recovery

The active slot is `(backend_id, platform, architecture, device)`. Minimum OS
version is checked during compatibility validation but is not part of the
slot key. Activation first writes the candidate as `pending_activation`, runs
Activation Doctor while holding the version use lock, and then commits the
candidate as `current` with the old current as `previous`. A failed Doctor
restores the old pointer and marks the candidate failed without deleting it.

`recover()` performs that restoration for an interrupted pending activation.
Rollback validates and activates `previous` through the same Doctor path. A
current, previous, pending, or in-use Runtime is protected from removal.
External Developer Mode records have `managed=false`: removing the record
does not touch the external Runtime directory.

The manager reports only operation phases such as `downloading`,
`verifying_archive`, `activating`, and `running_doctor`. It does not fabricate
percentages, byte progress, or ETA values.

## Scope

Phase 6.1 implements local/HTTPS Runtime packages, Faster Whisper CPU, and
native MLX Metal. It does not download models, migrate model caches, implement
CUDA12, publish release artifacts, or change the ordinary run default. Those
boundaries remain deferred to PR 6.2 and the later Release Closure work.
