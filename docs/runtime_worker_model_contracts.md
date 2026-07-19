# Phase 6 Runtime / Worker / Model Contracts

Phase 6.0 fixed the contracts used by Runtime and model work. Phase 6.1 adds
the local Runtime manager and real Worker transport described below. It still
does not download or install models, and it does not implement CUDA12,
Hugging Face, or ModelScope model management.

## Ownership and execution flow

Core owns media inspection, FFmpeg normalization, Job state, result
validation, and Artifact Store commits. A Worker owns only the backend-specific
ASR operation. The Worker receives normalized audio and a verified local model
directory; it does not receive `AppPaths`, `data_dir`, a remote repository ID,
or an FFmpeg command.

```text
Core
  → normalized audio
  → Worker Client Port
  → Runtime Worker
  → attempt/result.json
  → descriptor + SHA-256
  → Core validation
  → Artifact Store commit
```

The future transport is JSONL: Core writes protocol messages to Worker stdin,
Worker writes protocol messages only to stdout, and human-readable diagnostics
go to stderr. Complete Transcripts do not travel in JSONL; an operation result
contains a relative result path, byte size, SHA-256, schema ID, and schema
version. Core validates all five properties before an Artifact Store commit.

## Runtime contract

`RuntimeIdentity` contains only a stable ID and version. A `RuntimeManifest`
binds that identity to a normalized platform, architecture, device, backend,
Worker protocol version, capabilities, model formats, archive hash, and a
POSIX-relative file manifest. Absolute paths and `..` traversal are invalid.

`installed` means the archive and static files passed verification.
`available` additionally means Worker launch, protocol handshake, backend and
device checks, workspace round-trip, and clean shutdown passed activation
Doctor. An `external_unmanaged` record is not managed by Captioner for upgrade
or deletion. It can be active only after the same activation checks pass.

The active-runtime slot is `(backend_id, platform, architecture, device)`;
minimum OS version is a compatibility constraint, not a second slot key.
Selectors consume the repository's active pointers and do not guess between
multiple versions. A duplicate active candidate is an ambiguity error.

Static Doctor and Activation Doctor are separate operations. Static Doctor
validates the installed payload and sidecar/build metadata. Activation Doctor
starts the Runtime's own Python, performs the v1.1 handshake and Doctor probe,
checks backend imports/device visibility, verifies a workspace round-trip, and
shuts the Worker down cleanly.

Runtime packages are `.tar.gz` archives with an external `.runtime.json`
descriptor. The descriptor carries the archive size and the Manifest whose
`archive_sha256` is the digest of the archive bytes; the descriptor is never
inside the archive. Archives contain only regular files and directories under
the `payload/` tree. The manager validates every path, size, hash, and
executable bit before moving a staged transaction into
`<data_dir>/runtimes/<runtime-id>/<version>/`.

Activation uses `current`, `previous`, and `pending_activation` in an atomic
`active.json`. A failed activation restores the previous pointer and retains
the candidate for diagnosis. Startup recovery performs the same restoration
when a pending pointer is found. Managed current, previous, pending, or
in-use Runtimes cannot be removed. External Developer Mode Runtimes are only
registered after Manifest and Doctor validation; their files are never
copied, upgraded, or deleted by Captioner.

## Model contract

`ModelIdentity` separates backend, source, repository, revision, format, and
manifest hash from display text. Two sources or revisions may display the same
name while remaining different identities. A local absolute path never enters
the durable identity; it belongs only to an installation record or a single
Worker request.

Search results use `ModelSourceCandidate`, which contains safe display and
repository metadata and may have no revision yet. Exact resolution returns a
`ModelSourceReference` with an immutable revision. Neither type contains a
local directory or `manifest_sha256`, and neither is a durable Job model
identity. A final identity is created only after revision resolution,
materialization/import, file inventory, validation, and canonical Manifest
digest computation. `LocalModelInspector` is the separate Port for directories
that do not have a Captioner Manifest yet; its inspection projection carries
detection and validation results, while local import and external-path
registration remain later Model Manager work.

Managed models may be removed by a later repository implementation. External
models are advanced-mode references: Captioner does not copy or delete their
files. `installed` means file, size, hash, and manifest verification passed;
`load_verified` is a separate state earned after a compatible Runtime has
loaded the model. External models must first have a local validation projection
before they can be selected. A canonical Manifest digest is computed from
schema, stable identity fields (excluding `manifest_sha256` itself), display
metadata, sorted file entries, sorted backend/capability sets, source metadata,
and model constraints. JSON uses sorted keys, compact separators, UTF-8, and
finite values only; local absolute paths are never included.

Faster Whisper and MLX Whisper are separate backends and formats:

```text
faster-whisper + cpu/cuda  → faster-whisper-ct2
mlx-whisper + metal       → mlx-whisper
```

The Phase 6.0 MLX validator contract requires `config.json` and at least one
of `model.safetensors`, `weights.safetensors`, or `weights.npz`. No MLX
package, conversion step, SDK, or remote source is included here. Remote model
files are not hosted by this repository.

## Auto selection

Selection is a pure preflight policy. It never downloads, installs, mutates an
active pointer, converts a model, or reinterprets a persisted effective
selection. On native Apple Silicon with macOS 14 or later, `device=auto`
selects an available compatible MLX Metal Runtime for an MLX model. A Faster
Whisper CT2 model selects an available compatible CUDA Runtime first and then
CPU. An MLX model never falls back to Faster Whisper, and a Faster Whisper
model is never silently converted to MLX. Windows and Linux cannot select the
MLX Metal Runtime.

`auto` selection produces effective backend, Runtime identity, device, and
model identity values. Job creation will persist those values; Resume uses
the persisted effective values rather than running auto selection again.
For a Faster Whisper CT2 model, an available compatible CUDA Runtime is chosen
before CPU; if CUDA is unavailable, CPU is chosen. This priority applies on
Windows, Linux, and any host where a compatible CUDA Runtime is active. MLX
models remain MLX-only and never fall back to CPU/CUDA.

## Protocol and handshake

Every JSONL envelope is decoded against the typed payload schema for its
`message_type`; a valid envelope with an arbitrary placeholder object is not
accepted. Same-major protocol versions may add optional fields, but required
fields and field types remain validated. Core sends a `HandshakeRequest`, and
the activation policy checks the Worker Runtime identity, compatible backend
version, normalized platform/architecture, required capabilities and result
schemas, model formats, and target device before the Runtime becomes usable.

The real `SubprocessWorkerClient` launches the Runtime-provided interpreter in
an isolated environment. Core owns the stderr log; stdout is JSONL only. A
partial line, ordinary stdout text, unknown message, correlation mismatch, or
non-monotonic sequence is a structured protocol failure. Cooperative cancel
waits for `cancel.acknowledged` and `operation.cancelled`; a timeout escalates
from process-group termination to process-group kill. The scripted Worker fake
models the same lifecycle without starting a production Runtime.

## Progress and errors

Runtime, model, and ASR progress reports identify the current phase only, such
as `verifying_archive`, `loading_model`, or `transcribing`. Protocol v1.1 does
not carry percentages, completed-unit counts, ETA, or equivalent precision.

Public Worker errors contain a stable code, localized message code, retryable
flag, and safe details. Tracebacks belong to stderr/runtime logs in a later
implementation. Credentials are rejected from protocol objects and never
enter a message representation. Progress remains phase-only; no percentage,
byte count, or ETA is transmitted.

## Runtime implementations in Phase 6.1

The Faster Whisper CPU Runtime uses its bundled Python 3.12.9 and exact
`faster-whisper==1.2.1` / `ctranslate2==4.8.1` dependencies. It accepts only an
absolute local CTranslate2 model directory and uses CPU `int8` by default.
The MLX Metal Runtime is native macOS arm64 only, requires macOS 14 or newer,
and uses `mlx-whisper==0.4.3`. It accepts only a local MLX model directory;
`config.json` and one of the supported weight filenames are required. Neither
Worker downloads a model or resolves a remote repository ID.

The ordinary `captioner run --model ...` path remains on the existing
in-process Faster Whisper adapter in Phase 6.1. `WorkerBackedASREngine` is an
injection seam and is exercised by Runtime Doctor and isolated tests. PR 6.2
may construct it after Model Manager preflight; it must not make an uninstalled
model look available.
