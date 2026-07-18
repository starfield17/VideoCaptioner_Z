# Phase 6 Runtime / Worker / Model Contracts

Phase 6.0 fixes the contracts used by later Runtime and model work. It does
not install a Runtime, download a model, start a subprocess, or call a remote
model registry.

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

Static Doctor and Activation Doctor are separate Port operations. Phase 6.0
defines their structured reports and deterministic fake only; it does not run
either real Doctor.

## Model contract

`ModelIdentity` separates backend, source, repository, revision, format, and
manifest hash from display text. Two sources or revisions may display the same
name while remaining different identities. A local absolute path never enters
the durable identity; it belongs only to an installation record or a single
Worker request.

Managed models may be removed by a later repository implementation. External
models are advanced-mode references: Captioner does not copy or delete their
files. `installed` means file, size, hash, and manifest verification passed;
`load_verified` is a separate state earned after a compatible Runtime has
loaded the model.

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
selects an available compatible MLX Metal Runtime for an MLX model. Otherwise,
an available compatible Faster Whisper CPU Runtime is selected for a Faster
Whisper model. An MLX model never falls back to Faster Whisper, and a Faster
Whisper model is never silently converted to MLX. Windows and Linux cannot
select the MLX Metal Runtime.

`auto` selection produces effective backend, Runtime identity, device, and
model identity values. Job creation will persist those values; Resume uses
the persisted effective values rather than running auto selection again.

## Progress and errors

Runtime, model, and ASR progress reports identify the current phase only, such
as `verifying_archive`, `loading_model`, or `transcribing`. Protocol v1 does
not carry percentages, completed-unit counts, ETA, or equivalent precision.

Public Worker errors contain a stable code, localized message code, retryable
flag, and safe details. Tracebacks belong to stderr/runtime logs in a later
implementation. Credentials are rejected from protocol objects and never
enter a message representation.
