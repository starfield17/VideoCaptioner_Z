# Phase 5 Decision Record

**Status:** Accepted implementation baseline  
**Date:** 2026-07-17

## Delivery

- Use small vertical PRs.
- Phase 5 may proceed while Release Full Gate remains separately pending.
- Minimum Phase 5 milestone is an operable local desktop workflow.

## Queue

- One Queue row represents one Job.
- Queue projections are immutable.
- Queue schema version: 2.
- Pause state is projected from the exact durable `pause-batch` control marker.
- Updates use complete snapshots with session-monotonic revisions.
- The GUI receives refresh/change notifications through an Application boundary.
- Application state is reconstructed before being presented.
- Display all active Jobs plus the 100 most recent terminal Jobs.
- Preserve stable submission order; state changes do not reorder rows.
- Hiding a terminal row must not delete durable Job data.

## Progress

- Display current Stage and native Stage progress only.
- Do not invent weighted overall progress.
- Stale running state is presented as interrupted.
- GUI progress delivery may later be coalesced to approximately 5–10 updates per second.

## Inputs

- Support files, folders, recursion, and drag-and-drop.
- Duplicate input paths are allowed as separate Jobs.
- Unsupported entries do not invalidate otherwise valid selections.
- Full FFprobe remains a Pipeline Stage.

## Job actions

- Pause means stop scheduling new work after the current safe boundary.
- Pause is derived from the durable `control/pause-batch` marker, not a fake Journal Job state.
- Cancellation uses the existing cooperative and escalation path.
- Batch cancellation is supported and clears pause.
- Retry uses a simple default action from the earliest retryable Stage.
- Running a completed input again creates a new Job in a new Batch.

## GUI architecture

- Sidebar navigation.
- One Create page with collapsible configuration sections.
- MainWindow owns navigation and lifecycle only.
- Stateful pages use focused ViewModels.
- Queue uses QAbstractTableModel in PR5.2.
- Initial Queue is flat rather than hierarchical.
- Job details use a side panel or dialog.
- One dedicated Application runner bridge; no worker per Job.
- One serial Pipeline executor (`ThreadPoolExecutor(max_workers=1)`) outside the Qt worker.
- Long Pipeline work never runs on the GUI thread or inside the Qt worker event loop.
- One shared application-wide LLM runtime/semaphore reused across sequential Fast/Quality Batches.
- No detached process or tray mode in initial Phase 5.
- Active local work is cancelled before GUI shutdown; no tray/background continuation.

## Scope exclusions (retained)

- Diagnostics export
- Runtime/model installation
- OS keychain
- Dynamic language switching
- Native notifications
- Tray/background mode

## Configuration

- Persist only genuine global defaults.
- Built-in Fast and Quality profiles are immutable.
- User profiles are named copies.
- Use the strict TOML configuration path through an Application/config service.
- Invalid configuration loads safe defaults without overwriting the invalid file.
- Execution profiles use explicit Save; UI preferences may save immediately.

## Provider credentials

- `llm.toml` is the primary API-key source.
- When the selected profile has no API key in config, a future implementation may check an environment or OS credential source.
- API keys must never enter Job snapshots, logs, diagnostics, or ordinary error parameters.
- The precedence and GUI editing flow are deferred to the configuration PR.
- PR5.1 must not change credential loading behavior.

## Language

- English and Simplified Chinese are required.
- Locale is selected at startup.
- Changing the locale requires application restart.
- User data and user-defined names are not translated.
- Missing keys fail in development/tests and fall back safely in release.

## Diagnostics and recovery

- Activity history is derived from durable events rather than a second authoritative store.
- Diagnostic exports are redacted and exclude source/subtitle text by default.
- Interrupted Jobs require an explicit Resume action.
- Recovered queued Batches prompt before resuming.
- Runtime and Model pages may remain visible with disabled future controls.
- Notifications are in-application only.

## Testing and scope

- GUI behavior tests use Application fakes.
- Normal GUI tests do not execute real ASR or LLM providers.
- Queue models and controllers target high branch coverage.
- Pixel-exact screenshot tests remain limited.
- New unrelated defects are recorded unless they block correctness, safety, or the active PR.
