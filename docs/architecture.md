# Architecture

Phase 0 establishes the following direction:

```text
GUI / CLI
   ↓
Core application
   ↓
Core domain and ports
   ↓
Adapters / infrastructure
```

The presentation layers select and render commands. `core.application` will
coordinate future use cases. `core.domain` contains stable, localization-neutral
types and errors. `core.ports` defines small boundaries, while
`adapters/*` will own provider-specific implementations. GUI and CLI are not
allowed to import each other, and domain code cannot depend on either.

The long-term runtime direction keeps three concerns separate:

- Core App: business orchestration and policies.
- Runtime: installed/runtime capability management.
- Model: model loading and provider-specific execution.

Only the package boundaries, capability probes, and dependency-free fakes exist
in Phase 0. There is no pipeline, job state machine, media processing, ASR,
LLM, model management, queue, recovery journal, or release workflow yet.
