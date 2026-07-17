# LLM profiles and runtime

Phase 4 supports one provider kind, `openai-compatible`, through asynchronous
`/chat/completions` requests. There is no provider fallback, routing, automatic
model selection, streaming, alignment, or model-driven Cue segmentation.

## Configuration and credentials

Provider profiles live in `<AppPaths.config_dir>/llm.toml`:

```toml
[providers.default]
kind = "openai-compatible"
base_url = "https://example.com/v1"
api_key = "replace-manually"
model = "model-name"
max_concurrency = 4
request_timeout_sec = 120
max_retries = 5
temperature = 0.1
```

The API key is intentionally plaintext in the OS-specific config directory.
POSIX creation attempts mode `0600`. The application never generates a real
credential, accepts one on the CLI, or persists one in JobConfig, Journal,
Manifest, artifacts, LLM Cache metadata, exceptions, logs, or CLI JSON. Resume
loads the current credential from the same provider profile while retaining the
redacted result-affecting Job snapshot.

## Profiles and schemas

- `deterministic`: the Phase 3 six-Stage pipeline; no LLM runtime is initialized.
- `fast`: deterministic segmentation followed by one-pass source correction and
  translation, then validation, export, and publish.
- `quality`: terminology extraction, one-Word source correction, deterministic
  segmentation, contextual translation, deterministic anomaly selection, and
  anomaly-only Review.

Structured model responses have exact fields and reject duplicate JSON keys,
unknown fields, empty/noncanonical text, missing/extra/duplicate IDs, context
IDs, obvious language mismatch when applicable, and protected numeric loss.
Response schemas never contain timestamps, durations, Cue boundaries, or Word
IDs. Application code copies those values from validated domain inputs.

## Prompts and artifacts

Prompt identities contain `prompt_id`, `prompt_version`, `content_sha256`, and
content. Changing content requires a new version; runtime loading verifies the
persisted identity. Phase 4 v1 prompts are:

- `terminology.v1`
- `correct_source.v1`
- `translate_fast.v1`
- `translate_quality.v1`
- `review_anomalies.v1`
- `repair_structured.v1`

Quality writes `terminology.json` and `corrected-transcript.json`, then
`translated-track.<language>.json` and `translation-report.json`, then
`reviewed-track.<language>.json` and `review-report.json`. Raw Transcript JSON
is preserved. Final publication remains exactly Transcript JSON, Subtitle JSON,
SRT, WebVTT, and ASS.

## Chunking, validation, Cache, and retry

Chunk planning obeys item, token, context-item, and audio-context budgets.
Context IDs never enter the expected output set. ID mismatch deterministically
bisects the current Chunk and terminates at one item. Invalid schema/text may
receive at most one structured repair request.

Validated Cache entries live under `<AppPaths.cache_dir>/llm/sha256/`. Their key
binds task, provider identity, normalized base URL identity, model, temperature,
languages, profile, prompt ID/version/content hash, response schema, ordered
items/context, and Chunk configuration. It excludes credentials, Authorization,
timestamps, random IDs, and local temporary paths. Entries use canonical JSON,
flush, fsync, and atomic rename; corrupt or mismatched entries are misses.

The adapter classifies 429, 502/503/504, network failure, and timeout as
retryable. Authentication and request rejection are final. Cancellation is
never retried. Backoff is bounded, deterministic, and injectable. All Stages
and Jobs share one client and one application-wide Semaphore.

## Recovery and redaction

Each Chunk retries and commits Cache independently. Resume reuses validated
hits before Semaphore acquisition, so successful Chunks are not requested
again after a Stage crash. Once a Stage commit is durable, normal resume skips
the Stage entirely. Cancellation removes incomplete tasks and temporary Cache
files while retaining already committed valid entries.

Authorization and credentials are redacted before logging or structured error
creation. Provider credentials suppress dataclass representation. Automated
tests use scripted adapters or a local fake server and synthetic keys only.

## Known limitations

Phase 4 v1 supports one provider profile per Job, one OpenAI-compatible endpoint,
non-streaming responses, deterministic script-based language heuristics, and
one-Word Quality correction units. Terminology conflicts fail explicitly rather
than choosing a target silently. Real-provider validation is a manual or
protected-workflow smoke test, not default CI.
