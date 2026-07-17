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
tokenizer = "cl100k_base"
```

Supported tokenizer IDs are `cl100k_base` and `o200k_base`. The value may also
be `auto`, which maps only explicitly recognized model IDs; unknown models fail
closed with `llm.tokenizer_unknown`. Production budgeting uses tiktoken through
`ModelTokenCounter`. Character-length counters are test doubles only and must
not be reported as token budgets.

The API key is intentionally plaintext in the OS-specific config directory.
POSIX creation attempts mode `0600`. The application never generates a real
credential, accepts one on the CLI, or persists one in JobConfig, Journal,
Manifest, artifacts, LLM Cache metadata, exceptions, logs, or CLI JSON. Resume
loads the current credential from the same provider profile while retaining the
redacted result-affecting Job snapshot. Public provider identity includes
`tokenizer`; only the API key may differ on Resume.

The configured source-language option is separate from the language detected by
ASR. On `run`, `--language auto` stores no configured language. On `resume`, an
omitted option means no override, while `--language auto` explicitly clears the
configured language and invalidates from transcription.

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
Protected facts use exact ordered semantic token sequences: adding, deleting, or
reordering numbers, signs, currency/unit markers, date components, or AM/PM is
rejected. Explicit textual markers such as `100 dollars`, `10 percent`, and
`100 kilograms` are protected with the same strict kind identity as their
symbolic forms; only declared aliases such as `kilogram`/`kg` normalize.
Response schemas never contain timestamps, durations, Cue boundaries, or Word
IDs. Application code copies those values from validated domain inputs.

Provider `json_schema.name` values are stable task-based identities such as
`captioner_translate_fast_batch_v2`. They never derive from Python class
`__qualname__`; every batch uses a strict root object with a non-empty
`responses` array. Wire requests and Cache keys share the same schema identity
function, and the old root-array format is not accepted.

## Prompts and artifacts

Prompt identities contain `prompt_id`, `prompt_version`, `content_sha256`, and
content. Changing content requires a new version; runtime loading verifies the
persisted identity. The current Prompt identities are:

- `terminology.v2` (the sparse glossary contract)
- `correct_source.v1`
- `translate_fast.v1`
- `translate_quality.v1`
- `review_anomalies.v1`
- `repair_structured.v2`

Quality writes `terminology.json` and `corrected-transcript.json`, then
`translated-track.<language>.json` and `translation-report.json`, then
`reviewed-track.<language>.json` and `review-report.json`. Raw Transcript JSON
is preserved. Final publication remains exactly Transcript JSON, Subtitle JSON,
SRT, WebVTT, and ASS.

## Chunking, validation, Cache, and retry

Chunk planning obeys item, complete-request token, context-item, and
audio-context budgets. The complete request includes the system Prompt, user
JSON envelope, language/task metadata, dynamic context, and response schema.
Every network call — original, contextual repair, and ID-mismatch shrink — is
preflighted through the same `validate_request_budget` gate before transport.
Over-budget requests fail with `llm.item_too_large` and never reach the
provider. Context IDs never enter the expected output set. ID mismatch and
multi-item truncation deterministically bisect the current Chunk and terminate
at one item. Invalid schema/text may receive at most one
`repair_structured.v2` request, which keeps the original task, system
instruction, user envelope, invalid assistant candidate, and safe validation
diagnostics. Refusals and content filters are final; truncation is split-only
and a one-item truncation fails closed.

The adapter classifies the Provider envelope before decoding content. Only a
`stop` completion with non-empty string content reaches the strict JSON parser;
missing or unknown finish reasons produce `llm.provider_response_invalid`.

Stage semantic validation (Fast dual-field protected checks, terminology
unit/term matching and aggregate conflict detection) runs before Cache put.
Invalid Cache hits are deleted and re-requested.

Validated Cache entries live under `<AppPaths.cache_dir>/llm/sha256/`. Their key
binds task, provider identity, normalized base URL identity, model, temperature,
tokenizer, languages, profile, prompt ID/version/content hash, response schema,
ordered items/context, dynamic context, repair Prompt identity, and Chunk
configuration. It excludes credentials, Authorization, timestamps, random IDs,
and local temporary paths. Entries use canonical JSON, flush, fsync, and atomic
rename; corrupt or mismatched entries are misses.

The adapter classifies 429, 502/503/504, network failure, and timeout as
retryable. Authentication, refusal, content filtering, truncation, and invalid
Provider envelopes are final. Cancellation is never retried. Backoff is
bounded, deterministic, injectable, and immediately cancellable (sleep races
the cancellation token). All Stages and Jobs share one client and one
application-wide Semaphore.

## Recovery and redaction

Each Chunk retries and commits Cache independently. The key is derived from the
final actual `LLMRequest`, including dynamic context and repair Prompt identity.
Resume reuses validated
hits before Semaphore acquisition, so successful Chunks are not requested
again after a Stage crash. Once a Stage commit is durable, normal resume skips
the Stage entirely. Cancellation removes incomplete tasks and temporary Cache
files while retaining already committed valid entries. An aggregate semantic
failure removes every key written by that aggregate attempt. Structured repair
has one owner, the Chunk executor: transport retries do not consume its budget,
and an ID mismatch shrinks the Chunk instead of repairing it. The provider adapter
races every in-flight request against cancellation and awaits cleanup before
releasing the shared Semaphore.

Authorization and credentials are redacted before logging or structured error
creation. Provider credentials suppress dataclass representation. Automated
tests use scripted adapters or a local fake server and synthetic keys only.

Bundled `cl100k_base` and `o200k_base` resources are verified by SHA-256 before
initialization; token counting never downloads an encoding file. `auto` is
resolved to one of those durable IDs through the pinned tiktoken model map
before a Job snapshot is accepted.

## Known limitations

Phase 4 v1 supports one provider profile per Job, one OpenAI-compatible endpoint,
non-streaming responses, deterministic script-based language heuristics, and
one-Word Quality correction units. Terminology is sparse: ordinary units may
return no terms, source terms must match token boundaries in their input unit,
and conflicts fail explicitly rather than choosing a target silently.
Real-provider validation is a manual or
protected-workflow smoke test, not default CI.
