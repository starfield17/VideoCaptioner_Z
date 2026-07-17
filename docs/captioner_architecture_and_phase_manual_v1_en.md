# Captioner Architecture and Phased Implementation Manual

**Version:** 1.0
**Date:** 2026-07-15
**Status:** Implementation baseline
**Primary language:** English
**Target runtime language:** Python 3.13
**Primary GUI framework:** PySide6
**Primary packaging tool:** Nuitka
**Dependency and runtime manager:** uv

---

## 0. Purpose of This Document

This manual defines the complete engineering baseline for a desktop-oriented batch subtitle generation tool for video and audio files. The project takes inspiration from the product capabilities of `VideoCaptioner` and the engineering organization of `Video_compress_Encoder_gui`, but it does not reuse the former's overall architecture and does not require reuse of any specific code from it.

This manual is not intended to provide a rough "suggested directory structure." Instead, it serves as the single primary source of truth for subsequent implementation rounds, code reviews, automated acceptance testing, and phase gates. Any implementation submitted by a coding agent, human developer, or external contributor must conform to the module boundaries, state model, data invariants, and acceptance criteria defined in this document.

The project will be implemented in approximately eight phases. Full validation must be performed after each phase. Advancing to the next phase merely because "the feature runs" is not permitted.

---

## 1. Product Goals

### 1.1 Core Goals

The application accepts one or more local video/audio files and performs the following workflow:

```text
Media import
  → Media probing
  → Audio normalization
  → ASR speech recognition
  → Optional forced alignment
  → Source-text correction
  → Deterministic subtitle segmentation
  → LLM translation and subtitle optimization
  → Subtitle quality validation
  → SRT / VTT / ASS / JSON export
```

Primary capabilities include:

1. Batch scanning of files and directories.
2. Local ASR, with Faster Whisper as the first backend.
3. Future support for Qwen ASR, NVIDIA Parakeet, or other local/remote ASR backends.
4. Native LLM translation and subtitle optimization.
5. Traditional machine translation APIs are not used as an official product path.
6. Recoverable jobs, resumable execution, single-stage retry, and auditable intermediate artifacts.
7. GUI and CLI reuse the same Application layer.
8. Multilingual UI.
9. Testing, native packaging, and Release publication through GitHub Actions.
10. Mechanical quality gates for common mistakes introduced by coding agents.

### 1.2 Explicitly Out of Scope

The first major version explicitly excludes:

- Burning subtitles into video.
- Video re-encoding.
- Video/subtitle composition.
- TTS dubbing.
- Video downloading.
- Dynamic routing across multiple LLM providers.
- Failover across multiple LLM providers.
- Multi-GPU scheduling.
- Keeping multiple models resident on the GPU simultaneously.
- Distributed job scheduling.
- Server-side multi-user SaaS.
- Direct creation or modification of timestamps by an LLM.

If any of these capabilities are required in the future, they must be added as independent extensions without breaking the current domain model or dependency direction.

---

## 2. Overall Engineering Principles

### 2.1 Prefer a Modular Monolith

This project uses a modular monolith rather than microservices. Model runtime environments may use separate worker processes, but the main business system remains a local desktop application.

Reasons:

- Local file processing does not require network-service decomposition.
- Microservices would increase installation, debugging, logging, permission, and version-compatibility complexity.
- Coding agents are more likely to introduce errors in cross-service protocols, deployment, and synchronization.
- Single-machine GPU resources are not well suited to complex service-oriented scheduling.

### 2.2 Thin Entry Point

`main.py` is only responsible for selecting the GUI or CLI entry point. It must not contain:

- Configuration-loading details.
- Pipeline assembly.
- Model loading.
- Job execution.
- FFmpeg calls.
- LLM calls.

### 2.3 The GUI Must Not Contain Business Logic

The GUI is responsible only for:

- Collecting user input.
- Displaying state.
- Sending Application Commands.
- Subscribing to progress and result events.

The GUI must not:

- Import Faster Whisper, Torch, Transformers, or the OpenAI SDK.
- Execute FFmpeg directly.
- Read or write Job Manifests directly.
- Decide on its own whether a Pipeline Stage should be skipped.
- Implement a business state machine inside a `QThread`.

### 2.4 CLI Commands Must Not Call Other CLI Commands

A CLI command must not construct another command's `argparse.Namespace` and call its `run()` function.

Correct structure:

```text
CLI command
  → Application Service
  → Pipeline / Domain
```

The GUI and CLI call the same Application API.

### 2.5 Immutable Intermediate Artifacts

Each Stage reads input Artifacts and produces new Artifacts. In-place overwriting of structured results from a previous stage is prohibited.

Example:

```text
raw_transcript.json
aligned_transcript.json
corrected_transcript.json
source_track.json
translated_track.zh-CN.json
```

This enables:

- Any stage to be rerun independently.
- Comparison of outputs from different models or prompts.
- More reliable interruption recovery.
- Preservation of the last valid result when a failure occurs.

### 2.6 Separate Factual Data from Presentation Data

The ASR Transcript, time alignment, and final subtitle Cues are different objects.

A single mutable structure must not be used to store all of the following:

- Raw ASR text.
- Corrected source text.
- Translated text.
- Final display lines.
- Timeline data.

### 2.7 Concurrency Must Be Explicitly Bounded

Initial concurrency policy:

```text
Job concurrency = 1
ASR concurrency = 1
LLM request concurrency = configurable N, default 4
```

Model-internal batching is not equivalent to concurrent Jobs.

### 2.8 Errors Must Be Structured

The Core layer raises:

```python
AppError(
    code="asr.model_load_failed",
    params={"model": model_name},
    retryable=False,
)
```

The Core layer must not directly generate localized error sentences.

### 2.9 Every External Boundary Must Be Replaceable and Mockable

The following capabilities must be isolated through Ports/Adapters:

- ASR.
- LLM.
- FFmpeg/FFprobe.
- File storage.
- Runtime Worker.
- Time source.
- UUID/ID generator.

This allows unit tests to run without real models or real APIs.

---

## 3. Target Directory Structure

```text
main.py
pyproject.toml
uv.lock
README.md
AGENTS.md

src/captioner/
├── core/
│   ├── domain/
│   │   ├── media.py
│   │   ├── transcript.py
│   │   ├── subtitle.py
│   │   ├── job.py
│   │   ├── events.py
│   │   ├── result.py
│   │   └── errors.py
│   │
│   ├── application/
│   │   ├── batch_service.py
│   │   ├── job_service.py
│   │   ├── pipeline.py
│   │   ├── pipeline_builder.py
│   │   ├── stage_executor.py
│   │   ├── recovery_service.py
│   │   └── runtime_service.py
│   │
│   ├── ports/
│   │   ├── asr.py
│   │   ├── aligner.py
│   │   ├── llm.py
│   │   ├── media.py
│   │   ├── artifact_store.py
│   │   ├── job_store.py
│   │   ├── journal.py
│   │   └── runtime.py
│   │
│   └── policies/
│       ├── segmentation.py
│       ├── reading_speed.py
│       ├── line_breaking.py
│       ├── llm_chunking.py
│       └── subtitle_validation.py
│
├── adapters/
│   ├── asr/
│   │   ├── faster_whisper.py
│   │   ├── qwen_asr.py
│   │   ├── parakeet.py
│   │   └── fake.py
│   ├── aligners/
│   │   ├── qwen_forced_aligner.py
│   │   └── fake.py
│   ├── llm/
│   │   ├── openai_compatible.py
│   │   ├── fake.py
│   │   └── scripted.py
│   ├── media/
│   │   ├── ffmpeg.py
│   │   ├── ffprobe.py
│   │   └── fake.py
│   ├── persistence/
│   │   ├── filesystem_artifact_store.py
│   │   ├── filesystem_job_store.py
│   │   ├── jsonl_journal.py
│   │   └── atomic_write.py
│   └── exporters/
│       ├── srt.py
│       ├── vtt.py
│       ├── ass.py
│       └── json_exporter.py
│
├── runtime/
│   ├── manager.py
│   ├── protocol.py
│   ├── manifest.py
│   ├── worker_client.py
│   ├── process_controller.py
│   └── model_manager.py
│
├── cli/
│   ├── cli_entry.py
│   ├── output.py
│   └── commands/
│       ├── run.py
│       ├── status.py
│       ├── resume.py
│       ├── retry.py
│       ├── doctor.py
│       ├── runtime.py
│       └── model.py
│
├── gui/
│   ├── gui_entry.py
│   ├── main_window.py
│   ├── controllers/
│   ├── pages/
│   ├── dialogs/
│   ├── models/
│   ├── workers/
│   └── widgets/
│
├── i18n/
│   ├── service.py
│   ├── catalog.py
│   ├── locale.py
│   └── validation.py
│
└── infrastructure/
    ├── app_paths.py
    ├── config.py
    ├── logging.py
    ├── ids.py
    └── clock.py

resources/
├── i18n/
│   ├── en.json
│   └── zh-CN.json
├── prompts/
│   ├── correct_source.md
│   ├── translate.md
│   └── review_anomalies.md
└── runtime-manifests/

runtime-projects/
├── faster-whisper-cpu/
├── faster-whisper-cuda12/
├── qwen-asr/
└── parakeet/

scripts/
├── check.py
├── check_i18n.py
├── check_architecture.py
├── check_forbidden_patterns.py
├── build_nuitka.py
├── build_runtime.py
├── package_release.py
└── release_smoke.py

tests/
├── unit/
├── property/
├── contract/
├── integration/
├── recovery/
├── packaging/
├── golden/
└── fixtures/
```

---

## 4. Dependency Direction and Import Contracts

### 4.1 Allowed Dependency Direction

```text
GUI / CLI
    ↓
Application
    ↓
Domain / Policies / Ports

Adapters → Ports / Domain
Runtime → Runtime Port / Domain Types
Infrastructure → General foundational capabilities
```

### 4.2 Prohibited Dependencies

1. Domain must not depend on GUI, CLI, Adapters, PySide6, OpenAI, Torch, Transformers, or Faster Whisper.
2. Policies must not depend on the GUI or external SDKs.
3. Application must not depend on PySide6.
4. Adapters must not depend on the GUI.
5. The GUI must not depend on a concrete ASR SDK.
6. CLI commands must not call one another.
7. Exporters must not modify Domain Objects.
8. Runtime Workers must not import the GUI.

### 4.3 Automated Validation

Use Import Linter or a custom AST checker. CI must fail when an Import Contract fails.

---

## 5. Domain Model

### 5.1 Media

```python
@dataclass(frozen=True, slots=True)
class MediaAsset:
    id: str
    source_path: Path
    content_hash: str
    duration_ms: int
    audio_stream_index: int
    container: str
    metadata: Mapping[str, JsonValue]
```

### 5.2 Audio Artifact

```python
@dataclass(frozen=True, slots=True)
class AudioArtifact:
    artifact_id: str
    path: Path
    sha256: str
    sample_rate: int
    channels: int
    duration_ms: int
    codec: str
```

The default standardized ASR audio format is:

```text
16 kHz
mono
PCM WAV or lossless FLAC
```

MP3 must not be used as the internal standardized intermediate format.

### 5.3 Word Token

```python
@dataclass(frozen=True, slots=True)
class WordToken:
    id: str
    text: str
    start_ms: int
    end_ms: int
    confidence: float | None = None
    speaker_id: str | None = None
```

Invariants:

- `start_ms >= 0`
- `end_ms > start_ms`
- Token IDs are unique within a Transcript.
- Tokens are ordered by time.
- The `text.strip()` of a non-empty speech token is not empty.

### 5.4 Transcript

```python
@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    id: str
    word_ids: tuple[str, ...]
    raw_text: str
    start_ms: int
    end_ms: int
    confidence: float | None

@dataclass(frozen=True, slots=True)
class Transcript:
    id: str
    language: str
    words: tuple[WordToken, ...]
    segments: tuple[TranscriptSegment, ...]
    engine_id: str
    model_id: str
    metadata: Mapping[str, JsonValue]
```

### 5.5 Corrected Transcript

Source-text correction must not directly overwrite `Transcript`.

```python
@dataclass(frozen=True, slots=True)
class CorrectedSpan:
    source_word_ids: tuple[str, ...]
    corrected_text: str

@dataclass(frozen=True, slots=True)
class CorrectedTranscript:
    source_transcript_id: str
    spans: tuple[CorrectedSpan, ...]
    revision: int
```

### 5.6 Subtitle Cue

```python
@dataclass(frozen=True, slots=True)
class SubtitleCue:
    id: str
    start_ms: int
    end_ms: int
    source_word_ids: tuple[str, ...]
    source_text: str
    translated_text: str | None
    lines: tuple[str, ...]
```

Invariants:

- Cue times are ordered.
- `start_ms < end_ms`.
- Cues do not overlap by default.
- The same source word must not be assigned more than once.
- Source words must not be dropped except for explicitly filtered items.
- LLM output must not change timestamps.

### 5.7 Subtitle Track

```python
@dataclass(frozen=True, slots=True)
class SubtitleTrack:
    id: str
    source_transcript_id: str
    language: str
    cues: tuple[SubtitleCue, ...]
    revision: int
```

### 5.8 Job and Batch

```python
class JobState(StrEnum):
    DISCOVERED = "discovered"
    PREPARING = "preparing"
    TRANSCRIBING = "transcribing"
    ALIGNING = "aligning"
    CORRECTING = "correcting"
    SEGMENTING = "segmenting"
    TRANSLATING = "translating"
    VALIDATING = "validating"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_FINAL = "failed_final"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
```

State changes must go through a Domain Transition Function and must not be assigned directly in the GUI.

---

## 6. Pipeline Design

### 6.1 Stage Interface

```python
class PipelineStage(Protocol):
    stage_id: str
    stage_version: str

    def plan(
        self,
        context: StagePlanContext,
    ) -> StagePlan:
        ...

    async def execute(
        self,
        plan: StagePlan,
        context: StageExecutionContext,
    ) -> StageResult:
        ...
```

### 6.2 Stage Result

```python
@dataclass(frozen=True, slots=True)
class StageResult:
    stage_id: str
    status: StageStatus
    output_artifacts: tuple[ArtifactRef, ...]
    metrics: Mapping[str, JsonValue]
    warnings: tuple[WarningRecord, ...]
```

### 6.3 Base Stage List

1. `discover_media`
2. `probe_media`
3. `normalize_audio`
4. `transcribe`
5. `align`, optional depending on capabilities
6. `correct_source`, optional
7. `segment_subtitles`
8. `translate`
9. `validate_subtitles`
10. `export_subtitles`

### 6.4 Stage Cache Key

```text
SHA256(
    input artifact hashes
    + normalized stage config
    + stage version
    + model identifier
    + model revision
    + prompt version
)
```

Every field that can affect the result must be included in the cache key.

The following must not be omitted:

- ASR model revision.
- VAD parameters.
- Language.
- Compute type.
- Prompt content or Prompt version.
- Target language.
- Subtitle segmentation rules.
- Export-format parameters.

### 6.5 Skip Policy

A Stage may reuse existing output only when all of the following are true:

1. The input Artifact hash matches.
2. The Stage config hash matches.
3. The Stage version matches.
4. The output Artifact exists.
5. The output Artifact hash passes validation.
6. The Manifest state is `committed`.

A Stage must not be skipped merely because a file exists.

---
## 7. Deterministic Subtitle Segmentation

### 7.1 Basic Principle

An LLM may provide punctuation, terminology, and semantic hints, but it must not freely determine the timeline.

The primary segmentation logic must be a testable deterministic algorithm.

### 7.2 Inputs

- Word Tokens.
- Punctuation-correction results.
- Silence intervals.
- Optional speaker boundaries.
- Subtitle configuration.

### 7.3 Constraints

Recommended defaults:

```text
min_duration_ms = 800
max_duration_ms = 7000
max_lines = 2
max_cjk_chars_per_line = 18
max_latin_chars_per_line = 42
max_cps = 18
preferred_gap_ms = 250
hard_gap_ms = 700
```

### 7.4 Cost Function

```text
cost =
    duration_penalty
  + reading_speed_penalty
  + line_balance_penalty
  + semantic_break_penalty
  + silence_distance_penalty
  + orphan_word_penalty
  + punctuation_penalty
```

Use dynamic programming or an equivalent global algorithm to select breakpoints. Building the solution from a large stack of local `if` statements is not recommended.

### 7.5 Required Invariant Tests

- Time is monotonic.
- No negative timestamps.
- No zero-duration Cues.
- No Word is assigned more than once.
- No Word is dropped without an explicit reason.
- Identical input produces exactly identical output.
- Output does not depend on the system locale.
- Extremely long sentences still terminate.
- Mixed Chinese and English does not crash.
- Emoji, numbers, currency amounts, and units do not cause formatting errors.

---

## 8. ASR Architecture

### 8.1 ASR Port

```python
@dataclass(frozen=True, slots=True)
class ASRCapabilities:
    word_timestamps: bool
    segment_timestamps: bool
    language_detection: bool
    native_long_audio: bool
    internal_batching: bool
    supported_languages: frozenset[str] | None
    supported_devices: frozenset[str]

class ASREngine(Protocol):
    @property
    def engine_id(self) -> str: ...

    @property
    def capabilities(self) -> ASRCapabilities: ...

    async def transcribe(
        self,
        request: TranscriptionRequest,
        context: ExecutionContext,
    ) -> Transcript:
        ...
```

### 8.2 Faster Whisper

The first official backend.

Requirements:

- Use the Python API or Runtime Worker API.
- Load the model once and reuse it across calls.
- Support CPU and CUDA Runtime profiles.
- Provide word-level timestamps.
- Configure VAD parameters explicitly.
- Make internal batch size configurable.
- Reduce batch size progressively when an OOM occurs.
- Do not obtain structured results by parsing CLI output.

### 8.3 Qwen ASR

A future backend.

Requirements:

- Separate ASR from the Forced Aligner.
- Persist text results and time-alignment results independently.
- Qwen-specific configuration must not be added to Faster Whisper Config.
- Runtime dependencies must be locked independently.

### 8.4 NVIDIA Parakeet

A future backend.

Requirements:

- Implement it as an independent Adapter.
- Do not leak NeMo types into the Domain.
- Report language coverage through Capabilities.
- Fail during the Preflight stage for unsupported languages.

### 8.5 Concurrency Policy

Initial version:

```text
Active ASR model count = 1
Active ASR request count = 1
GPU device selection = one explicit device or auto
```

Not included:

- Multi-GPU fan-out.
- Multi-model parallelism.
- Dynamic VRAM preemption.
- Cross-model scheduling.

Model-internal batching is managed by the Adapter.

---

## 9. LLM Architecture

### 9.1 Provider Scope

The first version implements only an OpenAI-compatible API.

Allowed configuration:

```text
base_url
api_key
model
max_concurrency
request_timeout_sec
max_retries
temperature
```

Not implemented:

- Concurrent use of multiple Providers.
- Provider fallback.
- Provider load balancing.
- Automatic model selection.

### 9.2 LLM Client Port

```python
class LLMClient(Protocol):
    async def generate_structured(
        self,
        request: LLMRequest,
        response_schema: type[T],
    ) -> T:
        ...
```

### 9.3 Global Concurrency Control

The entire application shares one Semaphore:

```python
self._semaphore = asyncio.Semaphore(max_concurrency)
```

A new thread pool must not be created for every Job or Translator.

### 9.4 Chunk Planner

Chunks are determined jointly by the following limits:

- `max_items`
- `max_input_tokens`
- `context_before_items`
- `context_after_items`
- `max_audio_context_duration_ms`

Context items are provided only to help the model understand the content and must not appear in the output ID set.

### 9.5 Stable ID Protocol

Input:

```json
{
  "context_before": [],
  "items": [
    {
      "id": "cue-000001",
      "source": "..."
    }
  ],
  "context_after": []
}
```

Output:

```json
{
  "items": [
    {
      "id": "cue-000001",
      "corrected_source": "...",
      "translated_text": "..."
    }
  ]
}
```

Validation:

- The ID sets are exactly equal.
- No duplicate IDs.
- No new IDs.
- No missing IDs.
- Output language matches the target.
- Numbers and proper nouns are checked for consistency.
- Timestamps are not included in the LLM output Schema.

### 9.6 Retry Classification

| Error | Handling |
|---|---|
| 429 | Retry with exponential backoff |
| 502/503/504 | Limited retry |
| Network interruption | Limited retry |
| Timeout | Limited retry |
| 401/403 | Fail immediately |
| 400 | Fail immediately |
| Schema error | One repair request |
| ID mismatch | Reduce Chunk size and retry |
| User cancellation | Do not retry |

A blanket retry after `except Exception` is prohibited.

### 9.7 Fast and Quality Profiles

#### Fast

```text
ASR
→ deterministic segmentation
→ one-pass correction + translation
→ validation
```

#### Quality

```text
ASR
→ terminology extraction
→ source correction
→ deterministic segmentation
→ contextual translation
→ anomaly-only review
→ validation
```

By default, do not perform multi-round reflection for every Chunk.

---

## 10. Jobs, Workspaces, and Recovery

### 10.1 Workspace

```text
<data_root>/jobs/
└── <batch-id>/
    ├── batch.json
    ├── events.jsonl
    └── jobs/
        └── <job-id>/
            ├── manifest.json
            ├── events.jsonl
            ├── input/
            ├── audio/
            ├── transcript/
            ├── subtitles/
            ├── llm/
            ├── output/
            └── logs/
```

### 10.2 Manifest

The Manifest contains:

- Schema version.
- Job ID.
- Batch ID.
- Source path.
- Source hash.
- Current state.
- Active stage.
- Committed Artifacts.
- Stage attempts.
- Configuration snapshot.
- Model/runtime identity.
- Timestamps.
- Last error code.

### 10.3 Journal

`events.jsonl` is append-only.

Events should include:

```text
job_created
stage_planned
stage_started
artifact_written
stage_committed
stage_failed
job_cancel_requested
job_cancelled
job_interrupted
job_completed
```

Each event:

```json
{
  "sequence": 15,
  "event_id": "...",
  "job_id": "...",
  "type": "stage_committed",
  "timestamp": "...",
  "payload": {}
}
```

### 10.4 Atomic Commit Order

A successful Stage commit follows this order:

```text
1. Write a temporary Artifact
2. fsync
3. Atomic rename
4. Write artifact metadata
5. Append journal stage_committed
6. Atomically update the manifest projection
```

The Manifest must not be marked successful before the Artifact is written.

### 10.5 Recovery Rules

At startup:

1. Scan non-terminal Jobs.
2. Validate Manifests.
3. Replay Journals.
4. Verify committed Artifacts.
5. Mark any running Stage without a commit as interrupted.
6. Regenerate projections.
7. Continue from the last valid committed Stage.

### 10.6 Cancellation

Cancellation has three forms:

- Cooperative cancellation.
- Worker process termination.
- External subprocess-tree termination.

Cancellation must:

- Produce an explicit `cancel_requested` event.
- Not be mislabeled as a normal failure.
- Preserve committed Artifacts.
- Permit resume or retry on the next run.

---

## 11. Runtime and Model Assets

### 11.1 Three-Layer Structure

#### Core App

Packaged with Nuitka:

- GUI.
- CLI.
- Core/Application.
- LLM Client.
- Runtime Manager.
- FFmpeg/FFprobe, optionally bundled.
- Worker wheel.
- uv, optionally bundled or downloaded per platform.

#### ASR Runtime

```text
<data_root>/runtimes/
├── faster-whisper-cpu/
├── faster-whisper-cuda12/
├── qwen-asr/
└── parakeet/
```

#### Model Assets

```text
<data_root>/models/
├── faster-whisper/
├── qwen/
└── nvidia/
```

### 11.2 Runtime Manifest

```json
{
  "runtime_id": "faster-whisper-cpu",
  "runtime_version": "1.0.0",
  "protocol_version": 1,
  "python": "3.13",
  "platform": "windows-x86_64",
  "dependencies_lock_hash": "...",
  "worker_version": "1.0.0"
}
```

### 11.3 Installation Flow

```text
1. Read the Runtime Manifest
2. Check platform and architecture
3. Create .staging
4. Use uv to install the pinned Python and locked dependencies
5. Install the worker wheel
6. Run doctor
7. doctor succeeds
8. Atomically switch the current runtime
9. Preserve the previous version for rollback
```

### 11.4 Worker Protocol

The main application and Worker communicate through JSONL over stdin/stdout:

```json
{"type":"hello","protocol_version":1}
{"type":"load_model","request_id":"r1","model":"large-v3"}
{"type":"transcribe","request_id":"r2","audio_path":"..."}
{"type":"progress","request_id":"r2","percent":42}
{"type":"result","request_id":"r2","artifact_path":"...","sha256":"..."}
```

Requirements:

- stdout is used only for the protocol.
- Normal logs are written to stderr or log files.
- Each message is one complete line of JSON.
- Large results are written to Artifact files.
- Every request has a request ID.
- Protocol incompatibility must fail during the `hello` stage.

### 11.5 Model Manager

Responsibilities:

- Pin repository/revision.
- Download with resume support.
- Verify SHA256 or repository files.
- Use staging plus atomic activation.
- Check disk space.
- Support offline import.
- Delete models.
- Display model license notices.
- Display model storage usage.

Model weights are not included in the Core App Release.

---

## 12. Multilingual System

### 12.1 Locale

Use BCP 47-style locale identifiers:

```text
en
zh-CN
zh-TW
ja
```

### 12.2 Catalog

```json
{
  "_meta": {
    "locale": "zh-CN",
    "name": "简体中文",
    "fallback": "en",
    "schema_version": 1
  },
  "messages": {
    "app.title": "字幕批处理工具",
    "job.status.transcribing": "正在识别：{filename}"
  }
}
```

### 12.3 Fallback

```text
Built-in English
→ selected built-in locale
→ optional user override
```

Fallback is performed per key. Built-in language packs are not copied into writable configuration directories.

### 12.4 Core Error Localization

The Core persists only:

```text
error code
error params
technical details
```

The GUI/CLI translates them according to the current language.

### 12.5 i18n Validation

CI must check:

- No duplicate keys.
- English is 100% complete.
- Other languages contain no unknown keys.
- Placeholder sets match exactly.
- `_meta.locale` matches the filename.
- No empty strings.
- GUI-visible text is not hard-coded.
- Missing keys fail immediately in development mode.

---

## 13. GUI Architecture

### 13.1 MainWindow

Responsible only for:

- Page/Widget assembly.
- Menus and Toolbar.
- Top-level navigation.
- Window lifecycle.
- Controller binding.

Target: `main_window.py` must not exceed 500 lines.

### 13.2 Pages

Recommended pages:

- Input Page.
- ASR Settings Page.
- LLM Settings Page.
- Subtitle Settings Page.
- Output Settings Page.
- Queue Page.
- Model Manager Page.
- Diagnostics Page.

Each Page:

- Has an independent ViewModel.
- Has `retranslate()`.
- Does not access SDKs directly.
- Does not write TOML/JSON directly.

### 13.3 Queue

The GUI Queue is an Application Job Projection, not the sole source of truth.

The authoritative state is stored in:

```text
Job Manifest + Journal
```

After a GUI restart, the Queue is rebuilt through JobService.

### 13.4 GUI Worker

A Qt Worker does only:

```python
asyncio.run(batch_service.run(...))
```

It forwards:

- progress.
- state_changed.
- job_finished.
- job_failed.

A business Pipeline must not be implemented inside a Qt Worker.

---
## 14. CLI Design

```bash
captioner run ./videos --recursive --profile zh-quality
captioner status <batch-id>
captioner resume <batch-id>
captioner retry <job-id> --stage translate
captioner doctor
captioner runtime list
captioner runtime install faster-whisper-cpu
captioner model list
captioner model install faster-whisper-large-v3
captioner i18n validate
```

### 14.1 Exit Codes

Recommended:

```text
0  success
2  usage/configuration error
3  input file error
4  runtime unavailable
5  model unavailable
6  ASR failure
7  LLM failure
8  validation failure
9  partial batch failure
10 cancelled
11 internal consistency failure
```

The CLI and GUI use the same result model and do not independently interpret low-level exceptions.

---

## 15. Configuration System

Use TOML, validated by Pydantic or a strict dataclass parser.

Priority:

```text
CLI override
→ environment secret
→ user profile
→ app config
→ built-in default
```

### 15.1 Secrets

API Keys must not be uploaded to GitHub.

### 15.2 Tagged ASR Config

```toml
[asr]
kind = "faster-whisper"

[asr.faster_whisper]
model = "large-v3"
device = "auto"
compute_type = "auto"
batch_size = 4
```

Qwen parameters must not be mixed into Faster Whisper configuration.

---

## 16. Packaging and Release

### 16.1 Core App

Continue using Nuitka:

- Windows standalone.
- Linux standalone.
- macOS app-dist + DMG.

Local builds and CI use the same `scripts/build_nuitka.py`.

### 16.2 Separate the Core App from the Runtime

The Core App Release does not include:

- Model weights.
- Complete Torch/Qwen/NeMo Runtimes.
- Multiple sets of CUDA dependencies.

Runtimes are installed on demand by the application, or provided as separate offline packages.

### 16.3 Release Matrix

The first stable version prioritizes validation for:

- Windows x86_64.
- Linux x86_64.
- macOS arm64.
- macOS x86_64.

Windows ARM64 and Linux ARM64 Core Apps may continue to be built, but full local ASR Runtime support must not be claimed without validated backend support.

### 16.4 Release Checks

- Version format.
- Unique output directory.
- Binary architecture.
- CLI help.
- GUI offscreen startup.
- Built-in resources.
- i18n catalog.
- FFmpeg/FFprobe.
- Runtime manifest parsing.
- Checksums.
- SBOM.
- Nuitka compilation report.

---

## 17. Code Quality Gates

### 17.1 Toolchain

The following are mandatory:

- Ruff format.
- Ruff lint.
- Pyright strict.
- Pytest.
- Branch coverage.
- Hypothesis.
- Import Linter.
- Custom forbidden-pattern checker.

### 17.2 Unified Command

```bash
uv run python scripts/check.py --full
```

It must execute:

```text
uv lock --check
ruff format --check .
ruff check .
pyright
lint-imports
python scripts/check_i18n.py
python scripts/check_forbidden_patterns.py
pytest tests/unit tests/property tests/contract tests/recovery
coverage report
```

CI must not duplicate a separate implementation of the logic; it must call the same script.

### 17.3 Pyright

```toml
[tool.pyright]
typeCheckingMode = "strict"
```

Rules:

- First-party code uses strict mode.
- Missing third-party stubs are isolated through Adapter Shims.
- Broad `type: ignore` usage is prohibited.
- Every ignore must specify a concrete rule.
- CI tracks the number of newly added ignores.

### 17.4 Ruff and Complexity

Recommended rules:

- Function complexity limit: 10–12.
- Functions should not normally exceed 80 lines.
- Ordinary modules should not normally exceed 600 lines.
- MainWindow should not normally exceed 500 lines.

Code exceeding a limit must be split. Permanently exempting it instead is prohibited.

### 17.5 Forbidden Patterns

CI blocks:

```text
except Exception: pass
except: pass
bare print statements in core
GUI imports of faster_whisper/torch/openai
a CLI command importing another command's run function
PySide6 usage in domain
external requests without timeouts
subprocesses without a cancellation path
automatic overwriting of golden files
real LLM APIs in unit tests
```

### 17.6 Coverage

Recommended Branch Coverage:

| Module | Minimum |
|---|---:|
| domain | 95% |
| policies | 95% |
| application | 90% |
| persistence | 90% |
| adapters | 80% |
| GUI | 65% |
| overall | 85% |

### 17.7 Property Tests

The following must be verified:

- Subtitle timeline invariants.
- ID completeness.
- Stage cache-key stability.
- Job state transitions.
- Journal replay idempotency.
- SRT/VTT round-trip.
- LLM response validation.

### 17.8 Golden Tests

Golden data includes:

- Raw Transcript.
- Corrected Transcript.
- Subtitle Track.
- SRT/VTT/ASS.
- Scripted LLM responses.

Coding agents are prohibited from automatically updating all Golden files in bulk.

### 17.9 Fault Injection

Coverage must include:

- Worker startup failure.
- A partial JSON line from the Worker.
- Unexpected Worker exit.
- GPU OOM.
- Non-zero FFmpeg exit.
- Disk-write failure.
- Interrupted Manifest write.
- LLM succeeds after a 429.
- Permanent LLM 401.
- LLM omits a Cue.
- LLM duplicates a Cue.
- User cancellation at every Stage.

### 17.10 Mutation Testing

Run nightly against:

- segmentation.
- timestamp normalization.
- job transitions.
- cache keys.
- LLM validation.
- recovery reconciliation.

---

## 18. Coding Agent Constraints

The repository must contain an `AGENTS.md` that includes at least the following:

### 18.1 Prohibited Actions

- CLI must not call CLI.
- GUI must not directly import SDKs.
- Domain must not import Adapters.
- A Stage must not modify its input in place.
- An LLM must not modify timestamps.
- Exceptions must not be swallowed silently.
- `ignore`/`noqa` without an explanation is prohibited.
- Unit tests must not call real APIs.
- Platform/runtime support must not be claimed without validation.

### 18.2 Patch Report Format

Every Patch must include:

1. Changed behavior.
2. Preserved invariants.
3. New tests.
4. Failure-path tests.
5. Commands executed.
6. Static-check results.
7. Known limitations.
8. Files intentionally not modified.

### 18.3 Patch Principles

- A Patch handles only one clearly defined objective.
- Large-scale refactoring must not be performed in parallel with feature additions.
- Changes to the Domain Model must include corresponding changes to contract/property tests.
- Changes to an Artifact Schema must include a schema migration or explicitly versioned failure behavior.
- Changes to a Prompt must increment the prompt version.
- Changes to a cache key must include tests.

---

# 19. Phase Implementation Plan

There are eight phases in total. Each phase must pass complete acceptance before work proceeds to the next phase.

---

## Phase 0: Engineering Skeleton and Quality Gates

### Objective

Establish a repository structure, dependency management, CI, and architectural constraints that can support long-term growth. This phase does not pursue real ASR.

### Implementation Scope

1. Create the `src/captioner` package structure.
2. Create a thin `main.py`.
3. Create empty CLI/GUI entry points.
4. Create `pyproject.toml` and `uv.lock`.
5. Configure Ruff, Pyright strict, Pytest, Coverage, Hypothesis, and Import Linter.
6. Create `scripts/check.py`.
7. Create the i18n catalog loader and validator.
8. Create runtime directories using `platformdirs`.
9. Create structured `AppError`.
10. Create fake Ports/Adapters.
11. Create a GitHub Actions Fast Gate.
12. Create `AGENTS.md` and architecture documentation.

### Required Deliverables

- CLI `--help`.
- GUI can start and close in offscreen mode.
- English/Chinese catalogs.
- Architecture Import Contract.
- `scripts/check.py --full`.
- Initial Nuitka packaging wrapper.

### Required Tests

- Locale fallback.
- Placeholder mismatch.
- App paths are correct in source and compiled-simulation modes.
- Domain cannot import GUI/SDK code.
- `main.py` entry-point selection.
- Missing catalogs fail in development mode.
- Importing CLI and GUI does not trigger model SDK imports.

### Full Validation

```bash
uv sync --frozen
uv run python scripts/check.py --full
uv run pytest tests/packaging -q
uv run python scripts/build_nuitka.py --clean --version 0.0.0
<packaged executable> --cli --help
```

### Exit Criteria

- All quality checks pass.
- No Pyright errors.
- No architecture dependency violations.
- Nuitka completes at least one smoke build on the primary development platform.
- CI runs the Fast Gate on at least Windows, Linux, and macOS.

---

## Phase 1: Minimal End-to-End ASR Slice

### Objective

Complete the first genuinely usable end-to-end flow:

```text
Video/audio
→ FFprobe
→ FFmpeg audio normalization
→ Faster Whisper
→ Transcript JSON
→ Simple deterministic segmentation
→ SRT
```

### Implementation Scope

1. Media Domain Model.
2. FFprobe Adapter.
3. FFmpeg Audio Normalizer.
4. Faster Whisper Adapter, initially allowed to run in-process in the development environment.
5. Transcript Domain Model.
6. SRT Exporter.
7. Single-file CLI `captioner run`.
8. Fake ASR contract tests.
9. A small real-audio integration test, optionally marked `slow`.

### Key Constraints

- ASR concurrency is 1.
- The model is loaded once.
- No LLM.
- No complex Job Recovery.
- No multiple ASR backends.
- No GUI Queue.

### Required Tests

- Audio-normalization parameters.
- Paths containing spaces and Unicode.
- Missing audio stream.
- Invalid FFprobe JSON.
- Non-zero FFmpeg exit.
- Empty ASR result.
- Invalid Word timestamps.
- SRT time formatting.
- SRT Cue ordering.
- User cancellation during FFmpeg/ASR.

### Golden Fixtures

At minimum:

- 10 seconds of English speech.
- 10 seconds of Chinese speech.
- Speech containing silence.
- A video with no audio.

### Full Validation

In addition to all Phase 0 validation:

```bash
uv run pytest tests/integration/test_ffmpeg_pipeline.py -q
uv run pytest tests/integration/test_faster_whisper_smoke.py -q -m slow
uv run captioner run tests/fixtures/media/short.wav --output build/smoke
uv run python scripts/validate_subtitle.py build/smoke/*.srt
```

### Exit Criteria

- Repeated runs with the same input and configuration produce a stable structure.
- The model is not reloaded for every chunk.
- Failures do not leave behind a falsely successful SRT.
- ASR output retains structured Word timestamps.
- The CLI returns the correct exit code.

---

## Phase 2: Persistent Jobs, Artifacts, and Recovery

### Objective

Upgrade the one-shot Phase 1 flow into a real Job Pipeline.

### Implementation Scope

1. Batch/Job Domain Model.
2. Job State Transitions.
3. Manifest Store.
4. JSONL Journal.
5. Atomic Artifact Store.
6. Stage Executor.
7. Stage commit protocol.
8. Resume/Retry/Status CLI.
9. Cancellation and interrupted recovery.
10. Artifact hashes and cache keys.

### Required Tests

- Atomic Manifest writes.
- Repair or rejection policy for a partial trailing JSON line in the Journal.
- Crash before a Stage finishes writing an Artifact.
- Crash after the Artifact is written but before commit.
- Commit event exists but the Artifact is missing.
- Manifest and Journal disagree.
- Replaying twice produces the same result.
- Correct state after cancellation.
- Resume continues from the last valid Stage.
- Cache invalidation after configuration changes.

### Fault Injection

Inject at every Stage:

```text
before_execute
mid_execute
after_artifact_write
before_journal_commit
after_journal_commit
before_manifest_projection
```

### Full Validation

```bash
uv run python scripts/check.py --full
uv run pytest tests/recovery -q
uv run pytest tests/property/test_journal_replay.py -q
uv run pytest tests/property/test_job_transitions.py -q
uv run captioner run tests/fixtures/media/short.wav
# Interrupt manually or through a script
uv run captioner resume <batch-id>
uv run captioner status <batch-id> --json
```

### Exit Criteria

- Every crash point is recoverable or produces an explicit failure.
- A state of "Manifest successful but Artifact incomplete" cannot occur.
- A single-stage Retry does not rerun unrelated stages.
- Cancellation is not equivalent to failure.
- Job state can be reconstructed from the Journal.

---
## Phase 3: Complete Deterministic Subtitle Processing

### Objective

Implement reliable subtitle segmentation, line breaking, and quality validation without relying on an LLM.

### Implementation Scope

1. Complete Segmentation Policy.
2. Reading Speed Policy.
3. Line Breaking Policy.
4. CJK/Latin width measurement.
5. Silence boundaries and punctuation boundaries.
6. Dynamic-programming segmentation.
7. Subtitle Track Domain Model.
8. SRT/VTT/ASS/JSON Export.
9. Subtitle Validator.
10. Golden Regression Suite.

### Required Tests

- Extremely short words.
- Extremely long sentences.
- Mixed Chinese and English.
- Consecutive numbers.
- Currency amounts, dates, and units.
- Emoji.
- Missing punctuation.
- Long silence.
- Almost no silence.
- Overlapping Word timestamps.
- Identical timestamps.
- Unordered input.
- Two-line balance.
- CPS boundaries.

### Property Invariants

- Every Word is assigned exactly once.
- Cue times are valid.
- Output ordering is stable.
- Repeated runs are identical.
- Export → Parse round-trips remain within the permitted tolerance.

### Full Validation

```bash
uv run python scripts/check.py --full
uv run pytest tests/property/test_segmentation.py -q
uv run pytest tests/golden/test_subtitle_tracks.py -q
uv run pytest tests/golden/test_exporters.py -q
uv run python scripts/run_subtitle_corpus.py tests/fixtures/transcripts
```

### Exit Criteria

- All Domain invariants are covered by property tests.
- Golden updates require explicit human confirmation.
- Readable subtitles can already be produced before the LLM is added.
- Exporters do not modify the Track.

---

## Phase 4: LLM Source Correction and Translation

### Objective

Add structured correction and translation using a single OpenAI-compatible Provider.

### Implementation Scope

1. LLM Client Port.
2. Async OpenAI-compatible Adapter.
3. Global Semaphore.
4. Token-aware Chunk Planner.
5. Prompt Versioning.
6. Structured Output Schema.
7. Response Validator.
8. Retry Classification.
9. LLM Cache.
10. Fast/Quality Profiles.
11. Scripted/Fake LLM Adapter.
12. Anomaly-only Review.

### Required Tests

- Success after a 429.
- Success after a 503.
- Permanent 401.
- Timeout.
- Missing ID.
- Duplicate ID.
- Extra ID.
- Empty translation.
- Wrong language.
- Missing numbers.
- Retry with a smaller Chunk.
- User cancellation.
- Application-wide concurrency never exceeds N.
- Two Jobs do not each create N concurrent requests.

### Boundaries That Must Be Validated

- The LLM Schema contains no timestamp.
- LLM results cannot modify Cue times.
- The API Key is not written to ordinary logs or uploaded to GitHub.
- Request logs must be redacted.
- The cache key includes the model, prompt version, and target language.

### Full Validation

```bash
uv run python scripts/check.py --full
uv run pytest tests/contract/test_llm_client.py -q
uv run pytest tests/integration/test_llm_fake_server.py -q
uv run pytest tests/property/test_llm_response_validation.py -q
uv run pytest tests/recovery/test_llm_chunk_resume.py -q
```

Real API tests may run only manually or in a protected workflow.

### Exit Criteria

- No automated test depends on a real API.
- LLM concurrency has an application-wide upper bound.
- A failed Chunk can be retried locally.
- Schema errors are not silently written into final subtitles.
- The LLM does not affect timeline invariants.

---

## Phase 5: GUI, Queue, Multilingual Support, and Configuration Experience

### Objective

Build a usable desktop GUI while keeping business logic in the Application layer.

### Implementation Scope

1. Keep MainWindow thin.
2. Input/ASR/LLM/Subtitle/Output/Queue/Models/Diagnostics Pages.
3. Queue Table Model.
4. Batch Controller.
5. Qt Worker signal bridge.
6. Preset/Profile Store.
7. Language switching.
8. 100% complete English/Chinese catalogs.
9. Activity Log.
10. Recover historical Jobs.
11. Pause-after-current, Cancel, and Retry.

### GUI Constraints

- The GUI does not import SDKs.
- The GUI does not read the Journal.
- The GUI does not write the Manifest.
- Queue state can be reconstructed from the Application.
- MainWindow stays within the agreed size limit.
- All visible text comes from i18n.

### Required Tests

- GUI offscreen startup.
- Language switching.
- Missing translations fail in development mode.
- Add file/folder.
- Queue projection.
- Cancel signal.
- Queue recovery after restart.
- Invalid configuration displays a structured error.
- Behavior when the window closes while a task is running.
- Long-running tasks do not block the UI thread.

### Full Validation

```bash
uv run python scripts/check.py --full
QT_QPA_PLATFORM=offscreen uv run pytest tests/gui -q
uv run python scripts/gui_smoke.py --lang en
uv run python scripts/gui_smoke.py --lang zh-CN
uv run python scripts/check_i18n.py --strict
```

### Exit Criteria

- GUI operations do not block the main thread for extended periods.
- All state comes from the Application Projection.
- Switching between English and Chinese is complete.
- GUI close and cancellation paths pass testing.
- The CLI and GUI produce consistent results for the same Job.

---
## Phase 6: Independent Runtimes, Model Management, and Release Pipeline

### Objective

Separate ASR dependencies from the Core App and complete a release-ready Runtime/Model Manager and GitHub Release process.

### Implementation Scope

1. Worker JSONL Protocol.
2. Worker Client.
3. Runtime Manager.
4. uv-locked runtime projects.
5. Runtime install/upgrade/rollback.
6. Faster Whisper CPU Runtime.
7. Faster Whisper CUDA12 Runtime.
8. Model Manager.
9. Core App Nuitka packaging.
10. GitHub Actions packaging matrix.
11. Release checksums/SBOM.
12. Packaged-executable smoke tests.

### Required Tests

- Protocol version mismatch.
- Worker stdout contamination.
- Partial JSON line.
- Worker startup failure.
- Worker exits during execution.
- Runtime staging is interrupted.
- A failed doctor check does not replace the current runtime.
- Rollback after an upgrade.
- Incorrect model checksum.
- Insufficient disk space.
- Offline model import.
- Packaged App locates a writable data root.
- A macOS app does not write into `Contents/Resources`.

### Release Validation

On each platform:

```text
Build
→ locate unique output
→ verify executable architecture
→ run --cli --help
→ run doctor --skip-model-inference
→ validate i18n/resources
→ create archive/DMG
→ checksum
→ upload artifact
```

### Full Validation

```bash
uv run python scripts/check.py --full
uv run pytest tests/runtime -q
uv run pytest tests/packaging -q
uv run python scripts/build_nuitka.py --clean --version 0.0.0
<packaged app> --cli doctor --json --skip-model-inference
uv run python scripts/release_smoke.py dist/
```

Real Runtime smoke test:

```bash
captioner runtime install faster-whisper-cpu
captioner model install <small-test-model>
captioner doctor --runtime faster-whisper-cpu
captioner run tests/fixtures/media/short.wav
```

### Exit Criteria

- The Core App does not depend on system Python.
- The Core App does not include model weights.
- A failed Runtime installation can roll back safely.
- Worker protocol errors do not corrupt a Job.
- A GitHub tag can generate deterministically named Release Artifacts.

---

## Phase 7: Multiple ASR Backends and Release Hardening

### Objective

Add Qwen ASR and Parakeet without modifying the main Pipeline, and harden the first stable release.

### Implementation Scope

1. Qwen ASR Adapter.
2. Qwen Forced Aligner Adapter.
3. Independent Qwen Runtime.
4. Parakeet Adapter.
5. Independent Parakeet Runtime.
6. Capability-based Preflight.
7. Backend Contract Test Suite.
8. Backend Benchmark Tool.
9. Runtime support matrix.
10. Release documentation.
11. Migration policy.
12. Nightly mutation testing.

### Key Requirements

- Adding a backend must not modify the Domain Model.
- Adding a backend must not introduce large `if backend == ...` branches into the Pipeline.
- Adapter-specific configuration uses a tagged union.
- Unsupported languages must fail before execution begins.
- ASR concurrency remains 1.
- Multi-GPU scheduling is not added.

### Contract Tests

Every ASR Adapter must pass the same contract:

- It can report capabilities.
- It returns a valid Transcript.
- Timestamps are valid.
- Cancellation works.
- Empty-audio behavior is explicit.
- Missing-model errors are structured.
- Runtime version is included in metadata.

### Benchmark

Record the following using a fixed corpus:

- RTF.
- Peak memory.
- Peak VRAM.
- WER/CER, optional.
- Timestamp error.
- Model load time.

The Benchmark is not a hard gate for ordinary PRs, but a report must be generated before Release.

### Full Validation

```bash
uv run python scripts/check.py --full
uv run pytest tests/contract/test_all_asr_backends.py -q
uv run pytest tests/integration/test_qwen_runtime.py -q -m slow
uv run pytest tests/integration/test_parakeet_runtime.py -q -m slow
uv run python scripts/run_backend_benchmark.py --corpus tests/fixtures/benchmark
uv run mutmut run --paths-to-mutate src/captioner/core
```

Also run the complete Release Candidate workflow.

### Exit Criteria

- All three backends pass the same Contract.
- Backend-specific branching does not spread through the Pipeline.
- The Support Matrix matches real CI/hardware validation.
- Key modules meet the agreed mutation score.
- The Release Candidate completes packaging and smoke tests on every claimed platform.

---

# 20. Unified Acceptance Process for Every Phase

At the end of every Phase, perform the following steps in order:

## 20.1 Documentation Review

Confirm:

- The design remains consistent with this manual.
- New public APIs are documented.
- New error codes are documented.
- New Artifact schemas are versioned.
- New Prompts are versioned.

## 20.2 Static Checks

```bash
uv run python scripts/check.py --full
```

Failures must not be ignored.

## 20.3 Automated Tests

- Unit.
- Property.
- Contract.
- Recovery.
- Integration.
- Tests specific to the current Phase.

## 20.4 Failure-Path Tests

List the new failure modes introduced in the current Phase and prove that tests cover them.

## 20.5 Manual Smoke Test

Run at least one real user path.

## 20.6 Packaged Smoke Test

Beginning with Phase 0, run it at least on the primary platform. After Phase 6, run it on every claimed platform.

## 20.7 Acceptance Report

Format:

```text
Phase:
Commit/Tag:
Scope completed:
Acceptance criteria:
Static checks:
Unit/property/contract tests:
Integration tests:
Recovery/fault tests:
Packaging tests:
Known limitations:
Deferred items:
Final verdict: PASS / CONDITIONAL PASS / FAIL
```

Only `PASS` permits progression to the next Phase.

`CONDITIONAL PASS` may be used only for non-critical documentation or platform issues that do not affect the next phase, and it must include an explicit tracking item. Defects involving data correctness, recovery, cancellation, timelines, protocols, or package executability must not receive a Conditional Pass.

---

# 21. Definition of Done for the First Version

The project reaches its first stable version only when it satisfies at least the following:

1. Local Faster Whisper CPU and at least one GPU Runtime are available.
2. Single-file and batch-directory processing.
3. SRT/VTT/ASS/JSON output.
4. Deterministic subtitle segmentation.
5. Source correction/translation through one OpenAI-compatible LLM.
6. Global LLM concurrency limit.
7. Job Manifest + Journal + Resume.
8. Single-Stage Retry.
9. GUI and CLI share the Application layer.
10. Complete English/Chinese language packs.
11. Nuitka Core App Release.
12. On-demand Runtime/Model installation.
13. Release smoke tests pass on the core Windows/Linux/macOS platforms.
14. Pyright strict, Ruff, Import Contracts, Coverage, and Property/Recovery tests all pass.
15. No known P0/P1 data-correctness or recovery defects remain.

---

# 22. Summary of Architectural Red Lines

No Phase may violate the following:

1. CLI must not call CLI.
2. GUI must not execute core business logic.
3. Domain must not depend on external SDKs.
4. An LLM must not create or modify timestamps.
5. Raw ASR data and subtitle presentation must remain separate.
6. A Stage must not overwrite an input Artifact in place.
7. Job state must be reconstructable from persisted data.
8. Artifact success must go through an atomic commit.
9. Only one ASR task may run at a time.
10. The entire application shares one LLM concurrency limit.
11. Core App, Runtimes, and Model Assets remain separate.
12. Every Phase must pass complete validation before progression.
13. A coding agent must not resolve failures by weakening quality checks.
14. Every claimed platform/backend capability must have real validation evidence.

---

# 23. Recommended Initial Implementation Order

When formal implementation begins, follow this order strictly:

```text
Phase 0
→ Phase 1
→ Phase 2
→ Phase 3
→ Phase 4
→ Phase 5
→ Phase 6
→ Phase 7
```

Not recommended:

- Developing the GUI during Phase 1.
- Adding multiple ASR backends before Phase 2.
- Allowing an LLM to determine subtitle segmentation before Phase 3.
- Implementing complex Provider routing before Phase 4.
- Solving every platform Runtime-packaging problem before Phase 6.
- Implementing multi-GPU concurrency early in any Phase.

The core logic of this sequence is:

```text
First ensure the structure is correct
→ then ensure the minimum functionality is correct
→ then ensure state and recovery are correct
→ then ensure the subtitle algorithm is correct
→ then introduce the external uncertainty of an LLM
→ then build the GUI
→ then solve Release and Runtime concerns
→ finally add model backends
```

This minimizes the risk that a coding agent introduces several classes of errors simultaneously before complexity is under control.

---

## Appendix A: Recommended Default Configuration

```toml
[app]
language = "zh-CN"

[execution]
job_concurrency = 1

[asr]
kind = "faster-whisper"
concurrency = 1

[asr.faster_whisper]
model = "large-v3"
device = "auto"
compute_type = "auto"
batch_size = 4
vad_enabled = true

[llm]
enabled = true
provider = "openai-compatible"
model = ""
max_concurrency = 4
request_timeout_sec = 120
max_retries = 5

[subtitle]
min_duration_ms = 800
max_duration_ms = 7000
max_lines = 2
max_cjk_chars_per_line = 18
max_latin_chars_per_line = 42
max_cps = 18

[output]
formats = ["srt", "json"]
overwrite = false
```

## Appendix B: Recommended PR Template

```markdown
## Scope

## Changed behavior

## Preserved invariants

## Architecture impact

## New tests

## Failure-path tests

## Commands executed

## Static check results

## Packaging impact

## Known limitations

## Files intentionally not modified
```

## Appendix C: Recommended Issue Priorities

```text
P0: Data corruption, false-success state, unrecoverable state, timeline corruption, or security issue
P1: Primary workflow failure, unreliable cancellation, Runtime installation failure, or unusable Release
P2: Failure on some inputs, significant performance degradation, or important GUI malfunction
P3: General UX, wording, or non-critical optimization
```
