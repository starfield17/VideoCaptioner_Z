# Deterministic subtitle policy

Phase 3 converts a Transcript into an immutable `SubtitleTrack` without
semantic rewriting. The policy is pure and deterministic.

## Canonical input and text metrics

Words are ordered by `(start_ms, end_ms, word_id)`. Duplicate IDs, blank text,
negative timestamps, reversed ranges, unsupported controls, and unknown
references are rejected. Source timestamps are never changed. Text is NFC
normalized; CR, LF and tab become spaces, Unicode whitespace collapses to one
ASCII space, and surrounding whitespace is stripped.

Grapheme clusters use Unicode `\\X`. Reading characters exclude whitespace and
zero-width formatting components, so a ZWJ emoji sequence, skin-tone emoji, or
flag counts as one visible character. Display width uses `wcwidth`: ordinary
Latin is normally one column, CJK/full-width text and emoji are normally two,
and combining marks add no independent column.

## Speed, boundaries and protected spans

Reading speed is integer CPS-milli arithmetic:

```text
characters * 1_000_000 <= max_cps_milli * duration_ms
```

Equality is valid. Zero or negative duration is structural invalidity. Gaps at
least `hard_gap_ms` partition the input; gaps at least `preferred_gap_ms` are
preferred boundaries. Sentence punctuation (`.!?…。！？`) is stronger than
clause punctuation (`,;:、，；：`). Opening/closing punctuation and ellipses are
classified so line breaks do not orphan punctuation when an alternative exists.

Numbers, decimals, grouped numbers, dates, times, phone numbers, currency,
units, percentages, and common abbreviations receive a high protected-break cost.
Protected spans
are preferences rather than impossible constraints: if an atomic Word makes a
strict solution impossible, content is retained and validation reports the
stable readability issue.

## Dynamic programming and tie-break

Hard-silence partitions are solved with bounded dynamic programming. Candidate
spans score structural validity, overflow, protected-span breaks, CPS overflow,
duration overflow, width overflow, punctuation/silence quality, target-duration
deviation, minimum-duration shortfall, line balance, and cue count in that order.
Candidate windows are
bounded and every partition has a one-Word fallback.

Equal-cost paths prefer the lexicographically smaller tuple of boundary
indices, i.e. earlier boundaries. This tie-break is applied after the integer
cost tuple and never depends on set or dictionary iteration order.

## Timing and lines

A cue starts at the minimum assigned Word start and ends at the maximum end.
Cues are then processed in canonical order: a later start is clamped to the
previous end and an end is extended to at least one millisecond. This can make
positive, non-overlapping cues from overlapping or equal source timestamps
without mutating the Transcript.

Line breaking runs after cue selection. It searches legal grapheme boundaries
and minimizes overflow, protected/punctuation attachment, orphan, maximum
width, and absolute width difference. It emits at most two lines when feasible,
using configured limits (42 columns per line and 84 per cue by default). A
single indivisible token remains intact and receives a validator warning rather
than being dropped or split inside an emoji grapheme.

The complete immutable configuration is part of the policy signature and Track
ID. Phase 2’s three-field mapping is read as a legacy mapping with deterministic
Phase 3 defaults; new Jobs persist the full canonical mapping.
