# Subtitle formats

Exporters consume an immutable, validated `SubtitleTrack`. They are
observationally pure: serializing a Track twice produces identical bytes and
does not alter any Domain field.

## Canonical JSON

`final-subtitle.json` is UTF-8 JSON with sorted keys, compact separators, exact
integer milliseconds, NFC text, schema version 2, the policy signature, and a
final LF. Its strict decoder rejects unknown or missing fields and round-trips
the complete `SubtitleTrack` exactly.

## SRT

SRT uses one-based cue indexes and `HH:MM:SS,mmm` timestamps. Output is UTF-8
with LF line endings, exactly one blank line between cues, and a final LF. The
project parser accepts the canonical subset it emits and preserves timing and
lines exactly.

## WebVTT

WebVTT starts with the fixed `WEBVTT` header, uses `HH:MM:SS.mmm`, plain text
HTML escaping, LF line endings, deterministic blank lines, and no styles or
positioning metadata in Phase 3. Canonical parsing preserves timing and lines.

## ASS

ASS uses fixed `[Script Info]`, `[V4+ Styles]`, and `[Events]` sections with one
fixed `Default` style. Dialogue rows follow Cue order. Line breaks are `\\N`,
literal backslashes and braces are escaped, and timestamps use deterministic
half-up centisecond rounding. Canonical parsing preserves lines and stays
within 10 ms of source Cue timing. Arbitrary third-party override tags are not
an import target.

Publish commits this exact logical Export set:

```text
final-transcript.json
final-subtitle.json
final-subtitle.srt
final-subtitle.vtt
final-subtitle.ass
```

The PublicationReceipt uses that logical order and verifies the exact target
set, regular-file/symlink policy, path, size, SHA-256, and output generation.
