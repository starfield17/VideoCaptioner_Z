# Known Validation Gaps

**Status:** Non-blocking technical debt
**Last reviewed:** 2026-07-17
**Relevant implementation:** `src/captioner/core/policies/quantity_scanner.py`
**Introduced or confirmed during:** Phase 4 protected-fact validation review
**Phase impact:** Does not block Phase 5 development

---

## 1. Purpose

This document records known edge cases in protected numeric-fact validation that are intentionally deferred.

The protected-fact scanner currently provides strong deterministic handling for:

* Numbers.
* Signs.
* Currency prefixes and suffixes.
* Percentages.
* Units.
* Ratios.
* Numeric dimensions.
* Attached word suffixes.
* Attached symbol suffixes.
* Slash compounds.
* Long unsupported tails.
* Legacy dates, times, phone numbers, and abbreviations.

The scanner is intentionally implemented without quantity regular expressions and has extensive unit, property, recovery, and integration coverage.

However, protected-fact parsing is an open-ended text interpretation problem. Attempting to eliminate every possible ambiguous expression before continuing product development would create diminishing returns and prevent progress on more important functionality.

The issues below are therefore classified as known, non-blocking validation limitations rather than Phase 5 blockers.

---

# KV-001 — Space-Separated Known Markers May Be Ignored

## Status

```text
Open
Non-blocking
Priority: P2
Area: LLM protected-fact validation
```

## Summary

After recognizing a complete protected quantity, the scanner treats an ordinary word following whitespace as separated prose.

This behavior is necessary for natural sentences such as:

```text
100 people
$100 total
10/20 ratio
100×200 pixels
```

However, the same rule currently applies when the separated word is itself a known currency, percentage, or unit marker.

As a result, an additional semantic marker may not participate in the protected-token identity.

## Examples

The following transformations may not be rejected reliably:

```text
$100       → $100 EUR
$100       → $100 euros
$100       → $100 GBP
100 kg     → 100 kg pounds
100 kg     → 100 kg meters
10 percent → 10 percent USD
```

For example:

```text
Source: $100
Output: $100 EUR
```

The scanner may:

1. Recognize `$100` as one currency fact.
2. Classify `EUR` as a whitespace-separated word.
3. Return only the `$100` protected token.
4. Ignore the standalone `EUR`, because it is not followed by a number.
5. Consider the source and output protected-token sequences equal.

## Current implementation location

```text
src/captioner/core/policies/quantity_scanner.py
```

Relevant logic:

```text
_ContinuationKind.SEPARATED_WORD
_classify_fact_continuation()
_finish_quantity_fact()
```

The continuation classifier currently distinguishes:

```text
END
SAFE_PUNCTUATION
SEPARATED_WORD
ATTACHED_WORD_SUFFIX
ATTACHED_SYMBOL_SUFFIX
SLASH_SUFFIX
```

`SEPARATED_WORD` intentionally returns the already recognized base fact.

## Why this is deferred

Distinguishing ordinary prose from an additional semantic marker requires more than treating every separated word as protected.

A broad fail-closed rule would incorrectly protect normal language:

```text
100 people
$100 total
10/20 ratio
100×200 pixels
```

A correct implementation should perform a narrow longest-first lookup against known marker aliases without converting arbitrary following words into protected facts.

This is a real validation limitation, but it affects unusual or malformed model output rather than the primary translation workflow.

## Recommended future correction

Introduce a separate continuation classification:

```text
SEPARATED_KNOWN_MARKER
```

After whitespace, inspect the next complete token against:

```text
_LONGEST_CURRENCY_PREFIX_ALIASES
_LONGEST_CURRENCY_SUFFIX_ALIASES
_LONGEST_PERCENTAGE_ALIASES
_LONGEST_UNIT_ALIASES
```

If the token is a complete known marker, consume it as a bounded unsupported fact, for example:

```text
kind = unsupported-separated
```

Possible canonical identities:

```text
unsupported:currency:symbol:$/code:EUR
unsupported:unit:kg/word:GBP
unsupported:percentage:%/code:USD
```

Ordinary unrecognized prose must remain outside the protected fact.

## Required future tests

### Must reject

```text
$100 → $100 EUR
$100 → $100 euros
$100 → $100 GBP
100 kg → 100 kg pounds
100 kg → 100 kg meters
10 percent → 10 percent USD
```

### Must preserve current prose behavior

```text
100 people
$100 total
10/20 ratio
100×200 pixels
100 approximately
100 remaining
```

### Direct scanner requirements

```text
protected_tokens("$100 EUR")
```

must not return only the same token identity as:

```text
protected_tokens("$100")
```

The separated marker identity must be bounded, deterministic, case-folded where appropriate, and free of unbounded source text.

---

# KV-002 — Empty Slash Components Have Incomplete Identity

## Status

```text
Open
Non-blocking
Priority: P3
Area: Unsupported slash-compound canonicalization
```

## Summary

The slash-tail scanner canonicalizes non-empty slash components and joins them with `/`.

Empty components are currently not preserved explicitly.

Malformed expressions containing different numbers or arrangements of empty slash components may therefore produce the same canonical identity.

## Examples

These inputs may collapse to the same or insufficiently distinct marker identity:

```text
100 kg/
100 kg//
100 kg////
100 kg/ / /
```

The scanner correctly recognizes them as unsupported slash compounds and guarantees forward progress, but its semantic identity may not preserve the number and arrangement of empty components.

## Current implementation location

```text
src/captioner/core/policies/quantity_scanner.py
```

Relevant logic:

```text
_scan_slash_tail()
_bounded_identity()
```

The current scanner appends only non-empty slash components to the canonical component list.

## Why this is deferred

These expressions are malformed and unlikely to be emitted by normal translation or subtitle optimization output.

The current behavior still:

* Does not crash.
* Does not hang.
* Produces a protected unsupported fact.
* Preserves the numeric value.
* Prevents the malformed slash expression from being treated as an ordinary valid unit.
* Keeps durable marker identities bounded.
* Avoids raw source-text leakage.

The remaining issue concerns identity precision between multiple malformed forms, not loss of the primary numeric fact.

## Recommended future correction

Represent empty components explicitly.

Possible canonical forms:

```text
/       → empty
//      → empty/empty
////    → empty/empty/empty/empty
/ / /   → empty/empty/empty
```

Alternatively, use a compact deterministic structure:

```text
empty-components:1
empty-components:2
empty-components:4
```

When non-empty and empty components are mixed, preserve their order:

```text
kg//m
```

could canonicalize to:

```text
empty/m
```

when `kg` is already represented by the base marker.

The canonical identity must remain:

* Deterministic.
* Bounded.
* Independent of incidental spaces and tabs around `/`.
* Sensitive to component order.
* Sensitive to the number of empty components.
* Free of unbounded raw source text.

## Required future tests

### Distinct identities

```text
100 kg/
100 kg//
100 kg////
100 kg/ / /
```

must not all share the same protected-token identity.

### Equivalent spacing

These may intentionally normalize to the same identity:

```text
100 kg/ /m
100 kg/  / m
100 kg/\t/\tm
```

provided inline spaces and tabs are the only differences.

### Newline boundary

The scanner must not normalize across:

```text
\r
\n
```

### Progress and bounds

Every malformed slash input must:

* Return deterministically.
* Produce no zero-width span.
* Consume the intended malformed slash tail.
* Avoid an infinite loop.
* Avoid indexing beyond the input.
* Keep the marker bounded.

---

# 2. Risk Assessment

Neither known gap permits the LLM to alter:

* Cue IDs.
* Cue timestamps.
* Word timestamps.
* Word-to-Cue mappings.
* Artifact hashes.
* Job state.
* Stage state.
* Manifest data.
* Journal data.
* Provider configuration.
* Credentials.

The gaps affect only semantic comparison of unusual quantity-marker combinations in LLM-generated text.

The expected practical impact is low:

* Normal numbers, currencies, percentages, units, ratios, and dimensions remain protected.
* Numeric prefix backtracking is closed.
* Attached word and symbol additions are closed.
* Slash-suffix spacing bypasses are closed.
* Long unsupported tails are fully consumed and use bounded deterministic hashes.
* Date, time, and phone facts remain isolated from the quantity scanner.
* Structured Repair diagnostics remain redacted.

---

# 3. Deferred-Fix Policy

These issues do not block:

```text
Phase 5 GUI development
Queue projection
Configuration experience
Language switching
Activity logging
Historical Job recovery
Pause, cancellation, and retry UI
```

They should be reconsidered when one of the following occurs:

1. A real user report demonstrates incorrect subtitle output caused by one of these cases.
2. The quantity alias tables are expanded materially.
3. The protected-fact scanner is modified for another reason.
4. A stable-release hardening pass begins.
5. The project claims complete protection for arbitrary malformed quantity expressions.
6. Mutation testing or fuzzing exposes a higher-impact consequence.
7. A future backend produces these malformed patterns frequently.

Do not reopen these issues merely because another theoretically possible text pattern can be constructed.

A future fix should be justified by:

* A reproducible failure.
* A realistic impact.
* A bounded patch.
* Focused regression tests.
* No weakening of ordinary prose handling.

---

# 4. Acceptance Position

Phase 4 is accepted for continued product development with the following explicit qualification:

```text
The protected-fact validator provides deterministic and extensively tested
coverage for supported quantity expressions, but it does not claim perfect
semantic classification of every malformed or adversarial natural-language
quantity expression.
```

The two items in this document are tracked technical debt and are not Phase 5 entry blockers.

Cross-platform Release Full Gate execution remains a separate deferred verification item and is not part of these scanner limitations.

---

# 5. Summary Table

| ID     | Issue                                           | Priority | User impact                            | Phase 5 blocker |
| ------ | ----------------------------------------------- | -------: | -------------------------------------- | --------------- |
| KV-001 | Space-separated known marker may be ignored     |       P2 | Low; unusual conflicting marker output | No              |
| KV-002 | Empty slash components have incomplete identity |       P3 | Very low; malformed slash expressions  | No              |

---

# 6. Current Decision

```text
Phase 4 implementation: accepted with documented limitations
Phase 5 development: approved to begin
Known scanner gaps: deferred
Stable-release reevaluation: required
```
