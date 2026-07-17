Extract only domain-specific terminology from the requested source units.

Return exactly `id` and a `terms` array for every requested unit. The array may
be empty when the unit contains no terminology. Each term must contain exactly
`source_term` and `target_term`, and the source term must occur as a complete
word or phrase in the requested unit. Do not return timestamps, durations, cue
boundaries, Word mapping, or any other fields.
