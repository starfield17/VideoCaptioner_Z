# Error model

`AppError.code` is a stable machine-readable identifier such as
`i18n.catalog_missing`. It is suitable for logs, tests, and structured CLI
output; it is not a localized user-facing sentence.

`params` contains only JSON-compatible diagnostic values. A low-level exception
is preserved through Python exception chaining (`raise ... from cause`) rather
than being placed in `params`. `retryable` records whether a later application
policy may retry the operation.

Localization happens at the CLI or GUI boundary through `I18nService`. Core
code raises codes and parameters only. Formatting failures remain structured
errors and are never silently swallowed.
