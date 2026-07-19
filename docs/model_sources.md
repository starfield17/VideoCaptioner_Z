# Model sources

Captioner keeps source lookup and model installation separate.  A source
resolves a repository to an immutable revision; the Model Manager then copies
only the selected format's safe files into its own staging tree and validates
the clean payload before committing it.

## Hugging Face

Hugging Face supports search and exact repository resolution.  Search results
are candidates, not durable model identities.  Exact resolution must produce
a full commit SHA before materialization.  Public repositories may be used
without a token; private or gated repositories require one.

## ModelScope

Phase 6.2 supports exact repository resolution only.  The command must supply
`--revision`.  If the SDK cannot return a concrete immutable revision,
installation fails with `model.source_revision_unresolved`; a branch such as
`master` is never stored as durable identity.

## Credentials and cache

Optional configuration lives at `<config_dir>/model-sources.toml`.  Tokens are
plain text by design, but they are never copied into ModelIdentity,
ModelManifest, Job snapshots, Worker protocol, logs, or errors.  The config
file is not created automatically.  Environment variables are only fallback
credentials when the config has no token.

SDK caches are transaction-local download inputs.  They are not the Captioner
Model Store and are removed after materialization.  The Worker is offline and
never invokes a source SDK.
