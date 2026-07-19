# Model Manager

The Model Manager owns model installation records and the managed payload
layout:

```text
<data_dir>/models/
├── managed/<identity-digest>/{model-manifest.json,installation.json,payload/}
├── external/<identity-digest>.json
└── .use/<identity-digest>.lock
```

Remote materialization uses `<downloads_dir>/models/<transaction-id>/` as a
temporary SDK input and `<staging_dir>/models/<transaction-id>/` for the clean
installation payload. Both are removed after success or failure; neither is a
model Store.

Managed imports and remote installs use a transaction-local staging directory,
copy regular files into a clean payload, hash every file, validate the format,
write metadata, and atomically move the completed directory into the store.
An interrupted transaction is not an installed model.  Recovery is local and
never downloads again.

## Managed and external models

`model import` copies the source directory, so the original may be removed
after a successful import.  `model register-external` requires
`--developer-mode`, keeps the user's directory in place, and only stores a
registration record.  Captioner never modifies or deletes an external model.
Both forms are protected by a model use-lock while a Worker session is active.

Recovery is local-only: incomplete staging/download transactions are cleaned,
complete managed directories can reconstruct a missing registration record, and
incomplete final directories are moved to `.recovery`. External directories are
never modified by recovery.

## Validation and load verification

Static validation checks regular files, exact size/hash inventory, required
offline tokenizer assets, JSON metadata, and backend-specific format rules.
Load verification starts the selected installed Runtime Worker and sends
`model.load.request`; only a matching successful response changes an installed
model to `load_verified`.  A failed load leaves the model installed so another
compatible Runtime can be tried.

The CLI never downloads `tiny` or another implicit model.  Use `model search-hf`,
`model install-hf`, `model install-modelscope`, `model import`, or
`model register-external` explicitly.

New Jobs persist the exact Runtime identity, Model identity, backend, device,
and compute type in a schema-3 ASR snapshot. Resume uses those identities
without running `auto` selection again; only an explicit ASR override reruns
preflight and invalidates transcription.
