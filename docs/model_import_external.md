# Local import and external registration

Use managed import when Captioner should own a private copy:

```bash
captioner model import /path/to/model \
  --backend faster-whisper \
  --format faster-whisper-ct2
```

The source directory is inspected without modification.  Only regular files
are copied to a clean staging tree; symlinks, special files, hardlinks, code,
and missing offline assets are rejected.

Use external registration for a developer-owned directory:

```bash
captioner model register-external /path/to/model \
  --backend mlx-whisper \
  --format mlx-whisper \
  --developer-mode
```

External registration records a canonical identity and validation projection,
but never copies, chmods, upgrades, or removes the directory.  Every new Job
preflight rechecks its files and reports `model.external_content_changed` when
the content no longer matches the registration.  Re-register the model after
intentional changes.
