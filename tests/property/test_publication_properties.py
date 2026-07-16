from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from captioner.adapters.persistence.domain_codecs import encode_publication_receipt
from captioner.adapters.pipeline.stages import verify_publication
from captioner.core.domain.artifact import ArtifactRef
from captioner.core.domain.errors import AppError
from captioner.core.domain.publication import PublicationReceipt, PublishedTarget


@given(data=st.binary(min_size=1, max_size=256))
def test_publication_target_corruption_is_detected(data: bytes) -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        transcript_target = tmp_path / "sample.transcript.json"
        target = tmp_path / "sample.srt"
        transcript_data = b"{}"
        transcript_target.write_bytes(transcript_data)
        target.write_bytes(data)
        transcript_digest = hashlib.sha256(transcript_data).hexdigest()
        digest = hashlib.sha256(data).hexdigest()
        transcript_ref = ArtifactRef(
            transcript_digest,
            len(transcript_data),
            "final-transcript-json",
            "application/json",
            "final-transcript.json",
        )
        ref = ArtifactRef(
            digest,
            len(data),
            "final-subtitle-srt",
            "text/plain",
            "final-subtitle.srt",
        )
        receipt = PublicationReceipt(
            hashlib.sha256((transcript_digest + digest).encode()).hexdigest(),
            (
                PublishedTarget(
                    str(transcript_target.resolve()),
                    transcript_digest,
                    len(transcript_data),
                    "sample.transcript.json",
                ),
                PublishedTarget(str(target.resolve()), digest, len(data), "sample.srt"),
            ),
        )
        verify_publication(
            encode_publication_receipt(receipt),
            output_dir=tmp_path,
            input_path=tmp_path / "sample.wav",
            export_refs=(transcript_ref, ref),
            publication_version="publish-v1",
        )
        target.write_bytes(data + b"x")
        with pytest.raises(AppError, match=r"output\.publication_invalid"):
            verify_publication(
                encode_publication_receipt(receipt),
                output_dir=tmp_path,
                input_path=tmp_path / "sample.wav",
                export_refs=(transcript_ref, ref),
                publication_version="publish-v1",
            )
