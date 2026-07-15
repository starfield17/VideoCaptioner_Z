from __future__ import annotations

import pytest

from captioner.cli.outcomes import (
    ASR_ERROR,
    CANCELLED,
    CLI_ERROR,
    MEDIA_ERROR,
    OUTPUT_ERROR,
    exit_code_for_error,
)
from captioner.core.domain.errors import AppError


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("operation.cancelled", CANCELLED),
        ("cli.invalid", CLI_ERROR),
        ("i18n.unknown_key", CLI_ERROR),
        ("media.ffprobe_failed", MEDIA_ERROR),
        ("process.executable_not_found", MEDIA_ERROR),
        ("asr.runtime_missing", ASR_ERROR),
        ("transcript.invalid", ASR_ERROR),
        ("output.write_failed", OUTPUT_ERROR),
        ("export.srt_invalid", OUTPUT_ERROR),
        ("subtitle.invalid", OUTPUT_ERROR),
        ("unexpected.code", CLI_ERROR),
    ],
)
def test_exit_code_families_are_stable(code: str, expected: int) -> None:
    assert exit_code_for_error(AppError(code)) == expected
