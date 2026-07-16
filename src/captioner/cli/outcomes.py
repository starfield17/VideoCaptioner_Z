"""Stable CLI exit-code mapping for structured application errors."""

from __future__ import annotations

from captioner.core.domain.errors import AppError

SUCCESS = 0
CLI_ERROR = 2
MEDIA_ERROR = 3
ASR_ERROR = 4
OUTPUT_ERROR = 5
LLM_ERROR = 7
CANCELLED = 130


def exit_code_for_error(error: AppError) -> int:
    """Map stable error-code families at the presentation boundary."""
    if error.code == "operation.cancelled":
        return CANCELLED
    if error.code.startswith(("cli.", "i18n.")):
        return CLI_ERROR
    if error.code.startswith(("media.", "process.")):
        return MEDIA_ERROR
    if error.code.startswith(("asr.", "transcript.")):
        return ASR_ERROR
    if error.code.startswith(("output.", "export.", "subtitle.")):
        return OUTPUT_ERROR
    if error.code.startswith("llm."):
        return LLM_ERROR
    return CLI_ERROR
