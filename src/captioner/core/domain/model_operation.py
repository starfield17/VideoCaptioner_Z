"""Stage names for Model Manager operations."""

from __future__ import annotations

from enum import StrEnum


class ModelOperationPhase(StrEnum):
    RESOLVING_SOURCE = "resolving_source"
    DOWNLOADING = "downloading"
    INSPECTING = "inspecting"
    COPYING = "copying"
    HASHING = "hashing"
    VALIDATING = "validating"
    INSTALLING = "installing"
    LOAD_VERIFYING = "load_verifying"
    CLEANING_STAGING = "cleaning_staging"
    COMPLETED = "completed"


__all__ = ["ModelOperationPhase"]
