from __future__ import annotations

import pytest

from captioner.core.domain.errors import AppError
from captioner.core.policies.segmentation_config import SegmentationPolicyConfig


def test_legacy_configuration_is_completed_deterministically() -> None:
    policy = SegmentationPolicyConfig.from_mapping(
        {"max_duration_ms": 7_000, "max_text_units": 84, "hard_gap_ms": 700}
    )
    assert policy.max_cue_width == 84
    assert policy.to_mapping()["schema_version"] == 1
    assert policy.signature == SegmentationPolicyConfig.from_mapping(policy.to_mapping()).signature


def test_policy_rejects_partial_unknown_configuration() -> None:
    with pytest.raises(AppError, match=r"job\.config_invalid"):
        SegmentationPolicyConfig.from_mapping({"max_duration_ms": 7_000})
