from __future__ import annotations

from typing import cast

from hypothesis import given
from hypothesis import strategies as st

from captioner.core.domain.artifact import ArtifactRef
from captioner.core.domain.cache_key import derive_stage_cache_key
from captioner.core.domain.result import FrozenJsonValue, freeze_json_value


@given(st.dictionaries(st.text(min_size=1), st.integers(), max_size=8))
def test_cache_key_ignores_mapping_insertion_order(config: dict[str, int]) -> None:
    artifact = ArtifactRef("a" * 64, 1, "input", "application/octet-stream", "input.bin")
    forward = cast(dict[str, FrozenJsonValue], freeze_json_value(config))
    reverse = cast(dict[str, FrozenJsonValue], freeze_json_value(dict(reversed(config.items()))))
    assert derive_stage_cache_key(
        stage_name="inspect", stage_version="1", input_artifacts=(artifact,), config=forward
    ) == derive_stage_cache_key(
        stage_name="inspect", stage_version="1", input_artifacts=(artifact,), config=reverse
    )


@given(st.text(min_size=1), st.text(min_size=1))
def test_irrelevant_external_id_is_not_part_of_cache_key(batch_id: str, job_id: str) -> None:
    del batch_id, job_id
    artifact = ArtifactRef("a" * 64, 1, "input", "application/octet-stream", "input.bin")
    config = cast(dict[str, FrozenJsonValue], freeze_json_value({"language": "en"}))
    assert derive_stage_cache_key(
        stage_name="transcribe", stage_version="1", input_artifacts=(artifact,), config=config
    ) == derive_stage_cache_key(
        stage_name="transcribe", stage_version="1", input_artifacts=(artifact,), config=config
    )
