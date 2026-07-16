"""Small dependency-inversion ports used by the Phase 0 fakes."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from captioner.core.domain.result import FrozenJsonValue, JsonValue, freeze_json_value


@dataclass(frozen=True, slots=True)
class CapabilityProbe:
    """Report whether an adapter boundary is usable."""

    available: bool
    details: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        frozen = cast(Mapping[str, FrozenJsonValue], freeze_json_value(self.details))
        object.__setattr__(
            self,
            "details",
            cast(Mapping[str, JsonValue], frozen),
        )


__all__ = ["CapabilityProbe"]
