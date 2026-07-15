"""Small dependency-inversion ports used by the Phase 0 fakes."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from captioner.core.domain.result import JsonValue


@dataclass(frozen=True, slots=True)
class CapabilityProbe:
    """Report whether an adapter boundary is usable."""

    available: bool
    details: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


__all__ = ["CapabilityProbe"]
