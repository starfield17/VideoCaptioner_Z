"""Deterministic test fault injection."""

from __future__ import annotations

from dataclasses import dataclass, field


def _empty_hits() -> list[tuple[str, str, int]]:
    return []


class InjectedCrash(BaseException):
    """Simulate abrupt process disappearance outside ordinary error handling."""


@dataclass(slots=True)
class ScriptedFaultInjector:
    stage_name: str
    point: str
    remaining: int = 1
    hits: list[tuple[str, str, int]] = field(default_factory=_empty_hits)

    def hit(
        self,
        *,
        batch_id: str,
        job_id: str,
        stage_name: str,
        attempt: int,
        point: str,
    ) -> None:
        del batch_id, job_id
        self.hits.append((stage_name, point, attempt))
        if stage_name == self.stage_name and point == self.point and self.remaining > 0:
            self.remaining -= 1
            raise InjectedCrash
