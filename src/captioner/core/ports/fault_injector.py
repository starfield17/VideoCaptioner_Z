"""Named fault checkpoints owned by the generic Stage executor."""

from typing import Protocol


class FaultInjector(Protocol):
    def hit(
        self,
        *,
        batch_id: str,
        job_id: str,
        stage_name: str,
        attempt: int,
        point: str,
    ) -> None: ...


class NoOpFaultInjector:
    def hit(
        self,
        *,
        batch_id: str,
        job_id: str,
        stage_name: str,
        attempt: int,
        point: str,
    ) -> None:
        del batch_id, job_id, stage_name, attempt, point
