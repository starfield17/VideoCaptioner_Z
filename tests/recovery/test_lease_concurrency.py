from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

from captioner.adapters.persistence.batch_lease import BatchLease
from captioner.core.domain.errors import AppError


def test_simultaneous_stale_reclaim_has_one_token_safe_winner(tmp_path: Path) -> None:
    path = tmp_path / "lease.json"
    stale = BatchLease(path, "stale", 99, "host", "now", lambda _pid: False)
    stale.acquire()
    barrier = Barrier(2)
    winner_holds = Event()
    release_winner = Event()

    def claim(token: str, pid: int) -> tuple[str, BatchLease | AppError]:
        lease = BatchLease(path, token, pid, "host", "now", lambda owner: owner in {101, 102})
        barrier.wait()
        try:
            lease.acquire()
        except AppError as exc:
            winner_holds.wait(timeout=5)
            return "lost", exc
        winner_holds.set()
        release_winner.wait(timeout=5)
        return "won", lease

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(claim, "one", 101), pool.submit(claim, "two", 102)]
        assert winner_holds.wait(timeout=5)
        assert path.is_file()
        release_winner.set()
        results = [future.result(timeout=5) for future in futures]

    assert [result[0] for result in results].count("won") == 1
    assert [result[0] for result in results].count("lost") == 1
    winner = next(result[1] for result in results if result[0] == "won")
    assert isinstance(winner, BatchLease)
    assert path.is_file()
    winner.release()
