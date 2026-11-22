"""Reproducible runs: one thread for the linear algebra, so the same input
always gives the same numbers.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from threadpoolctl import threadpool_info, threadpool_limits

FIT_THREADS = 1


@contextmanager
def deterministic_blas(limit: int = FIT_THREADS) -> Iterator[None]:
    """Limit every BLAS/OpenMP pool to `limit` threads for the duration."""
    with threadpool_limits(limits=limit):
        yield


def blas_threads() -> list[dict[str, object]]:
    """The thread count of each loaded pool, for run manifests."""
    return [{"library": i.get("internal_api"), "threads": i.get("num_threads")}
            for i in threadpool_info()]
