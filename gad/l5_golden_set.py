"""L5, rule 4: the golden set, a fixed recording of clean operation captured once
and never updated automatically.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from gad.l2_residual import robust_scale


@dataclass(frozen=True)
class GoldenSet:
    """Frozen clean reference. `X`/`U` are (n_excerpts, length, n_channels)."""

    X: np.ndarray
    U: np.ndarray
    sensed: tuple[str, ...]
    commanded: tuple[str, ...]
    sigma: np.ndarray  # per-channel clean spread, fixed with the set
    source: str  # where these rows came from, in words
    captured_at: str
    digest: str

    def __post_init__(self) -> None:
        if self.X.ndim != 3 or self.U.ndim != 3:
            raise ValueError("X and U must be (n_excerpts, length, n_channels)")
        if self.X.shape[:2] != self.U.shape[:2]:
            raise ValueError("X and U must have the same excerpts and length")
        if self.X.shape[2] != len(self.sensed) or self.U.shape[2] != len(self.commanded):
            raise ValueError("X/U channel counts do not match the names")
        if self.sigma.shape != (len(self.sensed),):
            raise ValueError("sigma must have one entry per sensed channel")
        for a in (self.X, self.U, self.sigma):
            a.setflags(write=False)  # the golden set is frozen, literally

    def verify(self) -> bool:
        """True when the rows still hash to the digest recorded at capture."""
        return _digest(self.X, self.U) == self.digest


def _digest(X: np.ndarray, U: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in (np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(U, dtype=np.float64)):
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def capture(
    X: np.ndarray,
    U: np.ndarray,
    sensed, commanded,
    starts, length: int,
    source: str,
    captured_at: str,
    min_sigma: float = 1e-6,
    sigma: np.ndarray | None = None,
) -> GoldenSet:
    """Cut `len(starts)` excerpts of `length` rows from clean `X`/`U` and freeze
    them. The caller states `source` and `captured_at` for the record; this
    function never chooses rows for itself.
    """
    starts = [int(s) for s in starts]
    if not starts:
        raise ValueError("a golden set needs at least one excerpt")
    if length < 1:
        raise ValueError("length must be >= 1")
    for s in starts:
        if s < 0 or s + length > len(X):
            raise ValueError(f"excerpt [{s}, {s + length}) does not fit in {len(X)} rows")
    Xg = np.stack([np.array(X[s:s + length], dtype=np.float64) for s in starts])
    Ug = np.stack([np.array(U[s:s + length], dtype=np.float64) for s in starts])
    if not np.isfinite(Xg).all():
        raise ValueError("a golden set must not contain missing readings")

    if sigma is None:
        flat = Xg.reshape(-1, Xg.shape[2])
        sigma = np.empty(flat.shape[1])
        for j in range(flat.shape[1]):
            col = flat[:, j]
            s = robust_scale(np.abs(col - np.median(col)), min_sigma)
            sigma[j] = s if s is not None else min_sigma
    else:
        sigma = np.asarray(sigma, dtype=np.float64).copy()
        if sigma.shape != (Xg.shape[2],):
            raise ValueError("sigma must have one entry per sensed channel")
        sigma[~np.isfinite(sigma) | (sigma <= 0)] = min_sigma
    return GoldenSet(Xg, Ug, tuple(sensed), tuple(commanded), sigma,
                     source=source, captured_at=captured_at, digest=_digest(Xg, Ug))
