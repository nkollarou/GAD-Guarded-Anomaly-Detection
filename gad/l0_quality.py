"""L0: input quality. Flags each reading as missing, out of range or stale. A
flagged reading is still checked for attacks but never learned from.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gad.types import CLIPPED, MISSING, OOR, STALE, VALID

BLINDING_MASK = MISSING
LEARNING_EXCLUSION_MASK = MISSING | OOR
TRUSTED_LEARNING_MASK = MISSING | OOR | STALE


def is_blinding(q: int) -> bool:
    """The channel cannot be scored this sample (no reading)."""
    return bool(q & BLINDING_MASK)


@dataclass(frozen=True)
class ChannelRange:
    name: str
    lo: float
    hi: float
    clip_lo: float | None
    clip_hi: float | None
    stale_run_threshold: int


@dataclass(frozen=True)
class QualityRanges:
    channels: tuple[str, ...]
    ranges: tuple[ChannelRange, ...]

    def __post_init__(self) -> None:
        if len(self.channels) != len(self.ranges):
            raise ValueError("channels and ranges must have the same length")


def _stale_run_lengths(values: np.ndarray) -> np.ndarray:
    """Lengths of maximal runs of exactly-equal consecutive values."""
    if len(values) == 0:
        return np.array([], dtype=np.int64)
    change = np.flatnonzero(np.diff(values) != 0)
    boundaries = np.concatenate(([-1], change, [len(values) - 1]))
    return np.diff(boundaries)


def fit_quality_ranges(
    names: list[str],
    clean: np.ndarray,
    margin_k: float = 5.0,
    step_lag: int = 1,
    step_basis: str = "median",
) -> QualityRanges:
    """Fit per-channel OOR/CLIPPED/STALE parameters from a clean, steady-state
    window. `clean` is (n_samples, n_channels), no NaNs.
    """
    if clean.ndim != 2 or clean.shape[1] != len(names):
        raise ValueError("clean must be (n_samples, n_channels) matching names")
    if np.isnan(clean).any():
        raise ValueError("fit_quality_ranges requires a clean window with no NaNs")

    ranges = []
    for i, name in enumerate(names):
        values = clean[:, i]
        lo_obs = float(values.min())
        hi_obs = float(values.max())

        if step_basis == "sigma_d":
            d = np.diff(values)
            typical_step = float(np.std(d) / np.sqrt(2.0)) if len(d) else 0.0
        elif step_basis == "median":
            diffs = np.abs(np.diff(values, n=1) if step_lag == 1
                           else values[step_lag:] - values[:-step_lag])
            typical_step = float(np.median(diffs)) if len(diffs) else 0.0
        else:
            raise ValueError(f"unknown step_basis {step_basis!r}")
        if typical_step == 0.0:
            span = hi_obs - lo_obs
            typical_step = span * 1e-6 if span > 0 else 1e-9

        margin = margin_k * typical_step
        lo = lo_obs - margin
        hi = hi_obs + margin

        clip_lo = lo_obs if int((values == lo_obs).sum()) > 1 else None
        clip_hi = hi_obs if int((values == hi_obs).sum()) > 1 else None

        run_lengths = _stale_run_lengths(values)
        stale_run_threshold = int(run_lengths.max()) if len(run_lengths) else 1

        ranges.append(
            ChannelRange(
                name=name,
                lo=lo,
                hi=hi,
                clip_lo=clip_lo,
                clip_hi=clip_hi,
                stale_run_threshold=stale_run_threshold,
            )
        )

    return QualityRanges(channels=tuple(names), ranges=tuple(ranges))


class QualityChecker:
    """Streaming per-channel quality flagging. Holds its own state (previous value
    and current run length per channel) — call `update` once per incoming
    sample, in time order.
    """

    def __init__(self, ranges: QualityRanges) -> None:
        self._ranges = ranges.ranges
        n = len(ranges.ranges)
        self._prev = np.full(n, np.nan, dtype=np.float64)
        self._run_len = np.zeros(n, dtype=np.int64)

    def update(self, x: np.ndarray) -> np.ndarray:
        if x.shape != (len(self._ranges),):
            raise ValueError(f"expected shape ({len(self._ranges)},), got {x.shape}")

        q = np.zeros(len(self._ranges), dtype=np.uint8)
        for i, r in enumerate(self._ranges):
            v = x[i]
            if np.isnan(v):
                q[i] = MISSING
                self._run_len[i] = 0
                self._prev[i] = np.nan
                continue

            flags = VALID
            if v < r.lo or v > r.hi:
                flags |= OOR
            if (r.clip_lo is not None and v == r.clip_lo) or (r.clip_hi is not None and v == r.clip_hi):
                flags |= CLIPPED

            if not np.isnan(self._prev[i]) and v == self._prev[i]:
                self._run_len[i] += 1
            else:
                self._run_len[i] = 1
            if self._run_len[i] > r.stale_run_threshold:
                flags |= STALE

            self._prev[i] = v
            q[i] = flags

        return q
