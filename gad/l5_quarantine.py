"""L5, rule 1: which samples are trusted enough to learn from."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from gad.l0_quality import TRUSTED_LEARNING_MASK
from gad.types import MISSING

#: state codes as `gad.adaptation.schedule.RollingWindow` stores them
NORMAL, DEGRADED, ANOMALY = 0, 1, 2

QUARANTINE_FLAGS = TRUSTED_LEARNING_MASK

BLINDING_MASK = MISSING
PER_CHANNEL_FLAGS = QUARANTINE_FLAGS & ~BLINDING_MASK


@dataclass(frozen=True)
class QuarantinePolicy:
    """Stated in advance, never tuned against a
        result. `guard` is the aftermath: samples within this many steps after
        an ANOMALY sample are not clean, whatever the detector says about
        them, because the plant is recovering.
    """

    guard: int = 900
    min_segment: int = 600
    #: minimum clean samples across all segments
    min_samples: int = 21_600
    #: minimum distinct actuator contexts, each with `min_context_support`
    min_contexts: int = 8
    min_context_support: int = 100

    def __post_init__(self) -> None:
        if min(self.guard, self.min_segment, self.min_samples,
               self.min_contexts, self.min_context_support) < 0:
            raise ValueError("quarantine policy values must be >= 0")


Segment = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Batch:
    """What quarantine offers the refit, and why."""

    segments: tuple[Segment, ...]
    accepted: bool
    reasons: tuple[str, ...]
    n_clean: int
    n_kept: int
    n_contexts: int
    n_dropped_not_normal: int
    #: channel-rows barred by OOR/STALE on that channel (not whole samples)
    n_dropped_flagged: int
    n_dropped_guard: int
    #: whole samples removed because some channel had no reading at all
    n_dropped_blind: int = 0
    n_kept_per_channel: tuple[int, ...] = ()
    n_qualifying_channels: int = 0

    @property
    def n_segments(self) -> int:
        return len(self.segments)


def clean_mask(states: np.ndarray, q: np.ndarray, policy: QuarantinePolicy) -> dict[str, np.ndarray]:
    """What may be learned from, per sample **and per channel**."""
    states = np.asarray(states)
    q = np.asarray(q)
    if q.ndim != 2:
        raise ValueError(f"q must be (rows, channels), got shape {q.shape}")
    is_normal = states == NORMAL
    blind = (q & BLINDING_MASK).astype(bool).any(axis=1)
    per_channel_ok = (q & PER_CHANNEL_FLAGS).astype(bool) == 0

    # aftermath: the `guard` samples following every ANOMALY sample
    after = np.zeros(len(states), dtype=bool)
    since = policy.guard + 1  # nothing before the window's start counts as recent
    for i, s in enumerate(states):
        since = 0 if s == ANOMALY else since + 1
        after[i] = since <= policy.guard  # the `guard` samples following an alarm
    row = is_normal & ~blind & ~after
    return {"row": row,
            "keep": row[:, None] & per_channel_ok,
            "not_normal": ~is_normal, "blind": blind,
            "flagged": ~per_channel_ok, "guard": after}


def contiguous_runs(mask: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    """[lo, hi) ranges of consecutive True at least `min_length` long."""
    m = np.asarray(mask).astype(np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_length]


def count_contexts(U: np.ndarray, context_key: Callable[[np.ndarray], tuple], support: int) -> int:
    """Distinct actuator contexts seen at least `support` times."""
    counts: dict[tuple, int] = {}
    for u in U:
        key = tuple(context_key(u))
        counts[key] = counts.get(key, 0) + 1
    return sum(1 for c in counts.values() if c >= support)


def select_batch(
    X: np.ndarray,
    U: np.ndarray,
    q: np.ndarray,
    states: np.ndarray,
    policy: QuarantinePolicy,
    context_key: Callable[[np.ndarray], tuple] = tuple,
) -> Batch:
    """Gate 1a over one adaptation window, in time order."""
    masks = clean_mask(states, q, policy)
    row, keep = masks["row"], masks["keep"]
    runs = contiguous_runs(row, policy.min_segment)
    segments = tuple((np.array(X[lo:hi]), np.array(U[lo:hi]), np.array(keep[lo:hi]))
                     for lo, hi in runs)
    n_kept = sum(hi - lo for lo, hi in runs)
    # per channel, how many of the kept rows it may actually learn from
    per_channel = (np.concatenate([m for _, _, m in segments]).sum(axis=0)
                   if segments else np.zeros(keep.shape[1], dtype=int))

    n_contexts = 0
    if segments:
        n_contexts = count_contexts(np.concatenate([u for _, u, _ in segments]),
                                    context_key, policy.min_context_support)

    qualifying = np.flatnonzero(per_channel >= policy.min_samples)
    reasons = []
    if not qualifying.size:
        reasons.append(f"no channel has {policy.min_samples} learnable samples "
                       f"(best {int(per_channel.max()) if per_channel.size else 0})")
    if n_contexts < policy.min_contexts:
        reasons.append(f"only {n_contexts} actuator contexts with support, need {policy.min_contexts}")

    return Batch(
        segments=segments if not reasons else (),
        accepted=not reasons,
        reasons=tuple(reasons),
        n_clean=int(row.sum()),
        n_kept=n_kept,
        n_contexts=n_contexts,
        n_kept_per_channel=tuple(int(v) for v in per_channel),
        n_qualifying_channels=int(qualifying.size),
        n_dropped_not_normal=int(masks["not_normal"].sum()),
        n_dropped_flagged=int((masks["flagged"] & row[:, None]).sum()),
        n_dropped_guard=int((masks["guard"] & ~masks["not_normal"]).sum()),
        n_dropped_blind=int((masks["blind"] & ~masks["not_normal"]).sum()),
    )


def split_segments(
    segments: tuple[Segment, ...], fit_frac: float, warm_frac: float
) -> tuple[list[Segment], list[Segment], list[Segment]]:
    """Cut the segments, in time order, into fit / warm-up / calibration
        roles by sample count. A segment straddling a boundary is cut at it —
        a cut inside a run leaves both halves contiguous, so no role ever
        receives a gap.
    """
    total = sum(len(s[0]) for s in segments)
    fit_end = int(total * fit_frac)
    warm_end = fit_end + int(total * warm_frac)
    roles: tuple[list[Segment], list[Segment], list[Segment]] = ([], [], [])

    seen = 0
    for seg in segments:
        X, U, M = unpack(seg)
        start = seen
        seen += len(X)
        for role, (lo, hi) in enumerate(((0, fit_end), (fit_end, warm_end), (warm_end, total))):
            a, b = max(lo, start) - start, min(hi, seen) - start
            if b > a:
                roles[role].append((X[a:b], U[a:b], M[a:b]))
    return roles


def unpack(segment) -> Segment:
    """A segment as (X, U, keep), accepting the 2-tuple form."""
    if len(segment) == 3:
        X, U, M = segment
        return np.asarray(X), np.asarray(U), np.asarray(M, dtype=bool)
    X, U = segment
    X = np.asarray(X)
    return X, np.asarray(U), np.ones(X.shape, dtype=bool)
