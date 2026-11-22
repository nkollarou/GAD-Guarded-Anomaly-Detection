"""Mined invariants: relations between channels that hold in clean data.
Not used for alarms in the evaluated detector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from gad.l2_context import per_channel
from gad.types import Sample

DEFAULT_QUANTILE = 0.001
DEFAULT_MARGIN = 0.5
DEFAULT_MIN_SUPPORT = 100
DEFAULT_WIDTH_FLOOR_FRAC = 0.01

FEATURES = ("value", "delta")

DEFAULT_DRIFT_BLOCK = 6 * 3600
DRIFT_SPREAD_QUANTILES = (0.01, 0.99)

CONFIG_NOVELTY_VIOLATION = 1.0

Key = tuple[str, int, str]  # (channel, regime, feature)


def drifting_channels(X: np.ndarray, block: int = DEFAULT_DRIFT_BLOCK) -> np.ndarray:
    """Per column of `X`, whether its level wanders between blocks by more
        than the reading varies inside one — `shift > spread`, where `shift`
        is the range of the per-block medians and `spread` the median per-block
        inter-quantile width.
    """
    if block < 1:
        raise ValueError("block must be >= 1")
    n, d = X.shape
    out = np.zeros(d, dtype=bool)
    n_blocks = n // block
    if n_blocks < 2:
        return out
    blocks = X[:n_blocks * block].reshape(n_blocks, block, d)
    lo_q, hi_q = DRIFT_SPREAD_QUANTILES
    medians = np.median(blocks, axis=1)                       # (n_blocks, d)
    shift = medians.max(axis=0) - medians.min(axis=0)
    spread = np.median(np.quantile(blocks, hi_q, axis=1) - np.quantile(blocks, lo_q, axis=1), axis=0)
    return shift > spread


@dataclass(frozen=True)
class Invariant:
    """One mined envelope. `support` is how many clean samples backed it —
        reported so a thin invariant can be told from a well-evidenced one
        rather than both looking equally authoritative.
    """

    channel: str
    regime: int
    feature: str
    lo: float
    hi: float
    support: int

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def excess(self, v: float) -> float:
        """How far outside the envelope `v` sits, in envelope widths. 0.0
                inside; 1.0 means a full width beyond an edge.
        """
        over = max(0.0, v - self.hi, self.lo - v)
        return over / self.width if self.width > 0 else 0.0


@dataclass(frozen=True)
class InvariantState:
    """Deployed invariant state. Sized and rolled back with the rest of the
        bundle.
    """

    invariants: dict[Key, Invariant]
    sensed_channels: tuple[str, ...]
    quantile: float
    margin: float
    min_support: int
    known_stage_configs: tuple[frozenset[int], ...] = ()
    drifting_channels: tuple[str, ...] = ()

    def n_bytes(self) -> int:
        # lo, hi, support per invariant, as float64/int64; one id per config
        return len(self.invariants) * 3 * 8 + sum(len(k) for k in self.known_stage_configs) * 8

    def channels_covered(self) -> tuple[str, ...]:
        return tuple(sorted({k[0] for k in self.invariants}))

    def regimes_covered(self) -> tuple[int, ...]:
        return tuple(sorted({k[1] for k in self.invariants}))


def _envelope(
    values: np.ndarray, quantile: float, margin: float, width_floor: float
) -> tuple[float, float]:
    lo = float(np.quantile(values, quantile))
    hi = float(np.quantile(values, 1.0 - quantile))
    width = max(hi - lo, width_floor)
    pad = margin * width
    centre = (lo + hi) / 2.0
    half = width / 2.0 + pad
    return centre - half, centre + half


class InvariantMiner:
    """Mines conditional envelopes from clean data, then scores each new
        sample against them.
    """

    def __init__(
        self,
        sensed_channels: Iterable[str],
        regime_fn: Callable[[np.ndarray], object],
        quantile: float = DEFAULT_QUANTILE,
        margin: float = DEFAULT_MARGIN,
        min_support: int = DEFAULT_MIN_SUPPORT,
        width_floor_frac: float = DEFAULT_WIDTH_FLOOR_FRAC,
        stage_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        channel_stage: Iterable[int] | None = None,
        drifting: Iterable[str] | None = None,
        drift_block: int = DEFAULT_DRIFT_BLOCK,
    ) -> None:
        if (stage_fn is None) != (channel_stage is None):
            raise ValueError("stage_fn and channel_stage must be given together")
        if not 0.0 <= quantile < 0.5:
            raise ValueError("quantile must be in [0, 0.5)")
        if margin < 0.0:
            raise ValueError("margin must be >= 0")
        if min_support < 1:
            raise ValueError("min_support must be >= 1")
        if width_floor_frac < 0.0:
            raise ValueError("width_floor_frac must be >= 0")

        self.sensed_channels = tuple(sensed_channels)
        self.n_sensed = len(self.sensed_channels)
        self.regime_fn = regime_fn
        self.quantile = quantile
        self.margin = margin
        self.min_support = min_support
        self.width_floor_frac = width_floor_frac
        self.drifting = None if drifting is None else tuple(drifting)
        unknown = () if self.drifting is None else set(self.drifting) - set(sensed_channels)
        if unknown:
            raise ValueError(f"drifting names channels that are not sensed: {sorted(unknown)}")
        self.drift_block = drift_block
        self.stage_fn = stage_fn
        self.channel_stage = None if channel_stage is None else np.asarray(list(channel_stage), dtype=np.int64)
        if self.channel_stage is not None and self.channel_stage.shape != (self.n_sensed,):
            raise ValueError("channel_stage must have one entry per sensed channel")
        self.state: InvariantState | None = None
        self._prev_x: np.ndarray | None = None

    def novel_stages(self, u: np.ndarray) -> np.ndarray:
        """Per stage, whether this commanded vector puts it in a
                configuration clean commissioning never showed. Empty without a
                stage partition.
        """
        if self.stage_fn is None or self.state is None:
            return np.zeros(0, dtype=bool)
        ids = self.stage_fn(u)
        return np.array([int(i) not in known for i, known in zip(ids, self.state.known_stage_configs)])

    def fit_from_arrays(self, X: np.ndarray, U: np.ndarray) -> InvariantState:
        return self.fit_from_segments([(X, U)])

    def fit_from_segments(self, segments) -> InvariantState:
        """Mine from several contiguous blocks that are not contiguous with
                each other. A `delta` is never formed
                across a gap; every other statistic pools the segments, so with a
                single segment this is exactly `fit_from_arrays` was.
        """
        segments = [(np.asarray(X, dtype=np.float64), np.asarray(U, dtype=np.float64))
                    for X, U in segments]
        if not segments:
            raise ValueError("fit_from_segments requires at least one segment")
        for X, U in segments:
            if X.shape[0] != U.shape[0]:
                raise ValueError("X and U must have the same number of rows")
            if X.shape[1] != self.n_sensed:
                raise ValueError(f"X has {X.shape[1]} columns, expected {self.n_sensed}")
            if X.shape[0] < 2:
                raise ValueError("need at least 2 samples to mine a delta invariant")

        per_segment_regimes = [
            np.array([per_channel(self.regime_fn(U[t]), self.n_sensed) for t in range(len(U))])
            for _, U in segments
        ]
        X = np.concatenate([Xs for Xs, _ in segments])
        regimes = np.concatenate(per_segment_regimes)
        deltas = np.concatenate([np.diff(Xs, axis=0) for Xs, _ in segments])
        delta_regimes = np.concatenate([r[1:] for r in per_segment_regimes])

        known_stage_configs: tuple[frozenset[int], ...] = ()
        if self.stage_fn is not None:
            stage_ids = np.concatenate([
                np.array([self.stage_fn(U[t]) for t in range(len(U))]) for _, U in segments
            ])
            known = []
            for s in range(stage_ids.shape[1]):
                ids, counts = np.unique(stage_ids[:, s], return_counts=True)
                known.append(frozenset(int(i) for i, c in zip(ids, counts) if c >= self.min_support))
            known_stage_configs = tuple(known)

        if self.drifting is None:
            drifting = drifting_channels(X, self.drift_block)
        else:
            drifting = np.array([ch in self.drifting for ch in self.sensed_channels])

        invariants: dict[Key, Invariant] = {}
        for j, channel in enumerate(self.sensed_channels):
            floors = {
                "value": self.width_floor_frac * float(np.ptp(X[:, j])),
                "delta": self.width_floor_frac * float(np.ptp(deltas[:, j])),
            }
            for feature in FEATURES:
                if feature == "value":
                    if drifting[j]:
                        continue
                    series, series_regimes = X[:, j], regimes[:, j]
                else:
                    series, series_regimes = deltas[:, j], delta_regimes[:, j]

                for regime in np.unique(series_regimes):
                    in_regime = series[series_regimes == regime]
                    if len(in_regime) < self.min_support:
                        continue
                    lo, hi = _envelope(
                        in_regime, self.quantile, self.margin, floors[feature]
                    )
                    invariants[(channel, int(regime), feature)] = Invariant(
                        channel=channel,
                        regime=int(regime),
                        feature=feature,
                        lo=lo,
                        hi=hi,
                        support=int(len(in_regime)),
                    )

        self.state = InvariantState(
            invariants=invariants,
            sensed_channels=self.sensed_channels,
            quantile=self.quantile,
            margin=self.margin,
            min_support=self.min_support,
            known_stage_configs=known_stage_configs,
            drifting_channels=tuple(
                ch for ch, is_drifting in zip(self.sensed_channels, drifting) if is_drifting
            ),
        )
        return self.state

    def fit(self, samples: Iterable[Sample]) -> None:
        batch = list(samples)
        if not batch:
            raise ValueError("cannot fit on an empty sample batch")
        X = np.array([s.x for s in batch])
        U = np.array([s.u for s in batch])
        self.fit_from_arrays(X, U)

    def update(self, sample: Sample) -> np.ndarray:
        """Streaming: score this sample against the mined envelopes and
                return one violation magnitude per sensed channel — 0.0 inside,
                otherwise how far outside in envelope widths, taking the worst of
                that channel's features. NaN where no invariant covers the channel
                in this regime (or, for `delta` alone, on the very first sample).
        """
        if self.state is None:
            raise RuntimeError("fit() or fit_from_arrays() must run before update()")

        regimes = per_channel(self.regime_fn(sample.u), self.n_sensed)
        novel = self.novel_stages(sample.u)
        out = np.full(self.n_sensed, np.nan)

        for j, channel in enumerate(self.sensed_channels):
            observed: list[float] = []
            if len(novel) and novel[self.channel_stage[j]]:
                observed.append(CONFIG_NOVELTY_VIOLATION)
            for feature in FEATURES:
                invariant = self.state.invariants.get((channel, int(regimes[j]), feature))
                if invariant is None:
                    continue
                if feature == "value":
                    observed.append(invariant.excess(float(sample.x[j])))
                elif self._prev_x is not None:
                    delta = float(sample.x[j] - self._prev_x[j])
                    observed.append(invariant.excess(delta))
            if observed:
                out[j] = max(observed)

        self._prev_x = np.array(sample.x, dtype=np.float64)
        return out
