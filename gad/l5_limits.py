"""L5, rule 3: keeps the combined change of standard deviation and threshold
within the limit c of commissioning.
"""
from __future__ import annotations

import numpy as np

from gad.l2_context import per_channel
from gad.l2_residual import BoundedScaleResidual


class EventScaleResidual(BoundedScaleResidual):
    """Learns like `BoundedScaleResidual`; scores with committed values."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._committed: list[dict[int, tuple[float, float]]] | None = None

    def commit(self) -> None:
        """Make the learned centres and scales the ones used for scoring."""
        self._committed = [{ctx: (m, s) for ctx, (m, s) in per.items()} for per in self._state]

    def committed_scale(self, channel: int, regime_id: int) -> float | None:
        if self._committed is None:
            return self.current_scale(channel, regime_id)
        v = self._committed[channel].get(int(regime_id))
        return None if v is None else v[1]

    def update(self, residual: np.ndarray, regime_id, is_normal: bool,
               learn: np.ndarray | None = None) -> np.ndarray:
        live = super().update(residual, regime_id, is_normal, learn)  # learns in the background
        if self._committed is None:
            return live
        regimes = per_channel(regime_id, self.n_channels)
        scaled = np.full(self.n_channels, np.nan)
        for j in np.flatnonzero(np.isfinite(live)):
            ctx = int(regimes[j])
            committed = self._committed[j].get(ctx)
            if committed is None:
                m, s = self._state[j][ctx]
                committed = (m, s)
                self._committed[j][ctx] = committed
            m, s = committed
            scaled[j] = (residual[j] - m) / s
        return scaled

    def scale_factors(self) -> tuple[np.ndarray, np.ndarray]:
        """Per channel, the largest and smallest committed s / s_ref over its
                referenced contexts (1.0 where there is no reference).
        """
        hi = np.ones(self.n_channels)
        lo = np.ones(self.n_channels)
        if self._ref is None or self._committed is None:
            return hi, lo
        for j in range(self.n_channels):
            ratios = [self._committed[j][c][1] / s_ref for c, (_, s_ref) in self._ref[j].items()
                      if c in self._committed[j] and s_ref > 0]
            if ratios:
                hi[j], lo[j] = max(ratios), min(ratios)
        return hi, lo

    def export_state(self):
        """Exports the committed values: those are what the detector used."""
        if self._committed is None:
            return super().export_state()
        live = self._state
        self._state = [{c: [m, s] for c, (m, s) in per.items()} for per in self._committed]
        try:
            return super().export_state()
        finally:
            self._state = live


def joint_clamp(h: np.ndarray, h_ref: np.ndarray, f_hi: np.ndarray, f_lo: np.ndarray,
                cap: float) -> tuple[np.ndarray, np.ndarray]:
    """Clamp thresholds so that f_hi * (h / h_ref) <= cap and
        f_lo * (h / h_ref) >= 1 / cap. Returns (h, budget used), where used =
        log(f_hi * h / h_ref) / log(cap), NaN where h_ref is not finite.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    h_ref = np.asarray(h_ref, dtype=np.float64)
    ok = np.isfinite(h_ref) & np.isfinite(h) & (h_ref > 0)
    upper = h_ref * cap / f_hi
    lower = h_ref / (cap * f_lo)
    h[ok] = np.clip(h[ok], lower[ok], upper[ok])
    used = np.full(len(h), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        used[ok] = np.log(f_hi[ok] * h[ok] / h_ref[ok]) / np.log(cap)
    return h, used
