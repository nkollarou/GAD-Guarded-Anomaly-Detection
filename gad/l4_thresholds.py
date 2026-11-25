"""L4: thresholds that are re-estimated at each update, within a fixed limit of
their values at commissioning.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from gad.l3_cusum import EVT_MIN_PEAKS, EVT_TAIL_QUANTILE, _gpd_level


class AdaptiveThreshold:
    """Adaptive `h` for one branch's channels."""

    def __init__(
        self,
        h_ref: np.ndarray,
        calibration: list[np.ndarray],
        budget_time: float,
        budget_episodes_per_day: float,
        samples_per_day: float,
        cap: float = 2.0,
        period: float = 21_600.0,
        window: int = 345_600,
        tail_quantile: float = EVT_TAIL_QUANTILE,
        min_peaks: int = EVT_MIN_PEAKS,
    ) -> None:
        self.h_ref = np.asarray(h_ref, dtype=np.float64).copy()
        n = len(self.h_ref)
        if len(calibration) != n:
            raise ValueError("one calibration stream per channel")
        if cap < 1.0:
            raise ValueError("cap must be >= 1")
        self.h = self.h_ref.copy()
        self.budget_time = budget_time
        self.budget_eps = budget_episodes_per_day
        self.samples_per_day = samples_per_day
        self.cap = cap
        self.period = period
        self.window = window
        self.min_peaks = min_peaks
        self.u = np.full(n, np.nan)
        self._peaks = [deque() for _ in range(n)]    # (trusted index, peak)
        self._exceed = [deque() for _ in range(n)]   # (trusted index, excess)
        self._n_seen = np.zeros(n, dtype=np.int64)   # trusted samples so far
        self._run_max = np.full(n, np.nan)           # open excursion's max
        self._next_refresh: float | None = None
        self.n_refreshes = 0
        for j, ev in enumerate(calibration):
            ev = np.asarray(ev, dtype=np.float64)
            finite = ev[np.isfinite(ev)]
            if len(finite) and np.isfinite(self.h_ref[j]):
                self.u[j] = float(np.quantile(finite, tail_quantile))
                for e in ev:
                    self._observe_one(j, e, True)
                self._close_run(j)

    # -- learning -----------------------------------------------------------

    def _close_run(self, j: int) -> None:
        if np.isfinite(self._run_max[j]):
            self._peaks[j].append((int(self._n_seen[j]), float(self._run_max[j])))
            self._run_max[j] = np.nan

    def _observe_one(self, j: int, e: float, trusted: bool) -> None:
        if not trusted or not np.isfinite(e):
            self._close_run(j)  # an untrusted or abstaining sample ends a run
            return
        self._n_seen[j] += 1
        if e > self.u[j]:
            self._exceed[j].append((int(self._n_seen[j]), float(e - self.u[j])))
            self._run_max[j] = e if not np.isfinite(self._run_max[j]) else max(self._run_max[j], e)
        else:
            self._close_run(j)

    def observe(self, t: float, evidence: np.ndarray, trusted: np.ndarray) -> bool:
        """Feed one sample's evidence; `trusted` is per channel. Returns True if
        thresholds were refreshed on this sample.
        """
        for j in range(len(self.h)):
            if np.isfinite(self.u[j]):
                self._observe_one(j, float(evidence[j]), bool(trusted[j]))
        if self._next_refresh is None:
            self._next_refresh = t + self.period
            return False
        if t >= self._next_refresh:
            self.refresh()
            self._next_refresh = t + self.period
            return True
        return False

    # -- re-reading h ---------------------------------------------------------

    def _age_out(self, j: int) -> None:
        oldest = self._n_seen[j] - self.window
        for buf in (self._peaks[j], self._exceed[j]):
            while buf and buf[0][0] <= oldest:
                buf.popleft()

    def _estimate(self, j: int) -> float:
        self._age_out(j)
        n_window = min(int(self._n_seen[j]), self.window)
        peaks = np.array([p for _, p in self._peaks[j]])
        exceed = np.array([x for _, x in self._exceed[j]])
        if len(peaks) < self.min_peaks or len(exceed) < self.min_peaks or n_window == 0:
            return self.h_ref[j]
        days = n_window / self.samples_per_day
        target_ep = self.budget_eps / (len(peaks) / days)
        target_time = self.budget_time / (len(exceed) / n_window)
        try:
            h_ep = self.u[j]
            if target_ep < 1.0:
                h_ep = self.u[j] + _gpd_level(peaks - self.u[j], target_ep)[0]
            h_time = self.u[j]
            if target_time < 1.0:
                h_time = self.u[j] + _gpd_level(exceed, target_time)[0]
        except (ValueError, RuntimeError, FloatingPointError):
            return self.h_ref[j]
        h = max(h_ep, h_time)
        return float(h) if np.isfinite(h) else self.h_ref[j]

    def refresh(self) -> np.ndarray:
        for j in range(len(self.h)):
            if not np.isfinite(self.h_ref[j]) or not np.isfinite(self.u[j]):
                continue
            est = self._estimate(j)
            self.h[j] = float(np.clip(est, self.h_ref[j] / self.cap, self.h_ref[j] * self.cap))
        self.n_refreshes += 1
        return self.h

