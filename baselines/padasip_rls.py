"""Baseline: a recursive least squares filter (padasip) per sensor, predicting
it from the other sensors and the commands.
"""
from __future__ import annotations

import numpy as np

from baselines.common import PerPointCalibrationError, time_budget_threshold
from baselines.common import decide

MU = 2.0 ** (-1.0 / 43_200)
#: padasip FilterRLS's own default initialisation, R = I / eps
EPS = 0.001


class PadasipRLSMonitor:
    def __init__(self, sensed, commanded, mu: float = MU, eps: float = EPS,
                 dwell: int = 5, recover: int = 30) -> None:
        import padasip  # noqa: F401  fail now if missing
        self.sensed, self.commanded = list(sensed), list(commanded)
        self.mu, self.eps = float(mu), float(eps)
        self.dwell, self.recover = int(dwell), int(recover)
        self.filters: list = []
        self.inputs: list[np.ndarray] = []      # per sensor: input column indices
        self.thresholds: np.ndarray | None = None
        self.calibration: dict | None = None
        self.budget_met: bool | None = None
        self._mean = self._sd = None

    # --- inputs -------------------------------------------------------------

    def _rows(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        Z = np.hstack([np.asarray(X, float), np.asarray(U, float)])
        return (Z - self._mean) / self._sd

    def stream(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """|a-priori error| per sensor, (n, n_sensors); every filter adapts on
                every sample after it is scored.
        """
        Z = self._rows(X, U)
        X = np.asarray(X, float)
        E = np.empty((len(Z), len(self.sensed)))
        for i in range(len(Z)):
            for j, (f, cols) in enumerate(zip(self.filters, self.inputs)):
                x = np.append(Z[i, cols], 1.0)
                e = X[i, j] - f.predict(x)
                E[i, j] = abs(e)
                f.adapt(X[i, j], x)
        return E

    # --- commissioning ------------------------------------------------------

    def commission(self, X_fit, U_fit, X_warm, U_warm, X_cal, U_cal, *,
                   budget_time_fraction: float, samples_per_day: float = 86_400.0) -> None:
        import padasip as pa
        F = np.hstack([np.asarray(X_fit, float), np.asarray(U_fit, float)])
        self._mean = F.mean(axis=0)
        sd = F.std(axis=0)
        live = sd > 0
        self._sd = np.where(live, sd, 1.0)
        n_s = len(self.sensed)
        self.inputs = [np.array([k for k in range(F.shape[1]) if k != j and live[k]])
                       for j in range(n_s)]
        self.filters = [pa.filters.FilterRLS(len(c) + 1, mu=self.mu, eps=self.eps, w="zeros")
                        for c in self.inputs]
        self.stream(X_fit, U_fit)
        self.stream(X_warm, U_warm)
        E = self.stream(X_cal, U_cal)
        self.last_calibration_errors = E
        share = budget_time_fraction / n_s
        h, failed = np.full(n_s, np.inf), []
        for j, name in enumerate(self.sensed):
            try:
                h[j] = time_budget_threshold(E[:, j], budget_time_fraction=share,
                                             samples_per_day=samples_per_day,
                                             dwell=self.dwell, recover=self.recover)["threshold"]
            except PerPointCalibrationError:
                failed.append(name)
        self.thresholds = h
        self.calibration = {
            "rule": "time fraction, split equally across sensors (0160)",
            "mu": self.mu, "half_life_s": 43_200, "eps": self.eps,
            "n_inputs": {n: int(len(c) + 1) for n, c in zip(self.sensed, self.inputs)},
            "thresholds": {n: float(v) for n, v in zip(self.sensed, h)},
            "never_alarm": failed, "time_fraction": float(self.decide(E).mean()),
            "budget_time_fraction": float(budget_time_fraction),
            "dwell": self.dwell, "recover": self.recover, "n_clean_samples": int(len(E))}
        self.budget_met = True

    # --- deciding -----------------------------------------------------------

    def decide(self, E: np.ndarray) -> np.ndarray:
        out = np.zeros(len(E), dtype=bool)
        for j in range(E.shape[1]):
            if np.isfinite(self.thresholds[j]):
                out |= decide(E[:, j] > self.thresholds[j], dwell=self.dwell,
                              recover=self.recover)
        return out

    def summary_score(self, E: np.ndarray) -> np.ndarray:
        """max_j error_j / threshold_j: 1.0 is the alarm level."""
        h = np.where(np.isfinite(self.thresholds) & (self.thresholds > 0), self.thresholds, np.inf)
        return np.max(E / h, axis=1)
