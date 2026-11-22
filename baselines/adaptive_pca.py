"""Baseline: AdaptivePCA from the process-improve package, learning at the
same 12-hour half-life as GAD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from process_improve.multivariate import AdaptivePCA
from process_improve.multivariate._limits import hotellings_t2_limit, spe_calculation

from gad.types import DetectorState

MATCHED_RATE = 1.0 - 2.0 ** (-1.0 / 43_200)
PROVISIONAL_CONF = 0.9999
VARIANCE_EXPLAINED = 0.90
EXCLUDED = ("FIT601",)


def n_components_for(X: np.ndarray, share: float = VARIANCE_EXPLAINED) -> int:
    """Smallest number of components explaining `share` of the variance of
        the standardised data.
    """
    sd = X.std(axis=0)
    Z = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    ev = np.linalg.eigvalsh(np.cov(Z, rowvar=False))[::-1]
    return int(np.searchsorted(np.cumsum(ev) / ev.sum(), share) + 1)


def decide(out_of_control: np.ndarray, dwell: int = 5, recover: int = 30) -> np.ndarray:
    """The same decider as ours (gad.l4_decision.MonitorDecider, one monitor):
        ANOMALY after `dwell` consecutive out-of-control samples, NORMAL again
        after `recover` consecutive in-control ones. Returns a bool ANOMALY array.
    """
    state = np.zeros(len(out_of_control), dtype=bool)
    anomaly, bad, good = False, 0, 0
    for i, o in enumerate(out_of_control):
        bad, good = (bad + 1, 0) if o else (0, good + 1)
        if anomaly:
            anomaly = good < recover
        else:
            anomaly = bad >= dwell
        state[i] = anomaly
    return state


def _episodes(a: np.ndarray) -> int:
    return int(np.sum(np.diff(np.concatenate(([0], a.astype(np.int8)))) == 1))


class AdaptivePCAMonitor:
    """`speed="matched"` (main run) or `"default"` (the package's own
        adaptation arguments, labelled secondary).
    """

    def __init__(self, sensed, speed: str = "matched", dwell: int = 5, recover: int = 30,
                 update_when_out_of_control: bool = False) -> None:
        """`update_when_out_of_control` is the package's own switch. Its
                default, False, is what the library recommends: a sample flagged out
                of control does not update the model, so a fault is not absorbed as
                normal. True lets it
                keep adapting and recover. Both are run and both are reported; neither
                is "the fixed version".
        """
        if speed not in ("matched", "default"):
            raise ValueError("speed must be 'matched' or 'default'")
        self.update_when_out_of_control = bool(update_when_out_of_control)
        self.sensed = [s for s in sensed if s not in EXCLUDED]
        self.cols = [i for i, s in enumerate(sensed) if s not in EXCLUDED]
        self.speed = speed
        self.dwell, self.recover = dwell, recover
        self.model: AdaptivePCA | None = None
        self.conf_level = PROVISIONAL_CONF
        self._bad = self._good = 0
        self._anomaly = False

    def fit(self, X_fit: np.ndarray) -> "AdaptivePCAMonitor":
        """Rows of all sensed channels; the excluded ones are dropped here."""
        X_fit = X_fit[:, self.cols]
        A = n_components_for(X_fit)
        kw = dict(n_components=A, conf_level=PROVISIONAL_CONF,
                  update_when_out_of_control=self.update_when_out_of_control)
        if self.speed == "matched":
            kw.update(forgetting_factor=MATCHED_RATE, lambda_center=MATCHED_RATE,
                      alpha_scale=MATCHED_RATE, adaptive_spe_limit=False)
        self.model = AdaptivePCA(**kw).fit(pd.DataFrame(X_fit, columns=self.sensed))
        # training SPE, as the package computes it, for re-deriving its SPE limit
        m = self.model
        Xs = np.nan_to_num((X_fit - m._mx0) / m._sx0)
        resid = Xs - (Xs @ m.loadings0_) @ m.loadings0_.T
        self._train_spe = np.sqrt(np.sum(resid ** 2, axis=1))
        self._n_fit = len(X_fit)
        return self

    def _set_limits(self, conf: float) -> None:
        m = self.model
        self.conf_level = conf
        m.conf_level = conf
        m._t2_limit_0 = hotellings_t2_limit(conf_level=conf, n_components=m.n_components_, n_rows=self._n_fit)
        m._spe_limit_0 = spe_calculation(self._train_spe, conf_level=conf)

    def stream(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Feed rows through `update` (it adapts as it goes); return T2 and SPE."""
        X = X[:, self.cols]
        t2, spe = np.empty(len(X)), np.empty(len(X))
        for i, x in enumerate(X):
            r = self.model.update(x)
            t2[i], spe[i] = r.hotellings_t2, r.spe
        return t2, spe

    def calibrate(self, X_cal: np.ndarray, budget_eps_day: float, budget_time: float,
                  samples_per_day: float = 86_400.0) -> float:
        """Stream the calibration rows at the provisional level, then choose the
                most sensitive confidence level whose alarms, through the decider,
                stay within the budget on those rows (bisection on -log10(1 - conf)).
        """
        t2, spe = self.stream(X_cal)
        m = self.model
        days = len(X_cal) / samples_per_day

        def within(q: float) -> bool:
            conf = 1.0 - 10.0 ** (-q)
            lim_t2 = hotellings_t2_limit(conf_level=conf, n_components=m.n_components_, n_rows=self._n_fit)
            lim_spe = spe_calculation(self._train_spe, conf_level=conf)
            a = decide((t2 > lim_t2) | (spe > lim_spe), self.dwell, self.recover)
            return _episodes(a) / days <= budget_eps_day and a.mean() <= budget_time

        lo, hi = 1.0, 15.0
        if not within(hi):
            raise ValueError("no confidence level up to 1 - 1e-15 meets the budget")
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if within(mid) else (mid, hi)
        self._set_limits(1.0 - 10.0 ** (-hi))
        return self.conf_level

    def step(self, x: np.ndarray) -> DetectorState:
        r = self.model.update(x[self.cols])
        o = not r.in_control
        self._bad, self._good = (self._bad + 1, 0) if o else (0, self._good + 1)
        self._anomaly = (self._good < self.recover) if self._anomaly else (self._bad >= self.dwell)
        return DetectorState.ANOMALY if self._anomaly else DetectorState.NORMAL
