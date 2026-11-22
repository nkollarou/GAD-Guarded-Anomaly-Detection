"""Baseline: MemStream (Bhatia et al., WWW 2022) with a one-day memory.

The model is ported from https://github.com/Stream-AD/MemStream (file
code/memstream.py, commit 59b5ba0), Copyright the MemStream authors,
licensed under the Apache License 2.0 (see LICENSE-MemStream). Modified:
wrapped as a streaming monitor with a calibrated threshold and an alarm
rule. The model itself is unchanged.
"""
from __future__ import annotations

import numpy as np

from baselines.common import PerPointCalibrationError, time_budget_threshold
from baselines.common import DWELL, RECOVER

MEMORY_LEN = 2048
MEMORY_LEN_DAY = 86_400
EPOCHS = 5000
LR = 1e-2
#: the input noise of the denoising autoencoder, as published
NOISE = 1e-3


def _torch():
    """torch, imported on use. Never stored on an instance: the benchmark
        deep-copies the commissioned monitor per cell (see rrcf_forest._rrcf).
    """
    import torch
    return torch


class MemStreamMonitor:
    """MemStream over the sensed channels."""

    def __init__(self, channels, memory_len: int = MEMORY_LEN,
                 epochs: int = EPOCHS, lr: float = LR, seed: int = 0) -> None:
        _torch()                      # fail now if the dependency is missing
        self.channels = list(channels)
        self.memory_len = int(memory_len)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.seed = int(seed)
        self.beta = float("inf")
        self.threshold: float | None = None
        self.budget_met: bool | None = None
        self.calibration: dict | None = None
        self.last_calibration_scores: np.ndarray | None = None
        self.n_admitted = 0
        self.n_scored = 0
        self.encoder = self.decoder = None
        self.memory = self.mem_data = self.mean = self.std = None
        self._count = 0

    # --- construction -------------------------------------------------------

    def fit(self, X: np.ndarray) -> None:
        torch = _torch()
        X = np.asarray(X, dtype=np.float32)
        if len(X) < self.memory_len:
            raise ValueError(f"need at least {self.memory_len} rows to fill the memory")
        init = torch.from_numpy(np.ascontiguousarray(X[-self.memory_len:]))
        d = init.shape[1]
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            self.encoder = torch.nn.Sequential(torch.nn.Linear(d, 2 * d), torch.nn.Tanh())
            self.decoder = torch.nn.Sequential(torch.nn.Linear(2 * d, d))
            self.mem_data = init.clone()
            self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
            new = self._normalise(init)
            params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            opt = torch.optim.Adam(params, lr=self.lr)
            loss_fn = torch.nn.MSELoss()
            with torch.enable_grad():
                for _ in range(self.epochs):
                    opt.zero_grad()
                    out = self.decoder(self.encoder(new + NOISE * torch.randn_like(new)))
                    loss = loss_fn(out, new)
                    loss.backward()
                    opt.step()
        with torch.no_grad():
            self.memory = self.encoder(self._normalise(self.mem_data))
        self._count = 0
        self._reset_moments()


    def _reset_moments(self):
        torch = _torch()
        m = self.mem_data.to(torch.float64)
        self._shift = m.mean(0)
        d = m - self._shift
        self._s1, self._s2 = d.sum(0), (d * d).sum(0)
        self._since_reset = 0
        self._publish_moments()

    def _publish_moments(self):
        torch = _torch()
        n = self.memory_len
        mean = self._shift + self._s1 / n
        var = (self._s2 - self._s1 * self._s1 / n) / (n - 1)
        tiny = var <= 1e-12 * (1.0 + mean * mean)
        std = torch.sqrt(torch.clamp(var, min=0.0))
        std[tiny] = 0.0
        self.mean, self.std = mean.to(torch.float32), std.to(torch.float32)

    def _normalise(self, x):
        new = (x - self.mean) / self.std
        new[..., self.std == 0] = 0
        return new

    # --- streaming ----------------------------------------------------------

    def score(self, x: np.ndarray) -> float:
        """Score one sample, then admit it to the memory if the score is at
                most beta. One call per second, in order.
        """
        if self.memory is None:
            raise RuntimeError("score() before fit()")
        torch = _torch()
        with torch.no_grad():
            xt = torch.from_numpy(np.asarray(x, dtype=np.float32).reshape(1, -1))
            z = self.encoder(self._normalise(xt))
            s = torch.cdist(z, self.memory, p=1).min()
            if s <= self.beta:
                pos = self._count % self.memory_len
                old = self.mem_data[pos].to(torch.float64) - self._shift
                new = xt[0].to(torch.float64) - self._shift
                self.memory[pos] = z[0]
                self.mem_data[pos] = xt[0]
                self._count += 1
                self.n_admitted += 1
                self._since_reset += 1
                if self._since_reset >= self.memory_len:
                    self._reset_moments()
                else:
                    self._s1 += new - old
                    self._s2 += new * new - old * old
                    self._publish_moments()
        self.n_scored += 1
        return float(s)

    def stream(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return np.array([self.score(X[i]) for i in range(len(X))])

    # --- the threshold, on OUR budget --------------------------------------

    def calibrate(self, X: np.ndarray, *, budget_time_fraction: float,
                  samples_per_day: float = 86_400.0, dwell: int = DWELL,
                  recover: int = RECOVER, beta_rule: str = "threshold") -> float:
        """Smallest threshold at which the clean window is in alarm for at most
                `budget_time_fraction` of the time, then beta by `beta_rule`:
        """
        self.beta = float("inf")
        scores = self.stream(X)
        self.last_calibration_scores = scores
        try:
            cal = time_budget_threshold(scores, budget_time_fraction=budget_time_fraction,
                                        samples_per_day=samples_per_day,
                                        dwell=dwell, recover=recover)
        except PerPointCalibrationError as exc:
            self.budget_met = False
            self.threshold = None
            self.calibration = {"rule": "time fraction only", "failed": True,
                                "budget_time_fraction": float(budget_time_fraction),
                                "n_clean_samples": int(len(scores)),
                                "score_max": float(np.max(scores))}
            raise MemStreamCalibrationError(str(exc)) from exc
        if beta_rule not in ("threshold", "clean_max"):
            raise ValueError(f"unknown beta_rule {beta_rule!r}")
        h = cal["threshold"]
        self.threshold = h
        self.beta = h if beta_rule == "threshold" else float(np.max(scores))
        self.budget_met = True
        cal.update({"rule": cal["rule"] + f"; beta = {beta_rule} (docs/decisions/0155)",
                    "beta_rule": beta_rule,
                    "beta": self.beta, "memory_len": self.memory_len,
                    "epochs": self.epochs, "lr": self.lr})
        self.calibration = cal
        return h


class MemStreamCalibrationError(RuntimeError):
    """No threshold meets the budget on the clean window (as RRCFCalibrationError)."""
