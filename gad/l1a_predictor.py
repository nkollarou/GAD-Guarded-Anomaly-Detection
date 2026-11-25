"""L1a: predicts each sensor with a sparse linear model of recent sensor values
and actuator commands.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from gad.l1a_fit import (
    SufficientStats,
    default_lambda1,
    destandardize_coefficients,
    fit_elastic_net_multi_from_gram,
    standardize_gram_cross,
)
from gad.l1a_cross_sensor import CrossState, fit_cross, predict as cross_predict
from gad.types import Sample


class ArxWindow:
    """Maintains lagged sensed/commanded history and builds ARX feature vectors,
    one sample at a time. The sensed history buffer never contains the current
    sample.
    """

    def __init__(self, n_sensed: int, n_commanded: int, L: int) -> None:
        if L < 1:
            raise ValueError("L must be >= 1")
        self.n_sensed = n_sensed
        self.n_commanded = n_commanded
        self.L = L
        self.n_features = n_sensed * L + n_commanded * L
        self._x_hist: deque[np.ndarray] = deque(maxlen=L)  # x[t-1], x[t-2], ... most-recent-first
        self._u_hist: deque[np.ndarray] = deque(maxlen=L)  # u[t-1], u[t-2], ... most-recent-first

    def ready(self) -> bool:
        return len(self._x_hist) == self.L

    def features(self, u_now: np.ndarray) -> np.ndarray:
        """Feature vector for predicting the CURRENT sensed reading. Takes `u_now`
        (= u[t], commanded lag 0 — legitimate, it's the controller's own log,
        not a measurement of x[t]) but no sensed argument at all: the sensed
        lags come only from history already pushed for prior timesteps.
        """
        if not self.ready():
            raise ValueError(f"need {self.L} samples of history before features() can be called")
        x_lags = np.concatenate(list(self._x_hist))  # x[t-1], ..., x[t-L]
        u_lags = np.concatenate([u_now] + list(self._u_hist)[: self.L - 1])  # u[t], ..., u[t-L+1]
        return np.concatenate([x_lags, u_lags])

    def push(self, x_now: np.ndarray, u_now: np.ndarray) -> None:
        """Record this step's observed x/u for future `features()` calls. Call
        this AFTER `features()` for the same timestep, never before.
        """
        self._x_hist.appendleft(x_now)
        self._u_hist.appendleft(u_now)


def feature_names(sensed_channels: Iterable[str], commanded_channels: Iterable[str], L: int) -> list[str]:
    sensed_channels = list(sensed_channels)
    commanded_channels = list(commanded_channels)
    names = []
    for lag in range(1, L + 1):
        names.extend(f"{c}[t-{lag}]" for c in sensed_channels)
    for lag in range(0, L):
        suffix = "[t]" if lag == 0 else f"[t-{lag}]"
        names.extend(f"{c}{suffix}" for c in commanded_channels)
    return names


def _raw_autocorr(series: np.ndarray, lag: int) -> float:
    if lag == 0:
        return 1.0
    a, b = series[lag:], series[:-lag]
    if np.std(a) == 0 or np.std(b) == 0:
        return 1.0  # a constant channel is trivially "perfectly redundant" with itself
    return float(np.corrcoef(a, b)[0, 1])


def compute_autocorr_weights(X: np.ndarray, L: int, epsilon: float = 1e-3) -> np.ndarray:
    """Per (sensed channel, lag) elastic-net weight, shape (n_sensed, L): `w[c, 0]
    = 1` (lag 1, the baseline, never penalized beyond that), and for l=2..L,
    `w[c, l-1] = 1 / max(1 - autocorr_c(l-1), epsilon)` — how redundant lag l
    is *relative to that channel's own lag 1*, from `c`'s raw autocorrelation
    alone (independent of L1a's targets or fit).
    """
    n_sensed = X.shape[1]
    weights = np.ones((n_sensed, L), dtype=np.float64)
    for c in range(n_sensed):
        series = X[:, c]
        for l in range(2, L + 1):
            redundancy = _raw_autocorr(series, l - 1)
            weights[c, l - 1] = 1.0 / max(1.0 - redundancy, epsilon)
    return weights


def persistence_residual_variance(X: np.ndarray, L: int) -> np.ndarray:
    """Per sensed channel, Var(x[t] - x[t-1]) over exactly the same target rows
    `_build_lagged_batch(X, U, L)` produces (t = L..T-1) — one of the candidate
    baselines `fit_from_arrays` checks a fresh fit against.
    """
    targets = X[L:]
    persistence_pred = X[L - 1 : -1]
    return np.var(targets - persistence_pred, axis=0)


def velocity_residual_variance(X: np.ndarray, L: int) -> np.ndarray:
    """Per sensed channel, Var(x[t] - (2x[t-1] - x[t-2])) — linear extrapolation
    of the last observed step, over the same target rows as
    `persistence_residual_variance`. Requires L>=2 (needs a lag-2 value);
    callers must not call this for L<2.
    """
    if L < 2:
        raise ValueError("velocity_residual_variance requires L >= 2")
    targets = X[L:]
    lag1 = X[L - 1 : -1]
    lag2 = X[L - 2 : -2]
    velocity_pred = 2 * lag1 - lag2
    return np.var(targets - velocity_pred, axis=0)


CANDIDATE_PREFERENCE = ("persistence", "velocity", "fitted")


def candidate_coefficients(ch: str, L: int) -> dict[str, dict[str, float]]:
    """Named trivial candidates' sparse coefficient patterns for channel `ch` —
    feature name -> coefficient, everything else implicitly 0, no intercept.
    `velocity` only exists when L>=2 (needs a lag-2 feature). `fitted` isn't
    listed here: it's whatever the elastic-net fit already produced, not a
    fixed pattern.
    """
    candidates = {"persistence": {f"{ch}[t-1]": 1.0}}
    if L >= 2:
        candidates["velocity"] = {f"{ch}[t-1]": 2.0, f"{ch}[t-2]": -1.0}
    return candidates


def _select_best_candidate(variances: dict[str, float]) -> str:
    return min(variances, key=lambda name: (variances[name], CANDIDATE_PREFERENCE.index(name)))


def lag_penalty_weights(autocorr_weights: np.ndarray, n_commanded: int, L: int) -> np.ndarray:
    """Flattens a (n_sensed, L) per-channel weight matrix (from
    `compute_autocorr_weights`) into the full feature-weight vector, in the
    same order as `feature_names`/the design matrix: lag 1 for every sensed
    channel, then lag 2 for every sensed channel, ..., then every commanded
    feature.
    """
    sensed_weights = autocorr_weights.T.flatten()  # (L, n_sensed) -> lag-major order
    commanded_weights = np.ones(n_commanded * L, dtype=np.float64)
    return np.concatenate([sensed_weights, commanded_weights])


def _build_lagged_batch(X: np.ndarray, U: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of streaming `ArxWindow` over a contiguous block
    `X`(T, n_sensed)/`U`(T, n_commanded) — used only for the commissioning
    fit's speed; the returned arrays are transient (folded into
    `SufficientStats` and discarded, never retained).
    """
    T = X.shape[0]
    if T <= L:
        raise ValueError(f"need more than L={L} rows to build any lagged example, got {T}")
    x_blocks = [X[L - lag : T - lag] for lag in range(1, L + 1)]
    u_blocks = [U[L - lag : T - lag] for lag in range(0, L)]
    features = np.concatenate(x_blocks + u_blocks, axis=1)
    targets = X[L:T]
    return features, targets


def fit_stats_from_arrays(
    X: np.ndarray, U: np.ndarray, L: int, chunk_size: int = 50_000
) -> SufficientStats:
    """Accumulate `SufficientStats` over `X`/`U` in chunks — bounded peak memory
    regardless of how long the stream is, and a real demonstration of
    'recursive': each chunk's lagged design matrix is transient.
    """
    T, n_sensed = X.shape
    n_commanded = U.shape[1]
    n_features = n_sensed * L + n_commanded * L
    stats = SufficientStats(n_features, n_sensed)

    t = L
    while t < T:
        t_hi = min(t + chunk_size, T)
        features, targets = _build_lagged_batch(X[t - L : t_hi], U[t - L : t_hi], L)
        stats.add_batch(features, targets)
        t = t_hi
    return stats


@dataclass(frozen=True)
class ModelState:
    """The fitted L1a model: coefficients, scalers and intercepts."""

    L: int
    lambda1: float
    lambda2: float
    sensed_channels: tuple[str, ...]
    commanded_channels: tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    beta_raw: np.ndarray       # (n_features, n_sensed)
    intercept_raw: np.ndarray  # (n_sensed,)
    model_choice: dict[str, str] = field(default_factory=dict)
    residual_variance: np.ndarray | None = None
    cross: CrossState | None = None


DEFAULT_L = 10
DEFAULT_LAMBDA2 = 0.1


class RelationalPredictor:
    """L1a: predicts every sensor at each sample. Sensors with a stable relation
    to the others are predicted from them only (`l1a_cross_sensor`); the others
    from a sparse linear model of recent values.
    """

    def __init__(
        self,
        sensed_channels: Iterable[str],
        commanded_channels: Iterable[str],
        L: int = DEFAULT_L,
        lambda2: float = DEFAULT_LAMBDA2,
        lambda1: float | None = None,
    ) -> None:
        self._prev_x: np.ndarray | None = None
        self.sensed_channels = tuple(sensed_channels)
        self.commanded_channels = tuple(commanded_channels)
        self.L = L
        self.lambda2 = lambda2
        self._lambda1_override = lambda1
        self.n_sensed = len(self.sensed_channels)
        self.n_commanded = len(self.commanded_channels)
        self.state: ModelState | None = None
        self._window = ArxWindow(self.n_sensed, self.n_commanded, L)
        self.last_features: np.ndarray | None = None

    def fit_from_arrays(self, X: np.ndarray, U: np.ndarray) -> SufficientStats:
        """Fit on the clean rows `X` (sensors) and `U` (commands)."""
        stats = fit_stats_from_arrays(X, U, self.L)
        lambda1 = self._lambda1_override if self._lambda1_override is not None else default_lambda1(
            stats.n_features, stats.n
        )

        fm, fs, tm, ts, gram_std, cross_std = standardize_gram_cross(stats)
        autocorr_weights = compute_autocorr_weights(X, self.L)
        weights = lag_penalty_weights(autocorr_weights, self.n_commanded, self.L)
        beta_std = fit_elastic_net_multi_from_gram(gram_std, cross_std, lambda1, self.lambda2, weights=weights)
        beta_raw, intercept_raw = destandardize_coefficients(beta_std, fm, fs, tm, ts)

        features, targets = _build_lagged_batch(X, U, self.L)
        fitted_pred = intercept_raw + features @ beta_raw
        fitted_resid_var = np.var(fitted_pred - targets, axis=0)
        persistence_var = persistence_residual_variance(X, self.L)
        velocity_var = velocity_residual_variance(X, self.L) if self.L >= 2 else None

        names = feature_names(self.sensed_channels, self.commanded_channels, self.L)
        model_choice: dict[str, str] = {}
        residual_variance = np.empty(self.n_sensed)
        for j, ch in enumerate(self.sensed_channels):
            variances = {"fitted": float(fitted_resid_var[j]), "persistence": float(persistence_var[j])}
            if velocity_var is not None:
                variances["velocity"] = float(velocity_var[j])
            choice = _select_best_candidate(variances)
            model_choice[ch] = choice
            residual_variance[j] = variances[choice]

            if choice != "fitted":
                beta_raw[:, j] = 0.0
                for feat_name, coef in candidate_coefficients(ch, self.L)[choice].items():
                    beta_raw[names.index(feat_name), j] = coef
                intercept_raw[j] = 0.0

        cross = fit_cross(X, U, lambda2=self.lambda2)
        for j in cross.channels:
            model_choice[self.sensed_channels[j]] = "cross"

        self.state = ModelState(
            cross=cross,
            L=self.L,
            lambda1=lambda1,
            lambda2=self.lambda2,
            sensed_channels=self.sensed_channels,
            commanded_channels=self.commanded_channels,
            feature_mean=fm,
            feature_std=fs,
            target_mean=tm,
            target_std=ts,
            beta_raw=beta_raw,
            intercept_raw=intercept_raw,
            model_choice=model_choice,
            residual_variance=residual_variance,
        )
        return stats

    def update(self, sample: Sample) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("update() called before fit()")

        if not self._window.ready():
            x_hat = np.full(self.n_sensed, np.nan)
            self.last_features = None
        else:
            features = self._window.features(sample.u)
            x_hat = self.state.intercept_raw + features @ self.state.beta_raw
            self.last_features = features

        cross = self.state.cross
        if cross.channels:
            idx = list(cross.channels)
            x_now = np.asarray(sample.x, dtype=np.float64)
            if self._prev_x is None:
                x_hat[idx] = np.nan
            else:
                x_hat[idx] = cross_predict(cross, x_now, self._prev_x, np.asarray(sample.u, dtype=np.float64))
        self._prev_x = np.array(sample.x, dtype=np.float64)

        self._window.push(sample.x, sample.u)
        return x_hat


