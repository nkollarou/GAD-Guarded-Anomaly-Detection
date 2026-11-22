"""L1a: predicts each sensor with a sparse linear model of recent sensor
values and actuator commands.
"""
from __future__ import annotations

from collections import deque
import dataclasses
from dataclasses import dataclass, field
from typing import Iterable, Iterator

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
    """Maintains lagged sensed/commanded history and builds ARX feature
        vectors, one sample at a time. The sensed history buffer never
        contains the current sample.
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
        """Feature vector for predicting the CURRENT sensed reading. Takes
                `u_now` (= u[t], commanded lag 0 — legitimate, it's the controller's
                own log, not a measurement of x[t]) but no sensed argument at all:
                the sensed lags come only from history already pushed for prior
                timesteps.
        """
        if not self.ready():
            raise ValueError(f"need {self.L} samples of history before features() can be called")
        x_lags = np.concatenate(list(self._x_hist))  # x[t-1], ..., x[t-L]
        u_lags = np.concatenate([u_now] + list(self._u_hist)[: self.L - 1])  # u[t], ..., u[t-L+1]
        return np.concatenate([x_lags, u_lags])

    def push(self, x_now: np.ndarray, u_now: np.ndarray) -> None:
        """Record this step's observed x/u for future `features()` calls.
                Call this AFTER `features()` for the same timestep, never before.
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
    """Per (sensed channel, lag) elastic-net weight, shape (n_sensed, L):
        `w[c, 0] = 1` (lag 1, the baseline, never penalized beyond that), and
        for l=2..L, `w[c, l-1] = 1 / max(1 - autocorr_c(l-1), epsilon)` — how
        redundant lag l is *relative to that channel's own lag 1*, from `c`'s
        raw autocorrelation alone (independent of L1a's targets or fit).
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
    """Per sensed channel, Var(x[t] - x[t-1]) over exactly the same target
        rows `_build_lagged_batch(X, U, L)` produces (t = L..T-1) — one of the
        candidate baselines `fit_from_arrays` checks a fresh fit against.
    """
    targets = X[L:]
    persistence_pred = X[L - 1 : -1]
    return np.var(targets - persistence_pred, axis=0)


def velocity_residual_variance(X: np.ndarray, L: int) -> np.ndarray:
    """Per sensed channel, Var(x[t] - (2x[t-1] - x[t-2])) — linear
        extrapolation of the last observed step, over the same target rows as
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
    """Named trivial candidates' sparse coefficient patterns for channel
        `ch` — feature name -> coefficient, everything else implicitly 0, no
        intercept. `velocity` only exists when L>=2 (needs a lag-2 feature).
        `fitted` isn't listed here: it's whatever the elastic-net fit already
        produced, not a fixed pattern.
    """
    candidates = {"persistence": {f"{ch}[t-1]": 1.0}}
    if L >= 2:
        candidates["velocity"] = {f"{ch}[t-1]": 2.0, f"{ch}[t-2]": -1.0}
    return candidates


def _select_best_candidate(variances: dict[str, float]) -> str:
    return min(variances, key=lambda name: (variances[name], CANDIDATE_PREFERENCE.index(name)))


def lag_penalty_weights(autocorr_weights: np.ndarray, n_commanded: int, L: int) -> np.ndarray:
    """Flattens a (n_sensed, L) per-channel weight matrix (from
        `compute_autocorr_weights`) into the full feature-weight vector, in
        the same order as `feature_names`/the design matrix: lag 1 for every
        sensed channel, then lag 2 for every sensed channel, ..., then every
        commanded feature.
    """
    sensed_weights = autocorr_weights.T.flatten()  # (L, n_sensed) -> lag-major order
    commanded_weights = np.ones(n_commanded * L, dtype=np.float64)
    return np.concatenate([sensed_weights, commanded_weights])


def _chunks(T: int, L: int, chunk_size: int):
    """(lo, hi) row ranges covering targets L..T-1, so each chunk's lagged
        design matrix is transient — the same bounded-memory trick
        `fit_stats_from_arrays` uses.
    """
    t = L
    while t < T:
        hi = min(t + chunk_size, T)
        yield t, hi
        t = hi


def _build_lagged_batch(X: np.ndarray, U: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of streaming `ArxWindow` over a contiguous
        block `X`(T, n_sensed)/`U`(T, n_commanded) — used only for the
        commissioning fit's speed; the returned arrays are transient (folded
        into `SufficientStats` and discarded, never retained).
    """
    T = X.shape[0]
    if T <= L:
        raise ValueError(f"need more than L={L} rows to build any lagged example, got {T}")
    x_blocks = [X[L - lag : T - lag] for lag in range(1, L + 1)]
    u_blocks = [U[L - lag : T - lag] for lag in range(0, L)]
    features = np.concatenate(x_blocks + u_blocks, axis=1)
    targets = X[L:T]
    return features, targets


class _PooledVariance:
    """Var(r) per column, accumulated over blocks that must not be
        concatenated in memory. Var = E[r^2] - E[r]^2, the same quantity
        `np.var(..., axis=0)` reports for a single block.
    """

    def __init__(self, n: int) -> None:
        self.n = 0
        self.sum = np.zeros(n, dtype=np.float64)
        self.sum_sq = np.zeros(n, dtype=np.float64)

    def add(self, residual: np.ndarray) -> None:
        self.n += residual.shape[0]
        self.sum += residual.sum(axis=0)
        self.sum_sq += (residual * residual).sum(axis=0)

    def variance(self) -> np.ndarray:
        mean = self.sum / self.n
        return np.maximum(self.sum_sq / self.n - mean * mean, 0.0)


def fit_stats_from_arrays(
    X: np.ndarray, U: np.ndarray, L: int, chunk_size: int = 50_000
) -> SufficientStats:
    """Accumulate `SufficientStats` over `X`/`U` in chunks — bounded peak
        memory regardless of how long the stream is, and a real demonstration
        of 'recursive': each chunk's lagged design matrix is transient.
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
    """Deployed model state: coefficients, scalers, intercepts. Report its
        size separately from `SufficientStats.n_bytes()` (the adaptation
        working set) — never summed with it.
    """

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

    def n_bytes(self) -> int:
        arrays = (
            self.feature_mean,
            self.feature_std,
            self.target_mean,
            self.target_std,
            self.beta_raw,
            self.intercept_raw,
        )
        return sum(a.nbytes for a in arrays) + (self.cross.n_bytes() if self.cross is not None else 0)

    def beta_norms(self) -> dict[str, float]:
        """‖β‖ per channel — a certified sensitivity bound: a raw-unit
                perturbation δ to the inputs moves that channel's prediction by
                at most ‖β‖·‖δ‖.
        """
        return {
            ch: float(np.linalg.norm(self.beta_raw[:, j]))
            for j, ch in enumerate(self.sensed_channels)
        }


@dataclass(frozen=True)
class ConditioningReport:
    """Commissioning diagnostic — disclosure only. Nothing here filters or
        drops a channel; a channel flagged unidentifiable stays in the design
        matrix and simply gets (correctly) a near-zero coefficient.
    """

    raw_channel_distinct_values: dict[str, int]
    raw_channel_n_away_from_mode: dict[str, int]
    unidentifiable_channels: tuple[str, ...]
    near_duplicate_feature_pairs: tuple[tuple[str, str, float], ...]
    condition_number: float


def conditioning_diagnostic(
    X: np.ndarray,
    U: np.ndarray,
    sensed_channels: Iterable[str],
    commanded_channels: Iterable[str],
    L: int,
    gram_std: np.ndarray,
    feature_std: np.ndarray,
    min_effective_count: int = 30,
    duplicate_corr_threshold: float = 0.995,
) -> ConditioningReport:
    """3778 samples away from
        mode, for the nearest under/over cases) rather than an arbitrary
        round number alone.
    """
    sensed_channels = list(sensed_channels)
    commanded_channels = list(commanded_channels)

    distinct: dict[str, int] = {}
    n_away: dict[str, int] = {}
    unidentifiable = []
    for name, arr in [*zip(sensed_channels, X.T), *zip(commanded_channels, U.T)]:
        vals, counts = np.unique(arr, return_counts=True)
        distinct[name] = int(len(vals))
        away = int(len(arr) - counts.max())
        n_away[name] = away
        if away < min_effective_count:
            unidentifiable.append(name)

    names = feature_names(sensed_channels, commanded_channels, L)
    p = len(names)
    valid = feature_std > 0

    dup_pairs = []
    for i in range(p):
        if not valid[i]:
            continue
        for j in range(i + 1, p):
            if not valid[j]:
                continue
            corr = gram_std[i, j]
            if abs(corr) > duplicate_corr_threshold:
                dup_pairs.append((names[i], names[j], float(corr)))

    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) > 0:
        sub = gram_std[np.ix_(valid_idx, valid_idx)]
        eigvals = np.linalg.eigvalsh(sub)
        eigvals = np.abs(eigvals)
        condition_number = float(eigvals.max() / max(eigvals.min(), 1e-12))
    else:
        condition_number = float("nan")

    return ConditioningReport(
        raw_channel_distinct_values=distinct,
        raw_channel_n_away_from_mode=n_away,
        unidentifiable_channels=tuple(unidentifiable),
        near_duplicate_feature_pairs=tuple(dup_pairs),
        condition_number=condition_number,
    )


def predict_batch(state: ModelState, X: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized evaluation of a fitted `ModelState` over a contiguous
        block `X`/`U` — for sweep scoring and held-out gate metrics, where
        looping the streaming `update()` path per sample would be needlessly
        slow. Returns (predictions, targets), each (T-L, n_sensed).
    """
    features, targets = _build_lagged_batch(X, U, state.L)
    predictions = state.intercept_raw + features @ state.beta_raw
    return predictions, targets


DEFAULT_L = 10
DEFAULT_LAMBDA2 = 0.1


class RelationalPredictor:
    """L1a: implements the `Predictor` protocol (fit/update/errors_batch)."""

    def __init__(
        self,
        sensed_channels: Iterable[str],
        commanded_channels: Iterable[str],
        L: int = DEFAULT_L,
        lambda2: float = DEFAULT_LAMBDA2,
        lambda1: float | None = None,
        cross_channel: bool = False,
        cross_rule: str = "0044",
        own_lag_gate: float | None = None,
        gate_max_hold: int | None = None,
    ) -> None:
        """The prediction path needs no flag:
                it follows whatever the fitted state carries.
        """
        self.cross_channel = cross_channel
        self.cross_rule = cross_rule
        self.own_lag_gate = own_lag_gate
        self.gate_max_hold = gate_max_hold
        self._shadow = None
        self._hold_run = None
        self._gate_counts = None
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
        """Fit from raw arrays directly (fast path for commissioning/
                sweeps — avoids materializing `Sample` objects). Returns the
                `SufficientStats` used, so a caller can report the adaptation
                working set size.
        """
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

        cross = None
        if self.cross_channel:
            cross = fit_cross(X, U, lambda2=self.lambda2,
                              sensed=self.sensed_channels, rule=self.cross_rule)
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

    def fit_from_segments(self, segments, chunk_size: int = 50_000) -> SufficientStats:
        """Fit from several *contiguous* blocks that are not contiguous with
                each other — what quarantine leaves once flagged and non-NORMAL
                stretches are removed.
        """
        if self.cross_channel:
            raise NotImplementedError("docs/decisions/0044 is implemented for contiguous fits only")
        segments = [(np.asarray(X, dtype=np.float64), np.asarray(U, dtype=np.float64))
                    for X, U in segments]
        if not segments:
            raise ValueError("fit_from_segments requires at least one segment")
        for X, U in segments:
            if X.shape[0] != U.shape[0]:
                raise ValueError("each segment's X and U must have the same number of rows")
            if X.shape[0] <= self.L:
                raise ValueError(f"a segment of {X.shape[0]} rows is too short for L={self.L}")
            if X.shape[1] != self.n_sensed or U.shape[1] != self.n_commanded:
                raise ValueError("segment channel counts do not match this predictor")

        n_features = self.n_sensed * self.L + self.n_commanded * self.L
        stats = SufficientStats(n_features, self.n_sensed)
        weight_sum = np.zeros((self.n_sensed, self.L), dtype=np.float64)
        rows = 0
        for X, U in segments:
            for lo, hi in _chunks(len(X), self.L, chunk_size):
                features, targets = _build_lagged_batch(X[lo - self.L:hi], U[lo - self.L:hi], self.L)
                stats.add_batch(features, targets)
            weight_sum += len(X) * compute_autocorr_weights(X, self.L)
            rows += len(X)

        lambda1 = self._lambda1_override if self._lambda1_override is not None else default_lambda1(
            stats.n_features, stats.n
        )
        fm, fs, tm, ts, gram_std, cross_std = standardize_gram_cross(stats)
        weights = lag_penalty_weights(weight_sum / rows, self.n_commanded, self.L)
        beta_std = fit_elastic_net_multi_from_gram(gram_std, cross_std, lambda1, self.lambda2, weights=weights)
        beta_raw, intercept_raw = destandardize_coefficients(beta_std, fm, fs, tm, ts)

        # candidate competition, pooled over segments on identical rows
        fitted = _PooledVariance(self.n_sensed)
        persistence = _PooledVariance(self.n_sensed)
        velocity = _PooledVariance(self.n_sensed) if self.L >= 2 else None
        for X, U in segments:
            for lo, hi in _chunks(len(X), self.L, chunk_size):
                features, targets = _build_lagged_batch(X[lo - self.L:hi], U[lo - self.L:hi], self.L)
                fitted.add(intercept_raw + features @ beta_raw - targets)
            persistence.add(X[self.L:] - X[self.L - 1:-1])
            if velocity is not None:
                velocity.add(X[self.L:] - (2 * X[self.L - 1:-1] - X[self.L - 2:-2]))

        names = feature_names(self.sensed_channels, self.commanded_channels, self.L)
        fitted_var, persistence_var = fitted.variance(), persistence.variance()
        velocity_var = velocity.variance() if velocity is not None else None
        model_choice: dict[str, str] = {}
        residual_variance = np.empty(self.n_sensed)
        for j, ch in enumerate(self.sensed_channels):
            variances = {"fitted": float(fitted_var[j]), "persistence": float(persistence_var[j])}
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

        self.state = ModelState(
            L=self.L, lambda1=lambda1, lambda2=self.lambda2,
            sensed_channels=self.sensed_channels, commanded_channels=self.commanded_channels,
            feature_mean=fm, feature_std=fs, target_mean=tm, target_std=ts,
            beta_raw=beta_raw, intercept_raw=intercept_raw,
            model_choice=model_choice, residual_variance=residual_variance,
        )
        return stats

    def fit(self, samples: Iterable[Sample]) -> None:
        samples = list(samples)
        if not samples:
            raise ValueError("fit requires at least one sample")
        X = np.stack([s.x for s in samples])
        U = np.stack([s.u for s in samples])
        self.fit_from_arrays(X, U)

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
            if self.own_lag_gate is not None:
                x_hat = x_hat + self._own_lag_correction()

        cross = self.state.cross
        if cross is not None and cross.channels:
            idx = list(cross.channels)
            x_now = np.asarray(sample.x, dtype=np.float64)
            if self._prev_x is None:
                x_hat[idx] = np.nan
            else:
                x_hat[idx] = cross_predict(cross, x_now, self._prev_x, np.asarray(sample.u, dtype=np.float64))
        self._prev_x = np.array(sample.x, dtype=np.float64)

        if self.own_lag_gate is not None:
            self._update_gate(np.asarray(sample.x, dtype=np.float64), x_hat)

        self._window.push(sample.x, sample.u)
        return x_hat


    def reset_stream(self) -> None:
        """Drop every streaming-time variable, keeping the fitted model."""
        self._window = ArxWindow(self.n_sensed, self.n_commanded, self.L)
        self._prev_x = None
        self._shadow = None
        self._hold_run = None
        self._gate_counts = None
        self.last_features = None

    def _gateable(self) -> np.ndarray:
        """Which channels the gate may touch: those the plain ARX predicts.
                A cross-predicted channel's own columns are already zero, so holding
                its lag back would change nothing and claiming otherwise would be
                false.
        """
        mask = np.ones(self.n_sensed, dtype=bool)
        cross = self.state.cross if self.state is not None else None
        if cross is not None and cross.channels:
            mask[list(cross.channels)] = False
        return mask

    def _own_lag_idx(self, j: int) -> list[int]:
        """Where channel j's own lags sit in the feature vector, which is
                `[x[t-1] (n_sensed), x[t-2], ..., x[t-L], u...]`.
        """
        return [lag * self.n_sensed + j for lag in range(self._window.L)]

    def _own_lag_correction(self) -> np.ndarray:
        """The gated prediction, as a rank-1 correction rather than a second
                feature vector: the model is linear, so substituting the shadow for
                the measurement in channel j's own lags shifts only j's prediction,
                by `sum over lags of beta[own_lag, j] * (shadow_j - x_j[t-1-lag])`.
        """
        out = np.zeros(self.n_sensed, dtype=np.float64)
        if self._shadow is None or not self._window.ready():
            return out
        hist = list(self._window._x_hist)   # x[t-1], ..., x[t-L]
        beta = self.state.beta_raw
        for j in np.flatnonzero(np.isfinite(self._shadow)):
            j = int(j)
            for lag, idx in enumerate(self._own_lag_idx(j)):
                out[j] += beta[idx, j] * (self._shadow[j] - hist[lag][j])
        return out

    def _update_gate(self, x_now: np.ndarray, x_hat: np.ndarray) -> None:
        """Open the gate on a channel whose residual its own fit says it
                cannot be seeing; release it when the residual returns, or when the
                hold bound expires.
        """
        n = self.n_sensed
        if self._shadow is None:
            self._shadow = np.full(n, np.nan)
            self._hold_run = np.zeros(n, dtype=np.int64)
            self._gate_counts = {"held": np.zeros(n, dtype=np.int64),
                                 "forced": np.zeros(n, dtype=np.int64),
                                 "seen": 0}
        self._gate_counts["seen"] += 1
        var = self.state.residual_variance
        if var is None:
            return
        sd = np.sqrt(np.maximum(np.asarray(var, dtype=float), 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.abs(x_now - x_hat) / sd
        g = np.asarray(self.own_lag_gate, dtype=float)
        over = np.isfinite(z) & (z > g) & self._gateable()
        if self.gate_max_hold is not None:
            expired = (self._hold_run >= int(self.gate_max_hold))
            self._gate_counts["forced"] += over & expired
            over = over & ~expired
        held = np.isfinite(self._shadow)
        entering = over & ~held
        self._shadow = np.where(entering, x_hat, self._shadow)
        self._shadow = np.where(over, self._shadow, np.nan)
        self._hold_run = np.where(over, self._hold_run + 1, 0)
        self._gate_counts["held"] += over

    def gate_report(self) -> dict:
        """What the gate did, per channel. Holding a clean measurement back
                costs prediction accuracy, so the rate is reported rather than
                assumed negligible.
        """
        if self._gate_counts is None or not self._gate_counts["seen"]:
            return {"gate": self.own_lag_gate, "max_hold": self.gate_max_hold,
                    "n_seen": 0, "held": {}}
        n_seen = int(self._gate_counts["seen"])
        names = self.sensed_channels
        g = np.asarray(self.own_lag_gate, dtype=float)
        return {
            "gate": (float(g) if g.ndim == 0 else
                     {names[j]: float(v) for j, v in enumerate(g)
                      if np.isfinite(v)}),
            "max_hold": self.gate_max_hold,
            "n_seen": n_seen,
            "gateable": [names[j] for j in np.flatnonzero(self._gateable())],
            "held": {names[j]: int(c) for j, c in
                     enumerate(self._gate_counts["held"]) if c},
            "held_fraction": {names[j]: float(c / n_seen) for j, c in
                              enumerate(self._gate_counts["held"]) if c},
            "forced_release": {names[j]: int(c) for j, c in
                               enumerate(self._gate_counts["forced"]) if c},
        }

    def errors_batch(self, samples: Iterable[Sample]) -> Iterator[np.ndarray]:
        for sample in samples:
            x_hat = self.update(sample)
            yield sample.x - x_hat


GATE_SHARE_TOLERANCE = 0.10   # accept a realised share within 10% of target


def calibrate_own_lag_gate(
    predictor: "RelationalPredictor",
    X: np.ndarray,
    U: np.ndarray,
    target_share: float,
    *,
    max_passes: int = 12,
    lo: float = 0.5,
    hi: float = 1024.0,
) -> np.ndarray:
    """Per-channel gate thresholds such that the gate holds at most
        `target_share` of CLEAN samples on each gateable channel.
    """
    if not 0.0 < target_share < 1.0:
        raise ValueError(f"target_share must be in (0, 1), got {target_share}")
    if predictor.state is None:
        raise RuntimeError("calibrate_own_lag_gate requires a fitted predictor")
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    if len(X) != len(U):
        raise ValueError(f"X has {len(X)} rows and U has {len(U)}")

    n = predictor.n_sensed
    gateable = predictor._gateable()
    saved = predictor.own_lag_gate
    q = np.zeros(n, dtype=np.uint8)

    def realised(g: np.ndarray) -> np.ndarray:
        predictor.own_lag_gate = g
        predictor.reset_stream()
        for t in range(len(X)):
            predictor.update(Sample(t=float(t), x=X[t], u=U[t], q=q))
        counts = predictor._gate_counts
        if counts is None or not counts["seen"]:
            return np.zeros(n)
        return counts["held"] / float(counts["seen"])

    log_lo = np.full(n, np.log(lo))
    log_hi = np.full(n, np.log(hi))
    shares = np.zeros(n)
    try:
        for _ in range(int(max_passes)):
            mid = 0.5 * (log_lo + log_hi)
            shares = realised(np.exp(mid))
            too_much = shares > target_share
            log_lo = np.where(too_much, mid, log_lo)   # raise the floor
            log_hi = np.where(too_much, log_hi, mid)   # lower the ceiling
    finally:
        predictor.own_lag_gate = saved
        predictor.reset_stream()

    out = np.exp(log_hi)            # the end of the bracket that MEETS the target
    out[~gateable] = np.inf
    return out
