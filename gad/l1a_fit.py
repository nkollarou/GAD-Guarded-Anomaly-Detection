"""L1a: fits the elastic-net models from summary statistics of the data."""
from __future__ import annotations


import numpy as np


class SufficientStats:
    """Accumulates uncentered sums for a linear regression with `n_features`
    inputs and `n_targets` outputs. Centering/standardization is derived
    analytically from these sums when fitting — never by revisiting raw data.
    """

    def __init__(self, n_features: int, n_targets: int) -> None:
        self.n_features = n_features
        self.n_targets = n_targets
        self.n = 0
        self.sum_x = np.zeros(n_features, dtype=np.float64)
        self.sum_y = np.zeros(n_targets, dtype=np.float64)
        self.sum_yy = np.zeros(n_targets, dtype=np.float64)
        self.sum_xx = np.zeros((n_features, n_features), dtype=np.float64)
        self.sum_xy = np.zeros((n_features, n_targets), dtype=np.float64)

    def add_batch(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fold in a batch (rows = samples) in one vectorized step. `X`/`Y` are
        transient — nothing here retains them after this call returns.
        """
        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must be 2-D with matching row counts")
        if X.shape[1] != self.n_features or Y.shape[1] != self.n_targets:
            raise ValueError("X/Y column count does not match this SufficientStats")
        self.n += X.shape[0]
        self.sum_x += X.sum(axis=0)
        self.sum_y += Y.sum(axis=0)
        self.sum_yy += (Y * Y).sum(axis=0)
        self.sum_xx += X.T @ X
        self.sum_xy += X.T @ Y

    @property
    def mean_x(self) -> np.ndarray:
        return self.sum_x / self.n

    @property
    def mean_y(self) -> np.ndarray:
        return self.sum_y / self.n

    @property
    def var_x(self) -> np.ndarray:
        """Per-feature variance, from the diagonal of the centered Gram."""
        return np.diag(self.centered_gram()) / self.n

    @property
    def var_y(self) -> np.ndarray:
        """Per-target variance: sum_yy/n - mean_y^2."""
        return self.sum_yy / self.n - self.mean_y**2

    def centered_gram(self) -> np.ndarray:
        """n * Cov(x): sum((x - mean_x)(x - mean_x)^T)."""
        mx = self.mean_x
        return self.sum_xx - self.n * np.outer(mx, mx)

    def centered_cross(self) -> np.ndarray:
        """n * Cov(x, y): sum((x - mean_x)(y - mean_y)^T), shape (n_features,
        n_targets).
        """
        mx, my = self.mean_x, self.mean_y
        return self.sum_xy - self.n * np.outer(mx, my)


def _soft_threshold(z: np.ndarray, gamma: float) -> np.ndarray:
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)


def fit_elastic_net_multi_from_gram(
    gram: np.ndarray,
    cross: np.ndarray,
    lambda1: float,
    lambda2: float,
    max_iter: int = 1000,
    tol: float = 1e-7,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Coordinate descent for elastic net, operating only on a (mean) Gram matrix
    `gram` = (1/n) XᵀX and cross matrix `cross` = (1/n) XᵀY — never the raw
    design matrix. `cross` is (p, n_targets); every target shares `gram`, so
    each coordinate step updates all targets at once (one (n_targets,)-vector
    operation per feature, not `n_targets` separate scalar ones) — ~n_targets
    times fewer Python-level iterations than solving each target independently,
    for an identical result (coordinates are independent across targets given a
    shared Gram). Objective, per target column y:
    """
    if lambda2 <= 0:
        raise ValueError("lambda2 must be > 0 (guards zero-variance features from a zero denominator)")

    p, n_targets = cross.shape
    if weights is None:
        weights = np.ones(p, dtype=np.float64)
    elif weights.shape != (p,):
        raise ValueError(f"weights must have shape ({p},), got {weights.shape}")
    elif np.any(weights <= 0):
        raise ValueError("weights must be strictly positive")

    beta = np.zeros((p, n_targets), dtype=np.float64)
    diag = np.diag(gram)
    denom = diag + lambda2 * weights
    l1_penalty = lambda1 * weights

    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            rho_j = cross[j] - gram[j] @ beta + gram[j, j] * beta[j]
            beta[j] = _soft_threshold(rho_j, l1_penalty[j]) / denom[j]
        if np.max(np.abs(beta - beta_old)) < tol:
            break

    return beta


def fit_elastic_net_from_gram(
    gram: np.ndarray,
    cross: np.ndarray,
    lambda1: float,
    lambda2: float,
    max_iter: int = 1000,
    tol: float = 1e-7,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Single-target convenience wrapper around `fit_elastic_net_multi_from_gram`
    (`cross` is (p,), returns (p,)).
    """
    beta = fit_elastic_net_multi_from_gram(gram, cross[:, None], lambda1, lambda2, max_iter, tol, weights)
    return beta[:, 0]


def default_lambda1(n_features: int, n: int) -> float:
    """The 'universal' sparsity penalty sqrt(2 log(p) / n) (Bickel-Ritov-
    Tsybakov), for elastic net fit on features AND target standardized to unit
    variance — so this value is unitless and shared across every per-channel
    model regardless of the channel's physical units.
    """
    return float(np.sqrt(2.0 * np.log(n_features) / n))


def standardize_gram_cross(
    stats: SufficientStats,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive (feature_mean, feature_std, target_mean, target_std, gram_std,
    cross_std) analytically from `stats` — never revisits raw data. A zero-
    variance feature/target gets a std of 1.0 used only as a safe division
    placeholder: its row/column in the centered Gram/cross is already exactly
    zero, so the standardized entry is 0/1 = 0, not 0/0.
    `feature_std`/`target_std` are returned as *actually observed* (possibly 0)
    — callers needing the placeholder should guard again.
    """
    feature_mean = stats.mean_x
    target_mean = stats.mean_y
    feature_std = np.sqrt(stats.var_x)
    target_std = np.sqrt(stats.var_y)

    feature_std_safe = np.where(feature_std > 0, feature_std, 1.0)
    target_std_safe = np.where(target_std > 0, target_std, 1.0)

    gram = stats.centered_gram() / stats.n
    cross = stats.centered_cross() / stats.n

    gram_std = gram / np.outer(feature_std_safe, feature_std_safe)
    cross_std = cross / np.outer(feature_std_safe, target_std_safe)

    return feature_mean, feature_std, target_mean, target_std, gram_std, cross_std


def destandardize_coefficients(
    beta_std: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map coefficients fit in standardized space back to raw units, so `beta_raw`
    directly bounds a raw-unit perturbation's effect on the raw-unit prediction
    (‖β‖·‖δ‖). `beta_std` is (p,) or (p, n_targets).
    """
    feature_std_safe = np.where(feature_std > 0, feature_std, 1.0)
    if beta_std.ndim == 1:
        beta_raw = beta_std * target_std / feature_std_safe
        beta_raw = np.where(feature_std > 0, beta_raw, 0.0)
        intercept_raw = target_mean - feature_mean @ beta_raw
        return beta_raw, intercept_raw

    scale = target_std[np.newaxis, :] / feature_std_safe[:, np.newaxis]
    beta_raw = beta_std * scale
    beta_raw = np.where((feature_std > 0)[:, np.newaxis], beta_raw, 0.0)
    intercept_raw = target_mean - feature_mean @ beta_raw
    return beta_raw, intercept_raw
