"""L1a: predicts a sensor only from the other sensors and the commands, never from
its own past, so a faked reading cannot teach the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gad.l1a_fit import default_lambda1, fit_elastic_net_from_gram

#: the relation is fitted on the first half of the fitting data and checked on
#: the second half, split into four blocks
HOLD_FIT_FRAC = 0.5
HOLD_BLOCKS = 4
#: a sensor is predicted this way only if a shift of one spread of its normal
#: variation would still stand out by 1.5 spreads of the prediction error, in
#: the worst block
HOLD_VISIBILITY = 1.5


@dataclass(frozen=True)
class CrossState:
    """`channels` are the sensors predicted this way. `beta` has one column per
    such sensor, over the features [x(t), x(t-1), u(t)]. A sensor's own two
    rows are always 0.
    """

    channels: tuple[int, ...]
    beta: np.ndarray
    intercept: np.ndarray


def design(X: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows t = 1..T-1: features [x(t), x(t-1), u(t)], targets x(t)."""
    return np.column_stack([X[1:], X[:-1], U[1:]]), X[1:]


def _fit(D: np.ndarray, Y: np.ndarray, targets: list[int], n: int, lambda2: float):
    mean_d, std_d = D.mean(axis=0), D.std(axis=0)
    safe_d = np.where(std_d > 0, std_d, 1.0)
    Z = (D - mean_d) / safe_d
    gram = Z.T @ Z / len(Z)
    p = D.shape[1]
    beta = np.zeros((p, len(targets)))
    intercept = np.zeros(len(targets))
    for k, j in enumerate(targets):
        y = Y[:, j]
        mean_y, std_y = y.mean(), y.std()
        if std_y == 0:
            intercept[k] = mean_y
            continue
        cross = Z.T @ ((y - mean_y) / std_y) / len(Z)
        keep = np.array([c not in (j, n + j) for c in range(p)])
        b = fit_elastic_net_from_gram(gram[np.ix_(keep, keep)], cross[keep],
                                      default_lambda1(int(keep.sum()), len(Z)), lambda2)
        raw = np.zeros(p)
        raw[keep] = b * std_y / safe_d[keep]
        raw[std_d == 0] = 0.0
        beta[:, k] = raw
        intercept[k] = mean_y - mean_d @ raw
    return beta, intercept


def fit_cross(X: np.ndarray, U: np.ndarray, lambda2: float = 0.1) -> CrossState:
    """Choose the sensors that can be predicted from the others, and fit them.
    """
    X = np.asarray(X, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    n = X.shape[1]
    hcut = int(len(X) * HOLD_FIT_FRAC)
    D_h, Y_h = design(X[:hcut], U[:hcut])
    beta_h, ic_h = _fit(D_h, Y_h, list(range(n)), n, lambda2)
    sd_fit = np.maximum((Y_h - (D_h @ beta_h + ic_h)).std(axis=0), 1e-12)

    edges = np.linspace(hcut - 1, len(X), HOLD_BLOCKS + 1, dtype=int)
    pooled_res, pooled_y, ratios = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        D_b, Y_b = design(X[lo:hi], U[lo:hi])
        r_b = Y_b - (D_b @ beta_h + ic_h)
        pooled_res.append(r_b)
        pooled_y.append(Y_b)
        ratios.append(r_b.std(axis=0) / sd_fit)
    PR, PY = np.vstack(pooled_res), np.vstack(pooled_y)
    var_pool = np.var(PY, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2_pool = np.where(var_pool > 0, 1.0 - np.var(PR, axis=0) / var_pool, -np.inf)
    worst_ratio = np.maximum(1.0, np.vstack(ratios).max(axis=0))

    chosen = []
    for j in range(n):
        r2p, k = float(r2_pool[j]), float(worst_ratio[j])
        if np.isfinite(r2p) and r2p < 1.0 and 1.0 / (np.sqrt(1.0 - r2p) * k) >= HOLD_VISIBILITY:
            chosen.append(j)

    if chosen:
        D_all, Y_all = design(X, U)
        beta, intercept = _fit(D_all, Y_all, chosen, n, lambda2)
    else:
        beta, intercept = np.zeros((2 * n + U.shape[1], 0)), np.zeros(0)
    return CrossState(tuple(chosen), beta, intercept)


def predict(state: CrossState, x_now: np.ndarray, x_prev: np.ndarray, u_now: np.ndarray) -> np.ndarray:
    """Predictions for `state.channels`. A missing input only poisons the
    predictions that actually use it.
    """
    f = np.concatenate([x_now, x_prev, u_now])
    missing = np.isnan(f)
    pred = state.intercept + np.where(missing, 0.0, f) @ state.beta
    if missing.any():
        uses_missing = np.any(state.beta[missing] != 0.0, axis=0)
        pred = np.where(uses_missing, np.nan, pred)
    return pred
