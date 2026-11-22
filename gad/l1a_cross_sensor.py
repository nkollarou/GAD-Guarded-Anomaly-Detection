"""L1a: predicts a sensor only from the other sensors and the commands, never
from its own past, so a faked reading cannot teach the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gad.l1a_fit import default_lambda1, fit_elastic_net_from_gram

#: see module docstring — derived, not tuned
MIN_FIT = 0.75
ADF_ALPHA = 0.05
#: samples per block for the stability test (per-minute means at 1 Hz)
ADF_BLOCK = 60
INNER_FIT_FRAC = 0.7

HOLD_FIT_FRAC = 0.5
HOLD_BLOCKS = 4
HOLD_VISIBILITY = 1.5
HOLD_MIN_FIT = 1.0 - 1.0 / HOLD_VISIBILITY ** 2          # = 0.5556


@dataclass(frozen=True)
class CrossState:
    """`channels` are sensed-channel indices with a relation; `beta` is
        (2n + m, len(channels)) in raw units over [x(t), x(t-1), u(t)], with a
        channel's own two rows exactly 0; `selection` records, for every sensed
        channel, the validation figures and the verdict.
    """

    channels: tuple[int, ...]
    beta: np.ndarray
    intercept: np.ndarray
    selection: dict

    def n_bytes(self) -> int:
        return int(self.beta.nbytes + self.intercept.nbytes)


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


def _stable(resid: np.ndarray) -> float:
    """ADF p-value of the residual's block means (small = stable)."""
    m = len(resid) // ADF_BLOCK
    if m < 20:
        return 1.0
    means = resid[: m * ADF_BLOCK].reshape(m, ADF_BLOCK).mean(axis=1)
    if np.std(means) == 0:
        return 0.0  # a residual that never moves is as stable as it gets
    from statsmodels.tsa.stattools import adfuller
    return float(adfuller(means, regression="c", autolag="AIC", result_object=False)[1])


def fit_cross(X: np.ndarray, U: np.ndarray, lambda2: float = 0.1,
              sensed: tuple[str, ...] | None = None, rule: str = "0044") -> CrossState:
    X = np.asarray(X, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    n = X.shape[1]
    names = sensed or tuple(str(j) for j in range(n))
    cut = int(len(X) * INNER_FIT_FRAC)
    D_fit, Y_fit = design(X[:cut], U[:cut])
    D_val, Y_val = design(X[cut - 1:], U[cut - 1:])  # continuous across the cut
    beta_in, ic_in = _fit(D_fit, Y_fit, list(range(n)), n, lambda2)
    resid = Y_val - (D_val @ beta_in + ic_in)

    selection, chosen = {}, []
    if rule == "0083":
        hcut = int(len(X) * HOLD_FIT_FRAC)
        D_h, Y_h = design(X[:hcut], U[:hcut])
        beta_h, ic_h = _fit(D_h, Y_h, list(range(n)), n, lambda2)
        edges = np.linspace(hcut - 1, len(X), HOLD_BLOCKS + 1, dtype=int)
        per_block: list[np.ndarray] = []
        for b in range(HOLD_BLOCKS):
            lo, hi = edges[b], edges[b + 1]
            D_b, Y_b = design(X[lo:hi], U[lo:hi])
            res_b = Y_b - (D_b @ beta_h + ic_h)
            var_b = np.var(Y_b, axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                per_block.append(np.where(var_b > 0,
                                          1.0 - np.var(res_b, axis=0) / var_b,
                                          -np.inf))
        R = np.vstack(per_block)                      # (blocks, channels)

        res_fit = Y_h - (D_h @ beta_h + ic_h)
        sd_fit = np.maximum(res_fit.std(axis=0), 1e-12)
        pooled_res, pooled_y, ratios = [], [], []
        for b in range(HOLD_BLOCKS):
            lo, hi = edges[b], edges[b + 1]
            D_b, Y_b = design(X[lo:hi], U[lo:hi])
            r_b = Y_b - (D_b @ beta_h + ic_h)
            pooled_res.append(r_b); pooled_y.append(Y_b)
            ratios.append(r_b.std(axis=0) / sd_fit)
        PR, PY = np.vstack(pooled_res), np.vstack(pooled_y)
        var_pool = np.var(PY, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2_pool = np.where(var_pool > 0, 1.0 - np.var(PR, axis=0) / var_pool, -np.inf)
        worst_ratio = np.maximum(1.0, np.vstack(ratios).max(axis=0))

        for j in range(n):
            blocks = [float(v) for v in R[:, j]]
            r2p, k = float(r2_pool[j]), float(worst_ratio[j])
            vis = (1.0 / (np.sqrt(1.0 - r2p) * k)) if r2p < 1.0 else float("inf")
            ok = bool(np.isfinite(r2p) and r2p < 1.0 and vis >= HOLD_VISIBILITY)
            selection[names[j]] = {"rule": "0083", "r2_blocks": blocks,
                                   "r2_worst_block": min(blocks),
                                   "r2_pooled": r2p, "worst_scale_ratio": k,
                                   "visibility_spreads": (float(vis) if np.isfinite(vis)
                                                          else None),
                                   "target_spreads": HOLD_VISIBILITY,
                                   "selected": ok}
            if ok:
                chosen.append(j)
    else:
        for j in range(n):
            var_y = float(np.var(Y_val[:, j]))
            r2 = 1.0 - float(np.var(resid[:, j])) / var_y if var_y > 0 else float("-inf")
            p = _stable(resid[:, j])
            ok = r2 >= MIN_FIT and p < ADF_ALPHA
            selection[names[j]] = {"rule": "0044", "r2_validation": r2,
                                   "adf_p": p, "selected": bool(ok)}
            if ok:
                chosen.append(j)

    if chosen:
        D_all, Y_all = design(X, U)
        beta, intercept = _fit(D_all, Y_all, chosen, n, lambda2)
    else:
        beta, intercept = np.zeros((2 * n + U.shape[1], 0)), np.zeros(0)
    return CrossState(tuple(chosen), beta, intercept, selection)


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
