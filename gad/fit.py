"""Fits every layer of GAD from clean data at commissioning.

`BundleConfig` holds the options, and `settings.py` sets the values used in the
paper.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import chi2

from gad.detector import MONITOR_A, MONITOR_B
from gad.determinism import deterministic_blas
from gad.l0_quality import fit_quality_ranges
from gad.l1a_predictor import RelationalPredictor
from gad.l1b_physics import StateSpacePredictor, TankBalance
from gad.l2_context import SettlingTracker, per_channel
from gad.l2_residual import measurement_resolution
from gad.l3_cusum import (Cusum, calibrate_h_by_context, calibrate_h_evt,
                          calibrate_k, count_episodes)
from gad.l5_bundle import Bundle
from gad.l5_limits import EventScaleResidual
from gad.types import Sample


@dataclass(frozen=True)
class BundleConfig:
    """Everything the fit needs: the plant (channels, physics, contexts), the
    false-alarm budget and the settings of every layer.
    """

    sensed: tuple[str, ...]
    commanded: tuple[str, ...]
    table: tuple[TankBalance, ...]
    channel_ids: Callable[[np.ndarray], np.ndarray]
    budget_episodes_per_day: float
    budget_time_fraction: float
    monitor_share: dict[str, float]
    cusum_lambda: float
    samples_per_day: float
    l0_margin_k: float = 5.0
    # L1
    l1b_target_gain: float = 0.1
    l1b_innovation_gate: float | None = None
    l1b_gate_max_hold: int | None = None
    # L2
    l2_half_life: float = 21_600.0
    l2_clip: float = 3.0
    l2_cap: float = 2.0
    l2_band: float = 1.0
    l2_reference_from_commissioning_rows: bool = False
    l2_reference_min_rows: int = 1000
    l2_reference_min_gap: float = 4.0
    settle_samples: int = 0
    # L3 and L4
    cusum_cap_factor: float = 1.1
    l3_per_context_h: bool = False
    l3_per_context_min_rows: int = 1000
    l3_per_context_min_typical_z: float = 0.1
    l3_per_context_holdout: float = 0.0
    degraded_abstention_threshold: float = 0.5
    dwell_anomaly: int = 5
    dwell_recover: int = 30
    alarm_group_window: float = 30.0
    # L5
    learn_guard: int = 0
    adaptive_h_cap: float = 2.0
    adaptive_h_period: float = 21_600.0
    adaptive_h_window: int = 345_600
    sensitivity_budget: bool = False
    naive_learning: bool = False


@dataclass
class FitResult:
    """The fitted layers, warmed up to the end of the calibration data."""

    bundle: Bundle
    l1a: RelationalPredictor
    l1b: StateSpacePredictor
    l2a: EventScaleResidual
    l2b: EventScaleResidual
    l3a: Cusum
    l3b: Cusum
    resolution: np.ndarray
    h_detail_a: dict
    h_detail_b: dict
    per_channel_budget: dict
    calibration_a: list
    calibration_b: list


def make_l2(cfg: BundleConfig, n: int, floor: np.ndarray) -> EventScaleResidual:
    """L2 for one branch: robust means and standard deviations, learned slowly,
    clipped, and kept within the limits of rule 3.
    """
    return EventScaleResidual(n, half_life=cfg.l2_half_life, clip=cfg.l2_clip,
                              cap=cfg.l2_cap, band=cfg.l2_band, scale_floor=floor,
                              pooled_init_for_unseen=True)


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _poisson_rate_lower(count: int, exposure: float, alpha: float = 0.05) -> float:
    """Exact (Garwood) lower confidence bound on a Poisson rate. Zero events gives
    zero, which is what makes a short holdout unable to establish a breach from
    counts alone.
    """
    if count <= 0 or not (exposure > 0):
        return 0.0
    return float(chi2.ppf(alpha / 2.0, 2 * count) / 2.0 / exposure)


def _modes_over_budget(evidence: np.ndarray, contexts: np.ndarray,
                       bars: dict[int, float], budget_episodes_per_day: float,
                       budget_time_fraction: float, samples_per_day: float) -> set[int]:
    """The contexts whose own threshold breaks its share of the false-alarm budget
    on held-out clean rows. A context is dropped when the 95 % lower bound of
    its realised rate is above its share, or when it has fewer than 30 held-out
    rows.
    """
    ev = np.nan_to_num(np.asarray(evidence, dtype=float), nan=-np.inf)
    ctx = np.asarray(contexts, dtype=np.int64)
    days = len(ev) / samples_per_day
    if not bars or len(ev) == 0:
        return set()
    eps_share = budget_episodes_per_day / len(bars)
    time_share = budget_time_fraction / len(bars)
    drop = set()
    for c, hv in bars.items():
        mask = ctx == c
        alarm = (ev >= float(hv)) & mask
        t_lo, _ = _wilson_interval(int(alarm.sum()), len(ev))
        e_lo = _poisson_rate_lower(count_episodes(alarm), days)
        over = t_lo > time_share or (eps_share * days >= 1.0 and e_lo > eps_share)
        if int(mask.sum()) < 30 or over:
            drop.add(int(c))
    return drop


def _calibrate_h(cfg: BundleConfig, evidence_cal: np.ndarray, time_frac: float,
                 eps_day: float):
    """One threshold per channel from a tail fit of its clean evidence."""
    n = len(cfg.sensed)
    h = np.full(n, np.nan)
    detail = {}
    for i, ch in enumerate(cfg.sensed):
        col = evidence_cal[:, i]
        if not np.isfinite(col).any():
            continue
        try:
            res = calibrate_h_evt(col, eps_day, time_frac, cfg.samples_per_day)
        except ValueError:
            continue  # budget unreachable even empirically: stays NaN, never alarms
        h[i] = res.h
        detail[ch] = {"h": res.h, "method": res.method, "n_peaks": res.n_peaks,
                      "gpd_shape": res.shape}
    return h, detail


def _long_calibration_a(cfg: BundleConfig, l1a_state, X_fit, U_fit, n_warm: int,
                        resolution, k_a, evidence_a_cal, h_a, detail_a, budget,
                        X_warm=None, U_warm=None, regimes_cal=None,
                        scaled_a_cal=None):
    """Stream the fit rows through a separate copy of L1a and a separate L2 (the
    live components are not touched), score them with the calibrated slack
    `k_a` after `n_warm` rows of L2 warm-up, and re-read `h` from that evidence
    plus the calibration window's — for channels whose L1a model has no fitted
    coefficients only.
    """
    sensed, commanded = list(cfg.sensed), list(cfg.commanded)
    twin = RelationalPredictor(sensed, commanded)
    twin.state = l1a_state
    l2 = make_l2(cfg, len(sensed), resolution)
    cusum = Cusum(np.asarray(k_a), lam=cfg.cusum_lambda)
    q0 = np.zeros(len(sensed), dtype=np.uint8)
    settle = SettlingTracker(len(sensed), cfg.settle_samples)
    evidence, ctx_fit, z_fit = [], [], []
    for t in range(len(X_fit)):
        sample = Sample(t=float(t), x=X_fit[t], u=U_fit[t], q=q0)
        regime = settle.step(cfg.channel_ids(U_fit[t]))
        scaled = l2.update(X_fit[t] - twin.update(sample), regime, True)
        if t >= n_warm:
            evidence.append(cusum.update(scaled))
            ctx_fit.append(per_channel(regime, len(sensed)).copy())
            z_fit.append(np.asarray(scaled, dtype=float).copy())
    if not evidence:
        return h_a, detail_a, None, {}
    combined = np.concatenate([np.array(evidence), evidence_a_cal])

    def _with_cal(prefix, tail, dtype):
        if tail is None:
            return None
        return np.concatenate([np.asarray(prefix, dtype=dtype),
                               np.asarray(tail, dtype=dtype)])

    ctx_combined = _with_cal(ctx_fit, regimes_cal, np.int64)
    h_long, detail_long = _calibrate_h(cfg, combined, *budget)
    h, detail = np.array(h_a, dtype=np.float64), dict(detail_a)
    for i, ch in enumerate(sensed):
        if l1a_state.model_choice.get(ch) in ("fitted", "cross") or ch not in detail_long:
            continue
        h[i] = h_long[i]
        detail[ch] = {**detail_long[ch], "calibration": "fit+cal",
                      "n_calibration_rows": int(len(combined))}

    streams = {"fit+cal": (
        combined,
        ctx_combined,
        _with_cal(z_fit, scaled_a_cal, np.float64))}
    if X_warm is not None and len(X_warm):
        warm_ev, ctx_warm, z_warm = [], [], []
        for t in range(len(X_warm)):
            sample = Sample(t=float(len(X_fit) + t), x=X_warm[t], u=U_warm[t], q=q0)
            regime = settle.step(cfg.channel_ids(U_warm[t]))
            scaled = l2.update(X_warm[t] - twin.update(sample), regime, True)
            warm_ev.append(cusum.update(scaled))
            ctx_warm.append(per_channel(regime, len(sensed)).copy())
            z_warm.append(np.asarray(scaled, dtype=float).copy())
        if warm_ev:
            warm_combined = np.concatenate([np.array(warm_ev), evidence_a_cal])
            ctx_warm_combined = _with_cal(ctx_warm, regimes_cal, np.int64)
            streams["warm+cal"] = (
                warm_combined,
                ctx_warm_combined,
                _with_cal(z_warm, scaled_a_cal, np.float64))
            h_warm, detail_warm = _calibrate_h(cfg, warm_combined, *budget)
            for i, ch in enumerate(sensed):
                if (l1a_state.model_choice.get(ch) not in ("fitted", "cross")
                        or ch not in detail_warm):
                    continue
                h[i] = h_warm[i]
                detail[ch] = {**detail_warm[ch], "calibration": "warm+cal",
                              "n_calibration_rows": int(len(warm_combined))}
    return h, detail, combined, streams


def fit_bundle(
    cfg: BundleConfig,
    X_fit: np.ndarray, U_fit: np.ndarray,
    X_warm: np.ndarray, U_warm: np.ndarray,
    X_cal: np.ndarray, U_cal: np.ndarray,
    X_l0: np.ndarray,
    version: int,
) -> FitResult:
    """Fit all layers, with the linear algebra on one thread so the result is
    reproducible.
    """
    with deterministic_blas():
        return _fit_bundle(cfg, X_fit, U_fit, X_warm, U_warm, X_cal, U_cal, X_l0, version)


def _fit_bundle(
    cfg: BundleConfig,
    X_fit: np.ndarray, U_fit: np.ndarray,
    X_warm: np.ndarray, U_warm: np.ndarray,
    X_cal: np.ndarray, U_cal: np.ndarray,
    X_l0: np.ndarray,
    version: int,
) -> FitResult:
    sensed, commanded = list(cfg.sensed), list(cfg.commanded)
    n_sensed = len(sensed)

    # L1: the two predictors, fitted on the fit rows
    l1a = RelationalPredictor(sensed, commanded)
    l1a.fit_from_arrays(X_fit, U_fit)
    l1b = StateSpacePredictor(sensed, commanded, table=cfg.table,
                              target_gain=cfg.l1b_target_gain,
                              innovation_gate=cfg.l1b_innovation_gate,
                              gate_max_hold=cfg.l1b_gate_max_hold)
    l1b.fit_from_arrays(X_fit, U_fit)
    P0 = l1b.P.copy()

    # L2: warmed up on the warm-up rows, then scored on the calibration rows
    resolution = measurement_resolution(X_fit)
    l2a = make_l2(cfg, n_sensed, resolution)
    l2b = make_l2(cfg, n_sensed, resolution)
    q0 = np.zeros(n_sensed, dtype=np.uint8)
    settle = SettlingTracker(n_sensed, cfg.settle_samples)

    def residuals(t: int, x: np.ndarray, u: np.ndarray):
        sample = Sample(t=float(t), x=x, u=u, q=q0)
        return (x - l1a.update(sample), x - l1b.update(sample),
                settle.step(cfg.channel_ids(u)))

    keep_raw = cfg.l2_reference_from_commissioning_rows
    raw_a, raw_b, raw_ctx = [], [], []

    n_warm = len(X_warm)
    for t in range(n_warm):
        r_a, r_b, regime = residuals(t, X_warm[t], U_warm[t])
        l2a.update(r_a, regime, True)
        l2b.update(r_b, regime, True)
        if keep_raw:
            raw_a.append(r_a.copy())
            raw_b.append(r_b.copy())
            raw_ctx.append(per_channel(regime, n_sensed).copy())

    scaled_a_cal, scaled_b_cal, regimes_cal = [], [], []
    for t in range(len(X_cal)):
        r_a, r_b, regime = residuals(n_warm + t, X_cal[t], U_cal[t])
        scaled_a_cal.append(l2a.update(r_a, regime, True))
        scaled_b_cal.append(l2b.update(r_b, regime, True))
        if keep_raw:
            raw_a.append(r_a.copy())
            raw_b.append(r_b.copy())
            raw_ctx.append(per_channel(regime, n_sensed).copy())
        regimes_cal.append(per_channel(regime, n_sensed).copy())
    scaled_a_cal = np.array(scaled_a_cal)
    scaled_b_cal = np.array(scaled_b_cal)
    regimes_cal = np.array(regimes_cal)

    if keep_raw:
        # the reference mean and standard deviation of each context, fitted on
        # all commissioning rows
        R_a, R_b = np.asarray(raw_a), np.asarray(raw_b)
        CTX = np.asarray(raw_ctx)
        rep_a = l2a.fit_references(R_a, CTX, min_rows=cfg.l2_reference_min_rows,
                                   min_gap=cfg.l2_reference_min_gap)
        rep_b = l2b.fit_references(R_b, CTX, min_rows=cfg.l2_reference_min_rows,
                                   min_gap=cfg.l2_reference_min_gap)
        cal_from = len(R_a) - len(scaled_a_cal)
        for l2, rep, scaled in ((l2a, rep_a, scaled_a_cal),
                                (l2b, rep_b, scaled_b_cal)):
            moved = sorted({int(j) for j, *_ in rep["refitted"]})
            if not moved:
                continue
            rescored = l2.score_batch(R_a[cal_from:] if l2 is l2a else R_b[cal_from:],
                                      CTX[cal_from:])
            scaled[:, moved] = rescored[:, moved]
        raw_a = raw_b = raw_ctx = None

    # L3: slack k and the thresholds h, set for each predictor's share of the budget
    k_a = calibrate_k(scaled_a_cal)
    k_b = calibrate_k(scaled_b_cal)
    l3a = Cusum(k_a, lam=cfg.cusum_lambda)
    l3b = Cusum(k_b, lam=cfg.cusum_lambda)
    evidence_a_cal = np.array([l3a.update(r) for r in scaled_a_cal])
    evidence_b_cal = np.array([l3b.update(r) for r in scaled_b_cal])

    n_b_channels = len(l1b.state_names)
    per_channel_budget = {
        MONITOR_A: (cfg.monitor_share[MONITOR_A] * cfg.budget_time_fraction / n_sensed,
                    cfg.monitor_share[MONITOR_A] * cfg.budget_episodes_per_day / n_sensed),
        MONITOR_B: (cfg.monitor_share[MONITOR_B] * cfg.budget_time_fraction / n_b_channels,
                    cfg.monitor_share[MONITOR_B] * cfg.budget_episodes_per_day / n_b_channels),
    }
    h_a, h_detail_a = _calibrate_h(cfg, evidence_a_cal, *per_channel_budget[MONITOR_A])
    h_b, h_detail_b = _calibrate_h(cfg, evidence_b_cal, *per_channel_budget[MONITOR_B])

    # the CUSUM is capped a little above its threshold, so an alarm can end;
    # the thresholds are set again with the cap in place
    l3a = Cusum(k_a, h_a, lam=cfg.cusum_lambda, cap_factor=cfg.cusum_cap_factor,
                dwell=cfg.dwell_anomaly)
    l3b = Cusum(k_b, h_b, lam=cfg.cusum_lambda, cap_factor=cfg.cusum_cap_factor,
                dwell=cfg.dwell_anomaly)
    evidence_a_cal = np.array([l3a.update(r) for r in scaled_a_cal])
    evidence_b_cal = np.array([l3b.update(r) for r in scaled_b_cal])
    h_a, h_detail_a = _calibrate_h(cfg, evidence_a_cal, *per_channel_budget[MONITOR_A])
    h_b, h_detail_b = _calibrate_h(cfg, evidence_b_cal, *per_channel_budget[MONITOR_B])

    # L1a channels without a cross-sensor model are calibrated on more clean rows
    calibration_a = [evidence_a_cal[:, i] for i in range(n_sensed)]
    h_a, h_detail_a, combined, long_streams = _long_calibration_a(
        cfg, l1a.state, X_fit, U_fit, n_warm, resolution, k_a, evidence_a_cal,
        h_a, h_detail_a, per_channel_budget[MONITOR_A], X_warm, U_warm,
        regimes_cal=regimes_cal, scaled_a_cal=scaled_a_cal)
    if combined is not None:
        calibration_a = [combined[:, i] if h_detail_a.get(ch, {}).get("calibration") else evidence_a_cal[:, i]
                         for i, ch in enumerate(sensed)]
    calibration_b = [evidence_b_cal[:, i] for i in range(evidence_b_cal.shape[1])]
    l3a.h, l3b.h = h_a, h_b

    # L4: a threshold of its own for each operating context with enough data.
    # The last part of the clean rows is held out to check each one.
    bundle_h_by_context = None
    if cfg.l3_per_context_h:
        time_frac, eps_day = per_channel_budget[MONITOR_A]
        table: dict[int, np.ndarray] = {}
        for j, ch in enumerate(sensed):
            ev_j, ctx_j, z_j = evidence_a_cal[:, j], regimes_cal[:, j], scaled_a_cal[:, j]
            longer = long_streams.get(h_detail_a.get(ch, {}).get("calibration"))
            if longer is not None and longer[1] is not None:
                ev_j, ctx_j = longer[0][:, j], longer[1][:, j]
                z_j = longer[2][:, j] if longer[2] is not None else None
            cut = (int(len(ev_j) * (1.0 - cfg.l3_per_context_holdout))
                   if cfg.l3_per_context_holdout > 0.0 else len(ev_j))
            min_rows_fit = max(1, int(round(
                cfg.l3_per_context_min_rows * (cut / len(ev_j)))))
            got, _ = calibrate_h_by_context(
                ev_j[:cut], ctx_j[:cut], eps_day, time_frac,
                cfg.samples_per_day, min_rows=min_rows_fit,
                z=None if z_j is None else z_j[:cut],
                min_typical_z=cfg.l3_per_context_min_typical_z,
                refuse_empirical=(cfg.l3_per_context_holdout <= 0.0))
            drop = set()
            if cut < len(ev_j):
                drop = _modes_over_budget(ev_j[cut:], ctx_j[cut:], got,
                                          eps_day, time_frac, cfg.samples_per_day)
            for c, hv in got.items():
                if c not in drop:
                    table.setdefault(c, np.full(n_sensed, np.nan))[j] = hv
        if table:
            l3a.h_by_context = table
            bundle_h_by_context = table
    l3a.s_pos[:] = l3a.s_neg[:] = 0.0   # the calibration passes re-streamed the window
    l3b.s_pos[:] = l3b.s_neg[:] = 0.0

    # L5: the commissioning references every later update is held to
    l2a.commit_references()
    l2b.commit_references()
    l2a.commit()
    l2b.commit()
    bundle = Bundle(
        version=version,
        quality_ranges=fit_quality_ranges(sensed, X_l0, margin_k=cfg.l0_margin_k,
                                          step_lag=1, step_basis="median"),
        l1a_state=l1a.state,
        l1b_Q=l1b.Q, l1b_R=l1b.R, l1b_P0=P0,
        resolution=resolution,
        k_a=k_a, h_a=h_a, k_b=k_b, h_b=h_b,
        h_a_by_context=bundle_h_by_context,
        l2a_state=l2a.export_state(), l2b_state=l2b.export_state(),
    )
    return FitResult(bundle, l1a, l1b, l2a, l2b, l3a, l3b, resolution,
                     h_detail_a, h_detail_b, per_channel_budget, calibration_a,
                     calibration_b)
