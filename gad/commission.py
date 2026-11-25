"""Fit GAD once on clean data (commissioning).

`commission()` fits the two predictors, warms up the means and standard
deviations, sets the thresholds and records the golden set. The ready detector
is returned as `.detector`.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gad import settings
from gad.detector import MONITOR_A, MONITOR_B, Detector
from gad.fit import BundleConfig, fit_bundle
from gad.l0_quality import QualityChecker, fit_quality_ranges
from gad.l1b_physics import PHYSICS_TABLE
from gad.l4_decision import PerSensorDecider
from gad.l4_thresholds import AdaptiveThreshold
from gad.l5_detection_test import SensitivityGuard, SensitivityTest
from gad.l6_evidence import EvidenceBuilder
from gad.swat import build_contexts, find_commissioning_start, split_channels
from gad.types import Sample

#: the converted SWaT normal-operation recording (`build_datasets.py --convert`)
NORMAL_PARQUET = Path("data/swat_normal.parquet")

SAMPLES_PER_DAY = 86400  # 1 Hz
BUDGET_EPISODES_PER_DAY = settings.FALSE_ALARMS_PER_DAY
BUDGET_TIME_FRACTION = settings.SHARE_OF_TIME_IN_ALARM
#: the false-alarm budget is split equally between the two predictors
MONITOR_BUDGET_SHARE = {MONITOR_A: 0.5, MONITOR_B: 0.5}
CUSUM_DECAY_LAMBDA = settings.CUSUM_LAMBDA
L0_MARGIN_K = settings.L0_RANGE_MARGIN
GOLDEN_EXCERPTS = settings.GOLDEN_EXCERPTS
GOLDEN_LENGTH = settings.GOLDEN_EXCERPT_MINUTES * 60
GOLDEN_SPAN = settings.GOLDEN_SPAN_HOURS * 3600


def fill_gaps(frame: pd.DataFrame, channels: list[str]) -> tuple[pd.DataFrame, dict]:
    """Forward- then backward-fill missing values, for the data models are fitted
    on only (the stream itself keeps them, and L0 flags them).
    """
    out = frame.copy()
    counts = {}
    for ch in channels:
        v = out[ch].to_numpy(dtype=float)
        bad = ~np.isfinite(v)
        if bad.any():
            counts[ch] = int(bad.sum())
            out[ch] = pd.Series(np.where(bad, np.nan, v)).ffill().bfill().to_numpy()
    return out, {"n_channels": len(counts), "n_values": sum(counts.values()),
                 "per_channel": counts}


def l0_checker(steady: pd.DataFrame, sensed: list[str]) -> QualityChecker:
    """L0: valid ranges learned from the clean recording."""
    return QualityChecker(fit_quality_ranges(sensed, steady[sensed].to_numpy(),
                                             margin_k=L0_MARGIN_K, step_lag=1,
                                             step_basis="median"))


def physics_relation_names() -> dict[str, str]:
    """Each L1b level channel and the mass balance it belongs to, for L6."""
    return {
        tb.level_channel: (
            f"{tb.tank} mass balance ({tb.level_channel} <- {tb.inflow.flow_channel} via "
            f"{tb.inflow.driving_actuator}, -> {tb.outflow.flow_channel} via {tb.outflow.driving_actuator})"
        )
        for tb in PHYSICS_TABLE
    }


def clean_windows(n: int, split: tuple[float, float, float]) -> SimpleNamespace:
    """Where fitting, warm-up and threshold calibration end in the clean
    recording.
    """
    train_frac, warm_frac, cal_frac = split
    outer_train_end = int(n * train_frac)
    n_held = n - outer_train_end
    warmup_end = int(n_held * warm_frac)
    calib_end = warmup_end + int(n_held * cal_frac)
    return SimpleNamespace(outer_train_end=outer_train_end, warmup_end=warmup_end,
                           calib_end=calib_end)


def commission(cfg_overrides: dict, split: tuple[float, float, float] = settings.SPLIT,
               with_evidence: bool = False) -> SimpleNamespace:
    """Fit, warm up and calibrate GAD on the clean recording. `cfg_overrides` is a
    configuration from `settings.py`.
    """
    df = pd.read_parquet(NORMAL_PARQUET)
    sensed, commanded = split_channels(list(df.columns))
    steady = df.iloc[find_commissioning_start(df):].reset_index(drop=True)
    w = clean_windows(len(steady), split)
    outer_train_end, warmup_end, calib_end = w.outer_train_end, w.warmup_end, w.calib_end

    steady, _ = fill_gaps(steady, list(sensed) + list(commanded))
    X = steady[sensed].to_numpy()
    U = steady[commanded].to_numpy()
    X_train, U_train = X[:outer_train_end], U[:outer_train_end]
    X_held, U_held = X[outer_train_end:], U[outer_train_end:]
    n_sensed = len(sensed)

    ctx = build_contexts(sensed, commanded)
    cfg = BundleConfig(
        sensed=tuple(sensed), commanded=tuple(commanded), table=PHYSICS_TABLE,
        channel_ids=ctx.channel_ids,
        budget_episodes_per_day=BUDGET_EPISODES_PER_DAY, budget_time_fraction=BUDGET_TIME_FRACTION,
        monitor_share=MONITOR_BUDGET_SHARE, cusum_lambda=CUSUM_DECAY_LAMBDA,
        samples_per_day=SAMPLES_PER_DAY, l0_margin_k=L0_MARGIN_K,
    )
    cfg = dataclasses.replace(cfg, **cfg_overrides)
    fit = fit_bundle(
        cfg,
        X_train, U_train,
        X_held[:warmup_end], U_held[:warmup_end],
        X_held[warmup_end:calib_end], U_held[warmup_end:calib_end],
        X, version=1,
    )
    bundle = fit.bundle
    l1a, l1b = fit.l1a, fit.l1b
    l2a, l2b, l3a, l3b = fit.l2a, fit.l2b, fit.l3a, fit.l3b
    n_b_channels = len(l1b.state_names)

    evidence = None
    if with_evidence:
        evidence = EvidenceBuilder(sensed, commanded, l1a=l1a,
                                   physics_relations=physics_relation_names())

    def adaptive(h, calibration, monitor):
        time_frac, eps_day = fit.per_channel_budget[monitor]
        return AdaptiveThreshold(h, calibration, time_frac, eps_day, SAMPLES_PER_DAY,
                                 cap=cfg.adaptive_h_cap, period=cfg.adaptive_h_period,
                                 window=cfg.adaptive_h_window)

    detector = Detector(
        sensed, l1a, l1b, l2a, l2b, l3a, l3b,
        decider=PerSensorDecider(sensed, cfg.degraded_abstention_threshold,
                                 cfg.dwell_anomaly, cfg.dwell_recover,
                                 bundle_id=bundle.bundle_id,
                                 group_window=cfg.alarm_group_window),
        regime_fn=ctx.channel_ids,
        n_possible_tests=n_sensed + n_b_channels,
        adaptive_a=adaptive(l3a.h, fit.calibration_a, MONITOR_A),
        adaptive_b=adaptive(l3b.h, fit.calibration_b, MONITOR_B),
        joint_cap=cfg.adaptive_h_cap,
        evidence=evidence, learn_guard=cfg.learn_guard,
        settle_samples=cfg.settle_samples,
        naive_learning=cfg.naive_learning,
    )
    if cfg.sensitivity_budget:  # rule 4: the golden-set test
        guard = sensitivity_guard(cfg, fit, X, U, outer_train_end, ctx, cfg.adaptive_h_cap)
        detector.sensitivity = guard
        guard.commission(detector)  # each sensor's reference, with the commissioned settings
    return SimpleNamespace(detector=detector, steady=steady, sensed=list(sensed),
                           commanded=list(commanded), outer_train_end=outer_train_end,
                           calib_end=calib_end, cfg=cfg, fit=fit)


def sensitivity_guard(cfg, fit, X, U, fit_end: int, ctx, cap: float,
                      excerpts: int = GOLDEN_EXCERPTS, length: int = GOLDEN_LENGTH,
                      span: int = GOLDEN_SPAN) -> SensitivityGuard:
    """capture the golden set, store both branches' prediction errors on it (fresh
    copies of the frozen predictors), measure every sensor's commissioned
    smallest caught step.
    """
    import copy
    from gad.l5_golden_set import capture
    from gad.l2_context import SettlingTracker, per_channel
    from gad.l1a_predictor import ArxWindow

    starts = [fit_end - span + i * (span // excerpts) for i in range(excerpts)]
    golden = capture(X, U, cfg.sensed, cfg.commanded, starts=starts, length=length,
                     source="the last day of the commissioning fit rows",
                     captured_at=f"rows {starts[0]}-{fit_end}")
    from gad.l5_detection_test import STEP_AT
    E, T, n = golden.X.shape
    q0 = np.zeros(n, dtype=np.uint8)

    def errors(Xe, Ue):
        """Both branches' errors on one excerpt, from fresh predictor copies.
        """
        la, lb = copy.deepcopy(fit.l1a), copy.deepcopy(fit.l1b)
        la._window = ArxWindow(la.n_sensed, la.n_commanded, la.L)
        la._prev_x = None
        lb.x_hat, lb.P = None, np.array(fit.bundle.l1b_P0)
        ra, rb = np.full((T, n), np.nan), np.full((T, n), np.nan)
        for t in range(T):
            sample = Sample(t=float(t), x=Xe[t], u=Ue[t], q=q0)
            ra[t] = Xe[t] - la.update(sample)
            rb[t] = Xe[t] - lb.update(sample)
        return ra, rb

    res_a = np.full((E, T, n), np.nan)
    res_b = np.full((E, T, n), np.nan)
    g_a = np.zeros((E, T, n))
    g_b = np.zeros((E, T, n))
    regimes = np.zeros((E, T, n), dtype=np.int64)
    for e in range(E):
        Xe, Ue = np.array(golden.X[e]), golden.U[e]
        res_a[e], res_b[e] = errors(Xe, Ue)
        settle = SettlingTracker(n, cfg.settle_samples)
        for t in range(T):
            regimes[e, t] = per_channel(settle.step(ctx.channel_ids(Ue[t])), n)
        for j in range(n):  # the predictors are linear: a unit step's response is exact
            Xs = Xe.copy()
            Xs[STEP_AT:, j] += 1.0
            sa, sb = errors(Xs, Ue)
            g_a[e, :, j] = np.nan_to_num(sa[:, j] - res_a[e, :, j])
            g_b[e, :, j] = np.nan_to_num(sb[:, j] - res_b[e, :, j])
    sigma = np.array([fit.l2a._pooled_reference(j) or np.nan for j in range(n)], dtype=float)
    test = SensitivityTest(res_a, res_b, regimes, sigma, lam=cfg.cusum_lambda, dwell=cfg.dwell_anomaly,
                           g_a=g_a, g_b=g_b)
    return SensitivityGuard(test, cap, golden.digest)
