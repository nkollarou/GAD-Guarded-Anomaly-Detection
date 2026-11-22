"""Fits every layer of GAD from clean data. `BundleConfig` holds all the
options, and `settings.py` sets the ones used in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import math

import numpy as np
from scipy.stats import chi2

from gad.l5_bundle import Bundle
from gad.l5_quarantine import Batch, Segment, split_segments
from gad.l5_quarantine import unpack as quarantine_unpack
from gad.l5_limits import EventScaleResidual
from gad.l4_decision import MonitorDecider, PerSensorDecider
from gad.determinism import deterministic_blas
from gad.detector import MONITOR_A, MONITOR_B, Detector
from gad.l6_evidence import observed_states
from gad.l1a_invariants import InvariantMiner, InvariantState
from gad.l1a_predictor import (RelationalPredictor,
                                    calibrate_own_lag_gate)
from gad.l1b_physics import StateSpacePredictor, TankBalance
from gad.l0_quality import fit_quality_ranges
from gad.l1a_predictor import ArxWindow
from gad.l2_residual import (
    BoundedScaleResidual, RegimeScaledResidual, measurement_resolution,
    measurement_resolution_segments,
)
from gad.l2_context import SettlingTracker, per_channel
from gad.l3_cusum import (Cusum, calibrate_h_by_context, calibrate_h_evt,
                            calibrate_k, count_episodes, drain_bound_steps)
from gad.types import Sample


@dataclass(frozen=True)
class BundleConfig:
    """Everything fixed by design that the refit needs."""

    sensed: tuple[str, ...]
    commanded: tuple[str, ...]
    table: tuple[TankBalance, ...]
    channel_ids: Callable[[np.ndarray], np.ndarray]
    stage_ids: Callable[[np.ndarray], np.ndarray]
    channel_stage: tuple[int, ...]
    drifting_channels: tuple[str, ...]
    budget_episodes_per_day: float
    budget_time_fraction: float
    monitor_share: dict[str, float]
    cusum_lambda: float
    samples_per_day: float
    l0_margin_k: float = 5.0
    cusum_cap_factor: float | None = None
    l3_per_context_h: bool = False
    l3_per_context_min_rows: int = 1000
    l3_per_context_min_typical_z: float = 0.1
    l3_per_context_long_calibration: bool = False
    l3_per_context_estimator: str = "bisection"
    l3_per_context_holdout: float = 0.0
    l3_per_context_refuse_empirical: bool = True
    l1b_innovation_gate: float | None = None
    l1b_gate_max_hold: int | None = None
    l1b_include_flows: bool = False
    l3_per_context_holdout_scope: str = "channel"
    l1a_own_lag_gate: float | None = None
    l1a_gate_max_hold: int | None = None
    l1a_gate_clean_share: float | None = None
    l3_fallback_excludes_thin_contexts: bool = False
    l0_step_lag: int = 1
    l0_step_basis: str = "median"
    l1b_target_gain: float = 0.1
    #: L4 decider settings — design, not learned; the pipeline states them
    degraded_abstention_threshold: float = 0.5
    dwell_anomaly: int = 5
    dwell_recover: int = 30
    dwell_window: int | None = None
    l2_kind: str = "window"
    l2_reference_from_commissioning_rows: bool = False
    l2_reference_min_rows: int = 1000
    l2_reference_min_gap: float = 4.0
    l2_half_life: float = 21_600.0
    l2_clip: float = 3.0
    l2_cap: float = 2.0
    l2_band: float = 1.0
    trusted_learning: bool = False
    learn_guard: int = 0
    per_channel_learning: bool = False
    per_sensor_alarms: bool = False
    full_budget_split: bool = False
    cross_channel_warm_calibration: bool = False
    unseen_context_pooled_init: bool = False
    sensitivity_measure_only: bool = False
    #: seconds within which channels entering alarm are grouped into one event
    alarm_group_window: float = 30.0
    use_invariants: bool = False
    use_controller: bool = False
    use_l1a_long_calibration: bool = False
    l1a_cross: bool = False
    l1a_cross_rule: str = "0044"
    adaptive_h: bool = False
    adaptive_h_cap: float = 2.0
    adaptive_h_period: float = 21_600.0
    adaptive_h_window: int = 345_600
    unified_l5: bool = False
    freeze_scales: bool = False
    sensitivity_budget: bool = False
    sensitivity_cap_by_channel: tuple[tuple[str, float], ...] | None = None
    settle_samples: int = 0
    naive_learning: bool = False

    def __post_init__(self) -> None:
        if self.l2_kind not in ("window", "bounded"):
            raise ValueError("l2_kind must be 'window' or 'bounded'")
    fit_frac: float = 0.6
    warm_frac: float = 0.2


@dataclass
class FitResult:
    """A candidate bundle plus the components that produced it, warmed up
        to the end of the calibration data. Commissioning continues the stream
        with these components; a later refit uses only `bundle`.
    """

    bundle: Bundle
    l1a: RelationalPredictor
    l1b: StateSpacePredictor
    miner: InvariantMiner
    l2a: RegimeScaledResidual
    l2b: RegimeScaledResidual
    l3a: Cusum
    l3b: Cusum
    resolution: np.ndarray
    invariant_state: InvariantState
    h_detail_a: dict
    h_detail_b: dict
    per_channel_budget: dict
    calibration_a: list | None = None
    calibration_b: list | None = None
    reach_a: dict | None = None
    reach_b: dict | None = None
    detectable: np.ndarray | None = None
    cap_recalibration: dict | None = None
    per_context_h: dict | None = None
    l1b_innovation_gate: dict | None = None
    l1a_own_lag_gate: dict | None = None
    reach_a_best_context: dict | None = None
    l2_reference_fit: dict | None = None


def make_l2(cfg: BundleConfig, n: int, floor: np.ndarray):
    """L2 for one branch, of the kind the configuration names."""
    if cfg.l2_kind == "bounded":
        kind = EventScaleResidual if cfg.unified_l5 else BoundedScaleResidual
        return kind(n, half_life=cfg.l2_half_life, clip=cfg.l2_clip,
                    cap=cfg.l2_cap, band=cfg.l2_band, scale_floor=floor,
                    pooled_init_for_unseen=cfg.unseen_context_pooled_init)
    return RegimeScaledResidual(n_channels=n, scale_floor=floor)


def _reference_fit_summary(sensed, report: dict) -> dict:
    """..."""
    per: dict[str, dict] = {}
    for j, ctx, n_rows, before, after in report["refitted"]:
        e = per.setdefault(sensed[j], {"n_contexts": 0, "max_ratio": None,
                                       "contexts": {}})
        e["n_contexts"] += 1
        ratio = None if not before else after / before
        e["contexts"][int(ctx)] = {"n_rows": n_rows, "before": before,
                                   "after": after, "ratio": ratio}
        if ratio is not None and (e["max_ratio"] is None or ratio > e["max_ratio"]):
            e["max_ratio"] = ratio
    skipped: dict[str, int] = {}
    for j, _ctx, _n in report["too_few_rows"]:
        skipped[sensed[j]] = skipped.get(sensed[j], 0) + 1
    close: dict[str, dict] = {}
    for j, ctx, _n, reached, fitted in report.get("already_close", []):
        e = close.setdefault(sensed[j], {"n_contexts": 0, "max_ratio": None})
        e["n_contexts"] += 1
        ratio = fitted / reached if reached else None
        if ratio is not None and (e["max_ratio"] is None or ratio > e["max_ratio"]):
            e["max_ratio"] = ratio
    return {"channels": per, "n_contexts_too_few_rows": skipped,
            "n_contexts_already_close": close}


def _l2_fields(cfg: BundleConfig, l2a, l2b, n: int) -> dict:
    """The bundle's L2 fields. A bounded learner's references are committed
        here, at the end of commissioning or a refit: they are the anchor.
    """
    if cfg.l2_kind == "bounded":
        l2a.commit_references()
        l2b.commit_references()
        for l2 in (l2a, l2b):
            if hasattr(l2, "commit"):
                l2.commit()
        empty = tuple({} for _ in range(n))
        return {"l2a_windows": empty, "l2b_windows": empty,
                "l2a_state": l2a.export_state(), "l2b_state": l2b.export_state()}
    return {"l2a_windows": l2a.export_windows(), "l2b_windows": l2b.export_windows()}


REACHABILITY_SIGMAS = 4.0


def sigma_d_per_channel(X: np.ndarray) -> np.ndarray:
    """Each channel's one-second noise, the unit the reference attack is in."""
    d = np.diff(X, axis=0)
    return np.std(d, axis=0) / np.sqrt(2.0)


def pooled_scales(l2, n_channels: int) -> np.ndarray:
    """Each channel's commissioned reference scale, or NaN where this L2 cannot
        report one. Only `BoundedScaleResidual` keeps a pooled reference; the plain
        `RegimeScaledResidual` does not, and must not be made to look blind for
        lacking it.
    """
    getter = getattr(l2, "_pooled_reference", None)
    if getter is None:
        return np.full(n_channels, np.nan)
    out = np.full(n_channels, np.nan)
    for j in range(n_channels):
        try:
            v = getter(j)
        except Exception:                      # noqa: BLE001 - absent is not an error
            v = None
        if v:
            out[j] = float(v)
    return out


def reachability(h: np.ndarray, k: np.ndarray, lam: float, sigma_d: np.ndarray,
                 scale: np.ndarray, range_sigma: np.ndarray,
                 n_sigma: float = REACHABILITY_SIGMAS) -> dict:
    """Each channel's implied detection threshold, and whether it is physically
        reachable at all.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        z_min = k + h * (1.0 - lam)
        implied = np.where(sigma_d > 0, z_min * scale / sigma_d, np.nan)
        z_ref = np.where(scale > 0, n_sigma * sigma_d / scale, np.nan)
        h_reach = np.maximum(0.0, z_ref - k) / (1.0 - lam)
    assessed = np.isfinite(implied) & np.isfinite(range_sigma)
    detectable = assessed & (implied <= range_sigma)
    return {"z_ref": z_ref, "h_reach": h_reach, "detectable": detectable,
            "assessed": assessed, "implied_sigma": implied,
            "range_sigma": range_sigma,
            "catches_reference": np.isfinite(h) & (h <= h_reach),
            "n_sigma": float(n_sigma)}


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Kept here rather than
        imported from `pipeline` because `gad` must not depend on it.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _poisson_rate_lower(count: int, exposure: float, alpha: float = 0.05) -> float:
    """Exact (Garwood) lower confidence bound on a Poisson rate. Zero events
        gives zero, which is what makes a short holdout unable to establish a
        breach from counts alone.
    """
    if count <= 0 or not (exposure > 0):
        return 0.0
    return float(chi2.ppf(alpha / 2.0, 2 * count) / 2.0 / exposure)


def _judge_on_holdout(evidence: np.ndarray, contexts: np.ndarray,
                      bars: dict[int, float], h_channel: float,
                      budget_episodes_per_day: float,
                      budget_time_fraction: float,
                      samples_per_day: float) -> dict:
    """Does this channel stay inside its own false-alarm budget on clean rows
        its bars were never fitted on?
    """
    ev = np.asarray(evidence, dtype=float)
    ctx = np.asarray(contexts, dtype=np.int64)
    if ev.shape != ctx.shape:
        raise ValueError("evidence and contexts must have the same shape")
    days = len(ev) / samples_per_day if samples_per_day else float("nan")
    if len(ev) == 0 or not (days > 0):
        # nothing to judge on: say so rather than passing the channel silently
        return {"n_rows": int(len(ev)), "kept": True, "verdict": "no holdout rows",
                "n_bars_judged": len(bars)}
    h_eff = np.full(len(ev), float(h_channel), dtype=float)
    for c, hv in bars.items():
        h_eff[ctx == c] = float(hv)
    alarm = np.nan_to_num(ev, nan=-np.inf) >= h_eff
    n_ep = count_episodes(alarm)
    n_alarm = int(alarm.sum())
    eps_realised = n_ep / days
    time_realised = float(alarm.mean())
    per_mode: dict[int, dict] = {}
    if bars:
        eps_share = budget_episodes_per_day / len(bars)
        time_share = budget_time_fraction / len(bars)
        for c, hv in sorted(bars.items()):
            mask = ctx == c
            rows = int(mask.sum())
            a_c = np.nan_to_num(ev, nan=-np.inf) >= float(hv)
            a_c &= mask
            n_a = int(a_c.sum())
            n_e = count_episodes(a_c)
            t_lo_c, _ = _wilson_interval(n_a, len(ev))
            e_lo_c = _poisson_rate_lower(n_e, days)
            certifiable = bool(eps_share * days >= 1.0)
            over = (t_lo_c > time_share) or (certifiable and e_lo_c > eps_share)
            measurable = rows >= 30
            per_mode[int(c)] = {
                "h": float(hv), "n_rows": rows, "alarm_rows": n_a,
                "episodes": int(n_e),
                "time_fraction_realised": float(n_a / len(ev)),
                "time_fraction_allowed": float(time_share),
                "time_fraction_95_lower": float(t_lo_c),
                "episodes_per_day_realised": float(n_e / days),
                "episodes_per_day_allowed": float(eps_share),
                "episodes_per_day_95_lower": float(e_lo_c),
                "measurable": measurable,
                "kept": bool(measurable and not over),
                "verdict": ("held" if measurable and not over
                            else ("too few holdout rows" if not measurable
                                  else "over its share")),
            }

    time_lo, _ = _wilson_interval(n_alarm, len(ev))
    eps_lo = _poisson_rate_lower(n_ep, days)
    over_time = time_lo > budget_time_fraction
    eps_certifiable = bool(budget_episodes_per_day * days >= 1.0)
    over_eps = eps_certifiable and eps_lo > budget_episodes_per_day
    return {
        "n_rows": int(len(ev)), "days": float(days),
        "n_bars_judged": len(bars),
        "n_rows_on_a_context_bar": int(sum(int((ctx == c).sum()) for c in bars)),
        "episodes": int(n_ep), "alarm_rows": n_alarm,
        "episodes_per_day_realised": float(eps_realised),
        "episodes_per_day_95_lower": float(eps_lo),
        "episodes_per_day_allowed": float(budget_episodes_per_day),
        "episodes_allowed_over_this_holdout": float(budget_episodes_per_day * days),
        "episodes_per_day_certifiable": eps_certifiable,
        "time_fraction_realised": time_realised,
        "time_fraction_95_lower": float(time_lo),
        "time_fraction_allowed": float(budget_time_fraction),
        "test": ("95% lower bound on the realised rate exceeds the budget; "
                 "the episode half binds only when this holdout can certify it"),
        "per_mode": per_mode,
        "modes_kept": sorted(c for c, v in per_mode.items() if v["kept"]),
        "modes_dropped": sorted(c for c, v in per_mode.items() if not v["kept"]),
        "kept": not (over_eps or over_time),
        "verdict": ("held" if not (over_eps or over_time)
                    else ("episodes" if over_eps and not over_time
                          else ("time" if over_time and not over_eps else "both"))),
    }


def operating_range_sigma(X: np.ndarray, sigma_d: np.ndarray) -> np.ndarray:
    """Each channel's full clean excursion in its own noise units,
        (max - min)/sigma_d: the largest deviation that could still be read as this
        channel doing something it has actually done.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        span = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
        return np.where(sigma_d > 0, span / sigma_d, np.nan)


def _thin_context_mask(ctx_col: np.ndarray, min_rows: int) -> np.ndarray:
    """Rows belonging to a context with at least `min_rows` samples."""
    keep = np.ones(len(ctx_col), dtype=bool)
    labels, counts = np.unique(ctx_col, return_counts=True)
    for lab, cnt in zip(labels, counts):
        if cnt < min_rows:
            keep &= ctx_col != lab
    return keep


def _calibrate_h(cfg: BundleConfig, evidence_cal: np.ndarray, time_frac: float,
                 eps_day: float, contexts: np.ndarray | None = None):
    n = len(cfg.sensed)
    h = np.full(n, np.nan)
    detail = {}
    if cfg.l3_fallback_excludes_thin_contexts and contexts is None:
        raise ValueError(
            "l3_fallback_excludes_thin_contexts is on but this _calibrate_h "
            "call was given no contexts, so it would fit the fallback bar on "
            "every row including the ones the flag excludes")
    thin = (cfg.l3_fallback_excludes_thin_contexts
            and cfg.l3_per_context_min_rows > 0)
    for i, ch in enumerate(cfg.sensed):
        col = evidence_cal[:, i]
        if not np.isfinite(col).any():
            continue
        dropped = None
        spd = cfg.samples_per_day
        if thin:
            keep = _thin_context_mask(np.asarray(contexts)[:, i],
                                      int(cfg.l3_per_context_min_rows))
            if not keep.all() and keep.any():
                dropped = int((~keep).sum())
                spd = spd * len(col[keep]) / len(col)
                col = col[keep]
        try:
            res = calibrate_h_evt(col, eps_day, time_frac, spd)
        except ValueError:
            continue  # budget unreachable even empirically: stays NaN, never alarms
        h[i] = res.h
        detail[ch] = {"h": res.h, "method": res.method, "n_peaks": res.n_peaks,
                      "gpd_shape": res.shape}
        if dropped:
            detail[ch]["thin_context_rows_dropped"] = dropped
            detail[ch]["thin_context_row_share"] = dropped / len(evidence_cal)
    return h, detail


def _long_calibration_a(cfg: BundleConfig, l1a_state, X_fit, U_fit, n_warm: int,
                        resolution, k_a, evidence_a_cal, h_a, detail_a, budget,
                        X_warm=None, U_warm=None, regimes_cal=None,
                        scaled_a_cal=None):
    """Stream the fit rows through a separate copy of
        L1a and a separate L2 (the live components are not touched), score them
        with the calibrated slack `k_a` after `n_warm` rows of L2 warm-up, and
        re-read `h` from that evidence plus the calibration window's — for
        channels whose L1a model has no fitted coefficients only.
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
    h_long, detail_long = _calibrate_h(cfg, combined, *budget,
                                       contexts=ctx_combined)
    h, detail = np.array(h_a, dtype=np.float64), dict(detail_a)
    for i, ch in enumerate(sensed):
        if l1a_state.model_choice.get(ch) in ("fitted", "cross") or ch not in detail_long:
            continue
        h[i] = h_long[i]
        detail[ch] = {**detail_long[ch], "calibration": "fit+cal (0043)",
                      "n_calibration_rows": int(len(combined))}

    streams = {"fit+cal (0043)": (
        combined,
        ctx_combined,
        _with_cal(z_fit, scaled_a_cal, np.float64))}
    if cfg.cross_channel_warm_calibration and X_warm is not None and len(X_warm):
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
            streams["warm+cal (0059 3.7)"] = (
                warm_combined,
                ctx_warm_combined,
                _with_cal(z_warm, scaled_a_cal, np.float64))
            h_warm, detail_warm = _calibrate_h(cfg, warm_combined, *budget,
                                               contexts=ctx_warm_combined)
            for i, ch in enumerate(sensed):
                if (l1a_state.model_choice.get(ch) not in ("fitted", "cross")
                        or ch not in detail_warm):
                    continue
                h[i] = h_warm[i]
                detail[ch] = {**detail_warm[ch], "calibration": "warm+cal (0059 3.7)",
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
    """Fit one bundle, with the BLAS pools pinned so the result is
        bit-reproducible.
    """
    with deterministic_blas():
        return _fit_bundle(cfg, X_fit, U_fit, X_warm, U_warm, X_cal, U_cal, X_l0, version)


def _guarded_dwell(cfg) -> int | None:
    """The dwell `Cusum` must be able to sustain, or `None` when the guard does
        not apply.
    """
    return None if cfg.dwell_window is not None else cfg.dwell_anomaly


def _ratio(new: np.ndarray, old: np.ndarray) -> list:
    """`new/old` per channel, with non-finite entries as None so the record
        serialises and a channel without a threshold is visibly absent.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.asarray(new, dtype=float) / np.asarray(old, dtype=float)
    return [None if not np.isfinite(v) else float(v) for v in r]


def _finite_list(a: np.ndarray) -> list:
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, dtype=float)]


def _calibrate_gate(cfg: BundleConfig, l1a, row_blocks) -> None:
    """Replace L1a's fixed gate threshold with one derived per channel from
        clean rows. A no-op unless `l1a_gate_clean_share`
        is set. Mutates `l1a` because the derived thresholds are part of the
        fitted model and must version and roll back with it.
    """
    if cfg.l1a_gate_clean_share is None:
        return
    if cfg.l1a_own_lag_gate is not None:
        raise ValueError(
            "set l1a_gate_clean_share OR l1a_own_lag_gate, not both: the "
            "share derives the threshold the float would fix")
    blocks = [(np.asarray(X, dtype=float), np.asarray(U, dtype=float))
              for X, U in row_blocks if len(X)]
    if not blocks:
        raise ValueError("l1a_gate_clean_share needs clean rows to fit on")
    X = np.concatenate([b[0] for b in blocks])
    U = np.concatenate([b[1] for b in blocks])
    l1a.own_lag_gate = np.full(l1a.n_sensed, np.inf)
    l1a.own_lag_gate = calibrate_own_lag_gate(
        l1a, X, U, float(cfg.l1a_gate_clean_share))


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

    l1a = RelationalPredictor(sensed, commanded, cross_channel=cfg.l1a_cross,
                              cross_rule=cfg.l1a_cross_rule,
                              own_lag_gate=cfg.l1a_own_lag_gate,
                              gate_max_hold=cfg.l1a_gate_max_hold)
    l1a.fit_from_arrays(X_fit, U_fit)
    _calibrate_gate(cfg, l1a, [(X_warm, U_warm), (X_cal, U_cal)])
    l1b = StateSpacePredictor(sensed, commanded, table=cfg.table,
                             target_gain=cfg.l1b_target_gain,
                             innovation_gate=cfg.l1b_innovation_gate,
                             gate_max_hold=cfg.l1b_gate_max_hold,
                             include_flows=cfg.l1b_include_flows)
    l1b.fit_from_arrays(X_fit, U_fit)
    P0 = l1b.P.copy()
    miner = InvariantMiner(sensed, cfg.channel_ids, stage_fn=cfg.stage_ids,
                           channel_stage=cfg.channel_stage, drifting=cfg.drifting_channels)
    inv_state = miner.fit_from_arrays(X_fit, U_fit)

    resolution = measurement_resolution(X_fit)
    l2a = make_l2(cfg, n_sensed, resolution)
    l2b = make_l2(cfg, n_sensed, resolution)
    q0 = np.zeros(n_sensed, dtype=np.uint8)

    settle = SettlingTracker(n_sensed, cfg.settle_samples)

    def residuals(t: int, x: np.ndarray, u: np.ndarray):
        sample = Sample(t=float(t), x=x, u=u, q=q0)
        return (x - l1a.update(sample), x - l1b.update(sample),
                settle.step(cfg.channel_ids(u)))

    keep_raw = (cfg.l2_kind == "bounded"
                and cfg.l2_reference_from_commissioning_rows)
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

    l2_reference_fit = None
    if keep_raw:
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
        l2_reference_fit = {
            "min_rows": int(cfg.l2_reference_min_rows),
            "min_gap": float(cfg.l2_reference_min_gap),
            "n_commissioning_rows": int(len(R_a)),
            "rescored_channels_a": sorted({sensed[int(j)] for j, *_ in rep_a["refitted"]}),
            "rescored_channels_b": sorted({sensed[int(j)] for j, *_ in rep_b["refitted"]}),
            "a": _reference_fit_summary(sensed, rep_a),
            "b": _reference_fit_summary(sensed, rep_b),
        }
        raw_a = raw_b = raw_ctx = None   # commissioning-only; let it go

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
    h_a, h_detail_a = _calibrate_h(cfg, evidence_a_cal, *per_channel_budget[MONITOR_A],
                                   contexts=regimes_cal)
    h_b, h_detail_b = _calibrate_h(cfg, evidence_b_cal, *per_channel_budget[MONITOR_B],
                                   contexts=regimes_cal)

    cap_recalibration = None
    if cfg.cusum_cap_factor is not None:
        h_a_uncapped, h_b_uncapped = h_a, h_b
        l3a = Cusum(k_a, h_a, lam=cfg.cusum_lambda, cap_factor=cfg.cusum_cap_factor,
                    dwell=_guarded_dwell(cfg))
        l3b = Cusum(k_b, h_b, lam=cfg.cusum_lambda, cap_factor=cfg.cusum_cap_factor,
                    dwell=_guarded_dwell(cfg))
        evidence_a_cal = np.array([l3a.update(r) for r in scaled_a_cal])
        evidence_b_cal = np.array([l3b.update(r) for r in scaled_b_cal])
        h_a, h_detail_a = _calibrate_h(cfg, evidence_a_cal, *per_channel_budget[MONITOR_A],
                                       contexts=regimes_cal)
        h_b, h_detail_b = _calibrate_h(cfg, evidence_b_cal, *per_channel_budget[MONITOR_B],
                                       contexts=regimes_cal)
        cap_recalibration = {
            "cap_factor": float(cfg.cusum_cap_factor),
            "h_ratio_a": _ratio(h_a, h_a_uncapped),
            "h_ratio_b": _ratio(h_b, h_b_uncapped),
            "drain_bound_s_a": _finite_list(drain_bound_steps(
                cfg.cusum_cap_factor, k_a, h_a, cfg.cusum_lambda)),
            "drain_bound_s_b": _finite_list(drain_bound_steps(
                cfg.cusum_cap_factor, k_b, h_b, cfg.cusum_lambda)),
        }
    calibration_a = [evidence_a_cal[:, i] for i in range(n_sensed)]
    long_streams: dict = {}
    if cfg.use_l1a_long_calibration:
        h_a, h_detail_a, combined, long_streams = _long_calibration_a(
            cfg, l1a.state, X_fit, U_fit, n_warm, resolution, k_a, evidence_a_cal,
            h_a, h_detail_a, per_channel_budget[MONITOR_A], X_warm, U_warm,
            regimes_cal=regimes_cal, scaled_a_cal=scaled_a_cal)
        if combined is not None:
            calibration_a = [combined[:, i] if h_detail_a.get(ch, {}).get("calibration") else evidence_a_cal[:, i]
                             for i, ch in enumerate(sensed)]
    calibration_b = [evidence_b_cal[:, i] for i in range(evidence_b_cal.shape[1])]
    l3a.h, l3b.h = h_a, h_b
    per_context_h = None
    bundle_h_by_context = None
    if cfg.l3_per_context_h:
        time_frac, eps_day = per_channel_budget[MONITOR_A]
        table: dict[int, np.ndarray] = {}
        detail: dict[str, dict] = {}
        for j, ch in enumerate(sensed):
            ev_j, ctx_j = evidence_a_cal[:, j], regimes_cal[:, j]
            z_long = None
            fitted_on = "calibration window"
            if cfg.l3_per_context_long_calibration:
                label = h_detail_a.get(ch, {}).get("calibration")
                pair = long_streams.get(label) if label else None
                if pair is not None and pair[1] is not None:
                    ev_j, ctx_j = pair[0][:, j], pair[1][:, j]
                    z_long = pair[2][:, j] if pair[2] is not None else None
                    fitted_on = label
            how: dict[int, dict] = {}
            pooled: dict = {}
            cut = (int(len(ev_j) * (1.0 - cfg.l3_per_context_holdout))
                   if cfg.l3_per_context_holdout > 0.0 else len(ev_j))
            ev_fit, ctx_fit = ev_j[:cut], ctx_j[:cut]
            min_rows_fit = max(1, int(round(
                cfg.l3_per_context_min_rows * (cut / len(ev_j)))))
            z_fit = (scaled_a_cal[:, j] if fitted_on == "calibration window"
                     else z_long)
            if z_fit is not None and cut < len(ev_j):
                z_fit = z_fit[:cut]
            got, thin = calibrate_h_by_context(
                ev_fit, ctx_fit, eps_day, time_frac,
                cfg.samples_per_day, min_rows=min_rows_fit,
                z=z_fit,
                min_typical_z=cfg.l3_per_context_min_typical_z,
                detail=how, estimator=cfg.l3_per_context_estimator,
                h_channel=float(h_a[j]),
                pooled_report=pooled,
                refuse_empirical=(cfg.l3_per_context_refuse_empirical
                                  and cfg.l3_per_context_holdout <= 0.0))
            holdout = None
            if cut < len(ev_j):
                holdout = _judge_on_holdout(
                    ev_j[cut:], ctx_j[cut:], got, float(h_a[j]),
                    eps_day, time_frac, cfg.samples_per_day)
                if (cfg.l3_per_context_holdout_scope == "mode"
                        and holdout.get("per_mode")):
                    drop = set(holdout["modes_dropped"])
                    if drop:
                        holdout["refused_bars"] = {int(c): float(got[c])
                                                   for c in sorted(drop)
                                                   if c in got}
                        holdout["refused_how"] = {int(c): how[c] for c in drop
                                                  if c in how}
                        thin = sorted(set(thin) | drop)
                        got = {c: v for c, v in got.items() if c not in drop}
                        how = {c: v for c, v in how.items() if c not in drop}
                elif not holdout["kept"]:
                    holdout["refused_bars"] = {int(c): float(v)
                                               for c, v in sorted(got.items())}
                    holdout["refused_how"] = {int(c): v for c, v in how.items()}
                    thin = sorted(set(thin) | set(got))
                    got, how = {}, {}
            for c, hv in got.items():
                table.setdefault(c, np.full(n_sensed, np.nan))[j] = hv
            detail[ch] = {"n_own": len(got), "n_fallback": len(thin),
                          "fitted_on": fitted_on,
                          "n_rows_fitted_on": int(len(ev_j)),
                          "typical_z": float(np.median(np.abs(
                              scaled_a_cal[:, j][np.isfinite(scaled_a_cal[:, j])]))
                              if np.isfinite(scaled_a_cal[:, j]).any() else np.nan),
                          "h_channel": float(h_a[j]),
                          "h_by_context": {int(c): float(v) for c, v in got.items()},
                          "h_by_context_how": {int(c): v for c, v in how.items()}}
            if pooled:
                detail[ch]["pooled_fit"] = pooled
            if holdout is not None:
                detail[ch]["holdout"] = holdout
        if table:
            l3a.h_by_context = table
            bundle_h_by_context = table
            per_context_h = {"min_rows": cfg.l3_per_context_min_rows,
                             "holdout_fraction": float(cfg.l3_per_context_holdout),
                             "holdout_scope": cfg.l3_per_context_holdout_scope,
                             "min_rows_scaled_by_holdout": True,
                             "min_typical_z": cfg.l3_per_context_min_typical_z,
                             "long_calibration_for_per_context_bars":
                                 bool(cfg.l3_per_context_long_calibration),
                             "estimator": cfg.l3_per_context_estimator,
                             "n_calibration_rows": int(len(evidence_a_cal)),
                             "long_calibration_used_for_channel_h":
                                 bool(cfg.use_l1a_long_calibration),
                             "channels": detail}
    l3a.s_pos[:] = l3a.s_neg[:] = 0.0   # the ceiling pass re-streamed the window
    l3b.s_pos[:] = l3b.s_neg[:] = 0.0

    l1b_gate = l1b.gate_report() if hasattr(l1b, "gate_report") else None
    l1a_gate = l1a.gate_report() if hasattr(l1a, "gate_report") else None

    bundle = Bundle(
        version=version,
        quality_ranges=fit_quality_ranges(sensed, X_l0, margin_k=cfg.l0_margin_k,
                                          step_lag=cfg.l0_step_lag,
                                          step_basis=cfg.l0_step_basis),
        l1a_state=l1a.state,
        l1b_Q=l1b.Q, l1b_R=l1b.R, l1b_P0=P0,
        invariant_state=inv_state,
        resolution=resolution,
        k_a=k_a, h_a=h_a, k_b=k_b, h_b=h_b,
        h_a_by_context=bundle_h_by_context,
        commanded_states=observed_states(U_fit, commanded),
        **_l2_fields(cfg, l2a, l2b, n_sensed),
    )
    scale_a = pooled_scales(l2a, n_sensed)
    sd_fit = sigma_d_per_channel(X_fit)
    span_sigma = operating_range_sigma(X_fit, sd_fit)
    reach_a = reachability(h_a, np.asarray(k_a), cfg.cusum_lambda, sd_fit, scale_a,
                           span_sigma)
    reach_a_best_context = None
    if bundle_h_by_context:
        best = np.array(h_a, dtype=float)
        for v in bundle_h_by_context.values():
            m = np.isfinite(v)
            best[m] = np.minimum(best[m], np.asarray(v)[m])
        r = reachability(best, np.asarray(k_a), cfg.cusum_lambda, sd_fit, scale_a,
                         span_sigma)
        reach_a_best_context = {
            "h": [float(x) for x in best],
            "implied_sigma": _finite_list(r["implied_sigma"]),
            "detectable": [bool(x) for x in r["detectable"]],
            "n_detectable": int(np.sum(r["detectable"])),
            "note": ("the bar in each channel's most sensitive context; the gate "
                     "verdict uses the channel-wide bar, which is what holds "
                     "where a context has none of its own"),
        }
    scale_b = pooled_scales(l2b, n_sensed)
    reach_b = reachability(h_b, np.asarray(k_b), cfg.cusum_lambda, sd_fit, scale_b,
                           span_sigma)
    unknown = ~reach_a["assessed"] & ~reach_b["assessed"]
    detectable = reach_a["detectable"] | reach_b["detectable"] | unknown
    return FitResult(bundle, l1a, l1b, miner, l2a, l2b, l3a, l3b, resolution, inv_state,
                     h_detail_a, h_detail_b, per_channel_budget, calibration_a,
                     calibration_b, reach_a, reach_b, detectable, cap_recalibration,
                     per_context_h, l1b_gate, l1a_gate, reach_a_best_context,
                     l2_reference_fit)


def detector_from_bundle(cfg: BundleConfig, bundle: Bundle, decider=None,
                         learn_scales: bool = True) -> Detector:
    """A detector carrying `bundle`, with empty runtime state — lag
        history, Kalman estimate, CUSUM accumulators and dwell counters all
        start cold. Used to score a candidate bundle on the golden set without touching the live detector.
    """
    sensed, commanded = list(cfg.sensed), list(cfg.commanded)
    n = len(sensed)

    l1a = RelationalPredictor(sensed, commanded)
    l1a.state = bundle.l1a_state
    l1b = StateSpacePredictor(sensed, commanded, table=cfg.table,
                             target_gain=cfg.l1b_target_gain,
                             innovation_gate=cfg.l1b_innovation_gate,
                             gate_max_hold=cfg.l1b_gate_max_hold,
                             include_flows=cfg.l1b_include_flows)
    l1b.Q, l1b.R, l1b.P = np.array(bundle.l1b_Q), np.array(bundle.l1b_R), np.array(bundle.l1b_P0)
    miner = InvariantMiner(sensed, cfg.channel_ids, stage_fn=cfg.stage_ids,
                           channel_stage=cfg.channel_stage, drifting=cfg.drifting_channels)
    miner.state = bundle.invariant_state

    l2a = make_l2(cfg, n, bundle.resolution)
    l2b = make_l2(cfg, n, bundle.resolution)
    for l2, windows, state in ((l2a, bundle.l2a_windows, bundle.l2a_state),
                               (l2b, bundle.l2b_windows, bundle.l2b_state)):
        if state is not None:
            l2.load_state(state)
            if hasattr(l2, "commit"):
                l2.commit()
        else:
            l2.load_windows(windows)
    l3a = Cusum(np.array(bundle.k_a), np.array(bundle.h_a), lam=cfg.cusum_lambda,
                cap_factor=cfg.cusum_cap_factor, dwell=_guarded_dwell(cfg),
                h_by_context=(None if bundle.h_a_by_context is None else
                              {int(c): np.array(v)
                               for c, v in bundle.h_a_by_context.items()}))
    l3b = Cusum(np.array(bundle.k_b), np.array(bundle.h_b), lam=cfg.cusum_lambda,
                cap_factor=cfg.cusum_cap_factor, dwell=_guarded_dwell(cfg))

    if decider is None:
        decider = MonitorDecider(cfg.degraded_abstention_threshold, cfg.dwell_anomaly,
                                 cfg.dwell_recover, bundle_id=bundle.bundle_id)
    per_sensor = None
    if cfg.per_sensor_alarms:
        per_sensor = PerSensorDecider(sensed, cfg.degraded_abstention_threshold,
                                      cfg.dwell_anomaly, cfg.dwell_recover,
                                      bundle_id=bundle.bundle_id,
                                      group_window=cfg.alarm_group_window,
                                      dwell_window=cfg.dwell_window)
    return Detector(sensed, l1a, l1b, l2a, l2b, l3a, l3b, miner, decider, cfg.channel_ids,
                    settle_samples=cfg.settle_samples,
                    naive_learning=cfg.naive_learning,
                    n_possible_tests=n + len(l1b.state_names) + n, learn_scales=learn_scales,
                    trusted_learning=cfg.trusted_learning, learn_guard=cfg.learn_guard,
                    per_channel_learning=cfg.per_channel_learning, per_sensor=per_sensor,
                    use_invariants=cfg.use_invariants)


def fit_bundle_segments(cfg: BundleConfig, segments, version: int) -> FitResult:
    """Pinned for the same reason as `fit_bundle`."""
    with deterministic_blas():
        return _fit_bundle_segments(cfg, segments, version)


def _fit_bundle_segments(cfg: BundleConfig, segments, version: int) -> FitResult:
    """`fit_bundle` over segments that are contiguous individually but not
        with each other — what quarantine produces.
    """
    sensed, commanded = list(cfg.sensed), list(cfg.commanded)
    n_sensed = len(sensed)
    segments = [(np.asarray(X, dtype=np.float64), np.asarray(U, dtype=np.float64), M)
                for X, U, M in (quarantine_unpack(s) for s in segments)]
    if not segments:
        raise ValueError("fit_bundle_segments requires at least one segment")
    total = sum(len(X) for X, _, _ in segments)
    fit_end = int(total * cfg.fit_frac)
    warm_end = fit_end + int(total * cfg.warm_frac)
    if not 0 < fit_end < warm_end < total:
        raise ValueError(f"{total} clean rows are too few to split 60/20/20")

    fit_segments, _, _ = split_segments(tuple(segments), cfg.fit_frac, cfg.warm_frac)
    fit_rows = [(X, U) for X, U, _ in fit_segments]
    l1a = RelationalPredictor(sensed, commanded, cross_channel=cfg.l1a_cross,
                              cross_rule=cfg.l1a_cross_rule,
                              own_lag_gate=cfg.l1a_own_lag_gate,
                              gate_max_hold=cfg.l1a_gate_max_hold)
    l1a.fit_from_segments(fit_rows)
    _calibrate_gate(cfg, l1a, fit_rows)
    l1b = StateSpacePredictor(sensed, commanded, table=cfg.table,
                             target_gain=cfg.l1b_target_gain,
                             innovation_gate=cfg.l1b_innovation_gate,
                             gate_max_hold=cfg.l1b_gate_max_hold,
                             include_flows=cfg.l1b_include_flows)
    l1b.fit_from_segments(fit_rows)
    P0 = l1b.P.copy()
    miner = InvariantMiner(sensed, cfg.channel_ids, stage_fn=cfg.stage_ids,
                           channel_stage=cfg.channel_stage, drifting=cfg.drifting_channels)
    inv_state = miner.fit_from_segments(fit_rows)

    resolution = measurement_resolution_segments(fit_rows)
    l2a = make_l2(cfg, n_sensed, resolution)
    l2b = make_l2(cfg, n_sensed, resolution)
    q0 = np.zeros(n_sensed, dtype=np.uint8)

    scaled_a_cal, scaled_b_cal, cal_breaks, regimes_cal = [], [], [], []
    settle = SettlingTracker(n_sensed, cfg.settle_samples)
    g = 0
    for X, U, M in segments:
        seg_start, seg_end = g, g + len(X)
        g = seg_end
        if seg_end <= fit_end:
            continue
        lo = max(fit_end, seg_start) - seg_start
        l1a._window = ArxWindow(l1a.n_sensed, l1a.n_commanded, l1a.L)  # no lags across a gap
        l1b.x_hat = None  # re-bootstrap the level estimate
        settle.reset()
        for t in range(lo, len(X)):
            sample = Sample(t=float(t), x=X[t], u=U[t], q=q0)
            r_a = X[t] - l1a.update(sample)
            r_b = X[t] - l1b.update(sample)
            regime = settle.step(cfg.channel_ids(U[t]))
            scaled_a = l2a.update(r_a, regime, True, M[t])
            scaled_b = l2b.update(r_b, regime, True, M[t])
            if seg_start + t >= warm_end:
                if not scaled_a_cal or seg_start + t == max(warm_end, seg_start):
                    cal_breaks.append(len(scaled_a_cal))  # a fresh CUSUM starts here
                scaled_a_cal.append(scaled_a)
                scaled_b_cal.append(scaled_b)
                regimes_cal.append(per_channel(regime, n_sensed).copy())
    scaled_a_cal = np.array(scaled_a_cal)
    scaled_b_cal = np.array(scaled_b_cal)
    regimes_cal = np.array(regimes_cal)
    if len(scaled_a_cal) == 0:
        raise ValueError("no calibration rows survived the split")
    if cfg.use_l1a_long_calibration:
        raise NotImplementedError("docs/decisions/0043 is implemented for contiguous fits only")

    k_a = calibrate_k(scaled_a_cal)
    k_b = calibrate_k(scaled_b_cal)
    l3a = Cusum(k_a, lam=cfg.cusum_lambda)
    l3b = Cusum(k_b, lam=cfg.cusum_lambda)
    evidence_a, evidence_b = [], []
    breaks = set(cal_breaks)
    for i in range(len(scaled_a_cal)):
        if i in breaks and i > 0:  # evidence never carries across a gap
            l3a.s_pos[:] = l3a.s_neg[:] = 0.0
            l3b.s_pos[:] = l3b.s_neg[:] = 0.0
        evidence_a.append(l3a.update(scaled_a_cal[i]))
        evidence_b.append(l3b.update(scaled_b_cal[i]))
    evidence_a_cal, evidence_b_cal = np.array(evidence_a), np.array(evidence_b)

    n_b_channels = len(l1b.state_names)
    per_channel_budget = {
        MONITOR_A: (cfg.monitor_share[MONITOR_A] * cfg.budget_time_fraction / n_sensed,
                    cfg.monitor_share[MONITOR_A] * cfg.budget_episodes_per_day / n_sensed),
        MONITOR_B: (cfg.monitor_share[MONITOR_B] * cfg.budget_time_fraction / n_b_channels,
                    cfg.monitor_share[MONITOR_B] * cfg.budget_episodes_per_day / n_b_channels),
    }
    h_a, h_detail_a = _calibrate_h(cfg, evidence_a_cal, *per_channel_budget[MONITOR_A],
                                   contexts=regimes_cal)
    h_b, h_detail_b = _calibrate_h(cfg, evidence_b_cal, *per_channel_budget[MONITOR_B],
                                   contexts=regimes_cal)
    l3a.h, l3b.h = h_a, h_b

    bundle = Bundle(
        version=version,
        quality_ranges=fit_quality_ranges(
            sensed, np.concatenate([X for X, _, _ in segments]), margin_k=cfg.l0_margin_k,
            step_lag=cfg.l0_step_lag, step_basis=cfg.l0_step_basis),
        l1a_state=l1a.state,
        l1b_Q=l1b.Q, l1b_R=l1b.R, l1b_P0=P0,
        invariant_state=inv_state,
        resolution=resolution,
        k_a=k_a, h_a=h_a, k_b=k_b, h_b=h_b,
        commanded_states=observed_states(np.concatenate([U for _, U in fit_rows]), commanded),
        **_l2_fields(cfg, l2a, l2b, n_sensed),
    )
    return FitResult(bundle, l1a, l1b, miner, l2a, l2b, l3a, l3b, resolution, inv_state,
                     h_detail_a, h_detail_b, per_channel_budget)


def refit_batch(cfg: BundleConfig, batch: Batch, version: int) -> Bundle:
    """Candidate bundle from a quarantined batch: its segments are split
        60/20/20 by sample count into fit / warm-up / calibration, and L0's
        ranges come from all of them.
    """
    if not batch.accepted:
        raise ValueError(f"batch was refused by quarantine: {'; '.join(batch.reasons)}")
    return fit_bundle_segments(cfg, batch.segments, version).bundle


def split_window(n: int, cfg: BundleConfig) -> tuple[int, int]:
    """Row boundaries (fit_end, warm_end) of a refit window of n rows."""
    fit_end = int(n * cfg.fit_frac)
    warm_end = fit_end + int(n * cfg.warm_frac)
    if not 0 < fit_end < warm_end < n:
        raise ValueError(f"a window of {n} rows is too short to split")
    return fit_end, warm_end


def refit_window(cfg: BundleConfig, X: np.ndarray, U: np.ndarray, version: int) -> Bundle:
    """Candidate bundle from one time-ordered window: fit / warm-up /
        calibration in order on its first 60% / next 20% / last 20%, and L0's
        ranges on the whole window.
    """
    fit_end, warm_end = split_window(len(X), cfg)
    return fit_bundle(
        cfg,
        X[:fit_end], U[:fit_end],
        X[fit_end:warm_end], U[fit_end:warm_end],
        X[warm_end:], U[warm_end:],
        X, version,
    ).bundle

