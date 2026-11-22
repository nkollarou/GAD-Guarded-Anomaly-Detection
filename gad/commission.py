"""Fit GAD once on clean data (commissioning).

`commission()` fits the two predictors, warms up the means and standard
deviations, sets the thresholds and records the golden set. The ready
detector is returned as `.detector`.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gad.fit import BundleConfig, fit_bundle
from gad.swat import CONTROLLER_STRUCTURE
from gad.l4_decision import MonitorDecider, PerSensorDecider
from gad.detector import MONITOR_A, MONITOR_B, MONITOR_CTRL, MONITOR_INVARIANT, Detector
from gad.l6_evidence import EvidenceBuilder
from gad.l1b_controller import calibrate_controller, fit_controller
from gad.l1a_invariants import drifting_channels
from gad.l1b_physics import PHYSICS_TABLE
from gad.l0_quality import QualityChecker, fit_quality_ranges
from gad.l5_detection_test import SensitivityGuard, SensitivityTest
from gad.l4_thresholds import AdaptiveThreshold
from gad.types import Sample
from gad import settings

#: the converted SWaT normal-operation recording (`build_datasets.py --convert`)
NORMAL_PARQUET = Path("data/swat_normal.parquet")
#: print what commissioning fits
VERBOSE = False


def _log(*args, **kwargs) -> None:
    if VERBOSE:
        print(*args, **kwargs)


def domain(name: str = "swat") -> SimpleNamespace:
    """What commissioning needs to know about the plant: the clean recording,
        the adapter's channel split, context builder and commissioning start, and
        the physics table.
    """
    if name != "swat":
        raise ValueError(f"only SWaT is supported here, not {name!r}")
    import gad.swat as a
    return SimpleNamespace(parquet=NORMAL_PARQUET,
                           split_channels=a.split_channels, build_contexts=a.build_contexts,
                           find_commissioning_start=a.find_commissioning_start,
                           table=PHYSICS_TABLE, relations=physics_relation_names,
                           l0_step_basis="median", l0_step_lag=1)


OUTER_TRAIN_FRAC = 0.7


SAMPLES_PER_DAY = 86400  # 1 Hz


BUDGET_EPISODES_PER_DAY = settings.FALSE_ALARMS_PER_DAY


BUDGET_TIME_FRACTION = settings.SHARE_OF_TIME_IN_ALARM


MONITOR_BUDGET_SHARE = {MONITOR_INVARIANT: 0.60, MONITOR_A: 0.20, MONITOR_B: 0.20}


MONITOR_BUDGET_SHARE[MONITOR_CTRL] = 0.10


def equal_budget_share(use_invariants: bool = False, use_controller: bool = False,
                       use_mass_balance: bool = False) -> dict[str, float]:
    """..."""
    active = [MONITOR_A, MONITOR_B]
    if use_invariants:
        active.append(MONITOR_INVARIANT)
    if use_controller:
        active.append(MONITOR_CTRL)
    if use_mass_balance:
        active.append("l1b_mass_balance")
    return {m: 1.0 / len(active) for m in active}


CUSUM_DECAY_LAMBDA = settings.CUSUM_LAMBDA


DEGRADED_ABSTENTION_THRESHOLD = settings.DEGRADED_SHARE


L0_MARGIN_K = settings.L0_RANGE_MARGIN


DWELL_ANOMALY = settings.ALARM_RAISE_SECONDS


DWELL_RECOVER = settings.ALARM_CLEAR_SECONDS


class RecordingDecider:
    """Wraps `MonitorDecider` to record each monitor's raw verdict, so the
        gate can report every monitor's own contribution to the false-alarm
        budget separately. `Decision` stays quantised — this lives in the
        pipeline, not in the detector.
    """

    def __init__(self, inner: MonitorDecider) -> None:
        self.inner = inner
        self.fired: list[dict[str, bool]] = []
        self.abstentions: list[float] = []

    @property
    def bundle_id(self) -> str:
        return self.inner.bundle_id

    @bundle_id.setter
    def bundle_id(self, value: str) -> None:
        self.inner.bundle_id = value  # a bundle swap renames the decider it wraps

    def update(self, t, monitors, abstention_fraction):
        self.fired.append(dict(monitors))
        self.abstentions.append(abstention_fraction)
        return self.inner.update(t, monitors, abstention_fraction)


class RecordingPerSensorDecider:
    """Wraps `PerSensorDecider` the way `RecordingDecider` wraps
        `MonitorDecider`, and for the same reason: the gate reports where the
        false-alarm budget went, and it cannot do that from a decider the
        detector never calls.
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        self.fired: list[np.ndarray] = []
        self.abstentions: list[float] = []

    @property
    def bundle_id(self) -> str:
        return self.inner.bundle_id

    @bundle_id.setter
    def bundle_id(self, value: str) -> None:
        self.inner.bundle_id = value

    def update(self, t, per_channel_fired, abstention_fraction):
        self.fired.append(np.asarray(per_channel_fired, dtype=bool).copy())
        self.abstentions.append(abstention_fraction)
        return self.inner.update(t, per_channel_fired, abstention_fraction)

    def __getattr__(self, name):
        """Delegate to the wrapped decider."""
        try:
            inner = object.__getattribute__(self, "inner")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(inner, name)


def fill_gaps(frame: pd.DataFrame, channels: list[str]) -> tuple[pd.DataFrame, dict]:
    """Forward- then backward-fill non-finite values, for FITTING arrays only."""
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


def l0_checker(steady: pd.DataFrame, sensed: list[str], step_lag: int = 1,
               step_basis: str = "median") -> QualityChecker:
    """L0 as commissioned in pipeline/convert.py: ranges from the clean
        steady window, margin_k=5.0. `step_lag` is 1 for every reported result
        and only the WADI L0 control arm passes anything else.
    """
    return QualityChecker(fit_quality_ranges(sensed, steady[sensed].to_numpy(),
                                             margin_k=L0_MARGIN_K, step_lag=step_lag,
                                             step_basis=step_basis))


def bundle_config(sensed, commanded, ctx, drifting=(), table=None) -> BundleConfig:
    """The fixed design every refit uses: the constants above, plus the
        channels commissioned as slow wanderers.
    """
    return BundleConfig(
        drifting_channels=tuple(drifting),
        sensed=tuple(sensed), commanded=tuple(commanded),
        table=PHYSICS_TABLE if table is None else table,
        channel_ids=ctx.channel_ids, stage_ids=ctx.stage_ids, channel_stage=tuple(ctx.channel_stage),
        budget_episodes_per_day=BUDGET_EPISODES_PER_DAY, budget_time_fraction=BUDGET_TIME_FRACTION,
        monitor_share=MONITOR_BUDGET_SHARE, cusum_lambda=CUSUM_DECAY_LAMBDA,
        # overridden per variant; `full_budget_split` recomputes it below
        samples_per_day=SAMPLES_PER_DAY, l0_margin_k=L0_MARGIN_K,
    )


def physics_relation_names() -> dict[str, str]:
    """L1b-covered level channel -> the mass balance it belongs to, for L6."""
    return {
        tb.level_channel: (
            f"{tb.tank} mass balance ({tb.level_channel} <- {tb.inflow.flow_channel} via "
            f"{tb.inflow.driving_actuator}, -> {tb.outflow.flow_channel} via {tb.outflow.driving_actuator})"
        )
        for tb in PHYSICS_TABLE
    }


def clean_windows(n: int, split: tuple[float, float, float] | None = None) -> SimpleNamespace:
    """The clean recording's fit / warm-up / calibration boundaries."""
    train_frac, warm_frac, cal_frac = split or (OUTER_TRAIN_FRAC, 0.40, 0.40)
    outer_train_end = int(n * train_frac)
    n_held = n - outer_train_end
    warmup_end = int(n_held * warm_frac)
    if split is None:
        calib_end = warmup_end + int(n_held * 0.25) + int(n_held * 0.15)
    else:
        calib_end = warmup_end + int(n_held * cal_frac)
    return SimpleNamespace(outer_train_end=outer_train_end, n_held=n_held,
                           warmup_end=warmup_end, calib_end=calib_end,
                           split=(train_frac, warm_frac, cal_frac))


def format_cross_selection(sel: dict, n_sensed: int) -> list[str]:
    """Which channels got a cross-channel relation, and on what figures."""
    chosen = [ch for ch, v in sel.items() if v.get("selected")]
    rules = sorted({v.get("rule", "?") for v in sel.values()})
    lines = [f"  L1a cross-channel relations (rule {'/'.join(rules)}): "
             f"{len(chosen)}/{n_sensed} channels: {', '.join(chosen) or 'none'}"]

    def num(v, key, fmt):
        x = v.get(key)
        return format(x, fmt) if isinstance(x, (int, float)) else "-"

    for ch, v in sel.items():
        mark = "selected" if v.get("selected") else ""
        if v.get("rule") == "0083":
            lines.append(
                f"    {ch:8s} pooled R2 {num(v, 'r2_pooled', '8.3f')}  "
                f"worst scale x{num(v, 'worst_scale_ratio', '.2f')}  "
                f"visibility {num(v, 'visibility_spreads', '.2f')}"
                f"/{num(v, 'target_spreads', '.2f')} spreads  {mark}")
        else:
            lines.append(
                f"    {ch:8s} validation R2 {num(v, 'r2_validation', '8.3f')}  "
                f"ADF p {num(v, 'adf_p', '.3f')}  {mark}")
    return lines


def commission(with_evidence: bool = False, cfg_overrides: dict | None = None,
               split: tuple[float, float, float] | None = None,
               dataset: str = "swat") -> SimpleNamespace:
    """Fit, warm up and calibrate the full detector on clean data exactly
        as the gate below reports it, stopping at the start of the eval
        window. Shared with `pipeline/run.py`, which continues the
        same stream into the attack recording.
    """
    dom = domain(dataset)
    df = pd.read_parquet(dom.parquet)
    sensed, commanded = dom.split_channels(list(df.columns))
    start_idx = dom.find_commissioning_start(df)
    steady = df.iloc[start_idx:].reset_index(drop=True)
    n = len(steady)
    w = clean_windows(n, split)
    outer_train_end = w.outer_train_end

    X_stream = steady[sensed].to_numpy()
    U_stream = steady[commanded].to_numpy()
    steady, gaps = fill_gaps(steady, list(sensed) + list(commanded))
    if gaps["n_values"]:
        _log(f"  gap-filled for fitting only: {gaps['n_values']} values across "
              f"{gaps['n_channels']} channels ({gaps['n_values'] / (n * len(X_stream[0])):.2e} "
              f"of the window); the stream keeps them and L0 flags them MISSING")
    X = steady[sensed].to_numpy()
    U = steady[commanded].to_numpy()
    X_train, U_train = X[:outer_train_end], U[:outer_train_end]
    # fitted on (filled)
    X_fit_held, U_fit_held = X[outer_train_end:], U[outer_train_end:]
    # streamed by every caller (as recorded)
    X_held, U_held = X_stream[outer_train_end:], U_stream[outer_train_end:]
    n_held = len(X_held)
    n_sensed = len(sensed)

    warmup_end, calib_end = w.warmup_end, w.calib_end
    _log(f"steady={n:,} outer_train_end={outer_train_end:,} | held_out={n_held:,}: "
          f"warmup=[0,{warmup_end}) calibration=[{warmup_end},{calib_end}) "
          f"eval=[{calib_end},{n_held})")

    _log("\nFitting L1a, L1b and invariants on outer-train, warming L2, calibrating L3 ...")
    ctx = dom.build_contexts(sensed, commanded)
    drifting = [ch for ch, is_drifting in zip(sensed, drifting_channels(X_train)) if is_drifting]
    cfg = bundle_config(sensed, commanded, ctx, drifting, table=dom.table)
    cfg = dataclasses.replace(cfg, l0_step_basis=dom.l0_step_basis,
                              l0_step_lag=dom.l0_step_lag)
    #: how every caller must build its external L0 checker for this domain
    l0_kwargs = {"step_basis": dom.l0_step_basis, "step_lag": dom.l0_step_lag}
    if cfg_overrides:
        cfg = dataclasses.replace(cfg, **cfg_overrides)
    if cfg.full_budget_split:
        cfg = dataclasses.replace(cfg, monitor_share=equal_budget_share(
            use_invariants=cfg.use_invariants, use_controller=cfg.use_controller))
    fit = fit_bundle(
        cfg,
        X_train, U_train,
        X_fit_held[:warmup_end], U_fit_held[:warmup_end],
        X_fit_held[warmup_end:calib_end], U_fit_held[warmup_end:calib_end],
        X, version=1,
    )
    bundle = fit.bundle
    l1a, l1b, miner = fit.l1a, fit.l1b, fit.miner
    l2a, l2b, l3a, l3b = fit.l2a, fit.l2b, fit.l3a, fit.l3b
    inv_state, resolution = fit.invariant_state, fit.resolution
    h_detail_a, h_detail_b = fit.h_detail_a, fit.h_detail_b
    per_channel_budget = fit.per_channel_budget
    n_b_channels = len(l1b.state_names)
    _log(f"  {len(inv_state.invariants):,} invariants | {inv_state.n_bytes():,} bytes")
    _log(f"  no value envelope (level wanders, docs/decisions/0038): "
          f"{', '.join(inv_state.drifting_channels) or 'none'}")
    for label, detail, n_cov, monitor in (
        ("L1a", h_detail_a, n_sensed, MONITOR_A), ("L1b", h_detail_b, n_b_channels, MONITOR_B)
    ):
        n_evt = sum(d["method"] == "evt" for d in detail.values())
        _log(f"  {label}: h set on {len(detail)}/{n_cov} channels, {n_evt} by tail fit, "
              f"{len(detail) - n_evt} by fallback (per-channel budget "
              f"{per_channel_budget[monitor][0]:.2e} time, {per_channel_budget[monitor][1]:.2e} ep/day)")

    undetectable = []
    if fit.reach_a is not None:
        r = fit.reach_a
        det_mask = fit.detectable
        undetectable = [sensed[j] for j in range(n_sensed) if not det_mask[j]]
        imp = r["implied_sigma"][np.isfinite(r["implied_sigma"])]
        _log(f"  sensitivity (0060 14.4): implied detection threshold "
              f"median {np.median(imp):.1f} sigma_d, p90 {np.quantile(imp, 0.9):.1f}"
              if len(imp) else "  sensitivity: no implied thresholds")
        _log(f"  reachable within the channel's own clean span, either branch: "
              f"{int(det_mask.sum())}/{n_sensed} "
              f"(L1a {int(r['detectable'].sum())}, "
              f"L1b {int(fit.reach_b['detectable'].sum())})")
        if undetectable:
            _log(f"  UNDETECTABLE ({len(undetectable)}): " + ", ".join(undetectable))
            for ch in undetectable[:8]:
                j = sensed.index(ch)
                _log(f"     {ch:16s} needs {r['implied_sigma'][j]:12.1f} sigma_d "
                      f"but its whole clean span is only "
                      f"{r['range_sigma'][j]:8.1f} sigma_d"
                      + ("" if not fit.reach_b["detectable"][j] else "  (L1b reaches it)"))

    if l1a.state.cross is not None:
        for line in format_cross_selection(l1a.state.cross.selection, n_sensed):
            _log(line)

    n_possible_tests = n_sensed + n_b_channels + n_sensed  # L1a + L1b + invariants
    recorder = RecordingDecider(
        MonitorDecider(
            degraded_abstention_threshold=DEGRADED_ABSTENTION_THRESHOLD,
            dwell_anomaly=DWELL_ANOMALY, dwell_recover=DWELL_RECOVER,
            bundle_id=bundle.bundle_id,
        )
    )
    evidence = None
    if with_evidence:
        evidence = EvidenceBuilder(
            sensed, commanded, miner, ctx.channel_ids, bundle.commanded_states,
            l1a=l1a, physics_relations=dom.relations(),
        )
    controller = controller_state = None
    if cfg.use_controller:
        q0 = np.zeros(n_sensed, dtype=np.uint8)

        def as_samples(lo: int, hi: int) -> list[Sample]:
            return [Sample(t=float(t), x=X_fit_held[t], u=U_fit_held[t], q=q0)
                    for t in range(lo, hi)]

        share = MONITOR_BUDGET_SHARE[MONITOR_CTRL]
        controller, controller_state, _ = calibrate_controller(
            fit_controller(X_train, U_train, sensed, commanded, CONTROLLER_STRUCTURE,
                           budget_episodes_per_day=share * BUDGET_EPISODES_PER_DAY),
            as_samples(0, warmup_end), as_samples(warmup_end, calib_end),
            lam=CUSUM_DECAY_LAMBDA,
            budget_episodes_per_day=share * BUDGET_EPISODES_PER_DAY,
            budget_time_fraction=share * BUDGET_TIME_FRACTION,
            samples_per_day=SAMPLES_PER_DAY,
        )
        st = controller_state
        _log(f"  controller: {st.n_rules} rules ({len(st.triggers)} trigger, {len(st.pairs)} pair, "
              f"{len(st.dwells)} dwell, {len(st.unseen)} unseen) | {st.n_bytes():,} bytes | "
              f"h={st.h:.3f} ({st.h_method}) | sha256 {st.sha256()[:12]}")

    adaptive_a = adaptive_b = None
    if cfg.adaptive_h:
        def adaptive(h, calibration, monitor):
            time_frac, eps_day = per_channel_budget[monitor]
            return AdaptiveThreshold(h, calibration, time_frac, eps_day, SAMPLES_PER_DAY,
                                     cap=cfg.adaptive_h_cap, period=cfg.adaptive_h_period,
                                     window=cfg.adaptive_h_window)
        adaptive_a = adaptive(l3a.h, fit.calibration_a, MONITOR_A)
        adaptive_b = adaptive(l3b.h, fit.calibration_b, MONITOR_B)
        _log(f"  adaptive thresholds (0045): cap x{cfg.adaptive_h_cap:g}, every "
              f"{cfg.adaptive_h_period / 3600:g} h, window {cfg.adaptive_h_window / 3600:g} h")

    per_sensor_recorder = None
    if cfg.per_sensor_alarms:
        per_sensor_recorder = RecordingPerSensorDecider(
            PerSensorDecider(sensed, cfg.degraded_abstention_threshold,
                             cfg.dwell_anomaly, cfg.dwell_recover,
                             bundle_id=bundle.bundle_id,
                             group_window=cfg.alarm_group_window,
                             dwell_window=cfg.dwell_window))
    detector = Detector(
        sensed, l1a, l1b, l2a, l2b, l3a, l3b, miner, recorder, ctx.channel_ids, n_possible_tests,
        evidence=evidence, trusted_learning=cfg.trusted_learning, learn_guard=cfg.learn_guard,
        per_channel_learning=cfg.per_channel_learning,
        per_sensor=per_sensor_recorder,
        use_invariants=cfg.use_invariants, controller=controller,
        adaptive_a=adaptive_a, adaptive_b=adaptive_b, learn_scales=not cfg.freeze_scales,
        settle_samples=cfg.settle_samples,
        naive_learning=cfg.naive_learning,
        joint_cap=(cfg.adaptive_h_cap if cfg.unified_l5 else None),
    )
    if cfg.sensitivity_budget:
        if not cfg.unified_l5:
            raise ValueError("sensitivity_budget needs unified_l5 (docs/decisions/0051)")
        guard = sensitivity_guard(cfg, fit, X, U, outer_train_end, ctx, cfg.adaptive_h_cap,
                                  measure_only=cfg.sensitivity_measure_only)
        detector.sensitivity = guard
        guard.commission(detector)  # each sensor's reference, with the commissioned settings
        budgeted = [sensed[j] for j in guard.reference]
        if guard.cap_by_channel:
            own = ", ".join(f"{sensed[j]} x{c:g}"
                            for j, c in sorted(guard.cap_by_channel.items()))
            _log(f"  sensitivity budget (0051): golden set {guard.digest[:12]}, "
                  f"{len(budgeted)}/{n_sensed} sensors budgeted, shared cap "
                  f"x{cfg.adaptive_h_cap:g}, per-channel (0148): {own}")
        else:
            _log(f"  sensitivity budget (0051): golden set {guard.digest[:12]}, "
                  f"{len(budgeted)}/{n_sensed} sensors budgeted at x{cfg.adaptive_h_cap:g}")
    if cfg.unified_l5:
        if not cfg.adaptive_h or cfg.l2_kind != "bounded":
            raise ValueError("unified_l5 needs adaptive_h and the bounded L2 (docs/decisions/0046)")
        _log(f"  unified L5 trial (0046): scheduled updates, scale x threshold within x{cfg.adaptive_h_cap:g}")
    return SimpleNamespace(**locals())


GOLDEN_EXCERPTS = settings.GOLDEN_EXCERPTS
GOLDEN_LENGTH = settings.GOLDEN_EXCERPT_MINUTES * 60
GOLDEN_SPAN = settings.GOLDEN_SPAN_HOURS * 3600


def sensitivity_guard(cfg, fit, X, U, fit_end: int, ctx, cap: float,
                      excerpts: int = GOLDEN_EXCERPTS, length: int = GOLDEN_LENGTH,
                      span: int = GOLDEN_SPAN, measure_only: bool = False) -> SensitivityGuard:
    """capture the golden set, store both branches'
        prediction errors on it (fresh copies of the frozen predictors), measure
        every sensor's commissioned smallest caught step.
    """
    import copy
    from gad.l5_golden_set import capture
    from gad.l2_context import SettlingTracker, per_channel
    from gad.l1a_predictor import ArxWindow

    starts = [fit_end - span + i * (span // excerpts) for i in range(excerpts)]
    golden = capture(X, U, cfg.sensed, cfg.commanded, starts=starts, length=length,
                     source="last 24 h of the commissioning fit rows (docs/decisions/0051)",
                     captured_at=f"rows {starts[0]}-{fit_end}")
    from gad.l5_detection_test import STEP_AT
    E, T, n = golden.X.shape
    q0 = np.zeros(n, dtype=np.uint8)

    def errors(Xe, Ue):
        """Both branches' errors on one excerpt, from fresh predictor copies."""
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
    by_index = None
    if cfg.sensitivity_cap_by_channel:
        order = {name: k for k, name in enumerate(cfg.sensed)}
        by_index = {}
        for name, c in dict(cfg.sensitivity_cap_by_channel).items():
            if name not in order:
                raise ValueError(
                    f"sensitivity_cap_by_channel names {name!r}, which is not a "
                    f"sensed channel (docs/decisions/0148)")
            if not (c >= 1.0):
                raise ValueError(
                    f"sensitivity_cap_by_channel[{name!r}] = {c}: a cap below 1 would "
                    f"demand more sensitivity than commissioning (docs/decisions/0148)")
            by_index[order[name]] = float(c)
    return SensitivityGuard(test, cap, golden.digest, measure_only=measure_only,
                            cap_by_channel=by_index)
