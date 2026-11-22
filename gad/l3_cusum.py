"""L3: a two-sided CUSUM per sensor and predictor, so small but lasting shifts
build up into evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import genpareto

from gad.l2_context import per_channel
from gad.l2_residual import robust_scale

EVT_TAIL_QUANTILE = 0.98
EVT_MIN_PEAKS = 30


def min_cap_factor_for_dwell(lam: float, dwell: int) -> float:
    """The smallest accumulator ceiling that can sustain a `dwell`-sample alarm."""
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1); got {lam}")
    if dwell < 1:
        raise ValueError(f"dwell must be at least 1; got {dwell}")
    return float(lam ** -dwell)


def drain_bound_steps(cap_factor: float, k: np.ndarray, h: np.ndarray,
                      lam: float) -> np.ndarray:
    """Worst-case steps for a capped accumulator to fall back under `h`, per
        channel, **on the assumption that the deviation has ended** (residual
        back to zero).
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1) to drain; got {lam}")
    k = np.asarray(k, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    K = k / (1.0 - lam)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.log((h + K) / (cap_factor * h + K)) / np.log(lam)
    return np.where(np.isfinite(h) & np.isfinite(k), n, np.inf)


class Cusum:
    """Two-sided CUSUM for one branch's channels. A NaN input (the branch
        or L2 abstains on that channel this step) freezes that channel's
        accumulators — no update, no reset — and reports NaN evidence; an
        abstention is not evidence of normality, so it must not silently
        decay the accumulated evidence either.
    """

    def __init__(self, k: np.ndarray, h: np.ndarray | None = None, lam: float = 1.0,
                 cap_factor: float | np.ndarray | None = None,
                 dwell: int | None = None,
                 h_by_context: dict[int, np.ndarray] | None = None) -> None:
        self.k = np.asarray(k, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64) if h is not None else None
        self.h_by_context = (None if h_by_context is None
                             else {int(c): np.asarray(v, dtype=np.float64)
                                   for c, v in h_by_context.items()})
        n = len(self.k)
        if self.h is not None and len(self.h) != n:
            raise ValueError("h must have the same length as k")
        if not (0.0 < lam <= 1.0):
            raise ValueError("lam must be in (0, 1]")
        self.lam = lam
        if cap_factor is not None:
            cap_factor = np.asarray(cap_factor, dtype=np.float64)
            if np.any(cap_factor < 1.0):
                raise ValueError(
                    f"cap_factor below 1 would cap the accumulator under h, so an "
                    f"alarm could never be raised at all: min {cap_factor.min()}")
            if cap_factor.ndim and cap_factor.shape != self.k.shape:
                raise ValueError(f"cap_factor must be scalar or shape {self.k.shape}")
            if dwell is not None and 0.0 < lam < 1.0:
                floor = min_cap_factor_for_dwell(lam, dwell)
                if np.any(cap_factor < floor):
                    raise ValueError(
                        f"cap_factor {np.min(cap_factor):.4f} cannot sustain a "
                        f"{dwell}-sample dwell at lam={lam}: the accumulator is "
                        f"pinned at the ceiling with no reserve, so the evidence "
                        f"falls under h on the next sample that is not above the "
                        f"slack. Minimum is {floor:.4f} (docs/decisions/0079)")
        if self.h_by_context is not None:
            for c, v in self.h_by_context.items():
                if v.shape != self.k.shape:
                    raise ValueError(
                        f"h_by_context[{c}] must have shape {self.k.shape}, got {v.shape}")
            if self.h is None:
                raise ValueError("h_by_context needs a channel-wide h to fall back to")
        self.cap_factor = cap_factor
        self.s_pos = np.zeros(n)
        self.s_neg = np.zeros(n)

    def update(self, r: np.ndarray, regime=None) -> np.ndarray:
        if r.shape != self.k.shape:
            raise ValueError(f"expected shape {self.k.shape}, got {r.shape}")

        evidence = np.full(len(self.k), np.nan)
        for i, ri in enumerate(r):
            if np.isnan(ri) or not np.isfinite(self.k[i]):
                continue
            self.s_pos[i] = max(0.0, self.lam * self.s_pos[i] + ri - self.k[i])
            self.s_neg[i] = max(0.0, self.lam * self.s_neg[i] - ri - self.k[i])
            evidence[i] = max(self.s_pos[i], self.s_neg[i])
        self._apply_cap(regime)
        if self.cap_factor is None:
            return evidence
        return np.minimum(evidence, self._ceiling(regime))

    def _ceiling(self, regime=None) -> np.ndarray | None:
        """`cap_factor * h` per channel, or None while `h` is not yet set."""
        if self.cap_factor is None or self.h is None:
            return None
        return self.cap_factor * self.effective_h(regime)

    def _apply_cap(self, regime=None) -> None:
        ceiling = self._ceiling(regime)
        if ceiling is None:
            return
        # a channel with no calibrated h (NaN/inf) has no ceiling to apply
        ok = np.isfinite(ceiling)
        np.minimum(self.s_pos, ceiling, out=self.s_pos, where=ok)
        np.minimum(self.s_neg, ceiling, out=self.s_neg, where=ok)

    def rescale(self, factor: np.ndarray) -> None:
        """Multiply each channel's accumulators by `factor` (non-finite
                entries leave that channel unchanged). When a new bundle changes a
                channel's residual scale from sigma_old to sigma_new, a factor of
                sigma_old / sigma_new keeps the evidence already gathered in the
                same physical units (BUILD.md Part 6, step 7).
        """
        factor = np.asarray(factor, dtype=np.float64)
        if factor.shape != self.k.shape:
            raise ValueError(f"expected shape {self.k.shape}, got {factor.shape}")
        ok = np.isfinite(factor) & (factor > 0)
        self.s_pos[ok] *= factor[ok]
        self.s_neg[ok] *= factor[ok]
        # a swap that rescales evidence upward must not escape the ceiling
        self._apply_cap()

    def effective_h(self, regime=None) -> np.ndarray:
        """The threshold in force per channel. Without per-context bars, or
                without a regime, this is the channel-wide `h` -- so every existing
                caller is unchanged.
        """
        if self.h is None:
            raise RuntimeError("h was not set at construction")
        if self.h_by_context is None or regime is None:
            return self.h
        ids = per_channel(regime, len(self.k))
        out = self.h.copy()
        for c in np.unique(ids):
            v = self.h_by_context.get(int(c))
            if v is None:
                continue
            m = (ids == c) & np.isfinite(v)
            out[m] = v[m]
        return out

    def alarm(self, evidence: np.ndarray, regime=None) -> np.ndarray:
        # NaN >= h is False in numpy: an abstaining channel never alarms.
        return evidence >= self.effective_h(regime)


def calibrate_k(scaled_residuals: np.ndarray, factor: float = 0.5, min_scale: float = 1e-6) -> np.ndarray:
    """k = `factor` x the robust scale of each channel's clean scaled
        residual (columns of `scaled_residuals`, shape (T, n_channels), NaNs
        allowed). NaN where a channel has no
        usable clean data at all.
    """
    n = scaled_residuals.shape[1]
    k = np.full(n, np.nan)
    for i in range(n):
        col = scaled_residuals[:, i]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            continue
        center = np.median(col)
        scale = robust_scale(np.abs(col - center), min_scale)
        if scale is not None:
            k[i] = factor * scale
    return k


def count_episodes(alarm: np.ndarray) -> int:
    """Number of maximal contiguous runs of True."""
    if len(alarm) == 0:
        return 0
    ints = alarm.astype(np.int8)
    rising_edges = np.flatnonzero(np.diff(np.concatenate(([0], ints))) == 1)
    return int(len(rising_edges))


def episodes_and_time_in_alarm(evidence: np.ndarray, h: float) -> tuple[int, float]:
    alarm = evidence >= h  # NaN evidence -> False, never counted as alarm
    return count_episodes(alarm), float(np.mean(alarm))


def calibrate_h_by_context(
    evidence: np.ndarray,
    contexts: np.ndarray,
    budget_episodes_per_day: float,
    budget_time_fraction: float,
    samples_per_day: float,
    min_rows: int = 1000,
    h_max: float | None = None,
    z: np.ndarray | None = None,
    min_typical_z: float = 0.0,
    detail: dict[int, dict] | None = None,
    estimator: str = "bisection",
    h_channel: float | None = None,
    pooled_report: dict | None = None,
    refuse_empirical: bool = True,
) -> tuple[dict[int, float], list[int]]:
    """One threshold per operating context, for a single channel."""
    if evidence.shape != contexts.shape:
        raise ValueError("evidence and contexts must have the same shape")
    if min_rows < 1:
        raise ValueError("min_rows must be at least 1")
    total_days = len(evidence) / samples_per_day
    if total_days <= 0:
        raise ValueError("no samples to calibrate on")

    if z is not None and z.shape != evidence.shape:
        raise ValueError("z must have the same shape as evidence")
    ids, counts = np.unique(contexts, return_counts=True)
    own, thin = [], []
    for c, n in zip(ids, counts):
        c = int(c)
        if n < min_rows:
            thin.append(c)
            continue
        if min_typical_z > 0.0 and z is not None:
            zc = np.abs(z[contexts == c])
            zc = zc[np.isfinite(zc)]
            if zc.size == 0 or float(np.median(zc)) < min_typical_z:
                thin.append(c)      # L2 does not describe this context
                continue
        own.append(c)
    if not own:
        return {}, thin

    if estimator not in ("bisection", "evt", "pooled", "cascade"):
        raise ValueError(
            "estimator must be 'bisection', 'evt', 'pooled' or 'cascade'")
    if estimator in ("pooled", "cascade") and h_channel is None:
        raise ValueError("the pooled estimator needs h_channel as its ceiling")
    eps_each = budget_episodes_per_day / len(own)
    frac_each = budget_time_fraction / len(own)
    out: dict[int, float] = {}

    cascaded: dict[int, float] = {}
    if estimator in ("pooled", "cascade"):
        bars, rep = calibrate_h_evt_pooled(
            evidence, contexts, own, eps_each, frac_each, samples_per_day,
            float(h_channel))
        if pooled_report is not None:
            pooled_report.update(rep)
        for c in own:
            if c in bars:
                out[c] = float(bars[c].h)
                if detail is not None:
                    m = contexts == c
                    ev = evidence[m]
                    fin = ev[np.isfinite(ev)]
                    n_ep, _ = episodes_and_time_in_alarm(ev, bars[c].h)
                    in_alarm = float(np.sum(np.nan_to_num(ev, nan=-np.inf) >= bars[c].h))
                    detail[c] = {
                        "h": float(bars[c].h), "n_rows": int(m.sum()),
                        "n_contexts_sharing_budget": len(own),
                        "estimator": "pooled", "method": bars[c].method,
                        "n_peaks": int(bars[c].n_peaks),
                        "gpd_shape": float(bars[c].shape),
                        "episodes_per_day_allowed": float(eps_each),
                        "time_fraction_allowed": float(frac_each),
                        "episodes_per_day_realised": float(n_ep / total_days),
                        "time_fraction_realised": float(in_alarm / len(evidence)),
                        "evidence_max": (float(fin.max()) if fin.size else None),
                        "h_over_evidence_max": (float(bars[c].h / fin.max())
                                                if fin.size and fin.max() > 0 else None),
                    }
            elif estimator == "pooled":
                thin.append(c)
        if estimator == "pooled":
            return out, sorted(thin)
        cascaded = dict(out)
        own = [c for c in own if c not in cascaded]
        if not own:
            return out, sorted(thin)
        eps_each = budget_episodes_per_day / len(own)
        frac_each = budget_time_fraction / len(own)

    for c in own:
        m = contexts == c
        ev = evidence[m]
        finite = ev[np.isfinite(ev)]
        hi = h_max if h_max is not None else (float(finite.max()) * 2.0 + 1.0
                                              if finite.size else 1.0)

        def satisfies(h: float) -> bool:
            n_ep, _ = episodes_and_time_in_alarm(ev, h)
            in_alarm = float(np.sum(np.nan_to_num(ev, nan=-np.inf) >= h))
            return (n_ep / total_days <= eps_each
                    and in_alarm / len(evidence) <= frac_each)

        if estimator in ("evt", "cascade"):
            n_ctx = int(m.sum())
            if n_ctx <= 0:
                thin.append(c)
                continue
            ratio = len(evidence) / n_ctx
            try:
                res = calibrate_h_evt(ev, eps_each * ratio, frac_each * ratio,
                                      samples_per_day)
            except ValueError:
                thin.append(c)   # unreachable: leave it to the channel-wide bar
                continue
            if res.method != "evt" and refuse_empirical:
                thin.append(c)
                continue
            out[c] = float(res.h)
            if detail is not None:
                n_ep, _ = episodes_and_time_in_alarm(ev, res.h)
                in_alarm = float(np.sum(np.nan_to_num(ev, nan=-np.inf) >= res.h))
                detail[c] = {
                    "h": float(res.h), "n_rows": n_ctx,
                    "n_contexts_sharing_budget": len(own),
                    "estimator": "evt", "method": res.method,
                    "n_peaks": int(res.n_peaks), "gpd_shape": res.shape,
                    "episodes_per_day_allowed": float(eps_each),
                    "time_fraction_allowed": float(frac_each),
                    "episodes_per_day_realised": float(n_ep / total_days),
                    "time_fraction_realised": float(in_alarm / len(evidence)),
                    "evidence_max": (float(finite.max()) if finite.size else None),
                    "h_over_evidence_max": (float(res.h / finite.max())
                                            if finite.size and finite.max() > 0
                                            else None),
                }
            continue
        if not satisfies(hi):
            thin.append(c)
            continue
        lo, ceiling = 0.0, hi
        for _ in range(60):
            if hi - lo < 1e-4:
                break
            mid = (lo + hi) / 2.0
            if satisfies(mid):
                hi = mid
            else:
                lo = mid
        out[c] = hi
        if detail is not None:
            n_ep, _ = episodes_and_time_in_alarm(ev, hi)
            in_alarm = float(np.sum(np.nan_to_num(ev, nan=-np.inf) >= hi))
            detail[c] = {
                "h": float(hi),
                "n_rows": int(m.sum()),
                "n_contexts_sharing_budget": len(own),
                "estimator": "bisection",
                "h_over_evidence_max": (float(hi / finite.max())
                                        if finite.size and finite.max() > 0
                                        else None),
                "episodes_per_day_allowed": float(eps_each),
                "time_fraction_allowed": float(frac_each),
                "episodes_per_day_realised": float(n_ep / total_days),
                "time_fraction_realised": float(in_alarm / len(evidence)),
                "evidence_max": (float(finite.max()) if finite.size else None),
                "search_ceiling": float(ceiling),
                "at_search_ceiling": bool(hi >= ceiling * 0.999),
                # which of the two constraints is tight at the chosen bar
                "binding": ("episodes"
                            if n_ep / total_days > eps_each * 0.999
                            else ("time" if in_alarm / len(evidence) > frac_each * 0.999
                                  else "neither")),
            }
    return out, sorted(thin)


def calibrate_h_bisection(
    evidence: np.ndarray,
    budget_episodes_per_day: float,
    budget_time_fraction: float,
    samples_per_day: float,
    h_max: float | None = None,
    tol: float = 1e-4,
    max_iter: int = 60,
) -> float:
    """Smallest h (most sensitive) such that, on `evidence` (one channel's
        clean-data CUSUM statistic stream), BOTH the episode-rate and the
        time-in-alarm fraction stay within budget. Raises if even `h_max`
        can't satisfy the budget (refuse silently-wrong calibration).
    """
    finite = evidence[np.isfinite(evidence)]
    if h_max is None:
        h_max = float(finite.max()) * 2.0 if len(finite) else 1.0

    def satisfies(h: float) -> bool:
        n_ep, frac = episodes_and_time_in_alarm(evidence, h)
        days = len(evidence) / samples_per_day
        return (n_ep / days <= budget_episodes_per_day) and (frac <= budget_time_fraction)

    if not satisfies(h_max):
        raise ValueError(
            f"h_max={h_max} does not satisfy the stated budget "
            f"(episodes/day<={budget_episodes_per_day}, time_fraction<={budget_time_fraction}); "
            "widen h_max or the budget rather than silently returning an unsafe h"
        )

    lo, hi = 0.0, h_max
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2.0
        if satisfies(mid):
            hi = mid
        else:
            lo = mid
    return hi


@dataclass(frozen=True)
class EvtThreshold:
    """An alarm level for one channel, and how it was obtained. `method` is
        "evt" when a tail was fitted and "empirical" when there were too few
        excursions to fit one and `calibrate_h_bisection` was used instead —
        reported so a caller can say how many channels actually got a fitted
        tail. `shape` is the fitted GPD shape of the excursion peaks (NaN when
        empirical): above 0 a heavy tail, 0 exponential, below 0 bounded.
    """

    h: float
    method: str
    n_peaks: int
    shape: float


def excursion_peaks(evidence: np.ndarray, u: float) -> np.ndarray:
    """Maximum of each maximal run of samples above `u` — one value per
        excursion. A CUSUM that crosses a level stays above it for a stretch,
        so its samples above `u` are not independent; the runs are, closely
        enough, and one run is one alarm episode. NaN breaks a run.
    """
    above = np.nan_to_num(evidence, nan=-np.inf) > u
    edges = np.diff(np.concatenate(([0], above.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return np.array([evidence[s:e].max() for s, e in zip(starts, ends)])


def _gpd_level(excesses: np.ndarray, survival: float) -> tuple[float, float]:
    """(level, shape): the excess over the starting quantile at which a GPD
        fitted to `excesses` has survival probability `survival`, and the
        fitted shape.
    """
    shape, _, scale = genpareto.fit(excesses, floc=0.0)
    return float(genpareto.isf(survival, shape, loc=0.0, scale=scale)), float(shape)


POOLED_MIN_PEAKS_PER_CONTEXT = 3


def calibrate_h_evt_pooled(
    evidence: np.ndarray,
    contexts: np.ndarray,
    own: list[int],
    budget_episodes_per_day: float,
    budget_time_fraction: float,
    samples_per_day: float,
    h_channel: float,
    tail_quantile: float = EVT_TAIL_QUANTILE,
    min_peaks: int = EVT_MIN_PEAKS,
    min_peaks_per_context: int = POOLED_MIN_PEAKS_PER_CONTEXT,
) -> tuple[dict[int, EvtThreshold], dict]:
    """One tail SHAPE for a channel, each context's own SCALE."""
    evidence = np.asarray(evidence, dtype=float)
    contexts = np.asarray(contexts, dtype=np.int64)
    if evidence.shape != contexts.shape:
        raise ValueError("evidence and contexts must have the same shape")
    total_rows = len(evidence)
    total_days = total_rows / samples_per_day if samples_per_day else float("nan")
    if not (total_days > 0):
        raise ValueError("no samples to calibrate on")

    per: dict[int, dict] = {}
    pooled_peaks, pooled_exceed = [], []
    contributing: list[int] = []
    for c in own:
        m = contexts == c
        ev_c = evidence[m]
        finite = ev_c[np.isfinite(ev_c)]
        if finite.size == 0:
            continue
        u = float(np.quantile(finite, tail_quantile))
        peaks = excursion_peaks(ev_c, u) - u
        exceed = finite[finite > u] - u
        peaks = peaks[np.isfinite(peaks) & (peaks > 0)]
        exceed = exceed[np.isfinite(exceed) & (exceed > 0)]
        if peaks.size == 0 or exceed.size == 0:
            continue
        idx_p, idx_e = float(peaks.mean()), float(exceed.mean())
        if not (idx_p > 0 and idx_e > 0):
            continue
        per[c] = {"u": u, "peaks": peaks, "exceed": exceed,
                  "idx_p": idx_p, "idx_e": idx_e,
                  "ev_max": float(finite.max())}
        if peaks.size >= min_peaks_per_context:
            pooled_peaks.append(peaks / idx_p)
            pooled_exceed.append(exceed / idx_e)
            contributing.append(c)

    report = {"n_contexts_offered": len(own), "n_contexts_usable": len(per),
              "contributing_contexts": sorted(contributing),
              "min_peaks_per_context": int(min_peaks_per_context)}
    if not pooled_peaks:
        return {}, {**report, "n_pooled_peaks": 0, "reason": "no context met the floor"}
    pp = np.concatenate(pooled_peaks)
    pe = np.concatenate(pooled_exceed)
    report["n_pooled_peaks"] = int(pp.size)
    report["n_pooled_exceedances"] = int(pe.size)
    if pp.size < min_peaks or pe.size < min_peaks:
        return {}, {**report, "reason": f"pooled peaks {pp.size} < {min_peaks}"}

    try:
        shape_p, _, scale_p = genpareto.fit(pp, floc=0.0)
        shape_e, _, scale_e = genpareto.fit(pe, floc=0.0)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return {}, {**report, "reason": f"pooled fit failed: {exc}"}
    report["shape_peaks"] = float(shape_p)
    report["shape_exceedances"] = float(shape_e)

    bars: dict[int, EvtThreshold] = {}
    for c, d in per.items():
        # identical arithmetic to calibrate_h_evt, with this context's counts
        target_ep = budget_episodes_per_day * total_days / len(d["peaks"])
        target_time = budget_time_fraction * total_rows / len(d["exceed"])
        h_ep = h_time = d["u"]
        try:
            if target_ep < 1.0:
                h_ep = d["u"] + d["idx_p"] * float(
                    genpareto.isf(target_ep, shape_p, loc=0.0, scale=scale_p))
            if target_time < 1.0:
                h_time = d["u"] + d["idx_e"] * float(
                    genpareto.isf(target_time, shape_e, loc=0.0, scale=scale_e))
        except (ValueError, RuntimeError, FloatingPointError):
            continue
        h = max(h_ep, h_time)
        if not np.isfinite(h) or h <= d["ev_max"] or h > h_channel:
            continue
        bars[c] = EvtThreshold(h=float(h), method="evt_pooled",
                               n_peaks=len(d["peaks"]), shape=float(shape_p))
    report["n_bars"] = len(bars)
    return bars, report


def calibrate_h_evt(
    evidence: np.ndarray,
    budget_episodes_per_day: float,
    budget_time_fraction: float,
    samples_per_day: float,
    tail_quantile: float = EVT_TAIL_QUANTILE,
    min_peaks: int = EVT_MIN_PEAKS,
) -> EvtThreshold:
    """Alarm level for one channel from a peaks-over-threshold tail fit, for
        budgets finer than the calibration window can show directly.
    """
    finite = evidence[np.isfinite(evidence)]
    if len(finite) == 0:
        raise ValueError("no finite evidence to calibrate against")

    def empirical(n_peaks: int) -> EvtThreshold:
        h = calibrate_h_bisection(
            evidence, budget_episodes_per_day, budget_time_fraction, samples_per_day
        )
        return EvtThreshold(h=h, method="empirical", n_peaks=n_peaks, shape=float("nan"))

    u = float(np.quantile(finite, tail_quantile))
    peaks = excursion_peaks(evidence, u)
    exceed = finite[finite > u] - u
    if len(peaks) < min_peaks or len(exceed) < min_peaks:
        return empirical(len(peaks))

    days = len(evidence) / samples_per_day
    target_ep = budget_episodes_per_day / (len(peaks) / days)
    target_time = budget_time_fraction / (len(exceed) / len(evidence))

    try:
        shape = float("nan")
        h_ep = u
        if target_ep < 1.0:
            level, shape = _gpd_level(peaks - u, target_ep)
            h_ep = u + level
        h_time = u
        if target_time < 1.0:
            level, _ = _gpd_level(exceed, target_time)
            h_time = u + level
    except (ValueError, RuntimeError, FloatingPointError):
        return empirical(len(peaks))

    h = max(h_ep, h_time)
    if not np.isfinite(h):
        return empirical(len(peaks))
    return EvtThreshold(h=h, method="evt", n_peaks=len(peaks), shape=shape)
