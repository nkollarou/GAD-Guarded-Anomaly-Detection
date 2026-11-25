"""L5, rule 4: after each update, replays the golden set and measures the smallest
step attack each sensor can still catch. A sensor that became too insensitive
is pulled back towards its commissioning values.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gad.l2_context import per_channel

#: step sizes in sigma: 2^(k/4), k = 0..40 (1 to 1024 sigma)
SIZES = np.array([2.0 ** (k / 4) for k in range(41)])
WARM = 300      # samples of each excerpt discarded (predictor warm-up)
STEP_AT = 900   # the step starts here in every excerpt
WINDOW = 600    # and must be caught within this many samples


@dataclass
class Branch:
    """One branch's settings for one channel: committed (centre, scale) per
    context, CUSUM slack k, the channel-wide alarm level h, and the per-context
    bars that override it where a context has one.
    """

    centres: dict
    k: float
    h: float
    h_ctx: dict | None = None


class SensitivityTest:
    """Stored golden-set errors and the replay that measures a sensor's smallest
    caught step.
    """

    def __init__(self, res_a, res_b, regimes, sigma, lam: float, dwell: int,
                 g_a=None, g_b=None) -> None:
        self.res_a = np.asarray(res_a, dtype=np.float64)
        self.res_b = np.asarray(res_b, dtype=np.float64)
        unit = np.zeros_like(self.res_a)
        unit[:, STEP_AT:, :] = 1.0  # a predictor the step does not reach
        self.g_a = unit if g_a is None else np.asarray(g_a, dtype=np.float64)
        self.g_b = unit if g_b is None else np.asarray(g_b, dtype=np.float64)
        self.regimes = np.asarray(regimes)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.lam = float(lam)
        self.dwell = int(dwell)
        if self.res_a.shape[1] < STEP_AT + WINDOW:
            raise ValueError(f"golden excerpts must be at least {STEP_AT + WINDOW} samples long")

    def _centre_scale(self, j: int, b: Branch) -> tuple[np.ndarray, np.ndarray]:
        reg = self.regimes[:, :, j]
        m = np.full(reg.shape, np.nan)
        s = np.full(reg.shape, np.nan)
        for ctx, (mc, sc) in b.centres.items():
            sel = reg == ctx
            m[sel], s[sel] = mc, sc
        return m, s  # NaN where the context is uncommitted: that sample abstains

    def _levels(self, j: int, b: Branch) -> np.ndarray:
        """The alarm level in force at every (excerpt, sample) for channel j: the
        per-context bar wherever that context has a finite one, else the
        channel-wide `h`.
        """
        reg = self.regimes[:, :, j]
        h = np.full(reg.shape, float(b.h), dtype=np.float64)
        if b.h_ctx:
            for ctx, hv in b.h_ctx.items():
                if hv is None or not np.isfinite(hv):
                    continue
                h[reg == ctx] = float(hv)
        return h

    def smallest(self, j: int, sign: float, a: Branch | None, b: Branch | None) -> float | None:
        """Smallest size in SIZES (sigma) caught in every excerpt, for a step of
        `sign`; None if none is. The decider fires while either branch's CUSUM
        evidence is at or above its h, and ANOMALY needs `dwell` consecutive
        firing samples.
        """
        branches = []
        for res, g, br in ((self.res_a, self.g_a, a), (self.res_b, self.g_b, b)):
            if br is None or not np.isfinite(br.h) or not np.isfinite(br.k) or br.h <= 0:
                continue
            r = res[:, :, j]
            if not np.isfinite(r).any():
                continue
            m, s = self._centre_scale(j, br)
            branches.append((r, np.nan_to_num(g[:, :, j]), m, s, br.k,
                             self._levels(j, br)))
        if not branches:
            return None
        E, S = self.res_a.shape[0], len(SIZES)
        step = sign * SIZES * self.sigma[j]  # (S,) in the channel's own units
        states = []
        for r, g, m, s, k, hh in branches:
            z = (r - m) / s
            sp = np.zeros(E)
            sn = np.zeros(E)
            for t in range(WARM, STEP_AT):  # identical for every size
                zt = z[:, t]
                ok = np.isfinite(zt)
                zt = np.nan_to_num(zt)
                sp = np.where(ok, np.maximum(0.0, self.lam * sp + zt - k), sp)
                sn = np.where(ok, np.maximum(0.0, self.lam * sn - zt - k), sn)
            states.append([np.repeat(sp[:, None], S, 1), np.repeat(sn[:, None], S, 1)])
        streak = np.zeros((E, S), dtype=np.int64)
        caught = np.zeros((E, S), dtype=bool)
        for t in range(STEP_AT, STEP_AT + WINDOW):
            fire = np.zeros((E, S), dtype=bool)
            for (r, g, m, s, k, hh), st in zip(branches, states):
                zt = (((r[:, t] - m[:, t]) / s[:, t])[:, None]
                      + (step[None, :] * g[:, t][:, None]) / s[:, t][:, None])
                ok = np.isfinite(zt)
                zt = np.nan_to_num(zt)
                st[0] = np.where(ok, np.maximum(0.0, self.lam * st[0] + zt - k), st[0])
                st[1] = np.where(ok, np.maximum(0.0, self.lam * st[1] - zt - k), st[1])
                fire |= ok & (np.maximum(st[0], st[1]) >= hh[:, t][:, None])
            streak = np.where(fire, streak + 1, 0)
            caught |= streak >= self.dwell
        idx = np.flatnonzero(caught.all(axis=0))
        return float(SIZES[idx[0]]) if len(idx) else None


def branch_settings(l2, l3, h_ref: np.ndarray | None, j: int, lam_back: float = 1.0) -> Branch | None:
    """Channel j's committed settings in one branch; with `lam_back` < 1, pulled
    toward commissioning: s = s_ref^(1-lam) s^lam, m = m_ref + lam (m - m_ref),
    h = h_ref^(1-lam) h^lam, with `h_ref` the commissioned alarm levels (the
    adaptive threshold's reference).
    """
    committed = getattr(l2, "_committed", None)
    if committed is None:
        return None
    refs = l2._ref[j] if l2._ref is not None else {}
    centres = {}
    for ctx, (m, s) in committed[j].items():
        if ctx in refs and lam_back < 1.0:
            m_ref, s_ref = refs[ctx]
            m = m_ref + lam_back * (m - m_ref)
            s = s_ref ** (1.0 - lam_back) * s ** lam_back
        centres[ctx] = (m, s)
    h0 = float(l3.h[j])
    h = h0
    hr = float(h_ref[j]) if h_ref is not None else h
    if lam_back < 1.0 and np.isfinite(hr) and np.isfinite(h) and hr > 0:
        h = hr ** (1.0 - lam_back) * h ** lam_back
    table = getattr(l3, "h_by_context", None)
    h_ctx = None
    if table:
        factor = (h / h0) if (np.isfinite(h0) and h0 > 0 and np.isfinite(h)) else 1.0
        h_ctx = {int(ctx): float(v[j]) * factor
                 for ctx, v in table.items() if np.isfinite(float(v[j]))} or None
    return Branch(centres, float(l3.k[j]), h, h_ctx)


class SensitivityGuard:
    """Holds each sensor's commissioned smallest caught steps (both signs) and
    pulls a sensor's settings back whenever an update would take either beyond
    `cap` times them.
    """

    cap_by_channel: dict[int, float] | None = None

    def __init__(self, test: SensitivityTest, cap: float, digest: str,
                 measure_only: bool = False,
                 cap_by_channel: dict[int, float] | None = None) -> None:
        self.test = test
        self.cap = float(cap)
        self.cap_by_channel = (None if cap_by_channel is None
                               else {int(j): float(c) for j, c in cap_by_channel.items()})
        self.digest = digest  # the golden set's SHA-256, for the record
        self.measure_only = bool(measure_only)
        self.reference: dict[int, tuple[float | None, float | None]] = {}
        self.last_ratio: dict[int, float] = {}
        self.pulled: list[tuple[float, int, float]] = []  # (t, channel, lambda kept)
        self.reasons: list[tuple[float, int, str]] = []

    def cap_for(self, j: int) -> float:
        """Channel j's cap: its own if it has one, else the shared `cap`."""
        if self.cap_by_channel is None:
            return self.cap
        return self.cap_by_channel.get(int(j), self.cap)

    @staticmethod
    def _refs(det):
        return (getattr(det.adaptive_a, "h_ref", None) if det.adaptive_a is not None else None,
                getattr(det.adaptive_b, "h_ref", None) if det.adaptive_b is not None else None)

    def _sizes(self, det, j: int, lam_back: float = 1.0):
        ra, rb = self._refs(det)
        a = branch_settings(det.l2a, det.l3a, ra, j, lam_back)
        b = branch_settings(det.l2b, det.l3b, rb, j, lam_back)
        return tuple(self.test.smallest(j, sgn, a, b) for sgn in (1.0, -1.0))

    def commission(self, det) -> None:
        """Measure every sensor's reference with the commissioned settings."""
        for j in range(len(det.sensed_channels)):
            ref = self._sizes(det, j)
            if any(v is not None for v in ref):
                self.reference[j] = ref

    def _within(self, sizes, ref, j: int | None = None) -> bool:
        cap = self.cap if j is None else self.cap_for(j)
        for now, base in zip(sizes, ref):
            if base is None:
                continue
            if now is None or now > cap * base * (1.0 + 1e-9):
                return False
        return True

    def enforce(self, det, t: float) -> None:
        """After an update event: pull back every sensor that went too far."""
        regimes = per_channel(det._regime, len(det.sensed_channels))
        for j, ref in self.reference.items():
            sizes = self._sizes(det, j)
            if self._within(sizes, ref, j):
                self.last_ratio[j] = _ratio(sizes, ref)
                continue
            blind = any(now is None and base is not None for now, base in zip(sizes, ref))
            self.reasons.append((t, j, "blind" if blind else "over_cap"))
            if self.measure_only:
                self.pulled.append((t, j, 1.0))   # 1.0 = kept, nothing undone
                self.last_ratio[j] = _ratio(sizes, ref)
                continue
            lo, hi = 0.0, 1.0  # lam = 0 is the commissioned settings: within by construction
            for _ in range(8):
                mid = (lo + hi) / 2
                if self._within(self._sizes(det, j, mid), ref, j):
                    lo = mid
                else:
                    hi = mid
            self._apply(det, j, lo, regimes[j])
            self.pulled.append((t, j, lo))
            self.last_ratio[j] = _ratio(self._sizes(det, j), ref)

    def _apply(self, det, j: int, lam_back: float, ctx_now: int) -> None:
        for l2, l3, h_ref in zip((det.l2a, det.l2b), (det.l3a, det.l3b), self._refs(det)):
            br = branch_settings(l2, l3, h_ref, j, lam_back)
            if br is None:
                continue
            old = l2.committed_scale(j, ctx_now)
            for ctx, (m, s) in br.centres.items():
                l2._committed[j][ctx] = (m, s)
            new = l2.committed_scale(j, ctx_now)
            if old and new:
                factor = np.full(len(l3.k), np.nan)
                factor[j] = old / new
                l3.rescale(factor)  # evidence stays in the same units
            if np.isfinite(br.h):
                l3.h[j] = br.h


def _ratio(sizes, ref) -> float:
    vals = [now / base for now, base in zip(sizes, ref)
            if base is not None and now is not None]
    return max(vals) if vals else float("nan")
