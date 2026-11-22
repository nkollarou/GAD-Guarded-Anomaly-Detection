"""L2: turns each residual into a z-score, using a robust running mean and
standard deviation per sensor and context.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.stats import norm

from gad.l2_context import per_channel

ESCALATION_QUANTILES: tuple[float, ...] = (0.5, 0.75, 0.90, 0.95, 0.99)
ESCALATION_CONSTANTS: tuple[float, ...] = tuple(1.0 / norm.ppf(0.5 + q / 2) for q in ESCALATION_QUANTILES)


def _quantile_sorted(a: np.ndarray, q: float) -> float:
    """`np.quantile(a, q)` (default 'linear' method) for an already-sorted
        1-D array, without NumPy's per-call overhead. Reproduces its index
        arithmetic and its two-sided interpolation exactly, so results are
        bit-identical (tests/test_residual.py checks this against NumPy).
    """
    n = len(a)
    virtual = (n - 1) * q
    lo = int(np.floor(virtual))
    hi = min(lo + 1, n - 1)
    g = virtual - lo
    x0, x1 = a[lo], a[hi]
    diff = x1 - x0
    if g >= 0.5:
        return float(x1 - diff * (1 - g))
    return float(x0 + diff * g)


def _median_sorted(a: np.ndarray) -> float:
    """`np.median(a)` for an already-sorted 1-D array, bit-identical."""
    n = len(a)
    mid = n // 2
    if n % 2:
        return float(a[mid])
    return float((a[mid - 1] + a[mid]) / 2)


def robust_scale(abs_deviations: np.ndarray, min_scale: float) -> float | None:
    """The smallest-quantile scale in `ESCALATION_QUANTILES` that clears
        `min_scale`; `None` if every quantile up to and including 0.99 is
        still degenerate (nothing usable — the caller should abstain, not
        invent a floor value).
    """
    a = np.sort(abs_deviations)  # one sort serves every escalation level
    for q, c in zip(ESCALATION_QUANTILES, ESCALATION_CONSTANTS):
        scale = c * _quantile_sorted(a, q)
        if scale >= min_scale:
            return float(scale)
    return None


def measurement_resolution(X: np.ndarray) -> np.ndarray:
    """Per column, the smallest nonzero one-step change in `X` — the finest
        step that sensor ever reported. 0 for a column that never changed
        (no resolution can be read off it). NaNs are ignored.
    """
    d = np.abs(np.diff(X, axis=0))
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        nz = d[:, j][np.isfinite(d[:, j]) & (d[:, j] > 0)]
        if len(nz):
            out[j] = float(nz.min())
    return out


def measurement_resolution_segments(segments) -> np.ndarray:
    """`measurement_resolution` over blocks that are not contiguous with
        each other: the smallest nonzero one-step change within any segment,
        never across a gap.
    """
    per = [measurement_resolution(np.asarray(X, dtype=np.float64)) for X, _ in segments]
    if not per:
        raise ValueError("no segments")
    stacked = np.stack(per)
    out = np.zeros(stacked.shape[1])
    for j in range(stacked.shape[1]):
        nz = stacked[:, j][stacked[:, j] > 0]
        if len(nz):
            out[j] = float(nz.min())
    return out


class RegimeScaledResidual:
    """L2 for ONE branch. Call `update` once per sample with that branch's
        raw residual vector, the current operating regime id, and whether the
        detector's current state is NORMAL (from the *previous* decision —
        this sample's own decision doesn't exist yet).
    """

    def __init__(
        self,
        n_channels: int,
        window_size: int = 300,
        min_calibration: int = 30,
        min_scale: float = 1e-6,
        scale_floor: np.ndarray | None = None,
    ) -> None:
        if scale_floor is None:
            floor = np.zeros(n_channels)
        else:
            floor = np.asarray(scale_floor, dtype=np.float64)
            if floor.shape != (n_channels,):
                raise ValueError(f"scale_floor must have shape ({n_channels},)")
            if np.any(~np.isfinite(floor)) or np.any(floor < 0):
                raise ValueError("scale_floor must be finite and non-negative")
        self.n_channels = n_channels
        self.window_size = window_size
        self.min_calibration = min_calibration
        self.min_scale = min_scale
        self.scale_floor = floor
        # per channel: context id -> that channel's window in that context
        self._windows: list[dict[int, deque]] = [{} for _ in range(n_channels)]

    def window(self, channel: int, regime_id: int) -> deque:
        w = self._windows[channel].get(regime_id)
        if w is None:
            w = self._windows[channel][regime_id] = deque(maxlen=self.window_size)
        return w

    def update(
        self,
        residual: np.ndarray,
        regime_id,
        is_normal: bool,
        learn: np.ndarray | None = None,
    ) -> np.ndarray:
        """`regime_id` is one id for every channel (whole-plant keying) or
                one per channel (`gad.l2_context.ActuatorContexts.channel_ids`).
                `learn` (default: every channel) masks which channels may add this
                residual to their window — a channel flagged out of range by L0 is
                still scaled and scored, but does not teach the scale.
        """
        if residual.shape != (self.n_channels,):
            raise ValueError(f"expected shape ({self.n_channels},), got {residual.shape}")
        if learn is not None and learn.shape != (self.n_channels,):
            raise ValueError(f"learn must have shape ({self.n_channels},)")

        regimes = per_channel(regime_id, self.n_channels)
        scaled = np.full(self.n_channels, np.nan)

        for i in range(self.n_channels):
            r = residual[i]
            if np.isnan(r):
                continue  # branch abstains on this channel — nothing to scale or learn from

            w = self.window(i, int(regimes[i]))
            center_scale = self._center_scale(i, w)
            if center_scale is not None:
                center, scale = center_scale
                scaled[i] = (r - center) / scale
            # else: uncalibrated, or degenerate at every escalation level with no floor -> abstain (NaN)

            if is_normal and (learn is None or learn[i]):
                w.append(r)  # learning happens only while NORMAL — frozen otherwise

        return scaled

    def _center_scale(self, channel: int, w: deque) -> tuple[float, float] | None:
        if len(w) < self.min_calibration:
            return None
        arr = np.fromiter(w, dtype=np.float64)
        center = _median_sorted(np.sort(arr))
        scale = robust_scale(np.abs(arr - center), self.min_scale)
        floor = self.scale_floor[channel]
        if floor > 0:
            scale = max(scale if scale is not None else 0.0, floor)
        return None if scale is None else (center, scale)

    def current_scale(self, channel: int, regime_id: int) -> float | None:
        """The scale this channel's residual would be divided by right now
                in this context, or None if it would abstain. Read-only: used to
                rescale CUSUM evidence when a new bundle's windows replace these.
        """
        w = self._windows[channel].get(int(regime_id))
        cs = None if w is None else self._center_scale(channel, w)
        return None if cs is None else cs[1]

    def export_windows(self) -> tuple[dict[int, tuple[float, ...]], ...]:
        """An immutable copy of every (channel, context) window."""
        return tuple({r: tuple(w) for r, w in per.items()} for per in self._windows)

    def load_windows(self, windows: tuple[dict[int, tuple[float, ...]], ...]) -> None:
        """Replace every window with `windows` (as `export_windows` returns)."""
        if len(windows) != self.n_channels:
            raise ValueError(f"expected windows for {self.n_channels} channels, got {len(windows)}")
        self._windows = [
            {r: deque(vals, maxlen=self.window_size) for r, vals in per.items()} for per in windows
        ]

    def n_regimes(self) -> int:
        """Distinct context ids seen across all channels."""
        return len({r for per in self._windows for r in per})

    def n_bytes(self) -> int:
        """Deployed state size — bounded by window_size floats per (channel,
                context) seen, not by how long the stream has run.
        """
        return sum(len(per) for per in self._windows) * self.window_size * 8


def _clipped_second_moment(c: float) -> float:
    """E[min(z^2, c^2)] for z ~ N(0, 1): the consistency constant that keeps
        a clipped variance estimate unbiased on clean Gaussian residuals.
    """
    tail = float(norm.sf(c))
    body = float(norm.cdf(c) - norm.cdf(-c)) - 2.0 * c * float(norm.pdf(c))
    return body + 2.0 * c * c * tail


def half_life_rate(half_life_samples: float) -> float:
    """Per-update weight `alpha` of an exponential average whose memory
        halves after `half_life_samples` updates.
    """
    if half_life_samples <= 0:
        raise ValueError("half_life_samples must be > 0")
    return 1.0 - 2.0 ** (-1.0 / half_life_samples)


class BoundedScaleResidual:
    """L2 for one branch, with **bounded, anchored, continuous** learning. A drop-in for `RegimeScaledResidual`: same
        `update` signature, same NaN-means-abstain rule.
    """

    def __init__(
        self,
        n_channels: int,
        half_life: float,
        clip: float = 3.0,
        cap: float = 2.0,
        band: float = 1.0,
        min_calibration: int = 30,
        min_scale: float = 1e-6,
        scale_floor: np.ndarray | None = None,
        pooled_init_for_unseen: bool = False,
    ) -> None:
        if clip <= 0 or cap <= 1.0 or band < 0:
            raise ValueError("clip > 0, cap > 1 and band >= 0 are required")
        floor = np.zeros(n_channels) if scale_floor is None else np.asarray(scale_floor, dtype=np.float64)
        if floor.shape != (n_channels,) or np.any(~np.isfinite(floor)) or np.any(floor < 0):
            raise ValueError(f"scale_floor must be finite, non-negative and of shape ({n_channels},)")
        self.n_channels = n_channels
        self.half_life = float(half_life)
        self.alpha = half_life_rate(half_life)
        self.clip = float(clip)
        self.cap = float(cap)
        self.band = float(band)
        self.min_calibration = min_calibration
        self.min_scale = min_scale
        self.scale_floor = floor
        self.pooled_init_for_unseen = bool(pooled_init_for_unseen)
        self.kappa = _clipped_second_moment(self.clip)
        # per channel: context -> [m, s]; references once committed
        self._state: list[dict[int, list[float]]] = [{} for _ in range(n_channels)]
        self._ref: list[dict[int, tuple[float, float]]] | None = None
        self._warmup: list[dict[int, list[float]]] = [{} for _ in range(n_channels)]
        self.saturation: dict[tuple[int, int], int] = {}

    # -- the worst-case bound, one function the tests and the paper share --
    def max_growth_per_step(self) -> float:
        """The largest factor one learning step can multiply a scale by."""
        return float(np.sqrt(1.0 + self.alpha * (self.clip ** 2 / self.kappa - 1.0)))

    def min_steps_to_grow(self, factor: float) -> float:
        """Fewest learning steps in which any input can raise a scale by
                `factor`; infinite beyond the cap once references are committed.
        """
        if factor <= 1.0:
            return 0.0
        if self._ref is not None and factor > self.cap:
            return float("inf")
        return float(np.log(factor) / np.log(self.max_growth_per_step()))

    # -- commissioning-only reference fitting ---------------------------------
    def fit_references(self, residuals: np.ndarray, contexts: np.ndarray,
                       min_rows: int = 1000, min_gap: float = 1.0) -> dict:
        """Fit each context's centre and scale from ALL its commissioning rows."""
        if min_gap < 1.0:
            raise ValueError("min_gap < 1 would narrow references, which is "
                             "the opposite of the failure being corrected")
        if self._ref is not None:
            raise RuntimeError("fit_references is commissioning-only: references "
                               "are already committed, and re-fitting them here "
                               "would bypass the cap they anchor")
        residuals = np.asarray(residuals, dtype=np.float64)
        contexts = np.asarray(contexts, dtype=np.int64)
        if residuals.ndim != 2 or residuals.shape[1] != self.n_channels:
            raise ValueError(f"residuals must be (rows, {self.n_channels})")
        if contexts.shape != residuals.shape:
            raise ValueError("contexts must have the same shape as residuals")

        report: dict[str, list] = {"refitted": [], "too_few_rows": [],
                                   "already_close": []}
        for j in range(self.n_channels):
            col, ctx_col = residuals[:, j], contexts[:, j]
            for ctx in np.unique(ctx_col):
                ctx = int(ctx)
                vals = col[(ctx_col == ctx) & np.isfinite(col)]
                if vals.size < min_rows:
                    report["too_few_rows"].append((j, ctx, int(vals.size)))
                    continue
                m = float(np.median(vals))
                sc = robust_scale(np.abs(vals - m), self.min_scale)
                floor = self.scale_floor[j]
                if sc is None and floor <= 0:
                    continue  # degenerate with no floor: leave it abstaining
                sc = max(sc if sc is not None else 0.0, floor)
                before = self._state[j].get(ctx)
                if before is not None and min_gap > 1.0:
                    reached = float(before[1])
                    if reached > 0.0 and sc < reached * min_gap:
                        report["already_close"].append(
                            (j, ctx, int(vals.size), reached, float(sc)))
                        continue
                self._state[j][ctx] = [m, sc]
                self._warmup[j].pop(ctx, None)
                report["refitted"].append(
                    (j, ctx, int(vals.size),
                     None if before is None else float(before[1]), float(sc)))
        for j, ctx, *_ in report["refitted"]:
            self.saturation.pop((j, ctx), None)   # pressure against the old estimate
        return report

    def score_batch(self, residuals: np.ndarray, contexts: np.ndarray) -> np.ndarray:
        """`(r - m) / s` per sample with NO learning, abstaining where a
                context has no state.
        """
        residuals = np.asarray(residuals, dtype=np.float64)
        contexts = np.asarray(contexts, dtype=np.int64)
        if residuals.ndim != 2 or residuals.shape[1] != self.n_channels:
            raise ValueError(f"residuals must be (rows, {self.n_channels})")
        if contexts.shape != residuals.shape:
            raise ValueError("contexts must have the same shape as residuals")
        out = np.full(residuals.shape, np.nan)
        for j in range(self.n_channels):
            state = self._state[j]
            col, ctx_col = residuals[:, j], contexts[:, j]
            for ctx in np.unique(ctx_col):
                st = state.get(int(ctx))
                if st is None:
                    continue
                m = (ctx_col == ctx) & np.isfinite(col)
                out[m, j] = (col[m] - st[0]) / st[1]
        return out

    # -- references ------------------------------------------------------------
    def commit_references(self) -> None:
        """Freeze the current centre and scale of every context as the anchor
                all later learning is clamped against (called at commissioning).
        """
        self._ref = [{ctx: (m, s) for ctx, (m, s) in per.items()} for per in self._state]

    def _pooled_reference(self, channel: int) -> float | None:
        if self._ref is None or not self._ref[channel]:
            return None
        return float(np.median([s for _, s in self._ref[channel].values()]))

    def _bounds(self, channel: int, ctx: int) -> tuple[float, float, float, float] | None:
        """(s_lo, s_hi, m_lo, m_hi) for a context, or None before commissioning."""
        if self._ref is None:
            return None
        ref = self._ref[channel].get(ctx)
        floor = self.scale_floor[channel]
        if ref is None:
            pooled = self._pooled_reference(channel)
            if pooled is None:
                return None
            return max(pooled / self.cap, floor), pooled * self.cap, -np.inf, np.inf
        m_ref, s_ref = ref
        return (max(s_ref / self.cap, floor), max(s_ref * self.cap, floor),
                m_ref - self.band * s_ref, m_ref + self.band * s_ref)

    # -- streaming ---------------------------------------------------------------
    def _initialise(self, channel: int, ctx: int, values: list[float]) -> None:
        arr = np.asarray(values, dtype=np.float64)
        m = float(np.median(arr))
        s = robust_scale(np.abs(arr - m), self.min_scale)
        floor = self.scale_floor[channel]
        if s is None and floor <= 0:
            return  # degenerate and no floor: stay uninitialised (abstain)
        s = max(s if s is not None else 0.0, floor)
        if (self.pooled_init_for_unseen and self._ref is not None
                and self._ref[channel].get(ctx) is None):
            pooled = self._pooled_reference(channel)
            if pooled is not None:
                s = max(pooled, floor)
        bounds = self._bounds(channel, ctx)
        if bounds is not None:
            s = min(max(s, bounds[0]), bounds[1])
        self._state[channel][ctx] = [m, s]

    def update(self, residual: np.ndarray, regime_id, is_normal: bool,
               learn: np.ndarray | None = None) -> np.ndarray:
        if residual.shape != (self.n_channels,):
            raise ValueError(f"expected shape ({self.n_channels},), got {residual.shape}")
        if learn is not None and learn.shape != (self.n_channels,):
            raise ValueError(f"learn must have shape ({self.n_channels},)")
        regimes = per_channel(regime_id, self.n_channels)
        scaled = np.full(self.n_channels, np.nan)
        for j in range(self.n_channels):
            r = residual[j]
            if np.isnan(r):
                continue
            ctx = int(regimes[j])
            state = self._state[j].get(ctx)
            may_learn = is_normal and (learn is None or learn[j])
            if state is None:
                if may_learn:
                    buf = self._warmup[j].setdefault(ctx, [])
                    buf.append(float(r))
                    if len(buf) >= self.min_calibration:
                        self._initialise(j, ctx, buf)
                        del self._warmup[j][ctx]
                continue  # uncalibrated context: abstain
            m, s = state
            scaled[j] = (r - m) / s
            if not may_learn:
                continue
            z = min(max((r - m) / s, -self.clip), self.clip)
            m = m + self.alpha * z * s
            s = float(np.sqrt((1.0 - self.alpha) * s * s + self.alpha * (z * s) ** 2 / self.kappa))
            s = max(s, self.scale_floor[j])
            bounds = self._bounds(j, ctx)
            pinned = False
            if bounds is not None:
                s_lo, s_hi, m_lo, m_hi = bounds
                pinned = s >= s_hi or (self.band > 0 and (m <= m_lo or m >= m_hi))
                s = min(max(s, s_lo), s_hi)
                m = min(max(m, m_lo), m_hi)
            state[0], state[1] = m, s
            key = (j, ctx)
            if pinned:
                self.saturation[key] = self.saturation.get(key, 0) + 1
            elif key in self.saturation:
                del self.saturation[key]
        return scaled

    def current_scale(self, channel: int, regime_id: int) -> float | None:
        state = self._state[channel].get(int(regime_id))
        return None if state is None else state[1]

    def saturated(self, min_steps: int) -> list[tuple[int, int, int]]:
        """(channel, context, steps) pinned at a limit for >= `min_steps`."""
        return sorted((j, c, n) for (j, c), n in self.saturation.items() if n >= min_steps)

    # -- versioning: references belong to the bundle; m and s are runtime --
    def export_state(self) -> tuple[dict[int, tuple[float, float, float, float]], ...]:
        """(m, s, m_ref, s_ref) per (channel, context); references are NaN
                before commissioning. Warm-up buffers are not exported.
        """
        out = []
        for j, per in enumerate(self._state):
            refs = self._ref[j] if self._ref is not None else {}
            out.append({ctx: (m, s, *refs.get(ctx, (np.nan, np.nan))) for ctx, (m, s) in per.items()})
        return tuple(out)

    def load_state(self, state) -> None:
        if len(state) != self.n_channels:
            raise ValueError(f"expected state for {self.n_channels} channels, got {len(state)}")
        self._state = [{ctx: [v[0], v[1]] for ctx, v in per.items()} for per in state]
        refs = [{ctx: (v[2], v[3]) for ctx, v in per.items() if np.isfinite(v[3])} for per in state]
        self._ref = refs if any(refs) else None
        self._warmup = [{} for _ in range(self.n_channels)]
        self.saturation = {}

    def n_regimes(self) -> int:
        return len({c for per in self._state for c in per})

    def n_bytes(self) -> int:
        """Four floats per calibrated (channel, context)."""
        return sum(len(per) for per in self._state) * 4 * 8
