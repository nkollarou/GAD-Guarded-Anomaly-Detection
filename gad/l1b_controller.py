"""A model of the plant's control rules. Switched off in the evaluated detector."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from gad.l0_quality import is_blinding
from gad.l3_cusum import Cusum, calibrate_h_evt
from gad.types import Sample

OFF, ON = 1.0, 2.0
MIN_EVENTS = 10
PAIR_MIN_SUPPORT = 0.01
P_RANGE_FLOOR = 1e-4
RANGE_TAIL_QUANTILE = 0.90
RANGE_MIN_PEAKS = 10
H_CONSTRUCTION = 1.0
MAX_GAP_PERIODS = 10.0


@dataclass(frozen=True)
class ControllerStructure:
    """Design knowledge from the adapter: (actuator, controlling sensor)
        pairs for trigger rules, and where that knowledge comes from.
    """

    triggers: tuple[tuple[str, str], ...]
    source: str


@dataclass(frozen=True)
class TriggerRule:
    actuator: str
    sensor: str
    frm: float
    to: float
    lo: float          # sensor range at the switch instants
    hi: float
    width: float
    overdue_side: str  # "below" or "above": where the pending switch is due
    overdue_bound: float  # furthest the sensor went while the switch was pending

    @property
    def name(self) -> str:
        return f"trigger {self.actuator} {int(self.frm)}->{int(self.to)} on {self.sensor}"


@dataclass(frozen=True)
class PairRule:
    a: str
    a_state: float
    b: str
    b_state: float

    @property
    def name(self) -> str:
        return f"pair {self.a}={int(self.a_state)} & {self.b}={int(self.b_state)}"


@dataclass(frozen=True)
class DwellRule:
    actuator: str
    state: float
    min_s: float
    max_s: float
    spread: float

    @property
    def name(self) -> str:
        return f"dwell {self.actuator}={int(self.state)}"


@dataclass(frozen=True)
class UnseenRule:
    actuator: str
    state: float

    @property
    def name(self) -> str:
        return f"unseen {self.actuator}={int(self.state)}"


@dataclass(frozen=True)
class ControllerState:
    """The frozen, serialisable model. `h` is the CUSUM alarm level."""

    sensed: tuple[str, ...]
    commanded: tuple[str, ...]
    source: str
    triggers: tuple[TriggerRule, ...]
    pairs: tuple[PairRule, ...]
    dwells: tuple[DwellRule, ...]
    unseen: tuple[UnseenRule, ...]
    period: float = 1.0
    h: float = float("nan")
    h_method: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def n_bytes(self) -> int:
        return len(self.to_json().encode())

    @property
    def n_rules(self) -> int:
        return len(self.triggers) + len(self.pairs) + len(self.dwells) + len(self.unseen)


def settle(u: np.ndarray) -> np.ndarray:
    """Carry the last settled state (1 or 2) through transition samples
        (0). Columns of `u` are actuators; a leading 0 takes the first settled
        state that follows it.
    """
    u = np.asarray(u, dtype=np.float64)
    s = np.where(u == 0.0, np.nan, u)
    out = s.copy()
    for j in range(s.shape[1]):
        col = out[:, j]
        valid = ~np.isnan(col)
        if not valid.any():
            continue
        idx = np.where(valid, np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        col[:] = col[idx]
        first = int(np.argmax(valid))
        col[:first] = col[first]
    return out


def _tail_bound(v: np.ndarray, p: float, upper: bool) -> float:
    """The value a clean event exceeds (or falls below) with probability `p`."""
    from scipy.stats import genpareto, norm

    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    sign = 1.0 if upper else -1.0
    w = sign * v  # work on the upper tail of `w` either way
    observed = float(w.max())
    u = float(np.quantile(w, RANGE_TAIL_QUANTILE))
    excess = w[w > u] - u
    bound = None
    if len(excess) >= RANGE_MIN_PEAKS:
        p_u = len(excess) / len(w)
        if p < p_u:
            try:
                c, _, scale = genpareto.fit(excess, floc=0.0)
                x = u + float(genpareto.ppf(1.0 - p / p_u, c, loc=0.0, scale=scale))
                if np.isfinite(x):
                    bound = x
            except Exception:  # a failed fit falls back, it does not raise
                bound = None
    if bound is None:
        med = float(np.median(w))
        mad = float(np.median(np.abs(w - med))) * 1.4826
        if mad <= 0.0:
            mad = float(np.std(w))
        bound = med + float(norm.ppf(1.0 - p)) * mad
    return sign * max(bound, observed)


def _interval(v: np.ndarray, p: float) -> tuple[float, float]:
    """(lower, upper) bounds a clean event leaves with probability `p` each
        side; always contains the observed range.
    """
    return _tail_bound(v, p, upper=False), _tail_bound(v, p, upper=True)


def _p_per_event(n_events: int, span_s: float, budget_episodes_per_day: float,
                 n_rules_hint: int) -> float:
    """A rule's allowed probability of being left per event: its share of
        the monitor's episode budget spread over the events it sees per day.
    """
    days = max(span_s / 86_400.0, 1e-9)
    per_day = max(n_events / days, 1e-9)
    share = budget_episodes_per_day / max(n_rules_hint, 1)
    return float(max(min(share / per_day, 0.5), P_RANGE_FLOOR))


def _runs(s: np.ndarray) -> list[tuple[float, int, int]]:
    change = np.flatnonzero(np.diff(s) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(s)]])
    return [(float(s[a]), int(a), int(b - a)) for a, b in zip(starts, ends)]


def fit_controller(
    X: np.ndarray, U: np.ndarray,
    sensed: list[str] | tuple[str, ...], commanded: list[str] | tuple[str, ...],
    structure: ControllerStructure, period: float = 1.0,
    budget_episodes_per_day: float = 0.1,
) -> ControllerState:
    """Read every rule's numbers off clean, time-ordered data. Raises if the
        structure names a channel the stream does not carry.
    """
    sensed, commanded = tuple(sensed), tuple(commanded)
    for act, sen in structure.triggers:
        if act not in commanded or sen not in sensed:
            raise ValueError(f"structure names {act}/{sen}, not in the stream")
    X = np.asarray(X, dtype=np.float64)
    S = settle(U)
    n = len(S)

    unseen = []
    for j, a in enumerate(commanded):
        seen = set(np.unique(S[:, j][~np.isnan(S[:, j])]).tolist())
        unseen += [UnseenRule(a, st) for st in (OFF, ON) if st not in seen]

    span = n * period
    n_rules_hint = 2 * len(structure.triggers) + 2 * len(commanded)

    dwells = []
    for j, a in enumerate(commanded):
        complete = _runs(S[:, j])[1:-1]  # first/last runs are cut by the window
        for st in (OFF, ON):
            lengths = np.array([m * period for s, _, m in complete if s == st], dtype=float)
            if len(lengths) >= MIN_EVENTS:
                p = _p_per_event(len(lengths), span, budget_episodes_per_day, n_rules_hint)
                lo, hi = _interval(lengths, p)
                lo = max(lo, 0.0)
                dwells.append(DwellRule(a, st, float(lo), float(hi),
                                        max(float(hi - lo), period)))

    triggers = []
    for act, sen in structure.triggers:
        s = S[:, commanded.index(act)]
        x = X[:, sensed.index(sen)]
        idx = np.flatnonzero(np.diff(s) != 0) + 1
        for frm, to in ((OFF, ON), (ON, OFF)):
            at = idx[(s[idx - 1] == frm) & (s[idx] == to)]
            v = x[at][np.isfinite(x[at])]
            waiting = x[(s == frm) & np.isfinite(x)]
            if len(v) < MIN_EVENTS or len(waiting) == 0:
                continue
            p = _p_per_event(len(v), span, budget_episodes_per_day, n_rules_hint)
            lo, hi = _interval(v, p)
            width = max(hi - lo, 1e-9 * max(1.0, abs(float(np.median(v)))))
            below = np.median(v) <= np.median(waiting)
            reach = [(x[a:a + m][np.isfinite(x[a:a + m])]) for st_, a, m in _runs(s) if st_ == frm]
            reach = np.array([float(r.min() if below else r.max()) for r in reach if len(r)])
            if len(reach) == 0:
                continue
            p_reach = _p_per_event(len(reach), span, budget_episodes_per_day, n_rules_hint)
            if below:
                side = "below"
                bound = min(_tail_bound(reach, p_reach, upper=False), lo)
            else:
                side = "above"
                bound = max(_tail_bound(reach, p_reach, upper=True), hi)
            triggers.append(TriggerRule(act, sen, frm, to, lo, hi, width, side, bound))

    pairs = []
    switching = [j for j in range(len(commanded))
                 if np.all(np.isin([OFF, ON], S[:, j]))]
    for i_pos, i in enumerate(switching):
        for j in switching[i_pos + 1:]:
            for sa in (OFF, ON):
                for sb in (OFF, ON):
                    in_a, in_b = S[:, i] == sa, S[:, j] == sb
                    if in_a.mean() < PAIR_MIN_SUPPORT or in_b.mean() < PAIR_MIN_SUPPORT:
                        continue
                    if not np.any(in_a & in_b):  # any clean exception discards it
                        pairs.append(PairRule(commanded[i], sa, commanded[j], sb))

    return ControllerState(sensed, commanded, structure.source, tuple(triggers),
                           tuple(pairs), tuple(dwells), tuple(unseen), period)


@dataclass
class ControllerOutput:
    score: float            # summed rule scores this sample
    present_fraction: float  # share of rules not abstaining this sample
    rule_scores: dict[str, float] = field(default_factory=dict)  # nonzero only


class ControllerModel:
    """Streams one sample at a time against a frozen `ControllerState`.
        Runtime state is only the last settled actuator states, the time each
        entered its state, and a decayed per-rule total used to name the rules
        behind an alarm.
    """

    def __init__(self, state: ControllerState, lam: float = 0.99) -> None:
        self.state = state
        self.lam = lam
        self._a_index = {a: i for i, a in enumerate(state.commanded)}
        self._x_index = {x: i for i, x in enumerate(state.sensed)}
        self._prev: np.ndarray | None = None
        self._entered: np.ndarray | None = None  # NaN = entry time unknown
        self._last_t: float | None = None
        self.rule_totals: dict[str, float] = {}

    def reset(self) -> None:
        self._prev = None
        self._entered = None
        self._last_t = None
        self.rule_totals = {}

    def update(self, sample: Sample) -> ControllerOutput:
        st = self.state
        u = np.asarray(sample.u, dtype=np.float64)
        cur = u.copy()
        if self._prev is None:
            cur[cur == 0.0] = np.nan
        else:
            cur = np.where(cur == 0.0, self._prev, cur)  # carry through transitions
        switched = np.zeros(len(cur), dtype=bool)
        if self._prev is None:
            self._entered = np.full(len(cur), np.nan)  # dwell unknown until a switch
        else:
            switched = (cur != self._prev) & ~np.isnan(cur) & ~np.isnan(self._prev)
        t = float(sample.t)
        if self._last_t is not None and t - self._last_t > MAX_GAP_PERIODS * st.period:
            self._entered = np.full(len(cur), np.nan)
        self._last_t = t
        dwell_before = t - self._entered  # time spent in the state being left (NaN if unknown)
        entered_now = np.where(switched, t, self._entered)  # entry into the current state

        scores: dict[str, float] = {}
        n_rules = st.n_rules
        n_abstain = 0

        for r in st.triggers:
            j = self._a_index[r.actuator]
            k = self._x_index[r.sensor]
            if is_blinding(int(sample.q[k])):
                n_abstain += 1
                continue
            x = float(sample.x[k])
            s = 0.0
            if switched[j] and self._prev[j] == r.frm and cur[j] == r.to:
                s += max(r.lo - x, x - r.hi, 0.0) / r.width
            elif cur[j] == r.frm:  # still pending: overdue past anything clean
                beyond = (r.overdue_bound - x) if r.overdue_side == "below" else (x - r.overdue_bound)
                s += max(beyond, 0.0) / r.width
            if s > 0.0:
                scores[r.name] = s

        for r in st.pairs:
            if cur[self._a_index[r.a]] == r.a_state and cur[self._a_index[r.b]] == r.b_state:
                scores[r.name] = 1.0

        for r in st.dwells:
            j = self._a_index[r.actuator]
            s = 0.0
            if switched[j] and self._prev[j] == r.state and np.isfinite(dwell_before[j]):
                s = max(r.min_s - dwell_before[j], 0.0) / r.spread  # left too early
            elif cur[j] == r.state and np.isfinite(entered_now[j]):
                s = max((t - entered_now[j]) - r.max_s, 0.0) / r.spread  # held too long
            if s > 0.0:
                scores[r.name] = s

        for r in st.unseen:
            if cur[self._a_index[r.actuator]] == r.state:
                scores[r.name] = 1.0

        # advance runtime state
        self._entered = entered_now
        self._prev = cur

        for name in list(self.rule_totals):
            self.rule_totals[name] *= self.lam
        for name, s in scores.items():
            self.rule_totals[name] = self.rule_totals.get(name, 0.0) + s
        self.rule_totals = {k: v for k, v in self.rule_totals.items() if v > 1e-6}

        present = 1.0 - n_abstain / n_rules if n_rules else 0.0
        return ControllerOutput(float(sum(scores.values())), present, scores)

    def top_rules(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.rule_totals.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


class ControllerMonitor:
    """The controller model plus its one-channel CUSUM (slack 0: a rule
        score is already the excess beyond clean behaviour).
    """

    def __init__(self, model: ControllerModel, lam: float, h: float) -> None:
        self.model = model
        self.cusum = Cusum(np.zeros(1), np.array([h]), lam=lam)

    @property
    def h(self) -> float:
        return float(self.cusum.h[0])

    def update(self, sample: Sample) -> tuple[float, float, ControllerOutput]:
        """(evidence, present_fraction, output). Evidence is NaN only when
                every rule abstains.
        """
        out = self.model.update(sample)
        if out.present_fraction <= 0.0:
            return float("nan"), 0.0, out
        ev = float(self.cusum.update(np.array([out.score]))[0])
        return ev, out.present_fraction, out

    def alarm(self, evidence: float) -> bool:
        return bool(evidence >= self.h)  # NaN compares False: abstention never fires


def calibrate_controller(
    state: ControllerState,
    warm: list[Sample], cal: list[Sample],
    lam: float,
    budget_episodes_per_day: float, budget_time_fraction: float,
    samples_per_day: float,
) -> tuple[ControllerMonitor, ControllerState, np.ndarray]:
    """Stream `warm` (establishes the actuator states and dwell clocks),
        then `cal`, and set the alarm level to the monitor's budget share on
        the `cal` evidence. Returns the monitor — continued to the end of
        `cal`, ready to carry on the live stream — the state with `h` filled in,
        and the calibration evidence.
    """
    model = ControllerModel(state, lam=lam)
    for s in warm:
        model.update(s)
    cusum = Cusum(np.zeros(1), lam=lam)
    evidence = np.array([cusum.update(np.array([model.update(s).score]))[0] for s in cal])

    finite = evidence[np.isfinite(evidence)]
    peak = float(finite.max()) if len(finite) else 0.0
    h = H_CONSTRUCTION
    method = f"construction ({H_CONSTRUCTION:g} range-width); calibration peak {peak:.3g}"
    fitted = calibrate_h_evt(evidence, budget_episodes_per_day, budget_time_fraction,
                             samples_per_day) if peak > 0.0 else None
    if fitted is not None:
        method += f"; fitted level would be {fitted.h:.3g} ({fitted.method})"
    state = ControllerState(**{**asdict_shallow(state), "h": float(h), "h_method": method})
    monitor = ControllerMonitor(model, lam, h)
    monitor.model.state = state
    monitor.cusum.s_pos[:] = cusum.s_pos  # continue the accumulated evidence
    return monitor, state, evidence


def asdict_shallow(state: ControllerState) -> dict:
    """Field dict without recursing into the rule dataclasses."""
    return {f: getattr(state, f) for f in state.__dataclass_fields__}
