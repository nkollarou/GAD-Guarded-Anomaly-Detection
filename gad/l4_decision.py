"""L4: the decision. Raises an alarm when evidence stays above the threshold,
clears it after a quiet period, and reports DEGRADED when too little can be
checked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable

import numpy as np

from gad.types import Decision, DetectorState

Key = Hashable  # typically (branch_name, channel_name)


def conformal_p_value(s: float, sorted_calibration: np.ndarray) -> float:
    """p = (1 + #{clean >= s}) / (n + 1), against a sorted, frozen
        calibration array.
    """
    n = len(sorted_calibration)
    count = n - int(np.searchsorted(sorted_calibration, s, side="left"))
    return (1 + count) / (n + 1)


def simes_reject(p_values: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> bool:
    """Sort ascending, fire if any p_(i) <= i*alpha/m (1-indexed)."""
    m = len(p_values)
    if m == 0:
        return False
    if weights is None:
        adjusted = p_values
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != p_values.shape:
            raise ValueError("weights must have the same shape as p_values")
        if np.any(weights <= 0):
            raise ValueError("weights must be strictly positive")
        weights = weights * (m / weights.sum())  # present weights sum to m
        adjusted = p_values / weights
    sorted_adjusted = np.sort(adjusted)
    thresholds = (np.arange(1, m + 1) * alpha) / m
    return bool(np.any(sorted_adjusted <= thresholds))


class ConformalCalibration:
    """Frozen clean calibration sets, one per (branch, channel) key.
        Refuses at construction if `alpha`/`max_simultaneous_tests` would ever
        need a smaller p-value than any calibration set can express — the
        smallest possible p-value is 1/(n+1), and a set too small silently
        puts ordinary data in the grey band.
    """

    def __init__(
        self,
        calibration_sets: dict[Key, np.ndarray],
        alpha: float,
        max_simultaneous_tests: int,
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1)")
        if max_simultaneous_tests < 1:
            raise ValueError("max_simultaneous_tests must be >= 1")

        self.alpha = alpha
        self.max_simultaneous_tests = max_simultaneous_tests
        min_required_n = int(np.ceil(max_simultaneous_tests / alpha)) - 1

        self._sets: dict[Key, np.ndarray] = {}
        for key, arr in calibration_sets.items():
            arr = np.asarray(arr, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if len(arr) < min_required_n:
                raise ValueError(
                    f"calibration set for {key!r} has n={len(arr)}, but alpha={alpha} with up "
                    f"to {max_simultaneous_tests} simultaneous tests needs n>={min_required_n} "
                    "(smallest expressible p-value is 1/(n+1)) — refusing construction rather "
                    "than silently blinding this channel"
                )
            self._sets[key] = np.sort(arr)

    def p_value(self, key: Key, s: float) -> float:
        return conformal_p_value(s, self._sets[key])

    def keys(self):
        return self._sets.keys()


class Decider:
    """Three states with hysteresis and a minimum dwell. DEGRADED means
        abstention and nothing else: it is triggered purely by the fraction
        of (branch, channel) tests that abstained this step (L0 quality flags
        or an uncalibrated regime/branch), never by a grey-band p-value and
        never by evidence still accumulating toward the anomaly dwell.
    """

    def __init__(
        self,
        calibration: ConformalCalibration,
        n_possible_tests: int,
        degraded_abstention_threshold: float,
        dwell_anomaly: int,
        dwell_recover: int,
        bundle_id: str,
        reliability_weights: dict[Key, float] | None = None,
    ) -> None:
        """`reliability_weights`: a static, per-key
                base weight for the fusion (e.g. inverse-variance — how reliable
                that branch's prediction of that channel has measured to be), fed
                to `simes_reject`. A key with no entry gets weight 1.0. `None`
                (default) is exactly the unweighted test — no behavior change for
                a caller that doesn't supply any.
        """
        self.calibration = calibration
        self.n_possible_tests = n_possible_tests
        self.degraded_abstention_threshold = degraded_abstention_threshold
        self.dwell_anomaly = dwell_anomaly
        self.dwell_recover = dwell_recover
        self.bundle_id = bundle_id
        self.reliability_weights = reliability_weights or {}

        self.state = DetectorState.NORMAL
        self._reject_streak = 0
        self._accept_streak = 0

    def update(self, t: float, evidence_by_key: dict[Key, float]) -> Decision:
        n_present = len(evidence_by_key)
        abstention_fraction = 1.0 - (n_present / self.n_possible_tests if self.n_possible_tests else 0.0)

        if abstention_fraction > self.degraded_abstention_threshold:
            self.state = DetectorState.DEGRADED
            self._reject_streak = 0
            self._accept_streak = 0
            return Decision(t=t, state=self.state, bundle_id=self.bundle_id)

        keys = list(evidence_by_key.keys())
        p_values = np.array([self.calibration.p_value(k, evidence_by_key[k]) for k in keys])
        weights = np.array([self.reliability_weights.get(k, 1.0) for k in keys]) if n_present else None
        reject = simes_reject(p_values, self.calibration.alpha, weights) if n_present else False

        if reject:
            self._reject_streak += 1
            self._accept_streak = 0
        else:
            self._accept_streak += 1
            self._reject_streak = 0

        if self.state == DetectorState.ANOMALY:
            if self._accept_streak >= self.dwell_recover:
                self.state = DetectorState.NORMAL
            # else: stay ANOMALY — hysteresis, recovery needs sustained evidence
        else:
            # NORMAL, or just exited DEGRADED with streaks reset to 0
            self.state = (
                DetectorState.ANOMALY if self._reject_streak >= self.dwell_anomaly else DetectorState.NORMAL
            )

        return Decision(t=t, state=self.state, bundle_id=self.bundle_id)


class MonitorDecider:
    """Three states with hysteresis and a minimum dwell, over a set of
        independent monitors. Each monitor has already decided for itself,
        against its own stated share of the false-alarm budget; this class
        only combines those verdicts, applies the dwell, and reports a state.
    """

    def __init__(
        self,
        degraded_abstention_threshold: float,
        dwell_anomaly: int,
        dwell_recover: int,
        bundle_id: str,
    ) -> None:
        self.degraded_abstention_threshold = degraded_abstention_threshold
        self.dwell_anomaly = dwell_anomaly
        self.dwell_recover = dwell_recover
        self.bundle_id = bundle_id

        self.state = DetectorState.NORMAL
        self._reject_streak = 0
        self._accept_streak = 0

    def update(
        self, t: float, monitors: dict[str, bool], abstention_fraction: float
    ) -> Decision:
        if abstention_fraction > self.degraded_abstention_threshold:
            self.state = DetectorState.DEGRADED
            self._reject_streak = 0
            self._accept_streak = 0
            return Decision(t=t, state=self.state, bundle_id=self.bundle_id)

        reject = any(monitors.values())

        if reject:
            self._reject_streak += 1
            self._accept_streak = 0
        else:
            self._accept_streak += 1
            self._reject_streak = 0

        if self.state == DetectorState.ANOMALY:
            if self._accept_streak >= self.dwell_recover:
                self.state = DetectorState.NORMAL
            # else: stay ANOMALY — recovery needs sustained quiet
        else:
            self.state = (
                DetectorState.ANOMALY
                if self._reject_streak >= self.dwell_anomaly
                else DetectorState.NORMAL
            )

        return Decision(t=t, state=self.state, bundle_id=self.bundle_id)


@dataclass
class AlarmEvent:
    """One operator-facing alarm event."""

    t_open: float
    channels: list[str] = field(default_factory=list)
    t_entered: list[float] = field(default_factory=list)

    def add(self, channel: str, t: float) -> None:
        self.channels.append(channel)
        self.t_entered.append(t)


class _FiringWindow:
    """How many of the last `window` samples fired, per channel."""

    def __init__(self, n_channels: int, window: int) -> None:
        if window < 1:
            raise ValueError("dwell_window must be >= 1")
        self.window = window
        self._buf = np.zeros((window, n_channels), dtype=bool)
        self._i = 0
        self.count = np.zeros(n_channels, dtype=np.int64)

    def push(self, fired: np.ndarray) -> np.ndarray:
        """Add this sample's verdicts; return the per-channel count."""
        fired = np.asarray(fired, dtype=bool)
        self.count -= self._buf[self._i]
        self._buf[self._i] = fired
        self.count += fired
        self._i = (self._i + 1) % self.window
        return self.count

    def reset(self) -> None:
        self._buf[:] = False
        self.count[:] = 0
        self._i = 0


class PerSensorDecider:
    """Per-channel alarm states, and events formed from their onsets."""

    def __init__(
        self,
        channels: Iterable[str],
        degraded_abstention_threshold: float,
        dwell_anomaly: int,
        dwell_recover: int,
        bundle_id: str,
        group_window: float = 30.0,
        dwell_window: int | None = None,
    ) -> None:
        self.channels = tuple(channels)
        self.degraded_abstention_threshold = degraded_abstention_threshold
        self.dwell_anomaly = dwell_anomaly
        self.dwell_recover = dwell_recover
        self.bundle_id = bundle_id
        self.group_window = group_window
        self.dwell_window = dwell_window

        n = len(self.channels)
        self.in_alarm = np.zeros(n, dtype=bool)
        self._reject_streak = np.zeros(n, dtype=np.int64)
        self._accept_streak = np.zeros(n, dtype=np.int64)
        self._window = (None if dwell_window is None
                        else _FiringWindow(n, dwell_window))
        self.state = DetectorState.NORMAL
        self.events: list[AlarmEvent] = []
        self._open: AlarmEvent | None = None

    def update(self, t: float, fired: np.ndarray, abstention_fraction: float) -> Decision:
        """`fired` is one boolean per channel: did any monitor fire for it."""
        fired = np.asarray(fired, dtype=bool)
        if fired.shape != (len(self.channels),):
            raise ValueError(f"expected {len(self.channels)} channel verdicts, got {fired.shape}")

        if abstention_fraction > self.degraded_abstention_threshold:
            self.state = DetectorState.DEGRADED
            self._reject_streak[:] = 0
            self._accept_streak[:] = 0
            if self._window is not None:
                self._window.reset()
            return Decision(t=t, state=self.state, bundle_id=self.bundle_id)

        self._reject_streak = np.where(fired, self._reject_streak + 1, 0)
        self._accept_streak = np.where(fired, 0, self._accept_streak + 1)
        reached = (self._reject_streak if self._window is None
                   else self._window.push(fired))

        entering = (~self.in_alarm) & (reached >= self.dwell_anomaly)
        leaving = self.in_alarm & (self._accept_streak >= self.dwell_recover)
        self.in_alarm = (self.in_alarm | entering) & ~leaving

        for j in np.flatnonzero(entering):
            if self._open is None or t - self._open.t_open > self.group_window:
                self._open = AlarmEvent(t_open=t)
                self.events.append(self._open)
            self._open.add(self.channels[j], t)

        self.state = DetectorState.ANOMALY if self.in_alarm.any() else DetectorState.NORMAL
        return Decision(t=t, state=self.state, bundle_id=self.bundle_id)

    def channels_in_alarm(self) -> tuple[str, ...]:
        return tuple(c for c, on in zip(self.channels, self.in_alarm) if on)
