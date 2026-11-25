"""L4: the decision. Raises an alarm when evidence stays above the threshold,
clears it after a quiet period, and reports DEGRADED when too little can be
checked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from gad.types import Decision, DetectorState


@dataclass
class AlarmEvent:
    """One operator-facing alarm event."""

    t_open: float
    channels: list[str] = field(default_factory=list)
    t_entered: list[float] = field(default_factory=list)

    def add(self, channel: str, t: float) -> None:
        self.channels.append(channel)
        self.t_entered.append(t)


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
    ) -> None:
        self.channels = tuple(channels)
        self.degraded_abstention_threshold = degraded_abstention_threshold
        self.dwell_anomaly = dwell_anomaly
        self.dwell_recover = dwell_recover
        self.bundle_id = bundle_id
        self.group_window = group_window

        n = len(self.channels)
        self.in_alarm = np.zeros(n, dtype=bool)
        self._reject_streak = np.zeros(n, dtype=np.int64)
        self._accept_streak = np.zeros(n, dtype=np.int64)
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
            return Decision(t=t, state=self.state, bundle_id=self.bundle_id)

        self._reject_streak = np.where(fired, self._reject_streak + 1, 0)
        self._accept_streak = np.where(fired, 0, self._accept_streak + 1)

        entering = (~self.in_alarm) & (self._reject_streak >= self.dwell_anomaly)
        leaving = self.in_alarm & (self._accept_streak >= self.dwell_recover)
        self.in_alarm = (self.in_alarm | entering) & ~leaving

        for j in np.flatnonzero(entering):
            if self._open is None or t - self._open.t_open > self.group_window:
                self._open = AlarmEvent(t_open=t)
                self.events.append(self._open)
            self._open.add(self.channels[j], t)

        self.state = DetectorState.ANOMALY if self.in_alarm.any() else DetectorState.NORMAL
        return Decision(t=t, state=self.state, bundle_id=self.bundle_id)

