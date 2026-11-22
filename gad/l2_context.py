"""L2: operating contexts. Each sensor is judged within the current state of
the actuators it depends on, with a short settling context after a change.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _encode(u: np.ndarray, idx: np.ndarray) -> int:
    digits = np.rint(u[idx]).astype(np.int64)
    if np.any((digits < 0) | (digits > 2)):
        raise ValueError("actuator states must be in {0, 1, 2}")
    return int(np.dot(digits, 3 ** np.arange(len(idx), dtype=np.int64)))


class ActuatorContexts:
    """`channel_sets[j]`: indices into the commanded vector that sensed
        channel j is conditioned on. `stage_sets[s]`: the commanded indices of
        stage s. `channel_stage[j]`: the stage sensed channel j belongs to.
    """

    def __init__(
        self,
        channel_sets: Sequence[Sequence[int]],
        stage_sets: Sequence[Sequence[int]],
        channel_stage: Sequence[int],
    ) -> None:
        if len(channel_sets) != len(channel_stage):
            raise ValueError("channel_sets and channel_stage must have one entry per sensed channel")
        if any(not 0 <= s < len(stage_sets) for s in channel_stage):
            raise ValueError("channel_stage refers to a stage that stage_sets does not define")
        self.channel_sets = [np.asarray(sorted(set(s)), dtype=np.int64) for s in channel_sets]
        self.stage_sets = [np.asarray(sorted(set(s)), dtype=np.int64) for s in stage_sets]
        self.channel_stage = np.asarray(channel_stage, dtype=np.int64)
        self._last: dict[str, tuple[bytes, np.ndarray]] = {}

    @property
    def n_stages(self) -> int:
        return len(self.stage_sets)

    def _ids(self, kind: str, u: np.ndarray, sets: list[np.ndarray]) -> np.ndarray:
        key = np.asarray(u, dtype=np.float64).tobytes()
        hit = self._last.get(kind)
        if hit is not None and hit[0] == key:
            return hit[1].copy()
        ids = np.array([_encode(u, idx) for idx in sets], dtype=np.int64)
        self._last[kind] = (key, ids)
        return ids.copy()

    def channel_ids(self, u: np.ndarray) -> np.ndarray:
        """One context id per sensed channel — usable wherever a
                `regime_fn` is expected.
        """
        return self._ids("channel", u, self.channel_sets)

    def stage_ids(self, u: np.ndarray) -> np.ndarray:
        return self._ids("stage", u, self.stage_sets)


def per_channel(regime, n: int) -> np.ndarray:
    """A `regime_fn` result as one id per channel: an int means every
        channel shares it (the whole-plant keying), an array is used as is.
    """
    ids = np.asarray(regime)
    if ids.ndim == 0:
        return np.full(n, int(ids), dtype=np.int64)
    if ids.shape != (n,):
        raise ValueError(f"regime ids must be a scalar or have shape ({n},)")
    return ids.astype(np.int64)


SETTLING = -1


class SettlingTracker:
    """Marks each channel as SETTLING for `n_settle` samples after its context
        changes.
    """

    def __init__(self, n_channels: int, n_settle: int) -> None:
        if n_settle < 0:
            raise ValueError("n_settle must be >= 0")
        self.n_channels = int(n_channels)
        self.n_settle = int(n_settle)
        self.reset()

    def reset(self) -> None:
        """Forget the history: the next sample starts a new time series."""
        self._prev: np.ndarray | None = None
        self._since = np.zeros(self.n_channels, dtype=np.int64)

    def step(self, regime):
        if self.n_settle == 0:
            return regime
        ids = per_channel(regime, self.n_channels).astype(np.int64, copy=True)
        if self._prev is None:
            self._since[:] = 0
        else:
            changed = ids != self._prev
            self._since = np.where(changed, 0, np.minimum(self._since + 1, self.n_settle))
        self._prev = ids
        return np.where(self._since < self.n_settle, SETTLING, ids)
