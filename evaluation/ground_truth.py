"""Attack ground truth: which samples belong to which attack."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


NO_PHYSICAL_IMPACT = "No Physical Impact"

DOCUMENTED_DEFECTS: dict[int, str] = {
    26: "the attack sheet's own footnote states the historian stopped updating "
        "the attacked value while it was held constant and interpolated it "
        "afterwards, so the logged data is not what the plant did",
}


@dataclass(frozen=True)
class Attack:
    number: int
    start: pd.Timestamp
    end: pd.Timestamp  # inclusive
    raw_points: tuple[str, ...]
    points: tuple[str, ...]  # logged historian columns
    unlogged_points: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class GroundTruth:
    attacks: tuple[Attack, ...]
    excluded: tuple[tuple[int, str], ...]  # (attack number, reason)
    fixes: tuple[str, ...]  # every correction applied, human-readable
    notes: tuple[str, ...]
    unscoreable: tuple[Attack, ...] = ()


def attack_number_per_row(timestamps: pd.Series, gt: GroundTruth) -> np.ndarray:
    """For each timestamp, the number of the attack whose window contains it, or
    0. Windows are inclusive at both ends.
    """
    ts = pd.to_datetime(timestamps).to_numpy()
    out = np.zeros(len(ts), dtype=np.int64)
    for a in gt.attacks:
        inside = (ts >= a.start.to_datetime64()) & (ts <= a.end.to_datetime64())
        out[inside & (out == 0)] = a.number
    return out
