"""Where false alarms fall relative to the attacks."""
from __future__ import annotations


import numpy as np
import pandas as pd

from evaluation.ground_truth import GroundTruth
from evaluation.metrics import _ranges


def false_alarm_timing(states: np.ndarray, timestamps: pd.Series, gt: GroundTruth,
                       attack_period: np.ndarray,
                       recovery_window_s: float = 900.0) -> dict:
    """How many false alarms start soon after an attack ended."""
    alarm = np.asarray(states, dtype=object) == "ANOMALY"
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    clean = ~np.asarray(attack_period, dtype=bool)
    starts = [s for s, _ in _ranges(alarm) if clean[s]]
    ends = sorted(a.end for a in gt.attacks)

    def soon_after(s: int) -> bool:
        t = ts[s]
        return any(pd.Timedelta(0) < t - e <= pd.Timedelta(seconds=recovery_window_s)
                   for e in ends)

    after = sum(soon_after(s) for s in starts)
    return {"recovery_window_s": recovery_window_s,
            "n_false_alarm_episodes": len(starts),
            "starting_within_the_recovery_window": after,
            "elsewhere": len(starts) - after}


