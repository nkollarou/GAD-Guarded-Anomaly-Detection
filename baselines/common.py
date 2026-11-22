"""Alarm rule and threshold search shared by the baselines."""
from __future__ import annotations

import numpy as np

from gad.types import DetectorState

DWELL, RECOVER = 1, 1


def decide(out_of_control: np.ndarray, dwell: int = 5, recover: int = 30) -> np.ndarray:
    """ANOMALY after `dwell` consecutive out-of-control samples, NORMAL again
        after `recover` consecutive in-control ones. GAD and AdaptivePCA use
        5-in/30-out; RRCF uses 1/1, i.e. a raw threshold on its per-point score.
    """
    state = np.zeros(len(out_of_control), dtype=bool)
    on, run = False, 0
    for i, bad in enumerate(out_of_control):
        if on:
            run = run + 1 if not bad else 0
            if run >= recover:
                on, run = False, 0
        else:
            run = run + 1 if bad else 0
            if run >= dwell:
                on, run = True, 0
        state[i] = on
    return state


def states_from(scores: np.ndarray, threshold: float, dwell: int = DWELL,
                recover: int = RECOVER) -> np.ndarray:
    """ANOMALY/NORMAL strings, the shape `evaluation.metrics` scores."""
    alarm = decide(scores > threshold, dwell=dwell, recover=recover)
    return np.where(alarm, DetectorState.ANOMALY.value, DetectorState.NORMAL.value)


class PerPointCalibrationError(RuntimeError):
    """No threshold below the window's maximum meets the budget."""


def time_budget_threshold(scores: np.ndarray, *, budget_time_fraction: float,
                          samples_per_day: float = 86_400.0, dwell: int = DWELL,
                          recover: int = RECOVER) -> dict:
    """The calibration record, with `threshold` set. Raises
        `PerPointCalibrationError` rather than returning the window maximum, which
        would hold any budget in-sample and none out of sample.
    """
    if not 0.0 < budget_time_fraction < 1.0:
        raise ValueError(f"budget_time_fraction must be in (0, 1), got {budget_time_fraction}")
    scores = np.asarray(scores, dtype=float)
    days = len(scores) / samples_per_day
    top = float(np.max(scores))
    for q in np.linspace(0.90, 0.99999, 2000):
        h = float(np.quantile(scores, q))
        if h >= top:
            break
        alarm = decide(scores > h, dwell=dwell, recover=recover)
        frac = float(alarm.mean())
        if frac <= budget_time_fraction:
            episodes = int(np.sum(np.diff(np.concatenate(
                ([0], alarm.astype(np.int8)))) == 1))
            return {"rule": "time fraction only (docs/decisions/0061 7.4)",
                    "quantile": float(q), "threshold": h,
                    "budget_time_fraction": float(budget_time_fraction),
                    "time_fraction": frac,
                    # measured, NOT budgeted
                    "episodes_per_day": episodes / days,
                    "days": float(days), "n_clean_samples": int(len(scores)),
                    "score_max": top, "dwell": int(dwell), "recover": int(recover)}
    raise PerPointCalibrationError(
        f"no threshold below the window maximum ({top:.4f}) holds "
        f"{budget_time_fraction:.4%} time in alarm over {len(scores)} clean samples")
