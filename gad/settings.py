"""Every number we chose for GAD, in one place.

These are the values used in the paper. All of them can be changed for
another plant or another trade-off. `GAD` below turns them into the
configuration that `commission.py` fits.
"""
from __future__ import annotations

# --- how the clean recording is used ------------------------------------
#: fit on the first 45 % of the clean recording, warm up on the next 18 % of
#: the rest, set thresholds on the 27.5 % after that; the remainder is the
#: 40-hour test stream
SPLIT = (0.45, 0.18, 0.275)

# --- false alarms -------------------------------------------------------
FALSE_ALARMS_PER_DAY = 1.0
SHARE_OF_TIME_IN_ALARM = 0.01

# --- L0: input quality --------------------------------------------------
#: a reading is out of range if it is this many typical one-second steps
#: outside the range seen in clean data
L0_RANGE_MARGIN = 5.0
#: report DEGRADED when more than this share of the residuals cannot be computed
DEGRADED_SHARE = 0.5

# --- L1b: physics -------------------------------------------------------
#: a reading this many standard deviations away from the physics is ignored
PHYSICS_GATE_SD = 4.0
#: ... but for at most this long, so a real change in the plant is accepted
PHYSICS_MAX_IGNORE_MINUTES = 10

# --- L2: residual scaling -----------------------------------------------
#: how fast the means and standard deviations follow the plant
HALF_LIFE_HOURS = 12
#: time after a change of operating mode spent in a separate settling context
#: (one CUSUM memory length, 1 / (1 - 0.99))
SETTLING_SECONDS = 100

# --- L3: CUSUM ----------------------------------------------------------
CUSUM_LAMBDA = 0.99

# --- L4: decision -------------------------------------------------------
#: seconds above the threshold before an alarm is raised, and below it before
#: the alarm is cleared
ALARM_RAISE_SECONDS = 5
ALARM_CLEAR_SECONDS = 30

# --- L5: guarded adaptation ---------------------------------------------
UPDATE_EVERY_HOURS = 6
#: thresholds are re-estimated from this many hours of trusted data
THRESHOLD_WINDOW_HOURS = 96
#: rule 1: wait this long after an anomaly before learning again
WAIT_AFTER_ANOMALY_MINUTES = 15
#: rule 2: clip each residual at this many standard deviations
CLIP_SD = 3.0
#: rule 3: each standard deviation and threshold stays within a factor c of
#: its value at commissioning, each mean within MEAN_LIMIT_SD standard deviations
LIMIT_C = 2.0
MEAN_LIMIT_SD = 1.0
#: rule 4: the golden set, taken from the end of the fitting data
GOLDEN_EXCERPTS = 8
GOLDEN_EXCERPT_MINUTES = 30
GOLDEN_SPAN_HOURS = 24


HOUR = 3600

#: the detector as evaluated in the paper
GAD = {
    "learn_guard": WAIT_AFTER_ANOMALY_MINUTES * 60,
    "l2_cap": LIMIT_C,
    "adaptive_h_cap": LIMIT_C,
    "l2_band": MEAN_LIMIT_SD,
    "l2_clip": CLIP_SD,
    "l2_half_life": HALF_LIFE_HOURS * float(HOUR),
    "adaptive_h_period": UPDATE_EVERY_HOURS * float(HOUR),
    "adaptive_h_window": THRESHOLD_WINDOW_HOURS * HOUR,
    "l1b_innovation_gate": PHYSICS_GATE_SD,
    "l1b_gate_max_hold": PHYSICS_MAX_IGNORE_MINUTES * 60,
    "settle_samples": SETTLING_SECONDS,
    "dwell_anomaly": ALARM_RAISE_SECONDS,
    "dwell_recover": ALARM_CLEAR_SECONDS,
    # how the layers are built
    "l1a_cross": True,
    "l1a_cross_rule": "0083",
    "use_l1a_long_calibration": True,
    "cross_channel_warm_calibration": True,
    "l2_kind": "bounded",
    "l2_reference_from_commissioning_rows": True,
    "unseen_context_pooled_init": True,
    "l3_per_context_h": True,
    "l3_per_context_estimator": "evt",
    "l3_per_context_holdout": 0.2,
    "l3_per_context_holdout_scope": "mode",
    "l3_per_context_long_calibration": True,
    "cusum_cap_factor": 1.1,
    "full_budget_split": True,
    "per_sensor_alarms": True,
    "adaptive_h": True,
    "unified_l5": True,
    "sensitivity_budget": True,
    "trusted_learning": True,
    "per_channel_learning": True,
}

#: the same detector with the four rules of L5 switched off
UNGUARDED = {
    **GAD,
    "l2_cap": 1e6,
    "adaptive_h_cap": 1e6,
    "l2_band": 1e6,
    "sensitivity_budget": False,
    "naive_learning": True,
}
