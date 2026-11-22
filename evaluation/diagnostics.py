"""Where false alarms fall relative to the attacks."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from evaluation.ground_truth import GroundTruth, attack_number_per_row
from evaluation.metrics import _distribution, _ranges


def saturation(states: np.ndarray) -> dict:
    """Is this detector detecting, or just stuck on?"""
    alarm = np.asarray(states, dtype=object) == "ANOMALY"
    runs = [len(list(g)) for v, g in itertools.groupby(alarm.tolist()) if v]
    return {
        "time_in_alarm": float(alarm.mean()) if alarm.size else float("nan"),
        "n_alarm_runs": len(runs),
        "longest_alarm_run_samples": max(runs, default=0),
        "median_alarm_run_samples": float(np.median(runs)) if runs else None,
    }


def missed_rows_by_position(alarm: np.ndarray, timestamps: pd.Series, gt: GroundTruth,
                            classes: dict | None = None) -> dict:
    """*Where* inside each attack window are the missed rows?"""
    alarm = np.asarray(alarm, dtype=bool)
    ts = pd.to_datetime(timestamps).reset_index(drop=True).to_numpy()
    per_attack = []
    for a in gt.attacks:
        if classes is not None and classes.get(a.number, "sensor") == "actuator":
            continue
        inside = np.flatnonzero((ts >= a.start.to_datetime64())
                                & (ts <= a.end.to_datetime64()))
        if not inside.size:
            continue
        w = alarm[inside]
        n = int(w.size)
        runs = _ranges(w)
        if not runs:
            per_attack.append({"number": a.number, "n_rows": n, "detected": False,
                               "onset": n, "gaps": 0, "tail": 0, "n_gaps": 0,
                               "longest_gap": 0, "points": list(a.points)})
            continue
        gaps = [runs[i + 1][0] - runs[i][1] - 1 for i in range(len(runs) - 1)]
        per_attack.append({
            "number": a.number, "n_rows": n, "detected": True,
            "onset": int(runs[0][0]),
            "gaps": int(sum(gaps)),
            "tail": int(n - 1 - runs[-1][1]),
            "n_gaps": len(gaps),
            "longest_gap": int(max(gaps, default=0)),
            "points": list(a.points),
        })
    tot = {k: int(sum(p[k] for p in per_attack)) for k in ("onset", "gaps", "tail")}
    missed = sum(tot.values())
    return {
        "onset": tot["onset"], "gaps": tot["gaps"], "tail": tot["tail"],
        "missed_rows": missed,
        "shares": {k: (tot[k] / missed if missed else float("nan")) for k in tot},
        "n_never_detected": sum(1 for p in per_attack if not p["detected"]),
        "per_attack": per_attack,
        "source": "missed rows split by position in the attack window: before "
                  "the first alarm, between alarm runs, after the last "
                  "(docs/decisions/0066 §7)",
    }


def attack_observability(values: pd.DataFrame, timestamps: pd.Series, gt: GroundTruth,
                         min_constant_s: float = 1800.0, baseline_share: float = 0.2,
                         min_distinct: int = 50) -> dict:
    """How hard is each attack to see at all, before any detector runs?"""
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    eligible: dict[str, float] = {}
    for tag in values.columns:
        v = pd.to_numeric(values[tag], errors="coerce").to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        if v.size and len(np.unique(v)) >= min_distinct:
            _, counts = np.unique(v, return_counts=True)
            eligible[tag] = float(counts.max() / v.size)

    per_attack = []
    for a in gt.attacks:
        inside = ((ts >= a.start) & (ts <= a.end)).to_numpy()
        n = int(inside.sum())
        worst = {"tag": None, "share": 0.0, "run": 0, "baseline": None}
        flat = []
        for tag, base in eligible.items():
            if base >= baseline_share or not n:
                continue  # flat everywhere: its flatness here means nothing
            v = pd.to_numeric(values.loc[inside, tag], errors="coerce").to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            if not v.size:
                continue
            _, counts = np.unique(v, return_counts=True)
            share = float(counts.max() / v.size)
            run = int(np.diff(np.concatenate(
                ([-1], np.flatnonzero(np.diff(v) != 0), [len(v) - 1]))).max())
            if run >= min_constant_s:
                flat.append({"channel": tag, "longest_constant_run": run,
                             "share_at_modal_value": share})
            if run > worst["run"]:
                worst = {"tag": tag, "share": share, "baseline": base, "run": run}
        flat.sort(key=lambda f: -f["longest_constant_run"])
        per_attack.append({
            "number": a.number,
            "n_rows": n,
            "points": list(a.points),
            # the channel that goes flattest *relative to its own usual spread*
            "flattest_channel": worst["tag"],
            "is_an_attacked_point": worst["tag"] in a.points if worst["tag"] else None,
            "longest_constant_run": worst["run"],
            "share_at_modal_value": worst["share"],
            "baseline_share_recording_wide": worst["baseline"],
            "flat_channels": flat,
            "degenerate": bool(worst["run"] >= min_constant_s),
        })
    degenerate = [p for p in per_attack if p["degenerate"]]
    total_rows = sum(p["n_rows"] for p in per_attack)
    return {
        "min_constant_s": min_constant_s,
        "baseline_share_threshold": baseline_share,
        "min_distinct_values": min_distinct,
        "n_eligible_channels": sum(1 for b in eligible.values() if b < baseline_share),
        "n_degenerate": len(degenerate),
        "degenerate_attacks": [p["number"] for p in degenerate],
        "degenerate_share_of_attack_rows": (
            sum(p["n_rows"] for p in degenerate) / total_rows if total_rows else float("nan")),
        "per_attack": per_attack,
        "source": "a continuously-valued channel that holds one value for "
                  "min_constant_s or longer inside the attack window, while "
                  "varying across the recording (docs/decisions/0065 §6)",
    }


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


def per_sensor_alarm_events(event_rows, timestamps: pd.Series, gt: GroundTruth,
                            attack_period: np.ndarray,
                            samples_per_day: float) -> dict:
    """Detection and false alarms counted from per-sensor alarm EVENTS instead of the system state."""
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    number = attack_number_per_row(ts, gt)
    rows = sorted(int(r) for r in event_rows)
    period = np.asarray(attack_period, dtype=bool)

    per_attack = []
    for a in gt.attacks:
        inside = [r for r in rows if number[r] == a.number]
        first = inside[0] if inside else None
        per_attack.append({
            "number": a.number,
            "detected": first is not None,
            "n_events": len(inside),
            "detection_delay_s": (float((ts[first] - a.start).total_seconds())
                                  if first is not None else None),
        })

    false_rows = [r for r in rows if not period[r]]
    clean_days = float((~period).sum()) / samples_per_day if samples_per_day else float("nan")
    n_detected = sum(p["detected"] for p in per_attack)
    return {
        "n_attacks": len(per_attack),
        "n_detected": int(n_detected),
        "event_recall": n_detected / len(per_attack) if per_attack else float("nan"),
        "n_events": len(rows),
        "n_false_events": len(false_rows),
        "false_events_per_day": (len(false_rows) / clean_days if clean_days else float("nan")),
        "detection_delay_s": _distribution(
            [p["detection_delay_s"] for p in per_attack if p["detected"]]),
        "per_attack": per_attack,
    }


def localisation(alarm_events: list[dict], timestamps: pd.Series,
                 gt: GroundTruth) -> dict | None:
    """Of the attacks that raised a per-sensor event, how often the named
        channels include an attacked one, and what share of the attacked
        channels were named. Diagnostic; `metrics.top_k_attribution` is the
        measured explainability figure.
    """
    if not alarm_events:
        return None
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    hit = n = num = den = 0
    for a in gt.attacks:
        inside = [set(e["channels"]) for e in alarm_events
                  if a.start <= ts[int(e["row"])] <= a.end]
        if not inside:
            continue
        n += 1
        pts = set(a.points) | set(a.unlogged_points)
        named = set().union(*inside)
        hit += bool(named & pts)
        num += len(named & pts)
        den += len(pts)
    return {"n_attacks_with_an_event": n, "n_naming_an_attacked_channel": hit,
            "hit_rate": (hit / n) if n else None,
            "attacked_channel_recall": (num / den) if den else None}
