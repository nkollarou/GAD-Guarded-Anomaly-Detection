"""Event-wise metrics: attacks detected, false alarms per day, point-wise
precision, recall and F1, and detection delay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.ground_truth import GroundTruth, attack_number_per_row

RANGE_PARAMS = {"alpha_recall": 0.0, "cardinality": "reciprocal", "bias": "flat"}


# --- ranges ---------------------------------------------------------------

def _ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of `mask` as inclusive [start, end] index pairs."""
    m = np.asarray(mask, dtype=bool).astype(np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def _size(r: tuple[int, int]) -> int:
    return r[1] - r[0] + 1


def _existence(r: tuple[int, int], others: list[tuple[int, int]]) -> float:
    return 1.0 if any(_overlap(r, o) for o in others) else 0.0


def _overlap_reward(r: tuple[int, int], others: list[tuple[int, int]],
                    cardinality: str) -> float:
    """Tatbul et al. 2018 eq. for OverlapReward, with a flat positional bias."""
    hits = [o for o in others if _overlap(r, o)]
    if not hits:
        return 0.0
    overlapped = sum(_overlap(r, o) for o in hits) / _size(r)
    if cardinality == "one" or len(hits) <= 1:
        factor = 1.0
    elif cardinality == "reciprocal":
        factor = 1.0 / len(hits)
    else:
        raise ValueError(f"unknown cardinality rule {cardinality!r}")
    return factor * overlapped


def range_based(alarm: np.ndarray, label: np.ndarray, *,
                alpha_recall: float = RANGE_PARAMS["alpha_recall"],
                cardinality: str = RANGE_PARAMS["cardinality"],
                bias: str = RANGE_PARAMS["bias"]) -> dict:
    """Range-based precision / recall / F1 (Tatbul et al., NeurIPS 2018)."""
    if bias != "flat":
        raise ValueError("only the flat positional bias is implemented "
                         "(docs/decisions/0062); asking for another is a "
                         "research decision, not a parameter")
    real = _ranges(label)
    pred = _ranges(alarm)
    if real:
        recall = float(np.mean([
            alpha_recall * _existence(r, pred)
            + (1.0 - alpha_recall) * _overlap_reward(r, pred, cardinality)
            for r in real]))
    else:
        recall = float("nan")
    if pred:
        precision = float(np.mean([_overlap_reward(p, real, cardinality) for p in pred]))
    else:
        precision = float("nan")
    return {"precision": precision, "recall": recall, "f1": f1_score(precision, recall),
            "n_real_ranges": len(real), "n_predicted_ranges": len(pred),
            "params": {"alpha_recall": alpha_recall, "cardinality": cardinality, "bias": bias},
            "source": "Tatbul et al., Precision and Recall for Time Series, NeurIPS 2018"}


# --- point-wise -----------------------------------------------------------

def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean, NaN-safe. The only arithmetic this module shares."""
    if not (np.isfinite(precision) and np.isfinite(recall)):
        return float("nan")
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def point_wise(alarm: np.ndarray, label: np.ndarray) -> dict:
    """The classical per-sample confusion matrix. ANOMALY is positive;
        DEGRADED is not an alarm and is reported as abstention instead.
    """
    alarm = np.asarray(alarm, dtype=bool)
    label = np.asarray(label, dtype=bool)
    tp = int(np.sum(alarm & label))
    fp = int(np.sum(alarm & ~label))
    fn = int(np.sum(~alarm & label))
    tn = int(np.sum(~alarm & ~label))
    if tp + fp:
        precision = tp / (tp + fp)
    elif fn:
        precision = 0.0          # nothing raised, and there was something to find
    else:
        precision = float("nan")  # nothing raised and nothing to find
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1_score(precision, recall),
            "source": "classical per-sample confusion matrix"}


def composite(alarm: np.ndarray, label: np.ndarray, per_attack: list[dict]) -> dict:
    """Composite F1 (Garg et al., IEEE TNNLS 2021): event-wise recall x
        point-wise precision. `per_attack` supplies the event side, so there is
        exactly one definition of "this attack was detected" in the report.
    """
    n = len(per_attack)
    hit = sum(bool(p["detected"]) for p in per_attack)
    event_recall = hit / n if n else float("nan")
    precision = point_wise(alarm, label)["precision"]
    return {"n_attacks": n, "n_detected": hit, "event_recall": event_recall,
            "point_wise_precision": precision,
            "f1": f1_score(precision, event_recall),
            "source": "Garg et al., IEEE TNNLS 2021 (composite F1)"}


def auc_pr(scores: np.ndarray | None, label: np.ndarray) -> dict:
    """Average precision — the area under the precision-recall curve."""
    if scores is None:
        return {"average_precision": None,
                "reason": "the detector emits no continuous score; its output is "
                          "quantised by design, so there is no ranking to integrate",
                "source": "sklearn.metrics.average_precision_score"}
    from sklearn.metrics import average_precision_score
    label = np.asarray(label, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if label.all() or not label.any():
        return {"average_precision": None, "reason": "one class only",
                "source": "sklearn.metrics.average_precision_score"}
    return {"average_precision": float(average_precision_score(label, scores)),
            "source": "sklearn.metrics.average_precision_score"}


# --- per-event detection and delay ----------------------------------------

def detection(alarm: np.ndarray, timestamps: pd.Series, gt: GroundTruth) -> dict:
    """Per-attack detection and detection delay (time to detect)."""
    alarm = np.asarray(alarm, dtype=bool)
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    number = attack_number_per_row(ts, gt)
    per_attack = []
    for a in gt.attacks:
        inside = np.flatnonzero((number == a.number) & alarm)
        rows = np.flatnonzero(number == a.number)
        first = int(inside[0]) if len(inside) else None
        per_attack.append({
            "number": a.number,
            "points": list(a.points),
            "unlogged_points": list(a.unlogged_points),
            "duration_s": float((a.end - a.start).total_seconds()),
            "detected": first is not None,
            "detect_row": first,
            "detection_delay_s": (float((ts[first] - a.start).total_seconds())
                                  if first is not None else None),
            "alarm_active_at_start": bool(len(rows) and alarm[rows[0]]),
            "fraction_alarmed": float(alarm[rows].mean()) if len(rows) else 0.0,
        })
    delays = [p["detection_delay_s"] for p in per_attack if p["detected"]]
    n_detected = sum(p["detected"] for p in per_attack)
    return {
        "n_attacks": len(per_attack),
        "n_detected": int(n_detected),
        "event_recall": n_detected / len(per_attack) if per_attack else float("nan"),
        "n_detected_with_alarm_already_active": sum(
            p["detected"] and p["alarm_active_at_start"] for p in per_attack),
        "detection_delay_s": _distribution(delays),
        "per_attack": per_attack,
        "source": "any-overlap event detection; delay = first alarmed row - attack start",
    }


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "median": None, "p90": None, "max": None, "min": None}
    v = np.asarray(values, dtype=float)
    return {"n": int(v.size), "median": float(np.median(v)),
            "p90": float(np.quantile(v, 0.9)), "max": float(v.max()),
            "min": float(v.min())}


# --- false alarms and abstention -----------------------------------------

def false_alarms(alarm: np.ndarray, label: np.ndarray, samples_per_day: float) -> dict:
    """The attack-free portion, counted rather than scored."""
    alarm = np.asarray(alarm, dtype=bool)
    clean = ~np.asarray(label, dtype=bool)
    n_clean = int(clean.sum())
    starts = [s for s, _ in _ranges(alarm)]
    false_starts = [s for s in starts if clean[s]]
    clean_days = n_clean / samples_per_day if samples_per_day else float("nan")
    return {
        "clean_rows": n_clean,
        "clean_days": float(clean_days),
        "false_alarm_rows": int(np.sum(alarm & clean)),
        "fpr": float(np.mean(alarm[clean])) if n_clean else float("nan"),
        "n_alarm_episodes": len(starts),
        "n_false_alarm_episodes": len(false_starts),
        "episodes_per_day": (len(false_starts) / clean_days
                             if clean_days else float("nan")),
        "source": "false positive rate; alarm episodes per day of attack-free data",
    }


def abstention(abstained: np.ndarray, label: np.ndarray) -> dict:
    """Time spent in DEGRADED, overall and on the attack-free portion."""
    abstained = np.asarray(abstained, dtype=bool)
    clean = ~np.asarray(label, dtype=bool)
    return {"rate": float(abstained.mean()) if abstained.size else float("nan"),
            "rate_clean_portion": float(abstained[clean].mean()) if clean.any() else float("nan"),
            "source": "fraction of samples on which the detector abstained"}


# --- the one report -------------------------------------------------------

def report(states: np.ndarray, timestamps: pd.Series, gt: GroundTruth,
           attack_period: np.ndarray, samples_per_day: float,
           scores: np.ndarray | None = None) -> dict:
    """Every metric above, returned together."""
    states = np.asarray(states, dtype=object)
    alarm = states == "ANOMALY"
    degraded = states == "DEGRADED"
    label = np.asarray(attack_period, dtype=bool)
    det = detection(alarm, timestamps, gt)
    return {
        "point_wise": point_wise(alarm, label),
        "range_based": range_based(alarm, label),
        "composite": composite(alarm, label, det["per_attack"]),
        "auc_pr": auc_pr(scores, label),
        "detection": det,
        "false_alarms": false_alarms(alarm, label, samples_per_day),
        "abstention": abstention(degraded, label),
        "n_rows": int(states.size),
    }


# --- explainability -------------------------------------------------------

def top_k_attribution(
    records_by_row: dict[int, tuple[str, ...]],
    per_attack: list[dict],
    gt: GroundTruth,
    sensed: set[str],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """Top-k accuracy over the ground truth's attacked channels."""
    by_number = {a.number: a for a in gt.attacks}
    excluded = [p["number"] for p in per_attack if not by_number[p["number"]].points]
    scored = [p for p in per_attack if by_number[p["number"]].points]

    def tags_for(p: dict) -> tuple[str, ...]:
        return records_by_row.get(p["detect_row"], ()) if p["detected"] else ()

    out: dict = {"excluded_attacks_no_logged_point": excluded, "k": {},
                 "source": "top-k accuracy against the ground truth's attacked channels"}
    points = sorted({pt for p in scored for pt in by_number[p["number"]].points})
    for k in ks:
        hit_all, hit_det, n_det, hit_sens, n_sens = 0, 0, 0, 0, 0
        point_hit = {pt: False for pt in points}
        for p in scored:
            a = by_number[p["number"]]
            top = set(tags_for(p)[:k])
            hit = bool(top & set(a.points))
            hit_all += hit
            if p["detected"]:
                n_det += 1
                hit_det += hit
            sensed_points = set(a.points) & sensed
            if sensed_points:
                n_sens += 1
                hit_sens += bool(top & sensed_points)
            for pt in a.points:
                point_hit[pt] |= pt in top
        out["k"][k] = {
            "all_attacks": {"hits": hit_all, "denominator": len(scored)},
            "detected": {"hits": hit_det, "denominator": n_det},
            "sensor_points_only": {"hits": hit_sens, "denominator": n_sens},
            "per_point": {"hits": sum(point_hit.values()), "denominator": len(points),
                          "missed": sorted(pt for pt, h in point_hit.items() if not h)},
        }
    return out
