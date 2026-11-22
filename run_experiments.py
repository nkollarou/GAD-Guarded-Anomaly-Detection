"""Run the five detectors on the 100 datasets and print the table of the paper.

    python run_experiments.py

Each detector is commissioned once on the same clean data and then reads every
dataset. Results go to results/<detector>/s<seed>/, and the table shows the
median over the five attack orders.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.diagnostics import false_alarm_timing
from evaluation.ground_truth import Attack, GroundTruth
from evaluation.metrics import report
from gad.settings import GAD, SPLIT, UNGUARDED
from gad.types import Sample

DATA = Path("data/datasets")
RESULTS = Path("results")
NORMAL_PARQUET = Path("data/swat_normal.parquet")
DETECTORS = ("gad", "gad_unguarded", "adaptive_pca", "memstream", "rls")
SEEDS = (52, 53, 54, 55, 56)
SENSORS = ("DPIT301", "FIT401", "LIT101", "AIT503")
CONDITIONS = ("clean", "G_r0.25_m1", "G_r1_m1", "G_r4_m1", "G_r16_m1")

HOUR = 3600
SAMPLES_PER_DAY = 86_400.0
POISON_START, POISON_END = 1 * HOUR, 25 * HOUR
ATTACK_START = int(25.5 * HOUR)
MS_DWELL, MS_RECOVER = 5, 30
SEED = 52


def split_rows(n: int) -> tuple[int, int, int]:
    """(fit_end, warm_end, cal_end), as in `gad.commission.commission`."""
    fit_end = int(n * SPLIT[0])
    n_held = n - fit_end
    warm = int(n_held * SPLIT[1])
    cal = warm + int(n_held * SPLIT[2])
    return fit_end, fit_end + warm, fit_end + cal


def commission(detector: str, normal: Path):
    """Commission one detector on the clean recording. Returns what streaming
        needs: the detector, GAD's L0 checker, the channel lists and the row at
        which the test stream starts.
    """
    import gad.commission as gc
    from gad.swat import find_commissioning_start, split_channels
    gc.NORMAL_PARQUET = normal
    df = pd.read_parquet(normal)
    sensed, commanded = split_channels(list(df.columns))
    steady = df.iloc[find_commissioning_start(df):].reset_index(drop=True)
    fit_end, warm_end, cal_end = split_rows(len(steady))
    checker0 = None
    if detector in ("gad", "gad_unguarded"):
        c = gc.commission(with_evidence=True, split=SPLIT,
                          cfg_overrides=GAD if detector == "gad" else UNGUARDED)
        base, checker0 = c.detector, gc.l0_checker(c.steady, sensed, **c.l0_kwargs)
        assert c.outer_train_end == fit_end and c.outer_train_end + c.calib_end == cal_end
    elif detector == "memstream":
        import torch
        from baselines.memstream import MEMORY_LEN_DAY, MemStreamMonitor
        torch.set_num_threads(1)
        filled, _ = gc.fill_gaps(steady, list(sensed) + list(commanded))
        X = filled[sensed].to_numpy(dtype=float)
        base = MemStreamMonitor(sensed, memory_len=MEMORY_LEN_DAY, seed=SEED)
        base.fit(X[:fit_end])
        base.stream(X[fit_end:warm_end])
        base.calibrate(X[warm_end:cal_end], budget_time_fraction=gc.BUDGET_TIME_FRACTION,
                       samples_per_day=SAMPLES_PER_DAY, dwell=MS_DWELL, recover=MS_RECOVER,
                       beta_rule="clean_max")
    elif detector == "rls":
        from baselines.padasip_rls import PadasipRLSMonitor
        filled, _ = gc.fill_gaps(steady, list(sensed) + list(commanded))
        Xc = filled[sensed].to_numpy(dtype=float)
        Uc = filled[commanded].to_numpy(dtype=float)
        base = PadasipRLSMonitor(sensed, commanded, dwell=MS_DWELL, recover=MS_RECOVER)
        base.commission(Xc[:fit_end], Uc[:fit_end], Xc[fit_end:warm_end], Uc[fit_end:warm_end],
                        Xc[warm_end:cal_end], Uc[warm_end:cal_end],
                        budget_time_fraction=gc.BUDGET_TIME_FRACTION,
                        samples_per_day=SAMPLES_PER_DAY)
    elif detector == "adaptive_pca":
        from baselines.adaptive_pca import AdaptivePCAMonitor
        filled, _ = gc.fill_gaps(steady, list(sensed) + list(commanded))
        X = filled[sensed].to_numpy(dtype=float)
        base = AdaptivePCAMonitor(sensed, speed="matched", update_when_out_of_control=False)
        base.fit(X[:fit_end])
        base.stream(X[fit_end:warm_end])
        base.calibrate(X[warm_end:cal_end], gc.BUDGET_EPISODES_PER_DAY, gc.BUDGET_TIME_FRACTION)
    else:
        raise ValueError(f"unknown detector {detector!r}")
    return base, checker0, list(sensed), list(commanded), cal_end


def stream(detector: str, base, checker0, X: np.ndarray, U: np.ndarray, cal_end: int) -> np.ndarray:
    """The alarm state ("NORMAL", "ANOMALY" or "DEGRADED") at every sample of one dataset."""
    from gad.types import DetectorState
    if detector in ("gad", "gad_unguarded"):
        det, checker = copy.deepcopy(base), copy.deepcopy(checker0)
        det.start_stream()  # a dataset is a new time series
        states = np.empty(len(X), dtype=object)
        for i in range(len(X)):
            q = checker.update(X[i])
            states[i] = det.update(Sample(t=float(cal_end + i), x=X[i], u=U[i], q=q)).state.value
        return states
    if detector == "memstream":
        from baselines.common import states_from
        det = copy.deepcopy(base)
        return states_from(det.stream(X), det.threshold,
                           dwell=base.calibration["dwell"], recover=base.calibration["recover"])
    if detector == "rls":
        det = copy.deepcopy(base)
        E = det.stream(X, U)
        return np.where(det.decide(E), DetectorState.ANOMALY.value,
                        DetectorState.NORMAL.value).astype(object)
    mon = copy.deepcopy(base)
    states = np.empty(len(X), dtype=object)
    for i in range(len(X)):
        states[i] = mon.step(X[i]).value
    return states


def ground_truth(target: str, ts: pd.Series, data_dir: Path) -> GroundTruth:
    rows = [a for a in json.loads((data_dir / "attacks.json").read_text())
            if a["target"] == target]
    atk = tuple(Attack(a["number"], pd.Timestamp(a["start"]), pd.Timestamp(a["end"]), (target,), (target,), (),
                       f"step {a['direction'] * a['size']:+g}") for a in rows)
    return GroundTruth(atk, (), (), ("synthetic, docs/decisions/0052",)), rows


def score(states: np.ndarray, data: pd.DataFrame, target: str, poison_end: int | None = None,
          data_dir: Path | None = None, poison_start: int | None = None) -> dict:
    """harness/metrics.py on the attack phase; poisoning-phase alarms apart."""
    gt, rows = ground_truth(target, data["Timestamp"], data_dir)
    phase = slice(ATTACK_START, None)
    ts = data["Timestamp"].iloc[phase].reset_index(drop=True)
    period = data["label"].to_numpy()[phase] > 0
    rep = report(states[phase], ts, gt, period, SAMPLES_PER_DAY)
    by_size = {}
    for a, p in zip(rows, rep["detection"]["per_attack"]):
        s = by_size.setdefault(a["size"], {"detected": 0, "n": 0})
        s["n"] += 1
        s["detected"] += int(p["detected"])
    anomaly = states == "ANOMALY"
    stop = POISON_END if poison_end is None else int(poison_end)
    begin = POISON_START if poison_start is None else int(poison_start)
    poison = anomaly[begin:stop]
    between = anomaly[phase][~period]
    return {"point_wise": rep["point_wise"], "range_based": rep["range_based"],
            "composite": rep["composite"], "detection": rep["detection"],
            "false_alarms": rep["false_alarms"], "abstention": rep["abstention"],
            "recall_by_size": by_size,
            "false_alarm_timing": false_alarm_timing(states[phase], ts, gt, period),
            "attacks_caught": {"detected": sum(s["detected"] for s in by_size.values()),
                               "n": len(rows)},
            "alarm_between_steps": float(between.mean()),
            "poisoning_noticed": {"alarm_episodes": int(np.sum(np.diff(np.concatenate(([0], poison.astype(np.int8)))) == 1)),
                                  "time_fraction": float(poison.mean()),
                                  "window_rows": [begin, stop]}}


def run(detector: str, data: Path, results: Path, normal: Path) -> None:
    t0 = time.perf_counter()
    base, checker0, sensed, commanded, cal_end = commission(detector, normal)
    print(f"[{detector}] commissioned in {time.perf_counter() - t0:.0f} s", flush=True)
    for seed in SEEDS:
        data_dir = data / f"s{seed}"
        manifest = json.loads((data_dir / "manifest.json").read_text())
        out_dir = results / detector / f"s{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for target in SENSORS:
            for cond in CONDITIONS:
                name = f"{target}_{cond}"
                frame = pd.read_parquet(data_dir / f"{name}.parquet")
                X = frame[sensed].to_numpy(dtype=float)
                U = frame[commanded].to_numpy(dtype=float)
                states = stream(detector, base, checker0, X, U, cal_end)
                meta = manifest["datasets"][f"{name}.parquet"]
                begin = int(meta.get("poison_start_row", POISON_START))
                res = score(states, frame, target, data_dir=data_dir, poison_start=begin,
                            poison_end=None if cond == "clean" else begin + 24 * HOUR)
                res.update({"detector": detector, "seed": seed, "target": target,
                            "condition": cond})
                (out_dir / f"{name}.json").write_text(json.dumps(res, indent=2, default=float))
                print(f"[{detector}] s{seed} {name}: "
                      f"{res['attacks_caught']['detected']}/{res['attacks_caught']['n']} attacks",
                      flush=True)


def table(results: Path, detectors=DETECTORS) -> None:
    """Median over the attack orders, as in the paper's table."""
    def cell(det, target, cond):
        rs = [json.loads(p.read_text()) for p in
              sorted((results / det).glob(f"s*/{target}_{cond}.json"))]
        if not rs:
            return None
        med = lambda f: float(np.median([f(r) for r in rs]))
        return {"A": med(lambda r: r["attacks_caught"]["detected"]),
                "F": med(lambda r: r["false_alarms"]["episodes_per_day"]),
                "F1": med(lambda r: r["point_wise"]["f1"]),
                "P": med(lambda r: 100.0 * r["poisoning_noticed"]["time_fraction"]),
                "n": len(rs)}
    print("sensor   detector        | no poisoning: A    F    F1 | "
          + " | ".join(f"r={c.split('_r')[1].split('_')[0]}: A    P" for c in CONDITIONS[1:]))
    for target in SENSORS:
        for det in detectors:
            row = [f"{target:8s} {det:15s} |"]
            c0 = cell(det, target, "clean")
            row.append("     --" if c0 is None else
                       f"            {c0['A']:3.0f} {c0['F']:4.1f} {c0['F1']:5.2f} |")
            for cond in CONDITIONS[1:]:
                c = cell(det, target, cond)
                row.append("   --" if c is None else f"       {c['A']:3.0f} {c['P']:4.0f}")
            print(" ".join(row))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--detectors", nargs="+", choices=DETECTORS, default=list(DETECTORS))
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--normal", type=Path, default=NORMAL_PARQUET,
                    help="the converted normal-operation file (build_datasets.py --convert)")
    ap.add_argument("--table", action="store_true", help="only print the table")
    a = ap.parse_args()
    if not a.table:
        for d in a.detectors:
            run(d, a.data, a.results, a.normal)
    table(a.results, a.detectors)


if __name__ == "__main__":
    main()
