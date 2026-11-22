"""Build the 100 poisoning datasets of the paper from SWaT's normal operation.

    python build_datasets.py --convert --raw SWaT_Dataset_Normal_v1.xlsx

For each of four sensors and five random attack orders it writes one dataset
without poisoning and one for each poisoning rate, to data/datasets/s<seed>/.
Add --check DIR to compare the files with an earlier build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from gad.swat import clean_raw_frame, find_commissioning_start, split_channels
from gad.settings import GAD, SPLIT

HOUR = 3600
#: poisoning from hour 1 to hour 25 of the test stream, stepped at each update
POISON_START, POISON_END = 1 * HOUR, 25 * HOUR
UPDATE_PERIOD_S = 6 * HOUR
#: the 21 attacks: 10 minutes each, 30 minutes apart, from hour 25.5
ATTACK_START = int(25.5 * HOUR)
ATTACK_S, GAP_S, REPEATS = 600, 1800, 3
#: attack sizes: 7 steps from 1 sd of the clean readings to 0.44 x |mean|
LADDER_STEPS, LADDER_LO_SD, LADDER_HI_MEAN = 7, 1.0, 0.44
TARGETS = {"DPIT301": -1.0, "FIT401": -1.0, "LIT101": 1.0, "AIT503": 1.0}
RATES = (0.25, 1.0, 4.0, 16.0)
SEEDS = (52, 53, 54, 55, 56)

NORMAL_PARQUET = Path("data/swat_normal.parquet")

S_REF = {key: GAD[key] for key in (
    "learn_guard", "l2_band", "l2_clip", "l2_half_life", "l1a_cross", "l1a_cross_rule",
    "use_l1a_long_calibration", "cross_channel_warm_calibration", "l2_kind",
    "unseen_context_pooled_init", "cusum_cap_factor", "full_budget_split",
    "per_sensor_alarms", "adaptive_h", "unified_l5", "sensitivity_budget",
    "trusted_learning", "per_channel_learning")} | {"l2_cap": 4.0, "adaptive_h_cap": 4.0}
OUT = Path("data/datasets")


def split_rows(n: int) -> tuple[int, int, int]:
    """(fit_end, warm_end, cal_end) as rows of the steady window, the same
        arithmetic as `gad.commission.commission(split=SPLIT)`.
    """
    fit_end = int(n * SPLIT[0])
    n_held = n - fit_end
    warm = int(n_held * SPLIT[1])
    cal = warm + int(n_held * SPLIT[2])
    return fit_end, fit_end + warm, fit_end + cal


def load_steady(parquet: Path):
    """The clean recording from the end of the start-up fill, and its sensed
        and commanded channels.
    """
    df = pd.read_parquet(parquet)
    sensed, commanded = split_channels(list(df.columns))
    steady = df.iloc[find_commissioning_start(df):].reset_index(drop=True)
    return steady, sensed, commanded


def load_xlsx(path: Path) -> pd.DataFrame:
    """Read a SWaT xlsx export: row 1 is a process-group header, row 2 is
        the real header (both are stripped of the leading-whitespace quirk by
        `clean_raw_frame`, called by the caller on the returned frame).
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    header_row = next(ws.iter_rows(min_row=2, max_row=2, values_only=True))
    columns = [str(h) for h in header_row]
    data = list(ws.iter_rows(min_row=3, values_only=True))
    wb.close()
    return pd.DataFrame.from_records(data, columns=columns)


def measured_ladder(x: np.ndarray, steps: int = LADDER_STEPS,
                    lo_k: float = LADDER_LO_SD, hi_k: float = LADDER_HI_MEAN,
                    within_sd: float | None = None) -> dict:
    """The ladder for one channel from its clean fit rows, and its span."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        raise ValueError("not enough clean samples for a ladder")
    sd, mean = float(v.std()), float(abs(v.mean()))
    floor = sd if within_sd is None else float(within_sd)
    lo, hi = lo_k * floor, hi_k * mean
    out = {"sd": sd, "mean": mean, "lo": lo, "hi": hi,
           "floor_sd": floor, "floor_basis": ("sd" if within_sd is None
                                              else "within_mode"),
           "within_mode_sd": (None if within_sd is None else float(within_sd)),
           "headroom": (hi / lo) if lo > 0 else float("inf"),
           "steps": steps, "lo_k": lo_k, "hi_k": hi_k}
    if lo <= 0 or hi <= lo:
        out["sizes"], out["viable"] = (), False
        return out
    out["sizes"] = tuple(float(x) for x in np.geomspace(lo, hi, steps))
    out["viable"] = True
    return out


def schedule(rng: np.random.Generator, n_sizes: int = 7,
             repeats: int = REPEATS, n_rows: int | None = None
             ) -> list[tuple[int, int]]:
    """(size index, start row in the test stream), one entry per attack."""
    if n_sizes < 1 or repeats < 1:
        raise ValueError(f"n_sizes and repeats must be >= 1, got {n_sizes}, {repeats}")
    n = n_sizes * repeats
    if n_rows is not None:
        slots = (int(n_rows) - ATTACK_S - ATTACK_START) // (ATTACK_S + GAP_S) + 1
        if n > slots:
            raise SystemExit(
                f"{n_sizes} rungs x {repeats} repeats = {n} attacks, but the "
                f"{int(n_rows):,}-row stream holds {slots}. Lower --steps, or "
                f"use fewer repeats: {slots // repeats} rungs fit at "
                f"{repeats} repeats, {slots // 2} at 2, {slots} at 1.")
    order = rng.permutation(np.repeat(np.arange(n_sizes), repeats))
    return [(int(k), ATTACK_START + i * (ATTACK_S + GAP_S)) for i, k in enumerate(order)]


def _stepwise(n: int, start: int, stop: int, period: int) -> np.ndarray:
    """Step count k at each sample: 0 before `start`, then one more per
        `period` until `stop`, held afterwards (the lie stays in place).
    """
    k = np.zeros(n)
    idx = np.arange(n)
    active = (idx >= start)
    k[active] = np.floor((idx[active] - start) / period) + 1
    k[idx >= stop] = np.floor((stop - start) / period)
    return k


def campaign_window(start_h: float, duration_h: float) -> tuple[int, int]:
    """(start row, stop row) of a poisoning campaign, refused if it runs into
        the payload. The 21 attacks start at `ATTACK_START` and a campaign that
        overlapped them would poison the detector with the attacks it is being
        scored on.
    """
    start, stop = int(start_h * HOUR), int(start_h * HOUR + duration_h * HOUR)
    if stop > ATTACK_START:
        raise ValueError(
            f"a {duration_h} h campaign from {start_h} h ends at {stop / HOUR:.1f} h, "
            f"after the payload starts at {ATTACK_START / HOUR:.1f} h")
    return start, stop


def reference_scales(sensed: list[str], parquet: Path) -> dict[str, float]:
    """Each sensor's s_ref: the median over its operating contexts of the
        residual standard deviation the commissioned detector starts from.
    """
    import gad.commission as gc
    gc.NORMAL_PARQUET = parquet
    c = gc.commission(cfg_overrides=S_REF, split=SPLIT)
    refs = c.detector.l2a._ref  # commissioning references, per channel and context
    out = {}
    for j, ch in enumerate(sensed):
        scales = [sc for _, sc in (refs[j] or {}).values()] if refs else []
        if scales:
            out[ch] = float(np.median(scales))
    return out


def build_seed(seed: int, steady, sensed, commanded, s_ref: dict, out_root: Path,
               source: Path) -> Path:
    """One attack order: for every target, the clean dataset and one per rate."""
    start, stop = campaign_window(1, 24)
    fit_end, _, cal_end = split_rows(len(steady))
    test = steady.iloc[cal_end:].reset_index(drop=True)
    n = len(test)
    plan = schedule(np.random.default_rng(seed), n_sizes=LADDER_STEPS, n_rows=n)
    steps = _stepwise(n, start, stop, UPDATE_PERIOD_S)
    n_events = int(steps.max())
    rate_gain = int((24 * HOUR) // UPDATE_PERIOD_S) / n_events
    out_dir = out_root / f"s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = test[["Timestamp"] + sensed + commanded].copy()
    ladders = {t: (sign, measured_ladder(steady[t].to_numpy(dtype=float)[:fit_end])["sizes"])
               for t, sign in TARGETS.items()}
    manifest = {"seed": seed, "source": str(source), "attack_order_seed": seed,
                "ladder_lo_sd": LADDER_LO_SD, "ladder_hi_mean": LADDER_HI_MEAN,
                "attack_mode": "additive", "rates": list(RATES),
                "campaign_start_h": 1, "campaign_duration_h": 24,
                "n_update_events": n_events,
                "test_rows_of_steady_window": [int(cal_end), int(len(steady))],
                "s_ref": {t: s_ref[t] for t in TARGETS},
                "ladders": {t: {"direction": sign, "sizes": list(sizes)}
                            for t, (sign, sizes) in ladders.items()},
                "rates_over_cap": {}, "files": {}, "datasets": {}}
    attacks = []
    for target, (sign, sizes) in ladders.items():
        cap = sizes[-1]
        over = [{"rate": float(r), "installs": float(abs(n_events * r * rate_gain * s_ref[target])),
                 "cap": float(cap)} for r in RATES
                if abs(n_events * r * rate_gain * s_ref[target]) > cap]
        if over:
            manifest["rates_over_cap"][target] = over
        label = np.zeros(n, dtype=np.int64)
        step = np.zeros(n)
        for number, (k, at) in enumerate(plan, start=1):
            label[at:at + ATTACK_S] = number
            step[at:at + ATTACK_S] = sign * sizes[k]
            attacks.append({"target": target, "number": number, "size": sizes[k],
                            "direction": sign, "start_row": at,
                            "end_row": at + ATTACK_S - 1,
                            "start": str(test["Timestamp"].iloc[at]),
                            "end": str(test["Timestamp"].iloc[at + ATTACK_S - 1])})
        sr = s_ref[target]
        conditions = [("clean", np.zeros(n), {})] + [
            (f"G_r{r:g}_m1", sign * steps * r * rate_gain * sr,
             {"rate": r, "final_bias": float(sign * n_events * r * rate_gain * sr)})
            for r in RATES]
        for cond, bias, meta in conditions:
            out = base.copy()
            out[target] = test[target].to_numpy(dtype=float) + bias + step
            out["label"] = label
            path = out_dir / f"{target}_{cond}.parquet"
            out.to_parquet(path, index=False)
            manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest["datasets"][path.name] = {
                "target": target, "condition": cond,
                "poisoned": [] if cond == "clean" else [target],
                "duration_h": 24, "poison_start_row": start,
                "smallest_attack": sizes[0], **meta}
    (out_dir / "attacks.json").write_text(json.dumps(attacks, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir


def check(out_root: Path, old_root: Path) -> bool:
    """Compare every file built with an earlier build's manifests."""
    ok = True
    for seed in SEEDS:
        new = json.loads((out_root / f"s{seed}" / "manifest.json").read_text())["files"]
        old_path = old_root / f"s{seed}" / "manifest.json"
        if not old_path.exists():
            print(f"s{seed}: no manifest at {old_path}")
            ok = False
            continue
        old = json.loads(old_path.read_text())["files"]
        same = [f for f in new if old.get(f) == new[f]]
        diff = [f for f in new if f in old and old[f] != new[f]]
        missing = [f for f in new if f not in old]
        print(f"s{seed}: {len(same)} of {len(new)} files identical"
              + (f", DIFFERENT: {', '.join(diff)}" if diff else "")
              + (f", not in the old build: {', '.join(missing)}" if missing else ""))
        ok &= not diff and not missing
    print("all files identical" if ok else "the builds differ")
    return ok


def convert(raw: Path, out: Path) -> None:
    """SWaT's normal-operation xlsx export to a Parquet file."""
    df = clean_raw_frame(load_xlsx(raw))
    if not df["Timestamp"].is_monotonic_increasing:
        raise SystemExit(f"{raw}: timestamps are not sorted")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df):,} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--convert", action="store_true",
                    help="convert the SWaT normal-operation xlsx first")
    ap.add_argument("--raw", type=Path, default=Path("SWaT_Dataset_Normal_v1.xlsx"))
    ap.add_argument("--normal", type=Path, default=NORMAL_PARQUET,
                    help="the converted normal-operation file")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", type=Path, default=None, metavar="DIR",
                    help="compare the files with an earlier build's manifests")
    a = ap.parse_args()
    if a.convert:
        convert(a.raw, a.normal)
    steady, sensed, commanded = load_steady(a.normal)
    print("commissioning the detector to measure s_ref ...", flush=True)
    s_ref = reference_scales(sensed, a.normal)
    for seed in SEEDS:
        d = build_seed(seed, steady, sensed, commanded, s_ref, a.out, a.normal)
        print(f"built {d}", flush=True)
    if a.check is not None:
        sys.exit(0 if check(a.out, a.check) else 1)


if __name__ == "__main__":
    main()
