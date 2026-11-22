"""Attack ground truth: which samples belong to which attack."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gad.swat import attack_tag_to_column, split_attack_points

NO_PHYSICAL_IMPACT = "No Physical Impact"

DOCUMENTED_DEFECTS: dict[int, str] = {
    26: "the attack sheet's own footnote states the historian stopped updating "
        "the attacked value while it was held constant and interpolated it "
        "afterwards, so the logged data is not what the plant did "
        "(docs/decisions/0065 sec. 3)",
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

    def attacked_points(self) -> tuple[str, ...]:
        """Distinct logged channels attacked at least once."""
        return tuple(sorted({p for a in self.attacks for p in a.points}))

    def unlogged_points(self) -> tuple[str, ...]:
        return tuple(sorted({p for a in self.attacks for p in a.unlogged_points}))


def _as_timestamp(value) -> pd.Timestamp | None:
    if isinstance(value, dt.datetime):
        return pd.Timestamp(value)
    return None


def load_attacks(
    attack_list_path: str | Path,
    historian_columns: set[str],
    recording_start: pd.Timestamp,
    recording_end: pd.Timestamp,
    apply_defect_register: bool = True,
) -> GroundTruth:
    """`apply_defect_register=False` keeps the `DOCUMENTED_DEFECTS` attacks in
        the ground truth.
    """
    import openpyxl

    wb = openpyxl.load_workbook(attack_list_path, read_only=True, data_only=True)
    rows = list(wb[wb.sheetnames[0]].iter_rows(min_row=2, values_only=True))

    attacks: list[Attack] = []
    unscoreable: list[Attack] = []
    excluded: list[tuple[int, str]] = []
    fixes: list[str] = []
    notes: list[str] = []

    for r in rows:
        number, start_cell, end_cell, point_cell = r[0], r[1], r[2], r[3]
        if not isinstance(number, int):
            if isinstance(number, str) and "constant in 26" in number:
                notes.append(
                    "Attack 26: the sheet's own footnote says the historian stopped updating the "
                    "attacked value while it was held constant and interpolated it afterwards — "
                    "logged data for this attack is not what the plant did."
                )
            continue
        if point_cell is None or NO_PHYSICAL_IMPACT in str(point_cell):
            excluded.append((number, "No Physical Impact attack (excluded by the brief)"))
            continue

        start = _as_timestamp(start_cell)
        if start is None:
            raise ValueError(f"attack {number}: unparseable start {start_cell!r}")
        if not (recording_start <= start <= recording_end):
            for year in sorted({recording_start.year, recording_end.year}):
                candidate = start.replace(year=year)
                if recording_start <= candidate <= recording_end:
                    fixes.append(f"attack {number}: start year {start.year} -> {year} "
                                 f"(outside the recording as written)")
                    start = candidate
                    break
            else:
                raise ValueError(f"attack {number}: start {start} outside the recording")

        if isinstance(end_cell, dt.time):
            end = pd.Timestamp(dt.datetime.combine(start.date(), end_cell))
            if end < start:
                end += pd.Timedelta(days=1)
                fixes.append(f"attack {number}: end time rolled to the next day")
        elif isinstance(end_cell, dt.datetime):
            end = pd.Timestamp(end_cell)
        else:
            raise ValueError(f"attack {number}: unparseable end {end_cell!r}")

        raw_points = tuple(split_attack_points(str(point_cell)))
        columns = [attack_tag_to_column(p) for p in raw_points]
        for raw, col in zip(raw_points, columns):
            if raw.replace("-", "").upper() != col:
                fixes.append(f"attack {number}: tag {raw!r} -> column {col!r}")
        logged = tuple(dict.fromkeys(c for c in columns if c in historian_columns))
        unlogged = tuple(dict.fromkeys(c for c in columns if c not in historian_columns))

        attack = Attack(
            number=number,
            start=start,
            end=end,
            raw_points=raw_points,
            points=logged,
            unlogged_points=unlogged,
            description=str(r[5] or "").strip(),
        )
        if apply_defect_register and number in DOCUMENTED_DEFECTS:
            excluded.append((number, DOCUMENTED_DEFECTS[number]))
            unscoreable.append(attack)
            continue
        attacks.append(attack)

    return GroundTruth(
        attacks=tuple(attacks),
        excluded=tuple(excluded),
        fixes=tuple(fixes),
        notes=tuple(notes),
        unscoreable=tuple(unscoreable),
    )


def attack_period_mask(timestamps: pd.Series, labels: pd.Series, gt: GroundTruth) -> np.ndarray:
    """True wherever the plant is under attack by either account: inside a
        listed attack window, or labelled Attack in the data. Neither may count as
        clean operation.
    """
    return (attack_number_per_row(timestamps, gt) > 0) | (labels.to_numpy() == "Attack")


def unscoreable_mask(timestamps: pd.Series, gt: GroundTruth) -> np.ndarray:
    """True on rows inside an attack the defect register removed."""
    ts = pd.to_datetime(timestamps).to_numpy()
    out = np.zeros(len(ts), dtype=bool)
    for a in gt.unscoreable:
        out |= (ts >= a.start.to_datetime64()) & (ts <= a.end.to_datetime64())
    return out


def unlisted_attack_runs(timestamps: pd.Series, labels: pd.Series, gt: GroundTruth) -> list[dict]:
    """Runs labelled Attack that fall outside every listed window."""
    stray = np.flatnonzero((labels.to_numpy() == "Attack") & (attack_number_per_row(timestamps, gt) == 0))
    if len(stray) == 0:
        return []
    ts = pd.to_datetime(timestamps).to_numpy()
    runs = np.split(stray, np.flatnonzero(np.diff(stray) > 1) + 1)
    return [{"start": str(pd.Timestamp(ts[r[0]])), "end": str(pd.Timestamp(ts[r[-1]])),
             "n_rows": int(len(r))} for r in runs]


def attack_number_per_row(timestamps: pd.Series, gt: GroundTruth) -> np.ndarray:
    """For each timestamp, the number of the attack whose window contains
        it, or 0. Windows are inclusive at both ends.
    """
    ts = pd.to_datetime(timestamps).to_numpy()
    out = np.zeros(len(ts), dtype=np.int64)
    for a in gt.attacks:
        inside = (ts >= a.start.to_datetime64()) & (ts <= a.end.to_datetime64())
        out[inside & (out == 0)] = a.number
    return out
