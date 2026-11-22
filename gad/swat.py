"""Everything specific to SWaT: which channels are sensors and actuators, the
plant's stages, tanks and flows, and how to read the raw export.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import pandas as pd

from gad.l2_context import ActuatorContexts
from gad.l1b_controller import ControllerStructure
from gad.l0_quality import QualityChecker, QualityRanges
from gad.types import Sample

ChannelKind = Literal["sensed", "commanded"]

_SENSED_PREFIXES = ("FIT", "LIT", "AIT", "PIT")
_SENSED_EXACT = {"DPIT301"}
_COMMANDED_PREFIXES = ("MV",)
_COMMANDED_EXACT = {"UV401"}
_COMMANDED_PUMP_RE = re.compile(r"^P\d+$")

NON_DATA_COLUMNS = ("Timestamp", "label")

TANK_AREA_M2 = 1.4957  # from documented 1.38 m tank diameter
LEVEL_RATE_K = 0.18572  # mm/s per (m3/h) net flow == 1000/3600 / TANK_AREA_M2

MASS_BALANCES: dict[str, tuple[str, str, str]] = {
    # tank: (level_col, inflow_col, outflow_col)
    "T101": ("LIT101", "FIT101", "FIT201"),
    "T301": ("LIT301", "FIT201", "FIT301"),
    "T401": ("LIT401", "FIT301", "FIT401"),
}

DUTY_STANDBY_PAIRS: tuple[tuple[str, str], ...] = (
    ("P101", "P102"),
    ("P302", "P301"),
    ("P402", "P401"),
)

ACTUATOR_VALID_STATES = frozenset({0.0, 1.0, 2.0})

_ATTACK_TAG_TYPOS = {"DIT-301": "DPIT-301"}


def classify_channel(name: str) -> ChannelKind:
    """Classify a SWaT tag as 'sensed' or 'commanded'; raises on anything else."""
    if name in _SENSED_EXACT or name.startswith(_SENSED_PREFIXES):
        return "sensed"
    if name in _COMMANDED_EXACT or name.startswith(_COMMANDED_PREFIXES) or _COMMANDED_PUMP_RE.match(name):
        return "commanded"
    raise ValueError(f"channel {name!r} does not match any known sensed/commanded rule")


def split_channels(columns: list[str]) -> tuple[list[str], list[str]]:
    """Split data-column names into (sensed, commanded), each in input order."""
    sensed, commanded = [], []
    for name in columns:
        if name in NON_DATA_COLUMNS:
            continue
        (sensed if classify_channel(name) == "sensed" else commanded).append(name)
    return sensed, commanded


def _normalize_label(raw: str) -> str:
    """Collapse whitespace and match against the two known labels."""
    collapsed = re.sub(r"\s+", "", str(raw))
    if collapsed.lower() == "normal":
        return "Normal"
    if collapsed.lower() == "attack":
        return "Attack"
    raise ValueError(f"unrecognized label value: {raw!r}")


def clean_raw_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace everywhere, parse the timestamp, coerce numerics,
        normalize the label. Works on either the Normal or Attack raw sheet —
        both have the same 53-column schema but different whitespace quirks.
    """
    df = raw.rename(columns={c: str(c).strip() for c in raw.columns})
    df = df.rename(columns={"Normal/Attack": "label"})

    ts = df["Timestamp"].astype(str).str.strip()
    df["Timestamp"] = pd.to_datetime(ts, format="%d/%m/%Y %I:%M:%S %p")

    value_cols = [c for c in df.columns if c not in NON_DATA_COLUMNS]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(np.float64)

    df["label"] = df["label"].map(_normalize_label).astype("category")

    return df[["Timestamp"] + value_cols + ["label"]]


def stage_of(tag: str) -> int:
    """SWaT tags carry their process stage as the first digit of their
        number: FIT101 -> 1, DPIT301 -> 3, UV401 -> 4, P602 -> 6.
    """
    m = re.search(r"\d", tag)
    if m is None:
        raise ValueError(f"tag {tag!r} has no stage digit")
    return int(m.group())


def physics_drivers() -> dict[str, set[str]]:
    """Sensed channel -> the actuators that drive it according to the
        declarative physics table L1b already uses (gad.l1b_physics).
        A flow meter is driven by its source's actuator; a tank level by the
        actuators of its inflow and outflow. This is what crosses stage
        boundaries: FIT201 sits in stage 2 but measures what stage 1's P101
        pumps.
    """
    from gad.l1b_physics import PHYSICS_TABLE

    drivers: dict[str, set[str]] = {}
    for tb in PHYSICS_TABLE:
        for src in (tb.inflow, tb.outflow):
            drivers.setdefault(src.flow_channel, set()).add(src.driving_actuator)
            drivers.setdefault(tb.level_channel, set()).add(src.driving_actuator)
    return drivers


def build_contexts(sensed: list[str], commanded: list[str]) -> ActuatorContexts:
    """Each sensed channel is conditioned on its own stage's actuators plus
        the physics table's drivers of it; configuration novelty is judged per
        stage over that stage's actuators.
    """
    stages = sorted({stage_of(t) for t in list(sensed) + list(commanded)})
    index = {s: i for i, s in enumerate(stages)}
    stage_sets = [[k for k, t in enumerate(commanded) if stage_of(t) == s] for s in stages]
    drivers = physics_drivers()
    channel_sets = [
        stage_sets[index[stage_of(ch)]] + [commanded.index(a) for a in sorted(drivers.get(ch, ()))]
        for ch in sensed
    ]
    return ActuatorContexts(channel_sets, stage_sets, [index[stage_of(ch)] for ch in sensed])


def regime_id(u: np.ndarray) -> int:
    """Pack the commanded-channel state combination into one integer."""
    digits = np.rint(u).astype(np.int64)
    if np.any((digits < 0) | (digits > 2)):
        raise ValueError("regime_id requires commanded values in {0,1,2}")
    weights = 3 ** np.arange(len(digits), dtype=np.int64)
    return int(np.dot(digits, weights))


class RegimeTracker:
    """Streaming: regime id of the current sample, plus whether this sample
        is the first one after a switch (regime differs from the previous
        sample's).
    """

    def __init__(self) -> None:
        self._prev: int | None = None

    def update(self, u: np.ndarray) -> tuple[int, bool]:
        r = regime_id(u)
        after_switch = self._prev is not None and r != self._prev
        self._prev = r
        return r, after_switch


class SwatAdapter:
    """Replays a cleaned SWaT parquet file as a stream of `Sample`s."""

    def __init__(self, parquet_path: str | Path, quality_ranges: QualityRanges) -> None:
        df = pd.read_parquet(parquet_path)
        if "Timestamp" not in df.columns:
            raise ValueError("data must have a 'Timestamp' column")

        columns = [c for c in df.columns if c not in NON_DATA_COLUMNS]
        self.sensed_channels, self.commanded_channels = split_channels(columns)

        if tuple(self.sensed_channels) != quality_ranges.channels:
            raise ValueError(
                "quality_ranges channels do not match this file's sensed channels in order"
            )

        self._df = df
        self._checker = QualityChecker(quality_ranges)

    def samples(self) -> Iterator[Sample]:
        x_cols = self.sensed_channels
        u_cols = self.commanded_channels
        for row in self._df.itertuples(index=False):
            d = row._asdict()
            t = d["Timestamp"].timestamp()
            x = np.array([d[c] for c in x_cols], dtype=np.float64)
            u = np.array([d[c] for c in u_cols], dtype=np.float64)
            q = self._checker.update(x)
            yield Sample(t=t, x=x, u=u, q=q)

    def regimes(self) -> Iterator[tuple[int, bool]]:
        """Regime id and after-switch bit per row, in time order. A
                commissioning/analysis capability of the adapter — not part of the
                uniform Sample stream `samples()` exposes to the core.
        """
        tracker = RegimeTracker()
        u_cols = self.commanded_channels
        for row in self._df.itertuples(index=False):
            d = row._asdict()
            u = np.array([d[c] for c in u_cols], dtype=np.float64)
            yield tracker.update(u)

    def __len__(self) -> int:
        return len(self._df)


def mass_balance_correlation(
    df: pd.DataFrame, level_col: str, in_col: str, out_col: str, window_s: int = 60
) -> tuple[float, float, int]:
    """corr/slope of Δlevel[mm/s] vs LEVEL_RATE_K * net_flow[m3/h], both
        averaged over a `window_s`-second window (1 Hz sampling assumed). Returns
        (corr, slope, n).
    """
    level = df[level_col]
    net_flow = df[in_col] - df[out_col]

    d_level = (level.shift(-window_s) - level) / window_s
    net_flow_avg = net_flow.rolling(window=window_s).mean().shift(-window_s + 1)

    pred = LEVEL_RATE_K * net_flow_avg
    valid = d_level.notna() & pred.notna()
    corr = float(np.corrcoef(d_level[valid], pred[valid])[0, 1])
    slope = float(np.polyfit(pred[valid], d_level[valid], 1)[0])
    return corr, slope, int(valid.sum())


def actuator_encoding_means(df: pd.DataFrame, actuator: str, sensor: str) -> dict[float, tuple[float, int]]:
    """Mean sensor reading (and sample count) grouped by actuator state."""
    grouped = df.groupby(actuator)[sensor].agg(["mean", "count"])
    return {float(state): (float(row["mean"]), int(row["count"])) for state, row in grouped.iterrows()}


def duty_standby_fraction(df: pd.DataFrame, pump: str) -> float:
    """Fraction of samples where `pump` is running (state == 2)."""
    return float((df[pump] == 2).mean())


def find_commissioning_start(
    df: pd.DataFrame,
    level_cols: tuple[str, ...] = ("LIT101", "LIT301", "LIT401"),
    band: tuple[float, float] = (500.0, 1200.0),
) -> int:
    """First row index where every tank in `level_cols` is simultaneously
        inside `band` — the plant starts empty and fills before settling into
        normal cycling; rows before this are the startup transient.
    """
    in_band = np.ones(len(df), dtype=bool)
    for col in level_cols:
        values = df[col].to_numpy()
        in_band &= (values >= band[0]) & (values <= band[1])
    idx = np.flatnonzero(in_band)
    if len(idx) == 0:
        raise ValueError("no row has all tanks simultaneously in-band")
    return int(idx[0])


@dataclass(frozen=True)
class AttackPointReconciliation:
    n_attacks: int
    n_excluded_no_physical_impact: int
    raw_distinct_points: tuple[str, ...]
    semantic_distinct_points: tuple[str, ...]
    typo_fixes: dict[str, str]
    case_fixes: dict[str, str]
    matched_channels: tuple[str, ...]
    unmatched_points: tuple[str, ...]


def _normalize_semantic(raw_point: str) -> tuple[str, str | None, str | None]:
    """Returns (semantic_tag, typo_fix_applied, case_fix_applied)."""
    stripped = raw_point.strip()
    typo_fixed = _ATTACK_TAG_TYPOS.get(stripped)
    base = typo_fixed if typo_fixed else stripped
    upper = base.upper()
    semantic = upper.replace("-", "").replace(" ", "")
    case_fix = base if base != upper else None
    return semantic, (stripped if typo_fixed else None), case_fix


def _split_attack_points(cell: str) -> list[str]:
    parts = re.split(r",|;|\band\b", cell, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def split_attack_points(cell: str) -> list[str]:
    """Split one attack-list 'Attack Point' cell into its raw tags
        (separators vary: comma, semicolon, "and").
    """
    return _split_attack_points(cell)


def attack_tag_to_column(raw_point: str) -> str:
    """Attack-list tag -> historian column name: fixes the `DIT-301` typo,
        case-folds `Mv-303`, drops the hyphen (tags are hyphenated, columns are
        not). Whether the result is actually a logged column is the caller's
        check — `MV-504` maps to `MV504`, which is not.
    """
    return _normalize_semantic(raw_point)[0]


def reconcile_attack_points(
    attack_list_path: str | Path, historian_columns: set[str]
) -> AttackPointReconciliation:
    """Compare the attack-list 'Attack Point' tags against the historian's
        logged column names, after fixing the two known data-entry issues in
        the attack list (the `DIT-301` typo for `DPIT-301`, and the `Mv-303`
        case variant of `MV-303`).
    """
    import openpyxl

    wb = openpyxl.load_workbook(attack_list_path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(min_row=2, values_only=True))

    n_attacks = 0
    n_excluded = 0
    raw_points: list[str] = []
    semantic_points: list[str] = []
    typo_fixes: dict[str, str] = {}
    case_fixes: dict[str, str] = {}

    for r in rows:
        attack_no, point_cell = r[0], r[3]
        if attack_no is None or point_cell is None:
            continue
        if "No Physical Impact" in point_cell:
            n_excluded += 1
            continue
        n_attacks += 1
        for raw_point in _split_attack_points(point_cell):
            raw_points.append(raw_point)
            semantic, typo_fix, case_fix = _normalize_semantic(raw_point)
            semantic_points.append(semantic)
            if typo_fix:
                typo_fixes[typo_fix] = _ATTACK_TAG_TYPOS[typo_fix]
            if case_fix:
                case_fixes[case_fix] = case_fix.upper()

    raw_distinct = tuple(sorted(set(raw_points)))
    semantic_distinct = tuple(sorted(set(semantic_points)))
    matched = tuple(sorted(set(semantic_distinct) & historian_columns))
    unmatched = tuple(sorted(set(semantic_distinct) - historian_columns))

    return AttackPointReconciliation(
        n_attacks=n_attacks,
        n_excluded_no_physical_impact=n_excluded,
        raw_distinct_points=raw_distinct,
        semantic_distinct_points=semantic_distinct,
        typo_fixes=typo_fixes,
        case_fixes=case_fixes,
        matched_channels=matched,
        unmatched_points=unmatched,
    )


CONTROLLER_STRUCTURE = ControllerStructure(
    triggers=(("MV101", "LIT101"), ("P101", "LIT301"), ("MV201", "LIT301"),
              ("MV302", "LIT401"), ("P302", "LIT401")),
    source="Goh et al., CRITIS 2016 tag table + tank layout; sensor choice on clean "
           "training data (docs/decisions/0042)",
)
