"""Everything specific to SWaT: which channels are sensors and actuators, the
plant's stages, tanks and flows, and how to read the raw export.
"""
from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd

from gad.l2_context import ActuatorContexts

ChannelKind = Literal["sensed", "commanded"]

_SENSED_PREFIXES = ("FIT", "LIT", "AIT", "PIT")
_SENSED_EXACT = {"DPIT301"}
_COMMANDED_PREFIXES = ("MV",)
_COMMANDED_EXACT = {"UV401"}
_COMMANDED_PUMP_RE = re.compile(r"^P\d+$")

NON_DATA_COLUMNS = ("Timestamp", "label")

TANK_AREA_M2 = 1.4957  # from documented 1.38 m tank diameter
LEVEL_RATE_K = 0.18572  # mm/s per (m3/h) net flow == 1000/3600 / TANK_AREA_M2


def classify_channel(name: str) -> ChannelKind:
    """Classify a SWaT tag as 'sensed' or 'commanded'; raises on anything else.
    """
    if name in _SENSED_EXACT or name.startswith(_SENSED_PREFIXES):
        return "sensed"
    if name in _COMMANDED_EXACT or name.startswith(_COMMANDED_PREFIXES) or _COMMANDED_PUMP_RE.match(name):
        return "commanded"
    raise ValueError(f"channel {name!r} does not match any known sensed/commanded rule")


def split_channels(columns: list[str]) -> tuple[list[str], list[str]]:
    """Split data-column names into (sensed, commanded), each in input order.
    """
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
    normalize the label. Works on either the Normal or Attack raw sheet — both
    have the same 53-column schema but different whitespace quirks.
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
    """SWaT tags carry their process stage as the first digit of their number:
    FIT101 -> 1, DPIT301 -> 3, UV401 -> 4, P602 -> 6.
    """
    m = re.search(r"\d", tag)
    if m is None:
        raise ValueError(f"tag {tag!r} has no stage digit")
    return int(m.group())


def physics_drivers() -> dict[str, set[str]]:
    """Sensed channel -> the actuators that drive it according to the declarative
    physics table L1b already uses (gad.l1b_physics). A flow meter is driven by
    its source's actuator; a tank level by the actuators of its inflow and
    outflow. This is what crosses stage boundaries: FIT201 sits in stage 2 but
    measures what stage 1's P101 pumps.
    """
    from gad.l1b_physics import PHYSICS_TABLE

    drivers: dict[str, set[str]] = {}
    for tb in PHYSICS_TABLE:
        for src in (tb.inflow, tb.outflow):
            drivers.setdefault(src.flow_channel, set()).add(src.driving_actuator)
            drivers.setdefault(tb.level_channel, set()).add(src.driving_actuator)
    return drivers


def build_contexts(sensed: list[str], commanded: list[str]) -> ActuatorContexts:
    """Each sensed channel is conditioned on its own stage's actuators plus the
    physics table's drivers of it; configuration novelty is judged per stage
    over that stage's actuators.
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


def find_commissioning_start(
    df: pd.DataFrame,
    level_cols: tuple[str, ...] = ("LIT101", "LIT301", "LIT401"),
    band: tuple[float, float] = (500.0, 1200.0),
) -> int:
    """First row index where every tank in `level_cols` is simultaneously inside
    `band` — the plant starts empty and fills before settling into normal
    cycling; rows before this are the startup transient.
    """
    in_band = np.ones(len(df), dtype=bool)
    for col in level_cols:
        values = df[col].to_numpy()
        in_band &= (values >= band[0]) & (values <= band[1])
    idx = np.flatnonzero(in_band)
    if len(idx) == 0:
        raise ValueError("no row has all tanks simultaneously in-band")
    return int(idx[0])


