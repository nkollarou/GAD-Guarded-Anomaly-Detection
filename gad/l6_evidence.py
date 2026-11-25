"""L6: one record per alarm, naming the sensors, predictors and relations behind
it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from gad.l1a_predictor import RelationalPredictor, feature_names
from gad.types import OOR, VALID, Sample

MONITOR_NAMES = ("l1a_cusum", "l1b_cusum")


@dataclass(frozen=True)
class Contribution:
    tag: str
    score: float
    kind: str  # "sensed" | "commanded"
    via: tuple[str, ...]  # the monitors / relations that credited this tag


@dataclass(frozen=True)
class EvidenceRecord:
    t_detect: float
    onset: float
    detection_delay: float
    state: str
    bundle_id: str
    monitors_fired: tuple[str, ...]
    contributions: tuple[Contribution, ...]  # ranked, highest first
    relations_violated: tuple[str, ...]
    quality_flags: dict[str, int]  # tag -> L0 flag, non-VALID only


class EvidenceBuilder:
    """Tracks when each channel's evidence started to rise, and builds an
    `EvidenceRecord` when the detector enters ANOMALY.
    """

    def __init__(
        self,
        sensed_channels: Iterable[str],
        commanded_channels: Iterable[str],
        l1a: RelationalPredictor | None = None,
        physics_relations: dict[str, str] | None = None,
    ) -> None:
        """`physics_relations` maps an L1b channel to the name of the physical
        relation it belongs to (e.g. a tank's mass balance).
        """
        self.sensed = tuple(sensed_channels)
        self.commanded = tuple(commanded_channels)
        self.l1a = l1a
        self.physics_relations = physics_relations or {}
        d = len(self.sensed)
        self._run_start = {m: np.full(d, np.nan) for m in MONITOR_NAMES}
        self._active = {m: np.zeros(d, dtype=bool) for m in MONITOR_NAMES}
        self._feature_tags: np.ndarray | None = None
        if l1a is not None:
            names = feature_names(l1a.sensed_channels, l1a.commanded_channels, l1a.L)
            self._feature_tags = np.array([n.split("[")[0] for n in names])

    def update(
        self,
        sample: Sample,
        evidence_a: np.ndarray,
        evidence_b: np.ndarray,
        h_a: np.ndarray,
        h_b: np.ndarray,
        monitors: dict[str, bool],
        entering_anomaly: bool,
        state: str,
        bundle_id: str,
    ) -> EvidenceRecord | None:
        """Advance onset tracking by one sample; return a record if this is the
        sample on which the detector entered ANOMALY.
        """
        for m, sig in (("l1a_cusum", evidence_a), ("l1b_cusum", evidence_b)):
            known = ~np.isnan(sig)  # NaN: the channel abstains
            active = sig > 0.0
            started = known & active & ~self._active[m]
            self._run_start[m][started] = sample.t
            self._active[m] = np.where(known, active, self._active[m])
        if not entering_anomaly:
            return None
        return self._build(sample, evidence_a, evidence_b, h_a, h_b, monitors, state, bundle_id)

    def _build(self, sample, evidence_a, evidence_b, h_a, h_b,
               monitors, state, bundle_id) -> EvidenceRecord:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_a = np.nan_to_num(evidence_a / h_a, nan=0.0, posinf=0.0)
            ratio_b = np.nan_to_num(evidence_b / h_b, nan=0.0, posinf=0.0)

        scores: dict[str, float] = {}
        via: dict[str, set[str]] = {}
        kind: dict[str, str] = {}
        relations: list[str] = []

        def credit(tag: str, score: float, source: str, tag_kind: str) -> None:
            if score <= 0.0:
                return
            scores[tag] = max(scores.get(tag, 0.0), score)
            via.setdefault(tag, set()).add(source)
            kind[tag] = tag_kind

        onsets: list[float] = []
        for j, ch in enumerate(self.sensed):
            fired_here = False
            if ratio_a[j] >= 1.0:
                fired_here = True
                credit(ch, ratio_a[j], "l1a_cusum", "sensed")
                relations.append(self._l1a_relation(j, ch, ratio_a[j], credit))
                onsets.append(self._run_start["l1a_cusum"][j])
            if ratio_b[j] >= 1.0:
                fired_here = True
                credit(ch, ratio_b[j], "l1b_cusum", "sensed")
                name = self.physics_relations.get(ch, f"L1b dynamics of {ch}")
                relations.append(f"{name}: {ch} evidence {ratio_b[j]:.2f}x its alarm level")
                onsets.append(self._run_start["l1b_cusum"][j])
            if not fired_here:
                # below every threshold, still ranked by how close it came
                credit(ch, max(ratio_a[j], ratio_b[j]), "below_threshold", "sensed")
            elif int(sample.q[j]) & OOR:
                relations.append(f"{ch}: reading outside its commissioning range "
                                 f"(possible sensor fault or spoofing)")

        order = sorted(scores, key=lambda t: (-scores[t], kind[t] != "sensed", t))
        contributions = tuple(
            Contribution(tag=t, score=float(scores[t]), kind=kind[t], via=tuple(sorted(via[t])))
            for t in order
        )
        finite_onsets = [o for o in onsets if np.isfinite(o)]
        onset = float(min(finite_onsets)) if finite_onsets else float(sample.t)
        quality = {self.sensed[j]: int(q) for j, q in enumerate(sample.q) if int(q) != VALID}
        return EvidenceRecord(
            t_detect=float(sample.t),
            onset=onset,
            detection_delay=float(sample.t) - onset,
            state=state,
            bundle_id=bundle_id,
            monitors_fired=tuple(m for m in MONITOR_NAMES if monitors.get(m)),
            contributions=contributions,
            relations_violated=tuple(relations),
            quality_flags=quality,
        )

    def _l1a_relation(self, j: int, ch: str, ratio: float, credit) -> str:
        choice = "unknown"
        if self.l1a is not None and self.l1a.state is not None:
            choice = self.l1a.state.model_choice.get(ch, "fitted")
        text = f"L1a relation for {ch} ({choice}): evidence {ratio:.2f}x its alarm level"
        if (choice != "fitted" or self.l1a is None or self.l1a.last_features is None
                or self._feature_tags is None):
            return text

        st = self.l1a.state
        parts = np.abs(st.beta_raw[:, j] * (self.l1a.last_features - st.feature_mean))
        total = float(parts.sum())
        if total <= 0.0:
            return text
        by_tag: dict[str, float] = {}
        for tag, p in zip(self._feature_tags, parts):
            if p > 0.0:
                by_tag[str(tag)] = by_tag.get(str(tag), 0.0) + float(p)
        shares = sorted(by_tag.items(), key=lambda kv: (-kv[1], kv[0]))
        for tag, p in shares:
            credit(tag, ratio * p / total, f"l1a_input_of_{ch}",
                   "sensed" if tag in self.sensed else "commanded")
        top = ", ".join(f"{t} {100 * p / total:.0f}%" for t, p in shares[:3])
        return f"{text}; driven by {top}"
