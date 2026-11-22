"""L6: one record per alarm, naming the sensors, predictors and relations
behind it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import numpy as np

from gad.l2_context import per_channel
from gad.l1a_invariants import FEATURES, InvariantMiner
from gad.l1a_predictor import RelationalPredictor, feature_names
from gad.types import OOR, VALID, Sample

MONITOR_NAMES = ("l1a_cusum", "l1b_cusum", "invariant", "l1b_ctrl")


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

    def top_tags(self, k: int) -> tuple[str, ...]:
        return tuple(c.tag for c in self.contributions[:k])

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=float)


def observed_states(U: np.ndarray, commanded_channels: Iterable[str]) -> dict[str, tuple[float, ...]]:
    """Each actuator's distinct states in clean data — the only alternatives
        the actuator check will try.
    """
    return {
        tag: tuple(float(s) for s in np.unique(U[:, k][np.isfinite(U[:, k])]))
        for k, tag in enumerate(commanded_channels)
    }


class EvidenceBuilder:
    """Tracks per-channel onsets every sample and builds an `EvidenceRecord`
        when the detector enters ANOMALY. Owned by one `Detector`.
    """

    def __init__(
        self,
        sensed_channels: Iterable[str],
        commanded_channels: Iterable[str],
        invariants: InvariantMiner,
        regime_fn: Callable[[np.ndarray], int],
        commanded_states: dict[str, tuple[float, ...]],
        l1a: RelationalPredictor | None = None,
        physics_relations: dict[str, str] | None = None,
    ) -> None:
        """`physics_relations` maps an L1b-covered channel to the name of
                the physical relation it takes part in (e.g. a tank's mass
                balance), so a record names the relation rather than only the
                channel.
        """
        self.sensed = tuple(sensed_channels)
        self.commanded = tuple(commanded_channels)
        self.invariants = invariants
        self.regime_fn = regime_fn
        self.commanded_states = commanded_states
        self.l1a = l1a
        self.physics_relations = physics_relations or {}

        d = len(self.sensed)
        self._run_start = {m: np.full(1 if m == "l1b_ctrl" else d, np.nan) for m in MONITOR_NAMES}
        self._active = {m: np.zeros(1 if m == "l1b_ctrl" else d, dtype=bool) for m in MONITOR_NAMES}
        self._prev_x: np.ndarray | None = None

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
        violations: np.ndarray,
        monitors: dict[str, bool],
        entering_anomaly: bool,
        state: str,
        bundle_id: str,
        controller=None,
    ) -> EvidenceRecord | None:
        """Advance onset tracking by one sample; return a record if this
                sample is the one on which the detector entered ANOMALY.
        """
        signals = {
            "l1a_cusum": evidence_a,
            "l1b_cusum": evidence_b,
            "invariant": violations,
        }
        if controller is not None:
            signals["l1b_ctrl"] = np.array([controller[0]])
        for m, sig in signals.items():
            known = ~np.isnan(sig)  # NaN = abstaining: leave its run untouched
            active = sig > 0.0
            started = known & active & ~self._active[m]
            self._run_start[m][started] = sample.t
            self._active[m] = np.where(known, active, self._active[m])

        record = None
        if entering_anomaly:
            record = self._build(sample, evidence_a, evidence_b, h_a, h_b, violations,
                                 monitors, state, bundle_id, controller)
        self._prev_x = np.array(sample.x, dtype=np.float64)
        return record

    # -- record construction (once per alarm) ------------------------------

    def _build(self, sample, evidence_a, evidence_b, h_a, h_b, violations,
               monitors, state, bundle_id, controller=None) -> EvidenceRecord:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_a = np.nan_to_num(evidence_a / h_a, nan=0.0, posinf=0.0)
            ratio_b = np.nan_to_num(evidence_b / h_b, nan=0.0, posinf=0.0)
        viol = np.nan_to_num(violations, nan=0.0)
        inv_score = np.where(viol > 0.0, 1.0 + viol, 0.0)

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
            if viol[j] > 0.0:
                fired_here = True
                credit(ch, inv_score[j], "invariant", "sensed")
                relations.append(self._invariant_relation(j, ch, sample, inv_score[j], credit))
                onsets.append(self._run_start["invariant"][j])
            if not fired_here:
                # below every firing line, still ranked by how close it came
                credit(ch, max(ratio_a[j], ratio_b[j]), "below_threshold", "sensed")
            elif int(sample.q[j]) & OOR:
                relations.append(f"{ch}: reading outside its commissioning range "
                                 f"(possible sensor fault or spoofing)")

        if controller is not None and monitors.get("l1b_ctrl"):
            ev, h, model = controller
            ratio = float(ev / h) if h > 0 else 0.0
            for rule, total in model.top_rules(3):
                relations.append(f"L1b controller rule {rule}: recent excess {total:.2f} "
                                 f"(monitor {ratio:.2f}x its alarm level)")
                for tag in _rule_actuators(rule):
                    credit(tag, ratio, "l1b_ctrl", "commanded")
            onsets.append(float(self._run_start["l1b_ctrl"][0]))

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

    def _single_flips(self, u: np.ndarray):
        """Every commanded vector differing from `u` in exactly one actuator,
                moved to another state it held in clean operation.
        """
        for k, tag in enumerate(self.commanded):
            for s in self.commanded_states.get(tag, ()):
                if s != u[k]:
                    u_alt = u.copy()
                    u_alt[k] = s
                    yield tag, s, u_alt

    def _invariant_relation(self, j: int, ch: str, sample: Sample, score: float, credit) -> str:
        u = np.asarray(sample.u, dtype=np.float64)
        d = len(self.sensed)
        regime = int(per_channel(self.regime_fn(u), d)[j])
        observed = {"value": float(sample.x[j])}
        if self._prev_x is not None:
            observed["delta"] = float(sample.x[j] - self._prev_x[j])
        parts: list[str] = []

        novel = self.invariants.novel_stages(u)
        if len(novel) and novel[self.invariants.channel_stage[j]]:
            stage = self.invariants.channel_stage[j]
            known_if = []
            for tag, s, u_alt in self._single_flips(u):
                if not self.invariants.novel_stages(u_alt)[stage]:
                    credit(tag, score, f"stage_configuration_of_{ch}", "commanded")
                    known_if.append(f"{tag}={s:g}")
            text = f"{ch}: its stage is in an actuator configuration never seen in commissioning"
            if known_if:
                text += f"; a known one if {' or '.join(sorted(set(known_if)))}"
            parts.append(text)

        broken = {}
        for f in FEATURES:
            inv = self.invariants.state.invariants.get((ch, regime, f))
            if inv is not None and f in observed and inv.excess(observed[f]) > 0.0:
                broken[f] = inv
        if broken:
            explained_by: list[str] = []
            for tag, s, u_alt in self._single_flips(u):
                alt = int(per_channel(self.regime_fn(u_alt), d)[j])
                if all(
                    (alt_inv := self.invariants.state.invariants.get((ch, alt, f))) is not None
                    and alt_inv.excess(observed[f]) == 0.0
                    for f in broken
                ):
                    credit(tag, score, f"invariant_of_{ch}", "commanded")
                    explained_by.append(f"{tag}={s:g}")
            f, inv = next(iter(broken.items()))
            text = (f"invariant: {ch} {f} {observed[f]:.4g} outside its normal band "
                    f"[{inv.lo:.4g}, {inv.hi:.4g}] for this actuator configuration")
            if explained_by:
                text += f"; normal if {' or '.join(sorted(set(explained_by)))}"
            parts.append(text)

        return "; ".join(parts) if parts else f"invariant for {ch} violated"


def _rule_actuators(rule: str) -> list[str]:
    """Actuator tags a controller rule name refers to:
        "trigger A 1->2 on S", "pair A=1 & B=2", "dwell A=1", "unseen A=2".
    """
    kind, rest = rule.split(" ", 1)
    if kind == "trigger":
        return [rest.split(" ", 1)[0]]
    if kind == "pair":
        return [part.split("=")[0].strip() for part in rest.split("&")]
    return [rest.split("=")[0]]
