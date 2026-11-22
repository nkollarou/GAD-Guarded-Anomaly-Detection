"""The streaming detector: passes each sample through layers L0 to L6 and
returns NORMAL, ANOMALY or DEGRADED.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from gad.l5_limits import joint_clamp
from gad.l2_context import SettlingTracker, per_channel
from gad.l4_decision import MonitorDecider
from gad.l6_evidence import EvidenceBuilder, EvidenceRecord
from gad.l1_base import Predictor
from gad.l1a_invariants import InvariantMiner
from gad.l0_quality import TRUSTED_LEARNING_MASK, excluded_from_learning, is_blinding
from gad.l2_residual import RegimeScaledResidual
from gad.l3_cusum import Cusum
from gad.types import Decision, DetectorState, Sample

BRANCH_A = "l1a"
BRANCH_B = "l1b"

MONITOR_A = "l1a_cusum"
MONITOR_B = "l1b_cusum"
MONITOR_INVARIANT = "invariant"
MONITOR_CTRL = "l1b_ctrl"


class Detector:
    def __init__(
        self,
        sensed_channels: Iterable[str],
        l1a: Predictor,
        l1b: Predictor,
        l2a: RegimeScaledResidual,
        l2b: RegimeScaledResidual,
        l3a: Cusum,
        l3b: Cusum,
        invariants: InvariantMiner,
        decider: MonitorDecider,
        regime_fn: Callable[[np.ndarray], int],
        n_possible_tests: int,
        evidence: EvidenceBuilder | None = None,
        learn_scales: bool = True,
        trusted_learning: bool = False,
        learn_guard: int = 0,
        per_channel_learning: bool = False,
        per_sensor: "PerSensorDecider | None" = None,
        use_invariants: bool = True,
        controller=None,
        adaptive_a=None,
        adaptive_b=None,
        joint_cap: float | None = None,
        sensitivity=None,
        settle_samples: int = 0,
        naive_learning: bool = False,
    ) -> None:
        """`l3a`/`l3b` must already carry their per-channel `h`, bisected
                against that monitor's own share of the budget — this class does
                not decide significance, it only routes. `n_possible_tests` is how
                many per-channel checks could report if nothing abstained, across
                all three monitors; it sets the denominator for DEGRADED.
        """
        if l3a.h is None or l3b.h is None:
            raise ValueError("l3a and l3b must have calibrated h before use in Detector")
        if n_possible_tests < 1:
            raise ValueError("n_possible_tests must be >= 1")
        self.sensed_channels = tuple(sensed_channels)
        self.l1a, self.l1b = l1a, l1b
        self.l2a, self.l2b = l2a, l2b
        self.l3a, self.l3b = l3a, l3b
        self.invariants = invariants
        self.decider = decider
        self.regime_fn = regime_fn
        self.n_possible_tests = n_possible_tests
        self.evidence = evidence
        self.evidence_log: list[EvidenceRecord] = []
        self.learn_scales = learn_scales
        self.trusted_learning = trusted_learning
        self.learn_guard = learn_guard
        self.per_channel_learning = per_channel_learning
        self.per_sensor = per_sensor
        if per_sensor is not None and controller is not None:
            raise ValueError("per-sensor alarms and the control-rule monitor are not "
                             "implemented together (docs/decisions/0059 3.2)")
        self.use_invariants = use_invariants
        self.controller = controller
        self.adaptive_a, self.adaptive_b = adaptive_a, adaptive_b
        self.joint_cap = joint_cap
        self._settle = SettlingTracker(len(sensed_channels), settle_samples)
        self.naive_learning = bool(naive_learning)
        if self.naive_learning:
            for l2 in (l2a, l2b):
                if hasattr(l2, "clip"):
                    l2.clip = float("inf")   # rule 2 off: no bound on one sample
                    l2.kappa = 1.0           # the clipping correction goes with it
        self.sensitivity = sensitivity
        if joint_cap is not None:
            for l2 in (l2a, l2b):
                if not hasattr(l2, "commit"):
                    raise ValueError("joint_cap needs EventScaleResidual L2 learners")
        self.budget_used: dict[str, np.ndarray] = {}
        self.budget_warnings: dict[tuple[str, str], float] = {}
        self.n_events = 0
        self._regime = None
        self._n_tests = n_possible_tests - (0 if use_invariants else len(self.sensed_channels))
        self._n_tests += 1 if controller is not None else 0
        self.last_controller_output = None
        self.last_abstained: dict[str, np.ndarray] | None = None
        b_names = tuple(getattr(self.l1b, "state_names", ()) or ())
        self._l1b_slots = np.array(
            [i for i, ch in enumerate(self.sensed_channels) if ch in b_names],
            dtype=np.int64)
        self.abstention_slots: dict[str, tuple[str, ...]] = {
            "l1a": tuple(self.sensed_channels),
            "l1b": tuple(self.sensed_channels[i] for i in self._l1b_slots),
        }
        if use_invariants:
            self.abstention_slots["invariant"] = tuple(self.sensed_channels)
        if self._n_tests < 1:
            raise ValueError("no tests left to decide on")
        self._since_anomaly = learn_guard + 1  # nothing recent at start
        n_sensed = len(sensed_channels)
        self._fired_prev = np.zeros(n_sensed, dtype=bool)
        self._since_fire = np.full(n_sensed, learn_guard + 1, dtype=np.int64)
        self._prev_state = DetectorState.NORMAL

    def start_stream(self) -> None:
        """Tell the detector the next sample begins a new time series. Only the settling tracker is reset: every
                channel is then SETTLING for its first `settle_samples`, because what
                the plant was doing just before the stream is unknown. With
                `settle_samples == 0` this changes nothing.
        """
        self._settle.reset()

    def _l5_event(self, branch: str, t: float) -> None:
        """commit the learned scales, keep the CUSUM
                evidence in the same units, hold scale x threshold within the cap.
        """
        l2, l3, ada = ((self.l2a, self.l3a, self.adaptive_a) if branch == "a"
                       else (self.l2b, self.l3b, self.adaptive_b))
        n = len(self.sensed_channels)
        regimes = per_channel(self._regime, n)
        old = np.array([l2.committed_scale(j, regimes[j]) or np.nan for j in range(n)], dtype=float)
        l2.commit()
        new = np.array([l2.committed_scale(j, regimes[j]) or np.nan for j in range(n)], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            l3.rescale(old / new)
        f_hi, f_lo = l2.scale_factors()
        h, used = joint_clamp(ada.h, ada.h_ref, f_hi, f_lo, self.joint_cap)
        l3.h = h
        self.budget_used[branch] = used
        for j in np.flatnonzero(used > 0.5):
            self.budget_warnings.setdefault((branch, self.sensed_channels[j]), t)
        if branch == "a":
            self.n_events += 1
        last = "b" if self.adaptive_b is not None else "a"
        if self.sensitivity is not None and branch == last:
            self.sensitivity.enforce(self, t)  # both branches are committed by now

    def _channel_calm(self) -> np.ndarray:
        """Per channel: may this channel teach on this sample?"""
        return (~self._fired_prev) & (self._since_fire > self.learn_guard)

    def update(self, sample: Sample) -> Decision:
        regime_id = self._settle.step(self.regime_fn(sample.u))
        self._regime = regime_id
        not_anomaly = self._prev_state != DetectorState.ANOMALY

        x_hat_a = self.l1a.update(sample)
        x_hat_b = self.l1b.update(sample)
        r_a = sample.x - x_hat_a
        r_b = sample.x - x_hat_b

        blinding = np.array([is_blinding(int(q)) for q in sample.q])
        if self.trusted_learning:
            learn = (np.asarray(sample.q) & TRUSTED_LEARNING_MASK) == 0
            not_anomaly = not_anomaly and self._since_anomaly > self.learn_guard
        else:
            learn = ~np.array([excluded_from_learning(int(q)) for q in sample.q])
        if self.per_channel_learning:
            learn = learn & self._channel_calm()
            not_anomaly = True
        if self.naive_learning:
            not_anomaly = True
            learn = np.ones(len(self.sensed_channels), dtype=bool)
        if not self.learn_scales:
            learn[:] = False
        r_a = np.where(blinding, np.nan, r_a)
        r_b = np.where(blinding, np.nan, r_b)

        scaled_a = self.l2a.update(r_a, regime_id, not_anomaly, learn)
        scaled_b = self.l2b.update(r_b, regime_id, not_anomaly, learn)

        evidence_a = self.l3a.update(scaled_a, regime_id)
        evidence_b = self.l3b.update(scaled_b, regime_id)

        if self.use_invariants:
            violations = self.invariants.update(sample)
            violations = np.where(blinding, np.nan, violations)
        else:
            violations = np.full(len(self.sensed_channels), np.nan)

        # NaN compares False throughout: an abstaining channel never fires
        fired_a = np.asarray(self.l3a.alarm(evidence_a, regime_id), dtype=bool)
        fired_b = np.asarray(self.l3b.alarm(evidence_b, regime_id), dtype=bool)
        fired_inv = np.nan_to_num(violations, nan=0.0) > 0.0
        monitors = {
            MONITOR_A: bool(fired_a.any()),
            MONITOR_B: bool(fired_b.any()),
            MONITOR_INVARIANT: bool(fired_inv.any()),
        }

        n_present = (
            int(np.count_nonzero(~np.isnan(evidence_a)))
            + int(np.count_nonzero(~np.isnan(evidence_b)))
            + int(np.count_nonzero(~np.isnan(violations)))
        )
        evidence_ctrl = None
        if self.controller is not None:
            evidence_ctrl, present, out = self.controller.update(sample)
            self.last_controller_output = out
            monitors[MONITOR_CTRL] = self.controller.alarm(evidence_ctrl)
            n_present += present
        abstention_fraction = 1.0 - n_present / self._n_tests
        self.last_abstained = {
            "l1a": np.isnan(evidence_a),
            "l1b": np.isnan(np.asarray(evidence_b)[self._l1b_slots]),
        }
        if self.use_invariants:
            self.last_abstained["invariant"] = np.isnan(violations)

        if self.per_sensor is not None:
            per_channel_fired = fired_a | fired_b
            if self.use_invariants:
                per_channel_fired = per_channel_fired | fired_inv
            decision = self.per_sensor.update(sample.t, per_channel_fired, abstention_fraction)
        else:
            decision = self.decider.update(sample.t, monitors, abstention_fraction)
        if self.evidence is not None:
            record = self.evidence.update(
                sample, evidence_a, evidence_b, self.l3a.h, self.l3b.h, violations, monitors,
                entering_anomaly=(decision.state == DetectorState.ANOMALY
                                  and self._prev_state != DetectorState.ANOMALY),
                state=decision.state.value,
                bundle_id=decision.bundle_id,
                controller=(None if self.controller is None else
                            (evidence_ctrl, self.controller.h, self.controller.model)),
            )
            if record is not None:
                self.evidence_log.append(record)
        if self.adaptive_a is not None or self.adaptive_b is not None:
            quality_ok = (np.asarray(sample.q) & TRUSTED_LEARNING_MASK) == 0
            if self.per_channel_learning:
                trusted = quality_ok & self._channel_calm()
            else:
                calm = (self._prev_state != DetectorState.ANOMALY
                        and self._since_anomaly > self.learn_guard)
                trusted = quality_ok & calm
            if self.naive_learning:
                trusted = np.ones(len(self.sensed_channels), dtype=bool)
            if self.adaptive_a is not None and self.adaptive_a.observe(sample.t, evidence_a, trusted):
                self.l3a.h = self.adaptive_a.h.copy()
                if self.joint_cap is not None:
                    self._l5_event("a", sample.t)
            # both branches' evidence is aligned with the sensed channels
            if self.adaptive_b is not None and self.adaptive_b.observe(sample.t, evidence_b, trusted):
                self.l3b.h = self.adaptive_b.h.copy()
                if self.joint_cap is not None:
                    self._l5_event("b", sample.t)
        fired = fired_a | fired_b
        self._since_fire = np.where(fired, 0, self._since_fire + 1)
        self._fired_prev = fired
        self._prev_state = decision.state
        self._since_anomaly = 0 if decision.state == DetectorState.ANOMALY else self._since_anomaly + 1
        return decision
