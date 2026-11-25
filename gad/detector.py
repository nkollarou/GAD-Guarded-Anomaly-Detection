from __future__ import annotations
from typing import Callable, Iterable
import numpy as np
from gad.l0_quality import TRUSTED_LEARNING_MASK, is_blinding
from gad.l1a_predictor import RelationalPredictor
from gad.l1b_physics import StateSpacePredictor
from gad.l2_context import SettlingTracker, per_channel
from gad.l2_residual import BoundedScaleResidual
from gad.l3_cusum import Cusum
from gad.l4_decision import PerSensorDecider
from gad.l5_limits import joint_clamp
from gad.l6_evidence import EvidenceBuilder, EvidenceRecord
from gad.types import Decision, DetectorState, Sample

MONITOR_A = "l1a_cusum"
MONITOR_B = "l1b_cusum"


class Detector:
    def __init__(
        self,
        sensed_channels: Iterable[str],
        l1a: RelationalPredictor,
        l1b: StateSpacePredictor,
        l2a: BoundedScaleResidual,
        l2b: BoundedScaleResidual,
        l3a: Cusum,
        l3b: Cusum,
        decider: PerSensorDecider, # l4
        regime_fn: Callable[[np.ndarray], int],
        n_possible_tests: int,
        adaptive_a,
        adaptive_b,
        joint_cap: float,
        evidence: EvidenceBuilder | None = None, # l6
        learn_guard: int = 0,
        sensitivity=None,
        settle_samples: int = 0,
        naive_learning: bool = False, # for unguarded version of gad
    ) -> None:
        """`l3a`/`l3b` must already carry their thresholds. `n_possible_tests` is
        the number of residuals that could be checked on a sample (one per
        sensor for L1a, one per physics channel for L1b); DEGRADED is reported
        when too few of them can be.
        """
        if l3a.h is None or l3b.h is None:
            raise ValueError("l3a and l3b must have calibrated h before use in Detector")
        if n_possible_tests < 1:
            raise ValueError("n_possible_tests must be >= 1")
        self.sensed_channels = tuple(sensed_channels)
        self.l1a, self.l1b = l1a, l1b
        self.l2a, self.l2b = l2a, l2b
        self.l3a, self.l3b = l3a, l3b
        self.decider = decider
        self.regime_fn = regime_fn
        self._n_tests = n_possible_tests
        self.evidence = evidence
        self.evidence_log: list[EvidenceRecord] = []
        self.learn_guard = learn_guard
        self.adaptive_a, self.adaptive_b = adaptive_a, adaptive_b
        self.joint_cap = joint_cap
        self._settle = SettlingTracker(len(self.sensed_channels), settle_samples)
        self.naive_learning = bool(naive_learning) 
        if self.naive_learning:
            for l2 in (l2a, l2b):
                if hasattr(l2, "clip"):
                    l2.clip = float("inf")   # rule 2 off: no bound on one sample
                    l2.kappa = 1.0           # the clipping correction goes with it
        self.sensitivity = sensitivity
        self.budget_used: dict[str, np.ndarray] = {}
        self.budget_warnings: dict[tuple[str, str], float] = {}
        self.n_events = 0
        self._regime = None
        n_sensed = len(self.sensed_channels)
        self._fired_prev = np.zeros(n_sensed, dtype=bool)
        self._since_fire = np.full(n_sensed, learn_guard + 1, dtype=np.int64)
        self._prev_state = DetectorState.NORMAL

    def start_stream(self) -> None:
        """The next sample starts a new time series: every channel begins in its
        settling context, because what the plant did just before is unknown.
        """
        self._settle.reset()

    def _l5_event(self, branch: str, t: float) -> None:
        """commit the learned scales, keep the CUSUM evidence in the same units,
        hold scale x threshold within the cap.
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
        """Per channel: may this channel teach on this sample? (rule 1)"""
        return (~self._fired_prev) & (self._since_fire > self.learn_guard)

    def update(self, sample: Sample) -> Decision:
        regime_id = self._settle.step(self.regime_fn(sample.u))
        self._regime = regime_id

        # L1: two predictions, two residuals
        r_a = sample.x - self.l1a.update(sample)
        r_b = sample.x - self.l1b.update(sample)
        blinding = np.array([is_blinding(int(q)) for q in sample.q])
        r_a = np.where(blinding, np.nan, r_a)
        r_b = np.where(blinding, np.nan, r_b)

        # rule 1: learn only from trusted samples (all samples when unguarded)
        if self.naive_learning:
            learn = np.ones(len(self.sensed_channels), dtype=bool)
        else:
            learn = ((np.asarray(sample.q) & TRUSTED_LEARNING_MASK) == 0) & self._channel_calm()

        # L2 and L3
        scaled_a = self.l2a.update(r_a, regime_id, True, learn)
        scaled_b = self.l2b.update(r_b, regime_id, True, learn)
        evidence_a = self.l3a.update(scaled_a, regime_id)
        evidence_b = self.l3b.update(scaled_b, regime_id)

        # L4: per-sensor alarms; a missing residual never fires
        fired_a = np.asarray(self.l3a.alarm(evidence_a, regime_id), dtype=bool)
        fired_b = np.asarray(self.l3b.alarm(evidence_b, regime_id), dtype=bool)
        n_present = (int(np.count_nonzero(~np.isnan(evidence_a)))
                     + int(np.count_nonzero(~np.isnan(evidence_b))))
        abstention_fraction = 1.0 - n_present / self._n_tests
        decision = self.decider.update(sample.t, fired_a | fired_b, abstention_fraction)

        # L6: a record when an alarm starts
        if self.evidence is not None:
            monitors = {MONITOR_A: bool(fired_a.any()), MONITOR_B: bool(fired_b.any())}
            record = self.evidence.update(
                sample, evidence_a, evidence_b, self.l3a.h, self.l3b.h, monitors,
                entering_anomaly=(decision.state == DetectorState.ANOMALY
                                  and self._prev_state != DetectorState.ANOMALY),
                state=decision.state.value,
                bundle_id=decision.bundle_id,
            )
            if record is not None:
                self.evidence_log.append(record)

        # L5: thresholds and scales are updated at fixed times
        if self.naive_learning:
            trusted = np.ones(len(self.sensed_channels), dtype=bool)
        else:
            trusted = ((np.asarray(sample.q) & TRUSTED_LEARNING_MASK) == 0) & self._channel_calm()
        if self.adaptive_a.observe(sample.t, evidence_a, trusted):
            self.l3a.h = self.adaptive_a.h.copy()
            self._l5_event("a", sample.t)
        if self.adaptive_b.observe(sample.t, evidence_b, trusted):
            self.l3b.h = self.adaptive_b.h.copy()
            self._l5_event("b", sample.t)

        fired = fired_a | fired_b
        self._since_fire = np.where(fired, 0, self._since_fire + 1)
        self._fired_prev = fired
        self._prev_state = decision.state
        return decision
