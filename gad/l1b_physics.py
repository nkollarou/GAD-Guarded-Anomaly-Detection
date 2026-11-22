"""L1b: predicts tank levels from mass balance (inflow minus outflow) with a
Kalman filter. A reading the physics cannot explain is ignored, for a
limited time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from gad.types import Sample

LEVEL_RATE_K = 0.18572


@dataclass(frozen=True)
class FlowSource:
    """One physical flow, driven by one commanded actuator's discrete
        state. `state_to_flow` maps actuator state -> expected flow rate
        (m3/h). An actuator state not present in the table (never observed at
        commissioning) falls back to the mean of the states that were —
        a documented compromise, not a crash or a silent zero.
    """

    flow_channel: str
    driving_actuator: str
    state_to_flow: dict[float, float]
    provenance: str
    bin_edges: tuple[float, ...] = ()

    def state(self, actuator_state: float) -> float:
        """The raw command mapped to the operating state `state_to_flow` is
                keyed on — the command itself when the actuator is already discrete.
                Bin 0 is "commanded shut" (exactly zero), an exact state and not a
                quantile; bins 1..K-1 are the edges above. A non-finite command
                returns NaN, which misses the table and takes the documented pooled
                fallback below rather than silently landing in the top bin.
        """
        if not self.bin_edges:
            return actuator_state
        if not np.isfinite(actuator_state):
            return float("nan")
        if actuator_state == 0.0:
            return 0.0
        return float(np.searchsorted(np.asarray(self.bin_edges), actuator_state,
                                     side="right") + 1)

    def expected(self, actuator_state: float) -> float:
        key = self.state(actuator_state)
        if key in self.state_to_flow:
            return self.state_to_flow[key]
        return float(np.mean(list(self.state_to_flow.values())))


@dataclass(frozen=True)
class TankBalance:
    tank: str
    level_channel: str
    inflow: FlowSource
    outflow: FlowSource
    extra_outflows: tuple[FlowSource, ...] = ()
    k: float = LEVEL_RATE_K

    @property
    def outflows(self) -> tuple[FlowSource, ...]:
        return (self.outflow, *self.extra_outflows)


PHYSICS_TABLE: tuple[TankBalance, ...] = (
    TankBalance(
        tank="T101",
        level_channel="LIT101",
        inflow=FlowSource(
            "FIT101", "MV101", {0.0: 1.3423, 1.0: 0.0112, 2.0: 2.5422},
            "state1/state2 documented (Part 0 brief: 0.011/2.543); state0 (transitioning) "
            "measured on the steady-state Normal window, same commissioning data.",
        ),
        outflow=FlowSource(
            "FIT201", "P101", {1.0: 0.0182, 2.0: 2.4414},
            "measured on the steady-state Normal window with gad.swat."
            "actuator_encoding_means — the same method the brief documents for MV101/FIT101, "
            "extended here because no separate datasheet value was given for this pump.",
        ),
    ),
    TankBalance(
        tank="T301",
        level_channel="LIT301",
        inflow=FlowSource(
            "FIT201", "P101", {1.0: 0.0182, 2.0: 2.4414},
            "same physical flow as T101's outflow above (P101 pumps T101 -> T301 through the FIT201 meter).",
        ),
        outflow=FlowSource(
            "FIT301", "P302", {1.0: 0.0797, 2.0: 2.2055},
            "measured on the steady-state Normal window, same method as MV101/FIT101.",
        ),
    ),
    TankBalance(
        tank="T401",
        level_channel="LIT401",
        inflow=FlowSource(
            "FIT301", "P302", {1.0: 0.0797, 2.0: 2.2055},
            "same physical flow as T301's outflow above (P302 pumps T301 -> T401 through the FIT301 meter).",
        ),
        outflow=FlowSource(
            "FIT401", "P402", {2.0: 1.7103},
            "measured on the steady-state Normal window. P402 is observed only at state 2 "
            "(continuously running) throughout this recording — see docs/decisions/0009 for "
            "why that's a real plant-operation fact here, not a data gap.",
        ),
    ),
)


def build_state_space(
    table: tuple[TankBalance, ...] = PHYSICS_TABLE,
    include_flows: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Mechanically build (A, B, C) from `table` — every nonzero entry
        traces to exactly one TankBalance/FlowSource row. Returns
        (A, B, C, state_names, flow_channels): state_names is the order of the
        state vector, flow_channels is the order of B's input columns
        (a physical flow shared by two tanks — FIT201, FIT301 — gets exactly
        one column, entering with opposite signs in the two tanks it touches).
    """
    levels = [tb.level_channel for tb in table]

    flow_channels: list[str] = []
    for tb in table:
        for fs in (tb.inflow, *tb.outflows):
            if fs.flow_channel not in flow_channels:
                flow_channels.append(fs.flow_channel)

    state_names = levels + (list(flow_channels) if include_flows else [])
    n = len(state_names)
    A = np.zeros((n, n))
    A[:len(levels), :len(levels)] = np.eye(len(levels))   # levels random-walk
    C = np.eye(n)

    B = np.zeros((n, len(flow_channels)))
    for i, tb in enumerate(table):
        B[i, flow_channels.index(tb.inflow.flow_channel)] += tb.k
        for fs in tb.outflows:
            B[i, flow_channels.index(fs.flow_channel)] -= tb.k
    if include_flows:
        for j, fc in enumerate(flow_channels):
            B[len(levels) + j, j] = 1.0

    return A, B, C, state_names, flow_channels


def steady_state_R(Q: np.ndarray, target_gain: float) -> np.ndarray:
    """R (elementwise) such that a scalar random-walk-plus-noise Kalman
        filter (A=C=1) with process noise Q converges to the given steady-state
        gain: R = Q * (1 - K) / K^2.
    """
    if not (0 < target_gain < 1):
        raise ValueError("target_gain must be in (0, 1)")
    return Q * (1.0 - target_gain) / target_gain**2


def steady_state_P(R: np.ndarray, target_gain: float) -> np.ndarray:
    """The a-posteriori covariance at the same fixed point: P = K*R."""
    return target_gain * R


def innovation_diagnostics(innovations: np.ndarray, lags: int = 20) -> dict:
    """Per-channel mean, std, lag-1 autocorrelation, and a Ljung-Box
        whiteness test on a sequence of innovations (shape (T, n)). On clean
        data these should be ~zero-mean and ~white; significant autocorrelation
        means the model is structurally wrong (a missing dynamic term) — this
        is a report-only diagnostic, nothing here adjusts the model.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    T, n = innovations.shape
    report = {"mean": [], "std": [], "lag1_autocorr": [], "ljung_box_pvalue": []}
    for j in range(n):
        series = innovations[:, j]
        report["mean"].append(float(np.mean(series)))
        report["std"].append(float(np.std(series)))
        report["lag1_autocorr"].append(float(np.corrcoef(series[:-1], series[1:])[0, 1]))
        lb = acorr_ljungbox(series, lags=[lags], return_df=True)
        report["ljung_box_pvalue"].append(float(lb["lb_pvalue"].iloc[0]))
    return report


class StateSpacePredictor:
    """L1b: implements the `Predictor` protocol. Only predicts the 3
        channels named in `PHYSICS_TABLE`'s level_channel entries — every
        other sensed channel comes back NaN (abstention, never a pooled
        fallback), since this branch has no physics for them.
    """

    def __init__(
        self,
        sensed_channels: Iterable[str],
        commanded_channels: Iterable[str],
        table: tuple[TankBalance, ...] = PHYSICS_TABLE,
        target_gain: float = 0.1,
        innovation_gate: float | None = None,
        gate_max_hold: int | None = None,
        include_flows: bool = False,
    ) -> None:
        self.sensed_channels = tuple(sensed_channels)
        self.commanded_channels = tuple(commanded_channels)
        self.table = table
        self.target_gain = target_gain
        self.innovation_gate = innovation_gate
        self.gate_max_hold = gate_max_hold
        self._hold_run = None
        self.n_gated = None
        #: how often the bound above forced a channel back on
        self.n_forced = None
        self.n_seen = 0

        self.include_flows = include_flows
        self.A, self.B, self.C, self.state_names, self.flow_channels = \
            build_state_space(table, include_flows=include_flows)
        self.n = len(self.state_names)

        self._state_sensed_idx = [self.sensed_channels.index(name) for name in self.state_names]
        self._flow_sources = {fs.flow_channel: fs
                              for tb in table for fs in (tb.inflow, *tb.outflows)}
        self._actuator_idx = {
            fs.driving_actuator: self.commanded_channels.index(fs.driving_actuator)
            for fs in self._flow_sources.values()
        }

        self.Q: np.ndarray | None = None
        self.R: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.x_hat: np.ndarray | None = None

    def _u_effective(self, u_raw: np.ndarray) -> np.ndarray:
        out = np.empty(len(self.flow_channels))
        for j, fc in enumerate(self.flow_channels):
            fs = self._flow_sources[fc]
            out[j] = fs.expected(u_raw[self._actuator_idx[fs.driving_actuator]])
        return out

    def fit_from_arrays(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Calibrate Q (process noise) from the one-step mass-balance
                residual on clean data, then derive R/P0 from `target_gain`.
                A/B/C are untouched — they came from `PHYSICS_TABLE`, not from
                this data. Returns the per-channel residuals used, for reporting.
        """
        state = X[:, self._state_sensed_idx]
        u_eff = np.stack([self._u_effective(U[t]) for t in range(len(U))])
        predicted = u_eff @ self.B.T  # (T, n): B @ u_eff per row
        residual = state[1:] - (state[:-1] @ self.A.T + predicted[1:])

        Q = residual.var(axis=0)
        R = steady_state_R(Q, self.target_gain)
        P0 = steady_state_P(R, self.target_gain)

        self.Q, self.R = np.diag(Q), np.diag(R)
        self.P = np.diag(P0)
        self.x_hat = None
        return residual

    def fit_from_segments(self, segments) -> np.ndarray:
        """Calibrate Q from the one-step mass-balance residual over several
                contiguous blocks, never differencing across a gap.
        """
        residuals = []
        for X, U in segments:
            X = np.asarray(X, dtype=np.float64)
            U = np.asarray(U, dtype=np.float64)
            if len(X) < 2:
                continue
            state = X[:, self._state_sensed_idx]
            u_eff = np.stack([self._u_effective(U[t]) for t in range(len(U))])
            residuals.append(np.diff(state, axis=0) - (u_eff @ self.B.T)[1:])
        if not residuals:
            raise ValueError("no segment long enough to form a mass-balance residual")
        residual = np.concatenate(residuals)

        Q = residual.var(axis=0)
        R = steady_state_R(Q, self.target_gain)
        P0 = steady_state_P(R, self.target_gain)
        self.Q, self.R = np.diag(Q), np.diag(R)
        self.P = np.diag(P0)
        self.x_hat = None
        return residual

    def fit(self, samples: Iterable[Sample]) -> None:
        samples = list(samples)
        if not samples:
            raise ValueError("fit requires at least one sample")
        X = np.stack([s.x for s in samples])
        U = np.stack([s.u for s in samples])
        self.fit_from_arrays(X, U)

    def update(self, sample: Sample) -> np.ndarray:
        if self.Q is None:
            raise RuntimeError("update() called before fit()")

        x_hat_full = np.full(len(self.sensed_channels), np.nan)

        if self.x_hat is None:
            self.x_hat = sample.x[self._state_sensed_idx].copy()
            return x_hat_full

        u_eff = self._u_effective(sample.u)
        x_pred = self.A @ self.x_hat + self.B @ u_eff
        P_pred = self.A @ self.P @ self.A.T + self.Q

        x_hat_full[self._state_sensed_idx] = self.C @ x_pred

        z = sample.x[self._state_sensed_idx]
        innovation = z - self.C @ x_pred
        S = self.C @ P_pred @ self.C.T + self.R

        if self.n_gated is None:
            self.n_gated = np.zeros(self.C.shape[0], dtype=np.int64)
            self.n_forced = np.zeros(self.C.shape[0], dtype=np.int64)
        self.n_seen += 1

        keep = np.ones(self.C.shape[0], dtype=bool)
        if self._hold_run is None:
            self._hold_run = np.zeros(self.C.shape[0], dtype=np.int64)
        if self.innovation_gate is not None:
            sd = np.sqrt(np.maximum(np.diag(S), 0.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                standardised = np.abs(innovation) / sd
            keep = ~(np.isfinite(standardised)
                     & (standardised > float(self.innovation_gate)))
            if self.gate_max_hold is not None:
                forced = ~keep & (self._hold_run >= int(self.gate_max_hold))
                keep |= forced
                self.n_forced += forced
            self._hold_run = np.where(keep, 0, self._hold_run + 1)
            self.n_gated += ~keep
            # NaN readings were never usable; they are not the gate's doing
            keep &= np.isfinite(innovation)

        if not keep.all():
            idx = np.flatnonzero(keep)
            if idx.size == 0:
                self.x_hat, self.P = x_pred, P_pred
                return x_hat_full
            C_k = self.C[idx]
            S_k = C_k @ P_pred @ C_k.T + self.R[np.ix_(idx, idx)]
            K_k = P_pred @ C_k.T @ np.linalg.inv(S_k)
            self.x_hat = x_pred + K_k @ innovation[idx]
            self.P = (np.eye(self.n) - K_k @ C_k) @ P_pred
            return x_hat_full

        K = P_pred @ self.C.T @ np.linalg.inv(S)

        self.x_hat = x_pred + K @ innovation
        self.P = (np.eye(self.n) - K @ self.C) @ P_pred

        return x_hat_full

    def gate_report(self) -> dict:
        """What the gate did, for the commissioning report. Rejecting clean
                measurements costs tracking accuracy, so the rate is reported per
                channel rather than assumed negligible.
        """
        if self.n_gated is None or self.n_seen == 0:
            return {"gate": self.innovation_gate, "n_seen": int(self.n_seen),
                    "max_hold": self.gate_max_hold, "rejected": {}}
        return {
            "gate": self.innovation_gate,
            "max_hold": self.gate_max_hold,
            "n_seen": int(self.n_seen),
            "rejected": {name: int(c) for name, c in
                         zip(self.state_names, self.n_gated)},
            "rejected_fraction": {name: float(c / self.n_seen) for name, c in
                                  zip(self.state_names, self.n_gated)},
            "forced_reacquire": {name: int(c) for name, c in
                                 zip(self.state_names,
                                     self.n_forced if self.n_forced is not None
                                     else np.zeros(len(self.state_names)))},
        }

    def errors_batch(self, samples: Iterable[Sample]) -> Iterator[np.ndarray]:
        for sample in samples:
            x_hat = self.update(sample)
            yield sample.x - x_hat
