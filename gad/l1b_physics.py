"""L1b: predicts tank levels from mass balance (inflow minus outflow) with a
Kalman filter. A reading the physics cannot explain is ignored, for a limited
time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from gad.types import Sample

LEVEL_RATE_K = 0.18572


@dataclass(frozen=True)
class FlowSource:
    """One flow, set by the state of one actuator. `state_to_flow` gives the
    expected flow (m3/h) for each actuator state, measured on clean data. A
    state never seen at commissioning gets the mean of the known ones.
    """

    flow_channel: str
    driving_actuator: str
    state_to_flow: dict[float, float]
    note: str

    def expected(self, actuator_state: float) -> float:
        if actuator_state in self.state_to_flow:
            return self.state_to_flow[actuator_state]
        return float(np.mean(list(self.state_to_flow.values())))


@dataclass(frozen=True)
class TankBalance:
    """The level of one tank changes by k times (inflow - outflow)."""

    tank: str
    level_channel: str
    inflow: FlowSource
    outflow: FlowSource
    k: float = LEVEL_RATE_K

    @property
    def outflows(self) -> tuple[FlowSource, ...]:
        return (self.outflow,)


PHYSICS_TABLE: tuple[TankBalance, ...] = (
    TankBalance(
        tank="T101",
        level_channel="LIT101",
        inflow=FlowSource(
            "FIT101", "MV101", {0.0: 1.3423, 1.0: 0.0112, 2.0: 2.5422},
            "states 1 and 2 from the SWaT documentation, state 0 (moving) measured on clean data",
        ),
        outflow=FlowSource(
            "FIT201", "P101", {1.0: 0.0182, 2.0: 2.4414},
            "mean flow per pump state, measured on clean data",
        ),
    ),
    TankBalance(
        tank="T301",
        level_channel="LIT301",
        inflow=FlowSource(
            "FIT201", "P101", {1.0: 0.0182, 2.0: 2.4414},
            "the same flow as T101's outflow (P101 pumps T101 -> T301 through FIT201)",
        ),
        outflow=FlowSource(
            "FIT301", "P302", {1.0: 0.0797, 2.0: 2.2055},
            "mean flow per pump state, measured on clean data",
        ),
    ),
    TankBalance(
        tank="T401",
        level_channel="LIT401",
        inflow=FlowSource(
            "FIT301", "P302", {1.0: 0.0797, 2.0: 2.2055},
            "the same flow as T301's outflow (P302 pumps T301 -> T401 through FIT301)",
        ),
        outflow=FlowSource(
            "FIT401", "P402", {2.0: 1.7103},
            "P402 runs all the time in this recording, so only state 2 is known",
        ),
    ),
)


def build_state_space(
    table: tuple[TankBalance, ...] = PHYSICS_TABLE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Mechanically build (A, B, C) from `table` — every nonzero entry traces to
    exactly one TankBalance/FlowSource row. Returns (A, B, C, state_names,
    flow_channels): state_names is the order of the state vector, flow_channels
    is the order of B's input columns (a physical flow shared by two tanks —
    FIT201, FIT301 — gets exactly one column, entering with opposite signs in
    the two tanks it touches).
    """
    levels = [tb.level_channel for tb in table]

    flow_channels: list[str] = []
    for tb in table:
        for fs in (tb.inflow, *tb.outflows):
            if fs.flow_channel not in flow_channels:
                flow_channels.append(fs.flow_channel)

    state_names = list(levels)
    n = len(state_names)
    A = np.zeros((n, n))
    A[:len(levels), :len(levels)] = np.eye(len(levels))   # levels random-walk
    C = np.eye(n)

    B = np.zeros((n, len(flow_channels)))
    for i, tb in enumerate(table):
        B[i, flow_channels.index(tb.inflow.flow_channel)] += tb.k
        for fs in tb.outflows:
            B[i, flow_channels.index(fs.flow_channel)] -= tb.k

    return A, B, C, state_names, flow_channels


def steady_state_R(Q: np.ndarray, target_gain: float) -> np.ndarray:
    """R (elementwise) such that a scalar random-walk-plus-noise Kalman filter
    (A=C=1) with process noise Q converges to the given steady-state gain: R =
    Q * (1 - K) / K^2.
    """
    if not (0 < target_gain < 1):
        raise ValueError("target_gain must be in (0, 1)")
    return Q * (1.0 - target_gain) / target_gain**2


def steady_state_P(R: np.ndarray, target_gain: float) -> np.ndarray:
    """The a-posteriori covariance at the same fixed point: P = K*R."""
    return target_gain * R


class StateSpacePredictor:
    """L1b: a Kalman filter on the mass balance of the three tanks. It predicts
    only the tank levels. Every other sensor gets NaN, because the physics says
    nothing about it.
    """

    def __init__(
        self,
        sensed_channels: Iterable[str],
        commanded_channels: Iterable[str],
        table: tuple[TankBalance, ...] = PHYSICS_TABLE,
        target_gain: float = 0.1,
        innovation_gate: float | None = None,
        gate_max_hold: int | None = None,
    ) -> None:
        self.sensed_channels = tuple(sensed_channels)
        self.commanded_channels = tuple(commanded_channels)
        self.table = table
        self.target_gain = target_gain
        self.innovation_gate = innovation_gate
        self.gate_max_hold = gate_max_hold
        self._hold_run = None

        self.A, self.B, self.C, self.state_names, self.flow_channels = \
            build_state_space(table)
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
        """Calibrate Q (process noise) from the one-step mass-balance residual on
        clean data, then derive R/P0 from `target_gain`. A/B/C are untouched —
        they came from `PHYSICS_TABLE`, not from this data. Returns the per-
        channel residuals used, for reporting.
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
            self._hold_run = np.where(keep, 0, self._hold_run + 1)
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


