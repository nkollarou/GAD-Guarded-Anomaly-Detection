"""L5: everything the detector has learned, versioned as one unit so it can
be stored and restored.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from gad.l2_context import per_channel
from gad.detector import Detector
from gad.l2_residual import RegimeScaledResidual
from gad.l1a_invariants import InvariantState
from gad.l1a_predictor import ModelState
from gad.l0_quality import QualityChecker, QualityRanges

Windows = tuple[dict[int, tuple[float, ...]], ...]


def _frozen(a: np.ndarray) -> np.ndarray:
    out = np.array(a, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False)
class Bundle:
    version: int
    quality_ranges: QualityRanges
    l1a_state: ModelState
    l1b_Q: np.ndarray
    l1b_R: np.ndarray
    l1b_P0: np.ndarray
    invariant_state: InvariantState
    resolution: np.ndarray
    l2a_windows: Windows
    l2b_windows: Windows
    k_a: np.ndarray
    h_a: np.ndarray
    k_b: np.ndarray
    h_b: np.ndarray
    commanded_states: dict[str, tuple[float, ...]]
    l2a_state: tuple | None = None
    l2b_state: tuple | None = None
    h_a_by_context: dict[int, np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("bundle version must be >= 1")
        for name in ("l1b_Q", "l1b_R", "l1b_P0", "resolution", "k_a", "h_a", "k_b", "h_b"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        if self.h_a_by_context is not None:
            object.__setattr__(self, "h_a_by_context",
                               {int(c): _frozen(np.asarray(v, dtype=np.float64))
                                for c, v in self.h_a_by_context.items()})

    @property
    def bundle_id(self) -> str:
        return f"detector-v{self.version}"

    def n_bytes(self) -> int:
        """Approximate serialised size of the learned numbers."""
        arrays = (self.l1b_Q, self.l1b_R, self.l1b_P0, self.resolution,
                  self.k_a, self.h_a, self.k_b, self.h_b)
        windows = sum(len(w) for per in self.l2a_windows + self.l2b_windows for w in per.values())
        return (
            self.l1a_state.n_bytes()
            + self.invariant_state.n_bytes()
            + sum(a.nbytes for a in arrays)
            + windows * 8
            + len(self.quality_ranges.ranges) * 5 * 8
        )


def _check_windows(windows: Windows, n: int, label: str) -> None:
    if len(windows) != n:
        raise ValueError(f"{label}: expected windows for {n} channels, got {len(windows)}")


def _check_vector(a: np.ndarray, n: int, label: str) -> None:
    if a.shape != (n,):
        raise ValueError(f"{label}: expected shape ({n},), got {a.shape}")


def validate(bundle: Bundle, detector: Detector, checker: QualityChecker | None) -> None:
    """Raise if `bundle` does not fit `detector`'s fixed design. Called
        before anything is changed, so a rejected bundle leaves the detector
        exactly as it was.
    """
    sensed = detector.sensed_channels
    n = len(sensed)
    if bundle.quality_ranges.channels != sensed:
        raise ValueError("bundle quality ranges cover different channels")
    if checker is not None and len(bundle.quality_ranges.ranges) != len(checker._ranges):
        raise ValueError("bundle quality ranges do not match the checker")
    s = bundle.l1a_state
    if s.sensed_channels != detector.l1a.sensed_channels or s.commanded_channels != detector.l1a.commanded_channels:
        raise ValueError("bundle L1a state covers different channels")
    if s.L != detector.l1a.L:
        raise ValueError("bundle L1a lag length differs from the fixed design")
    m = detector.l1b.n
    for name in ("l1b_Q", "l1b_R", "l1b_P0"):
        if getattr(bundle, name).shape != (m, m):
            raise ValueError(f"{name}: expected shape ({m}, {m})")
    inv = bundle.invariant_state
    if inv.sensed_channels != sensed:
        raise ValueError("bundle invariants cover different channels")
    if detector.invariants.state is not None and (
        len(inv.known_stage_configs) != len(detector.invariants.state.known_stage_configs)
    ):
        raise ValueError("bundle invariants use a different stage partition")
    _check_vector(bundle.resolution, n, "resolution")
    _check_windows(bundle.l2a_windows, n, "l2a")
    _check_windows(bundle.l2b_windows, n, "l2b")
    for name in ("k_a", "h_a", "k_b", "h_b"):
        _check_vector(getattr(bundle, name), n, name)
    if detector.evidence is not None and set(bundle.commanded_states) != set(detector.evidence.commanded):
        raise ValueError("bundle commanded states cover different actuators")


def _current_scales(l2, regimes: np.ndarray) -> np.ndarray:
    out = np.full(l2.n_channels, np.nan)
    for i in range(l2.n_channels):
        s = l2.current_scale(i, int(regimes[i]))
        if s is not None:
            out[i] = s
    return out


def _scales_from_windows(l2, windows: Windows, floor: np.ndarray, regimes: np.ndarray) -> np.ndarray:
    """What `current_scale` would report once `windows` and `floor` were
        loaded, without loading them (keeps `apply` all-or-nothing). The
        candidate's own floor is used: it is part of the bundle, and the scale
        it produces is what the rescale factor must be measured against.
    """
    probe = RegimeScaledResidual(
        n_channels=l2.n_channels, window_size=l2.window_size,
        min_calibration=l2.min_calibration, min_scale=l2.min_scale, scale_floor=floor,
    )
    out = np.full(l2.n_channels, np.nan)
    for i in range(l2.n_channels):
        vals = windows[i].get(int(regimes[i]))
        if vals is None:
            continue
        cs = probe._center_scale(i, deque(vals, maxlen=l2.window_size))
        if cs is not None:
            out[i] = cs[1]
    return out


def _new_scales(l2, windows: Windows, state, floor: np.ndarray, regimes: np.ndarray) -> np.ndarray:
    """The scale each channel will have in its current context once the
        bundle is applied, from whichever L2 state the bundle carries.
    """
    if state is None:
        return _scales_from_windows(l2, windows, floor, regimes)
    out = np.full(l2.n_channels, np.nan)
    for i in range(l2.n_channels):
        entry = state[i].get(int(regimes[i]))
        if entry is not None:
            out[i] = entry[1]
    return out


def apply(
    bundle: Bundle,
    detector: Detector,
    checker: QualityChecker | None,
    u_now: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Swap `bundle` into the live `detector` (and L0 `checker`), keeping
        runtime state. `u_now` is the current commanded vector, which fixes the
        context the CUSUM rescaling is computed in; None skips the rescaling.
    """
    validate(bundle, detector, checker)

    n = len(detector.sensed_channels)
    factor_a = np.full(n, np.nan)
    factor_b = np.full(n, np.nan)
    if u_now is not None:
        regimes = per_channel(detector.regime_fn(u_now), n)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor_a = _current_scales(detector.l2a, regimes) / _new_scales(
                detector.l2a, bundle.l2a_windows, bundle.l2a_state, bundle.resolution, regimes)
            factor_b = _current_scales(detector.l2b, regimes) / _new_scales(
                detector.l2b, bundle.l2b_windows, bundle.l2b_state, bundle.resolution, regimes)

    # --- nothing below may raise on a validated bundle -------------------
    if checker is not None:
        checker.set_ranges(bundle.quality_ranges)
    detector.l1a.state = bundle.l1a_state
    detector.l1b.Q = np.array(bundle.l1b_Q)
    detector.l1b.R = np.array(bundle.l1b_R)
    detector.l1b.P = np.array(bundle.l1b_P0)
    detector.invariants.state = bundle.invariant_state
    for l2, windows, state in ((detector.l2a, bundle.l2a_windows, bundle.l2a_state),
                               (detector.l2b, bundle.l2b_windows, bundle.l2b_state)):
        l2.scale_floor = np.array(bundle.resolution)
        if state is not None:
            l2.load_state(state)
        else:
            l2.load_windows(windows)
    detector.l3a.rescale(factor_a)
    detector.l3b.rescale(factor_b)
    detector.l3a.k, detector.l3a.h = np.array(bundle.k_a), np.array(bundle.h_a)
    detector.l3b.k, detector.l3b.h = np.array(bundle.k_b), np.array(bundle.h_b)
    if detector.evidence is not None:
        detector.evidence.commanded_states = bundle.commanded_states
    detector.decider.bundle_id = bundle.bundle_id
    return {"l1a": factor_a, "l1b": factor_b}


class BundleHistory:
    """The committed bundles, newest last, keeping at most `keep`. The
        newest is the one in force; `rollback` re-applies the one before it.
    """

    def __init__(self, initial: Bundle, keep: int = 5) -> None:
        if keep < 2:
            raise ValueError("keep must be >= 2 so there is something to roll back to")
        self._bundles: deque[Bundle] = deque([initial], maxlen=keep)

    @property
    def current(self) -> Bundle:
        return self._bundles[-1]

    def __len__(self) -> int:
        return len(self._bundles)

    def next_version(self) -> int:
        return self.current.version + 1

    def commit(self, bundle: Bundle, detector: Detector, checker: QualityChecker | None,
               u_now: np.ndarray | None) -> dict[str, np.ndarray]:
        if bundle.version <= self.current.version:
            raise ValueError("a committed bundle must have a newer version than the one in force")
        factors = apply(bundle, detector, checker, u_now)
        self._bundles.append(bundle)
        return factors

    def rollback(self, detector: Detector, checker: QualityChecker | None,
                 u_now: np.ndarray | None) -> Bundle:
        """Re-apply the previous bundle exactly and drop the current one."""
        if len(self._bundles) < 2:
            raise RuntimeError("no earlier bundle to roll back to")
        previous = self._bundles[-2]
        apply(previous, detector, checker, u_now)
        self._bundles.pop()
        return previous
