"""L5, rule 4: the golden set, a fixed recording of clean operation captured
once and never updated automatically.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from gad.l5_bundle import Bundle
from gad.fit import BundleConfig, detector_from_bundle
from gad.l0_quality import QualityChecker
from gad.l2_residual import robust_scale
from gad.types import DetectorState, Sample

DEFAULT_WARMUP = 120
#: probe geometry, in samples
DEFAULT_PROBE_RAMP = 300
DEFAULT_PROBE_LENGTH = 900


@dataclass(frozen=True)
class GoldenSet:
    """Frozen clean reference. `X`/`U` are (n_excerpts, length, n_channels)."""

    X: np.ndarray
    U: np.ndarray
    sensed: tuple[str, ...]
    commanded: tuple[str, ...]
    sigma: np.ndarray  # per-channel clean spread, fixed with the set
    source: str  # where these rows came from, in words
    captured_at: str
    digest: str

    def __post_init__(self) -> None:
        if self.X.ndim != 3 or self.U.ndim != 3:
            raise ValueError("X and U must be (n_excerpts, length, n_channels)")
        if self.X.shape[:2] != self.U.shape[:2]:
            raise ValueError("X and U must have the same excerpts and length")
        if self.X.shape[2] != len(self.sensed) or self.U.shape[2] != len(self.commanded):
            raise ValueError("X/U channel counts do not match the names")
        if self.sigma.shape != (len(self.sensed),):
            raise ValueError("sigma must have one entry per sensed channel")
        for a in (self.X, self.U, self.sigma):
            a.setflags(write=False)  # the golden set is frozen, literally

    @property
    def n_excerpts(self) -> int:
        return self.X.shape[0]

    @property
    def length(self) -> int:
        return self.X.shape[1]

    def n_samples(self) -> int:
        return self.X.shape[0] * self.X.shape[1]

    def verify(self) -> bool:
        """True when the rows still hash to the digest recorded at capture."""
        return _digest(self.X, self.U) == self.digest


def _digest(X: np.ndarray, U: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in (np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(U, dtype=np.float64)):
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def capture(
    X: np.ndarray,
    U: np.ndarray,
    sensed, commanded,
    starts, length: int,
    source: str,
    captured_at: str,
    min_sigma: float = 1e-6,
    sigma: np.ndarray | None = None,
) -> GoldenSet:
    """Cut `len(starts)` excerpts of `length` rows from clean `X`/`U` and
        freeze them. The caller states `source` and `captured_at` for the
        record; this function never chooses rows for itself.
    """
    starts = [int(s) for s in starts]
    if not starts:
        raise ValueError("a golden set needs at least one excerpt")
    if length < 1:
        raise ValueError("length must be >= 1")
    for s in starts:
        if s < 0 or s + length > len(X):
            raise ValueError(f"excerpt [{s}, {s + length}) does not fit in {len(X)} rows")
    Xg = np.stack([np.array(X[s:s + length], dtype=np.float64) for s in starts])
    Ug = np.stack([np.array(U[s:s + length], dtype=np.float64) for s in starts])
    if not np.isfinite(Xg).all():
        raise ValueError("a golden set must not contain missing readings")

    if sigma is None:
        flat = Xg.reshape(-1, Xg.shape[2])
        sigma = np.empty(flat.shape[1])
        for j in range(flat.shape[1]):
            col = flat[:, j]
            s = robust_scale(np.abs(col - np.median(col)), min_sigma)
            sigma[j] = s if s is not None else min_sigma
    else:
        sigma = np.asarray(sigma, dtype=np.float64).copy()
        if sigma.shape != (Xg.shape[2],):
            raise ValueError("sigma must have one entry per sensed channel")
        sigma[~np.isfinite(sigma) | (sigma <= 0)] = min_sigma
    return GoldenSet(Xg, Ug, tuple(sensed), tuple(commanded), sigma,
                     source=source, captured_at=captured_at, digest=_digest(Xg, Ug))


@dataclass(frozen=True)
class Probe:
    excerpt: int
    channel: int
    size_sigma: float
    start: int
    ramp: int = DEFAULT_PROBE_RAMP
    length: int = DEFAULT_PROBE_LENGTH
    shape: str = "step"

    def __post_init__(self) -> None:
        if self.shape not in ("step", "ramp"):
            raise ValueError("shape must be 'step' or 'ramp'")


def probe_plan(
    golden: GoldenSet,
    sizes=(4.0,),
    warmup: int = DEFAULT_WARMUP,
    ramp: int = DEFAULT_PROBE_RAMP,
    length: int = DEFAULT_PROBE_LENGTH,
    shape: str = "step",
) -> tuple[Probe, ...]:
    """One probe per (channel, size), spread round-robin over the excerpts
        so no excerpt carries them all. Deterministic: same golden set and
        sizes, same plan.
    """
    if warmup + length > golden.length:
        raise ValueError("excerpts are too short for a probe of this length")
    start = warmup
    probes = []
    i = 0
    for size in sizes:
        for channel in range(len(golden.sensed)):
            probes.append(Probe(excerpt=i % golden.n_excerpts, channel=channel,
                                size_sigma=float(size), start=start, ramp=ramp,
                                length=length, shape=shape))
            i += 1
    return tuple(probes)


def probe_plan_per_channel(
    golden: GoldenSet,
    size_by_channel: dict[str, float],
    warmup: int = DEFAULT_WARMUP,
    ramp: int = DEFAULT_PROBE_RAMP,
    length: int = DEFAULT_PROBE_LENGTH,
    shape: str = "step",
) -> tuple[Probe, ...]:
    """One probe per named channel, each at its own size — the size that
        channel's own detection threshold calls for.
    """
    if warmup + length > golden.length:
        raise ValueError("excerpts are too short for a probe of this length")
    probes = []
    for i, (channel, size) in enumerate(sorted(size_by_channel.items())):
        if channel not in golden.sensed:
            raise ValueError(f"{channel} is not a sensed channel of this golden set")
        probes.append(Probe(excerpt=i % golden.n_excerpts, channel=golden.sensed.index(channel),
                            size_sigma=float(size), start=warmup, ramp=ramp,
                            length=length, shape=shape))
    return tuple(probes)


def with_probe(golden: GoldenSet, probe: Probe) -> np.ndarray:
    """The probe's excerpt with its bias applied — a copy; the golden set
        itself never changes.
    """
    X = np.array(golden.X[probe.excerpt])
    offset = probe.size_sigma * golden.sigma[probe.channel]
    t = np.arange(probe.length)
    profile = np.ones(probe.length) if probe.shape == "step" else np.minimum(1.0, t / max(probe.ramp, 1))
    X[probe.start:probe.start + probe.length, probe.channel] += offset * profile
    return X


@dataclass(frozen=True)
class GoldenReport:
    bundle_id: str
    n_scored: int
    n_anomaly: int
    n_episodes: int
    n_probes: int
    n_probes_detected: int
    detected: tuple[bool, ...]

    @property
    def time_in_anomaly(self) -> float:
        return self.n_anomaly / self.n_scored if self.n_scored else float("nan")

    def episodes_per_day(self, samples_per_day: float) -> float:
        return self.n_episodes / (self.n_scored / samples_per_day) if self.n_scored else float("nan")


def _stream(det, checker, X, U, warmup: int) -> np.ndarray:
    """States for rows `warmup:`; the first `warmup` rows are streamed but
        not scored (cold lag history, cold Kalman estimate).
    """
    states = np.empty(len(X) - warmup, dtype=np.int8)
    for i in range(len(X)):
        q = checker.update(X[i]) if checker is not None else np.zeros(X.shape[1], dtype=np.uint8)
        state = det.update(Sample(t=float(i), x=X[i], u=U[i], q=q)).state
        if i >= warmup:
            states[i - warmup] = 1 if state == DetectorState.ANOMALY else 0
    return states


def evaluate(
    cfg: BundleConfig,
    bundle: Bundle,
    golden: GoldenSet,
    probes: tuple[Probe, ...] = (),
    warmup: int = DEFAULT_WARMUP,
    use_quality: bool = True,
) -> GoldenReport:
    """Score `bundle` on the golden set: how much it alarms on clean data,
        and how many probes it catches. Each excerpt and each probe is streamed
        by its own cold detector built from the bundle, so the result depends
        on the bundle alone and never on the live detector's runtime state.
    """
    if golden.sensed != cfg.sensed:
        raise ValueError("golden set covers different sensed channels")

    n_anomaly = n_scored = n_episodes = 0
    for e in range(golden.n_excerpts):
        det = detector_from_bundle(cfg, bundle)
        checker = QualityChecker(bundle.quality_ranges) if use_quality else None
        states = _stream(det, checker, golden.X[e], golden.U[e], warmup)
        n_scored += len(states)
        n_anomaly += int(states.sum())
        n_episodes += int(np.count_nonzero(np.diff(np.concatenate(([0], states))) == 1))

    detected = []
    for probe in probes:
        det = detector_from_bundle(cfg, bundle)
        checker = QualityChecker(bundle.quality_ranges) if use_quality else None
        X = with_probe(golden, probe)
        end = probe.start + probe.length
        states = _stream(det, checker, X[:end], golden.U[probe.excerpt][:end], warmup)
        # detected when it alarms anywhere from the probe's onset onwards
        detected.append(bool(states[probe.start - warmup:].any()))

    return GoldenReport(
        bundle_id=bundle.bundle_id, n_scored=n_scored, n_anomaly=n_anomaly, n_episodes=n_episodes,
        n_probes=len(probes), n_probes_detected=int(sum(detected)), detected=tuple(detected),
    )


@dataclass(frozen=True)
class RegressionVerdict:
    admitted: bool
    reasons: tuple[str, ...]
    incumbent: GoldenReport
    candidate: GoldenReport


def regression_test(
    incumbent: GoldenReport,
    candidate: GoldenReport,
    delta_fa: float,
    delta_sens: int,
) -> RegressionVerdict:
    """Gate 1c: the candidate must be no noisier than the incumbent on
        clean data (within `delta_fa`, a fraction of time in ANOMALY) and no
        less sensitive to the probes (within `delta_sens` probes).
    """
    if delta_fa < 0 or delta_sens < 0:
        raise ValueError("tolerances must be >= 0")
    if incumbent.n_probes != candidate.n_probes:
        raise ValueError("reports must come from the same probe plan")

    reasons = []
    noisier = candidate.time_in_anomaly - incumbent.time_in_anomaly
    if noisier > delta_fa:
        reasons.append(
            f"noisier on the golden set: {candidate.time_in_anomaly:.4%} vs "
            f"{incumbent.time_in_anomaly:.4%}, tolerance {delta_fa:.4%}"
        )
    lost = incumbent.n_probes_detected - candidate.n_probes_detected
    if lost > delta_sens:
        reasons.append(
            f"less sensitive: catches {candidate.n_probes_detected}/{candidate.n_probes} probes vs "
            f"{incumbent.n_probes_detected}, tolerance {delta_sens}"
        )
    return RegressionVerdict(not reasons, tuple(reasons), incumbent, candidate)
