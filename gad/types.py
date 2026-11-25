from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np

VALID = 0
MISSING = 1
STALE = 2
OOR = 4
CLIPPED = 8


class DetectorState(Enum):
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    ANOMALY = "ANOMALY"


@dataclass(frozen=True)
class Sample:
    t: float          # unix seconds
    x: np.ndarray     # sensor readings
    u: np.ndarray     # commands
    q: np.ndarray     # quality tags from l0

    def __post_init__(self) -> None:
        if self.x.ndim != 1 or self.u.ndim != 1:
            raise ValueError("x and u must be 1-D")
        if self.q.shape != self.x.shape:
            raise ValueError(f"q must align with x: q.shape={self.q.shape} x.shape={self.x.shape}")
        if self.q.dtype != np.uint8:
            raise ValueError(f"q must be uint8, got {self.q.dtype}")


@dataclass(frozen=True)
class Decision:

    t: float
    state: DetectorState
    bundle_id: str #which version of the detector made the decision
