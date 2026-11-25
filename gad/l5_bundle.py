"""L5: everything the detector learned at commissioning, kept together as one
versioned unit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gad.l0_quality import QualityRanges
from gad.l1a_predictor import ModelState


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
    resolution: np.ndarray
    k_a: np.ndarray
    h_a: np.ndarray
    k_b: np.ndarray
    h_b: np.ndarray
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
