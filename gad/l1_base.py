"""The interface both predictors (L1a and L1b) follow."""
from __future__ import annotations

from typing import Iterable, Iterator, Protocol

import numpy as np

from gad.types import Sample


class Predictor(Protocol):
    def fit(self, samples: Iterable[Sample]) -> None:
        """Commissioning fit from a batch of clean samples."""
        ...

    def update(self, sample: Sample) -> np.ndarray:
        """Streaming: consume one new sample, return x_hat for it (using
                only information available before this sample's x was observed),
                and update internal history/state for the next call.
        """
        ...

    def errors_batch(self, samples: Iterable[Sample]) -> Iterator[np.ndarray]:
        """Batch evaluation: x[t] - x_hat[t] for each sample in order,
                via the same causal update() path — for held-out error reporting,
                never used to leak information back into fit().
        """
        ...
