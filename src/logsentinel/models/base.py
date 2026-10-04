"""Common interface for all anomaly detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Detector(ABC):
    """An unsupervised detector. Higher scores mean "more anomalous".

    ``context`` is the number of consecutive windows the detector needs to
    score the most recent one (1 for stateless models, >1 for sequence models).
    """

    name: str = "detector"
    context: int = 1

    @abstractmethod
    def fit(self, X: np.ndarray, feature_names: Sequence[str]) -> Detector:
        """Learn what normal looks like from windows of a healthy period."""

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return one anomaly score per row. Rows must be in chronological order."""
