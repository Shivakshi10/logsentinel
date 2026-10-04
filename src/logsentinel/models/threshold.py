"""Baseline: a fixed rule on the share of error lines, like classic alerting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import Detector


class StaticThresholdDetector(Detector):
    """Scores each window by its error ratio.

    This mirrors how most alerting starts out ("alert if more than X% of
    requests fail"). The evaluation tunes the best possible X on validation
    data, so the comparison with the ML models is fair.
    """

    name = "static_threshold"

    def __init__(self, feature: str = "share:errors"):
        self.feature = feature
        self._index: int | None = None

    def fit(self, X: np.ndarray, feature_names: Sequence[str]) -> StaticThresholdDetector:
        self._index = list(feature_names).index(self.feature)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._index is None:
            raise RuntimeError("Detector must be fitted before scoring")
        return X[:, self._index].astype(float)
