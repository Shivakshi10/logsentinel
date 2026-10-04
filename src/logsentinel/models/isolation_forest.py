"""Isolation Forest: isolates unusual windows with random splits."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import Detector


class IsolationForestDetector(Detector):
    """Anomalies are easier to separate from the rest, so they need fewer
    random splits to isolate. Fast, robust and a strong default.

    Limitation worth knowing: a feature that was constant during training
    (e.g. "unseen error templates" is always 0 on healthy data) can never be
    used for a split, so novel log messages are invisible to this model. The
    streaming detector covers that gap with an explicit novelty rule.
    """

    name = "isolation_forest"

    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    def fit(self, X: np.ndarray, feature_names: Sequence[str]) -> IsolationForestDetector:
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)
