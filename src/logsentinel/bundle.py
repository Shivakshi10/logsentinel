"""A trained model packaged with everything needed to use it in production."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import Window, WindowFeaturizer
from .models import Detector


@dataclass
class ModelBundle:
    featurizer: WindowFeaturizer
    detector: Detector
    threshold: float
    window_seconds: int
    metadata: dict = field(default_factory=dict)

    def score_windows(self, windows: list[Window]) -> np.ndarray:
        return self.detector.score(self.featurizer.transform(windows))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(self, fh)
        tmp.replace(path)  # atomic, so a running service never reads a half-written file
        return path

    @staticmethod
    def load(path: str | Path) -> ModelBundle:
        # Only load bundles you trained yourself: pickle can execute code.
        with Path(path).open("rb") as fh:
            bundle = pickle.load(fh)
        if not isinstance(bundle, ModelBundle):
            raise TypeError(f"{path} does not contain a ModelBundle")
        return bundle
