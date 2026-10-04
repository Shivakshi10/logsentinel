"""Anomaly detection models."""

from __future__ import annotations

from .base import Detector
from .isolation_forest import IsolationForestDetector
from .threshold import StaticThresholdDetector

DETECTORS = ["static_threshold", "isolation_forest", "lstm_autoencoder"]


def build_detector(name: str, **kwargs) -> Detector:
    if name == "static_threshold":
        return StaticThresholdDetector(**kwargs)
    if name == "isolation_forest":
        return IsolationForestDetector(**kwargs)
    if name == "lstm_autoencoder":
        from .lstm_autoencoder import LSTMAutoencoderDetector

        return LSTMAutoencoderDetector(**kwargs)
    raise ValueError(f"Unknown detector {name!r}; choose from {DETECTORS}")


__all__ = ["DETECTORS", "Detector", "IsolationForestDetector", "StaticThresholdDetector", "build_detector"]
