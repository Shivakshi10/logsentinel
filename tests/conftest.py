from datetime import datetime, timezone

import numpy as np
import pytest

from logsentinel import synthetic
from logsentinel.bundle import ModelBundle
from logsentinel.evaluation import best_threshold
from logsentinel.features import WindowFeaturizer, label_by_intervals, time_windows
from logsentinel.models import build_detector

START = datetime(2026, 3, 2, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def healthy():
    return synthetic.generate(START, hours=6, incidents_per_day=0, seed=1)


@pytest.fixture(scope="session")
def with_incidents(healthy):
    return synthetic.generate(healthy.end, hours=8, incidents_per_day=24, seed=2)


@pytest.fixture(scope="session")
def bundle(healthy, with_incidents) -> ModelBundle:
    """A small Isolation Forest bundle trained on 6 healthy hours."""
    train = time_windows(healthy.records, 60, healthy.start, healthy.end)
    val = time_windows(with_incidents.records, 60, with_incidents.start, with_incidents.end)
    label_by_intervals(val, with_incidents.incidents)
    featurizer = WindowFeaturizer().fit(train)
    detector = build_detector("isolation_forest", n_estimators=100).fit(
        featurizer.transform(train), featurizer.feature_names
    )
    threshold = best_threshold(detector.score(featurizer.transform(val)), np.array([w.label for w in val]))
    return ModelBundle(featurizer, detector, threshold, 60, {"detector": "isolation_forest"})
