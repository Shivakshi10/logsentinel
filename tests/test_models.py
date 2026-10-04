import pickle

import numpy as np
import pytest

from logsentinel.evaluation import best_threshold, incident_metrics, window_metrics
from logsentinel.features import WindowFeaturizer, label_by_intervals, time_windows
from logsentinel.models import build_detector


@pytest.fixture(scope="module")
def data(healthy, with_incidents):
    train = time_windows(healthy.records, 60, healthy.start, healthy.end)
    test = time_windows(with_incidents.records, 60, with_incidents.start, with_incidents.end)
    label_by_intervals(test, with_incidents.incidents)
    featurizer = WindowFeaturizer().fit(train)
    return featurizer, featurizer.transform(train), featurizer.transform(test), np.array([w.label for w in test]), test


def test_static_threshold_scores_error_share(data):
    featurizer, X_train, X_test, _, _ = data
    detector = build_detector("static_threshold").fit(X_train, featurizer.feature_names)
    idx = featurizer.feature_names.index("share:errors")
    np.testing.assert_array_equal(detector.score(X_test), X_test[:, idx])


def test_isolation_forest_ranks_incidents_higher(data):
    featurizer, X_train, X_test, y, _ = data
    detector = build_detector("isolation_forest", n_estimators=100).fit(X_train, featurizer.feature_names)
    scores = detector.score(X_test)
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_lstm_autoencoder_trains_scores_and_pickles(data):
    pytest.importorskip("torch")
    featurizer, X_train, X_test, y, _ = data
    detector = build_detector("lstm_autoencoder", epochs=3, hidden=16, latent=4, seq_len=5)
    detector.fit(X_train, featurizer.feature_names)
    assert detector.train_losses[-1] < detector.train_losses[0]
    scores = detector.score(X_test)
    assert scores.shape == (len(X_test),)
    assert scores[y == 1].mean() > scores[y == 0].mean()
    clone = pickle.loads(pickle.dumps(detector))
    np.testing.assert_allclose(clone.score(X_test), scores, rtol=1e-5)


def test_unknown_detector():
    with pytest.raises(ValueError):
        build_detector("magic")


def test_best_threshold_and_metrics():
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold = best_threshold(scores, labels)
    predictions = (scores >= threshold).astype(int)
    metrics = window_metrics(labels, scores, predictions)
    assert metrics["f1"] == 1.0 and metrics["roc_auc"] == 1.0


def test_incident_metrics_counts_detection_and_false_alarms(data):
    *_, test = data
    predictions = np.array([w.label for w in test])
    predictions[0] = 1  # one false alarm at the very start
    with_incidents_intervals = [type("I", (), {"start": w.start, "end": w.end, "kind": "x"}) for w in test if w.label][:1]
    result = incident_metrics(test, predictions, with_incidents_intervals)
    assert result["incident_recall"] == 1.0
    assert result["false_alarms_per_day"] > 0
