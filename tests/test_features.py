from datetime import timedelta

import numpy as np

from logsentinel.features import Window, WindowFeaturizer, align, count_windows, label_by_intervals, time_windows
from logsentinel.parsing import parse_app_line

from .conftest import START


def _line(offset_s: float, level="INFO", service="svc", message="hello 1"):
    ts = (START + timedelta(seconds=offset_s)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return parse_app_line(f"{ts} {level} [{service}] {message}")


def test_align_rounds_down_to_window_start():
    assert align(START + timedelta(seconds=95), 60) == START + timedelta(seconds=60)


def test_time_windows_keep_empty_windows():
    records = [_line(5), _line(10), _line(200)]
    windows = time_windows(records, 60)
    assert [w.n_lines for w in windows] == [2, 0, 0, 1]
    assert windows[0].start == START


def test_count_windows():
    records = [_line(i) for i in range(50)]
    windows = count_windows(records, size=20, step=10)
    assert len(windows) == 4
    assert all(w.n_lines == 20 for w in windows)


def test_label_by_intervals():
    windows = time_windows([_line(0), _line(250)], 60)

    class Interval:
        start = START + timedelta(seconds=70)
        end = START + timedelta(seconds=100)

    label_by_intervals(windows, [Interval()])
    assert [w.label for w in windows] == [0, 1, 0, 0, 0]


def test_featurizer_uses_proportions(healthy):
    train = time_windows(healthy.records, 60)
    featurizer = WindowFeaturizer().fit(train)
    X = featurizer.transform(train)
    n_templates = len(featurizer.templates)
    assert X.shape == (len(train), len(featurizer.feature_names))
    # template shares plus the unknown share sum to 1 for non-empty windows
    unknown = featurizer.feature_names.index("share:unknown_templates")
    np.testing.assert_allclose(X[:, :n_templates].sum(axis=1) + X[:, unknown], 1.0, atol=1e-9)


def test_unseen_error_templates_and_explanations(healthy):
    featurizer = WindowFeaturizer().fit(time_windows(healthy.records, 60))
    window = Window(start=START, end=START + timedelta(minutes=1))
    for i in range(50):
        window.add(_line(i, service="api-gateway", message="Health check OK"))
    window.add(_line(51, level="ERROR", service="payment-service", message="Disk /dev/sda1 is full"))
    assert featurizer.unseen_error_templates(window) == ["Disk /dev/sda1 is full"]
    explanation = featurizer.explain(featurizer.transform_one(window))
    assert explanation and all(abs(d["z_score"]) > 2 for d in explanation)


def test_featurizer_survives_pickling(healthy):
    import pickle

    featurizer = WindowFeaturizer().fit(time_windows(healthy.records[:5000], 60))
    clone = pickle.loads(pickle.dumps(featurizer))
    window = time_windows(healthy.records[:500], 60)[0]
    np.testing.assert_array_equal(clone.transform_one(window), featurizer.transform_one(window))
