"""Group log records into windows and turn each window into a feature vector.

Features are *proportions* rather than raw counts, so the model is not fooled
by normal daily traffic cycles: 30 errors in a busy minute of 3,000 lines is
fine, 30 errors in a quiet minute of 60 lines is not.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from .parsing import ERROR_LEVELS, LogRecord

MAX_SAMPLES = 5


@dataclass
class Window:
    """Aggregated statistics for the log lines that fall in one window."""

    start: datetime
    end: datetime
    templates: Counter = field(default_factory=Counter)
    services: Counter = field(default_factory=Counter)
    template_levels: dict[str, str] = field(default_factory=dict)
    n_lines: int = 0
    n_errors: int = 0
    n_warnings: int = 0
    label: int = 0
    samples: list[str] = field(default_factory=list)
    warning_samples: list[str] = field(default_factory=list)

    def example_lines(self) -> list[str]:
        """A few raw lines worth showing a human: errors first, then warnings."""
        return (self.samples + self.warning_samples)[:MAX_SAMPLES]

    def add(self, record: LogRecord) -> None:
        self.n_lines += 1
        self.templates[record.template] += 1
        self.services[record.service] += 1
        self.template_levels.setdefault(record.template, record.level)
        if record.is_error:
            self.n_errors += 1
            if len(self.samples) < MAX_SAMPLES:
                self.samples.append(record.raw)
        elif record.is_warning:
            self.n_warnings += 1
            if len(self.warning_samples) < MAX_SAMPLES:
                self.warning_samples.append(record.raw)
        self.label = max(self.label, record.label)


def align(ts: datetime, seconds: int) -> datetime:
    """Round a timestamp down to the start of its window (aligned to the Unix epoch)."""
    epoch = int(ts.timestamp()) // seconds * seconds
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def time_windows(
    records: Iterable[LogRecord],
    seconds: int = 60,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[Window]:
    """Tumbling time windows. Empty windows are kept, because silence is a signal."""
    records = list(records)
    if not records and (start is None or end is None):
        return []
    size = timedelta(seconds=seconds)
    first = align(start or records[0].timestamp, seconds)
    last = end or records[-1].timestamp + timedelta(microseconds=1)
    windows: list[Window] = []
    cursor = first
    while cursor < last:
        windows.append(Window(start=cursor, end=cursor + size))
        cursor += size
    for record in records:
        idx = int((record.timestamp - first) / size)
        if 0 <= idx < len(windows):
            windows[idx].add(record)
    return windows


def count_windows(records: Sequence[LogRecord], size: int = 20, step: int | None = None) -> list[Window]:
    """Fixed-size windows of ``size`` lines, used for sparse datasets such as BGL."""
    step = step or size
    windows: list[Window] = []
    for i in range(0, max(len(records) - size + 1, 1), step):
        chunk = records[i : i + size]
        if not chunk:
            break
        window = Window(start=chunk[0].timestamp, end=chunk[-1].timestamp)
        for record in chunk:
            window.add(record)
        windows.append(window)
    return windows


def label_by_intervals(windows: Sequence[Window], intervals: Iterable) -> None:
    """Mark windows overlapping any interval (objects with ``start``/``end``) as anomalous.

    Needed for incidents that remove log lines (a crashed service) rather than add them.
    """
    intervals = list(intervals)
    for window in windows:
        if any(i.start < window.end and window.start < i.end for i in intervals):
            window.label = 1


class WindowFeaturizer:
    """Learns a vocabulary of templates and services, then vectorises windows."""

    def __init__(self, max_templates: int = 200, min_count: int = 3, std_floor: float = 0.01):
        self.max_templates = max_templates
        self.min_count = min_count
        self.std_floor = std_floor
        self.templates: list[str] = []
        self.services: list[str] = []
        self.template_levels: dict[str, str] = {}
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    # -- fitting -----------------------------------------------------------------
    def fit(self, windows: Sequence[Window]) -> WindowFeaturizer:
        template_counts: Counter = Counter()
        service_counts: Counter = Counter()
        for window in windows:
            template_counts.update(window.templates)
            service_counts.update(window.services)
            for template, level in window.template_levels.items():
                self.template_levels.setdefault(template, level)
        self.templates = [t for t, c in template_counts.most_common(self.max_templates) if c >= self.min_count]
        self.services = sorted(service_counts)
        self._template_index = {t: i for i, t in enumerate(self.templates)}
        self._service_index = {s: i for i, s in enumerate(self.services)}
        matrix = self.transform(windows)
        self.mean_ = matrix.mean(axis=0)
        self.std_ = np.maximum(matrix.std(axis=0), self.std_floor)
        return self

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._template_index = {t: i for i, t in enumerate(self.templates)}
        self._service_index = {s: i for i, s in enumerate(self.services)}

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    # -- transforming ------------------------------------------------------------
    @property
    def feature_names(self) -> list[str]:
        return (
            [f"template:{t}" for t in self.templates]
            + [f"service:{s}" for s in self.services]
            + ["share:unknown_templates", "share:errors", "share:warnings", "log_volume", "log_unseen_error_templates"]
        )

    def unseen_error_templates(self, window: Window) -> list[str]:
        return [
            t for t in window.templates
            if t not in self._template_index and window.template_levels.get(t) in ERROR_LEVELS
        ]

    def transform_one(self, window: Window) -> np.ndarray:
        n_t, n_s = len(self.templates), len(self.services)
        vec = np.zeros(n_t + n_s + 5, dtype=np.float64)
        total = max(window.n_lines, 1)
        unknown = 0
        for template, count in window.templates.items():
            idx = self._template_index.get(template)
            if idx is None:
                unknown += count
            else:
                vec[idx] = count / total
        for service, count in window.services.items():
            idx = self._service_index.get(service)
            if idx is not None:
                vec[n_t + idx] = count / total
        base = n_t + n_s
        vec[base] = unknown / total
        vec[base + 1] = window.n_errors / total
        vec[base + 2] = window.n_warnings / total
        vec[base + 3] = np.log1p(window.n_lines)
        vec[base + 4] = np.log1p(len(self.unseen_error_templates(window)))
        return vec

    def transform(self, windows: Sequence[Window]) -> np.ndarray:
        if not windows:
            return np.zeros((0, len(self.feature_names)))
        return np.vstack([self.transform_one(w) for w in windows])

    # -- explaining --------------------------------------------------------------
    def explain(self, vector: np.ndarray, top_k: int = 3) -> list[dict]:
        """Return the features that deviate most from what was normal in training."""
        if self.mean_ is None:
            return []
        z = (vector - self.mean_) / self.std_
        names = self.feature_names
        order = np.argsort(-np.abs(z))[:top_k]
        return [
            {
                "feature": names[i],
                "value": round(float(vector[i]), 4),
                "expected": round(float(self.mean_[i]), 4),
                "z_score": round(float(z[i]), 1),
            }
            for i in order
            if abs(z[i]) > 2
        ]
