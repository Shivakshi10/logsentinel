"""Real-time detection: turn a stream of log records into scored windows."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

import numpy as np

from .bundle import ModelBundle
from .features import Window, align
from .parsing import LogRecord

MAX_GAP_WINDOWS = 24 * 60  # a jump bigger than this resets the stream instead of emitting empty windows


@dataclass
class WindowResult:
    window_start: datetime
    window_end: datetime
    log_count: int
    error_count: int
    warning_count: int
    score: float
    threshold: float
    is_anomaly: bool
    reasons: list[str] = field(default_factory=list)
    deviations: list[dict] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["window_start"] = self.window_start.isoformat()
        data["window_end"] = self.window_end.isoformat()
        return data


def describe_deviation(dev: dict) -> str | None:
    """Turn a feature deviation into a sentence a human can act on."""
    if dev["feature"] == "log_unseen_error_templates":
        return None  # already reported explicitly, with the message text
    kind, _, name = dev["feature"].partition(":")
    value, expected = dev["value"], dev["expected"]
    if kind == "template":
        if expected <= 0:
            return f"'{name}' appeared ({value:.1%} of lines) although it is normally absent"
        ratio = value / expected
        if ratio >= 1:
            return f"'{name}' is {ratio:.1f}x more frequent than usual ({value:.1%} of lines vs {expected:.1%})"
        return f"'{name}' dropped to {value:.1%} of lines (usually {expected:.1%})"
    if kind == "service":
        direction = "rose" if value > expected else "dropped"
        return f"share of logs from {name} {direction} to {value:.1%} (usually {expected:.1%})"
    if kind == "share":
        label = {"errors": "error lines", "warnings": "warning lines", "unknown_templates": "never-seen messages"}.get(name, name)
        return f"{label} make up {value:.1%} of the window (usually {expected:.1%})"
    if dev["feature"] == "log_volume":
        direction = "higher" if value > expected else "lower"
        return f"log volume is {direction} than usual (~{np.expm1(value):.0f} lines vs ~{np.expm1(expected):.0f})"
    return f"{dev['feature']} = {value} (usually {expected})"


class StreamingDetector:
    """Groups records into tumbling windows and scores each window as it closes.

    Windows close when a record from a *later* window arrives (event-time
    processing), so replaying old logs at any speed gives the same result as
    live traffic. Empty windows are still emitted and scored, because a
    service going silent is itself a symptom.
    """

    def __init__(self, bundle: ModelBundle, flag_unseen_errors: bool = True):
        self.bundle = bundle
        self.window_seconds = bundle.window_seconds or 60
        self.size = timedelta(seconds=self.window_seconds)
        self.flag_unseen_errors = flag_unseen_errors
        self._history: deque[np.ndarray] = deque(maxlen=max(bundle.detector.context, 1))
        self._current: Window | None = None
        self.late_records = 0

    def process(self, record: LogRecord) -> list[WindowResult]:
        results: list[WindowResult] = []
        if self._current is None:
            self._current = self._new_window(align(record.timestamp, self.window_seconds))
        elif record.timestamp >= self._current.end:
            gap = int((record.timestamp - self._current.end) / self.size)
            if gap > MAX_GAP_WINDOWS:
                results.append(self._close(self._current))
                self._history.clear()
                self._current = self._new_window(align(record.timestamp, self.window_seconds))
            else:
                while record.timestamp >= self._current.end:
                    results.append(self._close(self._current))
                    self._current = self._new_window(self._current.end)
        elif record.timestamp < self._current.start:
            self.late_records += 1  # counted in the current window rather than dropped
        self._current.add(record)
        return results

    def flush(self) -> list[WindowResult]:
        if self._current is None or self._current.n_lines == 0:
            return []
        result = self._close(self._current)
        self._current = None
        return [result]

    def score_window(self, window: Window) -> WindowResult:
        """Score a single, self-contained window (used by the REST API)."""
        vector = self.bundle.featurizer.transform_one(window)
        score = float(self.bundle.detector.score(vector[None, :])[-1])
        return self._result(window, vector, score)

    # -- internals ---------------------------------------------------------------------
    def _new_window(self, start: datetime) -> Window:
        return Window(start=start, end=start + self.size)

    def _close(self, window: Window) -> WindowResult:
        vector = self.bundle.featurizer.transform_one(window)
        self._history.append(vector)
        score = float(self.bundle.detector.score(np.vstack(self._history))[-1])
        return self._result(window, vector, score)

    def _result(self, window: Window, vector: np.ndarray, score: float) -> WindowResult:
        featurizer = self.bundle.featurizer
        threshold = self.bundle.threshold
        reasons: list[str] = []
        is_anomaly = score >= threshold
        if is_anomaly:
            reasons.append(f"anomaly score {score:.3f} is above the alert threshold {threshold:.3f}")
        if self.flag_unseen_errors:
            for template in featurizer.unseen_error_templates(window)[:3]:
                is_anomaly = True
                reasons.append(f"new error message never seen during training: '{template}'")
        deviations = featurizer.explain(vector) if is_anomaly else []
        reasons.extend(text for text in map(describe_deviation, deviations) if text)
        return WindowResult(
            window_start=window.start,
            window_end=window.end,
            log_count=window.n_lines,
            error_count=window.n_errors,
            warning_count=window.n_warnings,
            score=score,
            threshold=threshold,
            is_anomaly=is_anomaly,
            reasons=reasons,
            deviations=deviations,
            samples=window.example_lines(),
        )
