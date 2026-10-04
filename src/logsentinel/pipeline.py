"""Glue between parsing, detection, incident grouping, summaries and storage."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from . import metrics
from .bundle import ModelBundle
from .detector import StreamingDetector, WindowResult
from .parsing import parse_line
from .storage import Incident, Repository
from .summarizer import Summarizer, rule_based_summary

logger = logging.getLogger(__name__)


class Pipeline:
    """Consumes raw lines, scores windows and manages incidents.

    Consecutive anomalous windows are merged into one incident, which is
    resolved after ``resolve_after`` normal windows in a row. That avoids
    paging someone once per minute for the same problem.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        repository: Repository,
        summarizer: Summarizer | None = None,
        log_format: str = "app",
        resolve_after: int = 2,
        on_incident: Callable[[Incident], None] | None = None,
    ):
        self.detector = StreamingDetector(bundle)
        self.repository = repository
        self.summarizer = summarizer or Summarizer()
        self.log_format = log_format
        self.resolve_after = resolve_after
        self.on_incident = on_incident
        self.open_incident: Incident | None = None
        self._normal_streak = 0
        metrics.THRESHOLD.set(bundle.threshold)

    def handle_line(self, line: str) -> list[WindowResult]:
        record = parse_line(line, self.log_format)
        if record is None:
            metrics.PARSE_FAILURES.inc()
            return []
        metrics.LINES_PROCESSED.inc()
        started = time.perf_counter()
        results = self.detector.process(record)
        if results:
            metrics.SCORING_SECONDS.observe((time.perf_counter() - started) / len(results))
        for result in results:
            self._handle_result(result)
        return results

    def flush(self) -> None:
        for result in self.detector.flush():
            self._handle_result(result)
        if self.open_incident is not None:
            self._resolve()

    # -- internals ---------------------------------------------------------------------
    def _handle_result(self, result: WindowResult) -> None:
        metrics.WINDOWS_SCORED.inc()
        metrics.LATEST_SCORE.set(result.score)
        self.repository.save_window(result)
        if result.is_anomaly:
            metrics.ANOMALOUS_WINDOWS.inc()
            self._normal_streak = 0
            if self.open_incident is None:
                incident = Incident.from_window(result)
                incident.summary = rule_based_summary(incident)
                self.open_incident = self.repository.save_incident(incident)
                metrics.INCIDENTS_OPENED.inc()
                metrics.OPEN_INCIDENTS.set(1)
                logger.warning("Incident %s opened at %s: %s", incident.id, incident.started_at, incident.summary)
            else:
                self.open_incident.absorb(result)
                self.repository.save_incident(self.open_incident)
        elif self.open_incident is not None:
            self._normal_streak += 1
            if self._normal_streak >= self.resolve_after:
                self._resolve()

    def _resolve(self) -> None:
        incident = self.open_incident
        incident.status = "resolved"
        incident.summary, incident.summary_source = self.summarizer.summarize(incident)
        self.repository.save_incident(incident)
        logger.warning("Incident %s resolved after %s windows: %s", incident.id, incident.windows, incident.summary)
        if self.on_incident:
            self.on_incident(incident)
        self.open_incident = None
        self._normal_streak = 0
        metrics.OPEN_INCIDENTS.set(0)
