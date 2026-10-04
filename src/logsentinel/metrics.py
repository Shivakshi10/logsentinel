"""Prometheus metrics exposed by the detector and the API."""

from prometheus_client import Counter, Gauge, Histogram

LINES_PROCESSED = Counter("logsentinel_lines_processed_total", "Log lines consumed")
PARSE_FAILURES = Counter("logsentinel_parse_failures_total", "Log lines that could not be parsed")
WINDOWS_SCORED = Counter("logsentinel_windows_scored_total", "Windows scored by the model")
ANOMALOUS_WINDOWS = Counter("logsentinel_anomalous_windows_total", "Windows flagged as anomalous")
INCIDENTS_OPENED = Counter("logsentinel_incidents_opened_total", "Incidents opened")
LATEST_SCORE = Gauge("logsentinel_latest_window_score", "Anomaly score of the most recent window")
THRESHOLD = Gauge("logsentinel_alert_threshold", "Alert threshold of the loaded model")
OPEN_INCIDENTS = Gauge("logsentinel_open_incidents", "Incidents currently open")
SCORING_SECONDS = Histogram(
    "logsentinel_scoring_seconds", "Time to featurise and score one window",
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
