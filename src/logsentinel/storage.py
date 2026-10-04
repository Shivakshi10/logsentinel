"""Persist window scores and incidents.

``InMemoryRepository`` is used for tests and quick local runs;
``PostgresRepository`` backs the Docker deployment and the Grafana dashboards.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Protocol

from .detector import WindowResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS window_scores (
    window_start   TIMESTAMPTZ PRIMARY KEY,
    window_end     TIMESTAMPTZ NOT NULL,
    log_count      INTEGER NOT NULL,
    error_count    INTEGER NOT NULL,
    warning_count  INTEGER NOT NULL,
    score          DOUBLE PRECISION NOT NULL,
    threshold      DOUBLE PRECISION NOT NULL,
    is_anomaly     BOOLEAN NOT NULL,
    reasons        JSONB NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS incidents (
    id              BIGSERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL,
    ended_at        TIMESTAMPTZ NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('open', 'resolved')),
    windows         INTEGER NOT NULL,
    peak_score      DOUBLE PRECISION NOT NULL,
    error_count     INTEGER NOT NULL,
    reasons         JSONB NOT NULL DEFAULT '[]',
    samples         JSONB NOT NULL DEFAULT '[]',
    summary         TEXT NOT NULL DEFAULT '',
    summary_source  TEXT NOT NULL DEFAULT 'rules',
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS incidents_started_at_idx ON incidents (started_at DESC);
"""


@dataclass
class Incident:
    started_at: datetime
    ended_at: datetime
    windows: int = 1
    peak_score: float = 0.0
    error_count: int = 0
    reasons: list[str] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)
    status: str = "open"
    summary: str = ""
    summary_source: str = "rules"
    id: int | None = None

    def absorb(self, result: WindowResult) -> None:
        """Extend the incident with another anomalous window."""
        self.ended_at = result.window_end
        self.windows += 1
        self.peak_score = max(self.peak_score, result.score)
        self.error_count += result.error_count
        for reason in result.reasons:
            if reason not in self.reasons and len(self.reasons) < 15:
                self.reasons.append(reason)
        for line in result.samples:
            if len(self.samples) < 10:
                self.samples.append(line)

    @classmethod
    def from_window(cls, result: WindowResult) -> Incident:
        return cls(
            started_at=result.window_start,
            ended_at=result.window_end,
            peak_score=result.score,
            error_count=result.error_count,
            reasons=list(result.reasons),
            samples=list(result.samples)[:10],
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["started_at"] = self.started_at.isoformat()
        data["ended_at"] = self.ended_at.isoformat()
        return data


class Repository(Protocol):
    def save_window(self, result: WindowResult) -> None: ...
    def save_incident(self, incident: Incident) -> Incident: ...
    def recent_windows(self, limit: int = 60) -> list[dict]: ...
    def recent_incidents(self, limit: int = 20) -> list[dict]: ...
    def get_incident(self, incident_id: int) -> dict | None: ...


class InMemoryRepository:
    def __init__(self) -> None:
        self.windows: dict[datetime, WindowResult] = {}
        self.incidents: dict[int, Incident] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def save_window(self, result: WindowResult) -> None:
        with self._lock:
            self.windows[result.window_start] = result

    def save_incident(self, incident: Incident) -> Incident:
        with self._lock:
            if incident.id is None:
                incident.id = self._next_id
                self._next_id += 1
            self.incidents[incident.id] = incident
        return incident

    def recent_windows(self, limit: int = 60) -> list[dict]:
        items = sorted(self.windows.values(), key=lambda r: r.window_start, reverse=True)[:limit]
        return [r.to_dict() for r in items]

    def recent_incidents(self, limit: int = 20) -> list[dict]:
        items = sorted(self.incidents.values(), key=lambda i: i.started_at, reverse=True)[:limit]
        return [i.to_dict() for i in items]

    def get_incident(self, incident_id: int) -> dict | None:
        incident = self.incidents.get(incident_id)
        return incident.to_dict() if incident else None


class PostgresRepository:
    """PostgreSQL storage. Uses one autocommit connection guarded by a lock."""

    def __init__(self, dsn: str):
        import psycopg
        from psycopg.rows import dict_row

        self._psycopg = psycopg
        self._dsn = dsn
        self._row_factory = dict_row
        self._lock = threading.Lock()
        self._conn = self._connect()
        with self._conn.cursor() as cur:
            cur.execute(SCHEMA)

    def _connect(self):
        return self._psycopg.connect(self._dsn, autocommit=True, row_factory=self._row_factory)

    def _execute(self, sql: str, params: tuple = (), fetch: str | None = None):
        with self._lock:
            for attempt in (1, 2):
                try:
                    with self._conn.cursor() as cur:
                        cur.execute(sql, params)
                        if fetch == "one":
                            return cur.fetchone()
                        if fetch == "all":
                            return cur.fetchall()
                        return None
                except self._psycopg.OperationalError:
                    if attempt == 2:
                        raise
                    self._conn = self._connect()  # reconnect once, e.g. after a database restart

    def save_window(self, result: WindowResult) -> None:
        self._execute(
            """
            INSERT INTO window_scores (window_start, window_end, log_count, error_count, warning_count,
                                       score, threshold, is_anomaly, reasons)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (window_start) DO UPDATE SET
                window_end = EXCLUDED.window_end, log_count = EXCLUDED.log_count,
                error_count = EXCLUDED.error_count, warning_count = EXCLUDED.warning_count,
                score = EXCLUDED.score, threshold = EXCLUDED.threshold,
                is_anomaly = EXCLUDED.is_anomaly, reasons = EXCLUDED.reasons
            """,
            (result.window_start, result.window_end, result.log_count, result.error_count,
             result.warning_count, result.score, result.threshold, result.is_anomaly, json.dumps(result.reasons)),
        )

    def save_incident(self, incident: Incident) -> Incident:
        values = (incident.started_at, incident.ended_at, incident.status, incident.windows, incident.peak_score,
                  incident.error_count, json.dumps(incident.reasons), json.dumps(incident.samples),
                  incident.summary, incident.summary_source)
        if incident.id is None:
            row = self._execute(
                """
                INSERT INTO incidents (started_at, ended_at, status, windows, peak_score, error_count,
                                       reasons, samples, summary, summary_source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                """,
                values, fetch="one",
            )
            incident.id = row["id"]
        else:
            self._execute(
                """
                UPDATE incidents SET started_at = %s, ended_at = %s, status = %s, windows = %s, peak_score = %s,
                    error_count = %s, reasons = %s, samples = %s, summary = %s, summary_source = %s,
                    updated_at = now()
                WHERE id = %s
                """,
                (*values, incident.id),
            )
        return incident

    @staticmethod
    def _serialise(row: dict) -> dict:
        return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in row.items()}

    def recent_windows(self, limit: int = 60) -> list[dict]:
        rows = self._execute("SELECT * FROM window_scores ORDER BY window_start DESC LIMIT %s", (limit,), fetch="all")
        return [self._serialise(r) for r in rows]

    def recent_incidents(self, limit: int = 20) -> list[dict]:
        rows = self._execute("SELECT * FROM incidents ORDER BY started_at DESC LIMIT %s", (limit,), fetch="all")
        return [self._serialise(r) for r in rows]

    def get_incident(self, incident_id: int) -> dict | None:
        row = self._execute("SELECT * FROM incidents WHERE id = %s", (incident_id,), fetch="one")
        return self._serialise(row) if row else None


def make_repository(database_url: str | None) -> Repository:
    return PostgresRepository(database_url) if database_url else InMemoryRepository()
