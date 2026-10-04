"""Generate realistic microservice logs with labelled incidents.

Real, labelled production logs are rarely public, so this generator produces a
controllable stand-in. It deliberately includes things that make naive alerting
rules struggle:

* **Daily traffic cycles** - volume swings roughly 4x between night and day, so
  fixed "more than N errors per minute" rules fire during busy hours.
* **Background errors** - a healthy system still logs some errors (declined
  payments, client timeouts).
* **Benign events** - deployments and a nightly batch job change the log mix
  but are *not* incidents.

Six incident types are injected, each with a random intensity so some are
subtle:

=====================  =========================================================
``db_outage``          burst of connection errors cascading into 503s
``memory_leak``        a never-before-seen OutOfMemoryError plus GC warnings
``latency_degradation`` slow-query warnings and upstream timeouts increase
``service_silence``    a service crashes and simply stops logging
``retry_storm``        a known, normally rare retry warning becomes very frequent
``credential_stuffing`` failed-login warnings spike from many IPs
=====================  =========================================================
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from .parsing import LogRecord, extract_template, format_app_line

MessageFn = Callable[[random.Random], str]

SERVICES = ["api-gateway", "auth-service", "payment-service", "inventory-service", "database"]
_ENDPOINTS = ["products", "orders", "cart", "users", "search"]
_TABLES = ["orders", "payments", "inventory", "users", "sessions"]


def _ms(lo: int, hi: int) -> Callable[[random.Random], int]:
    return lambda r: int(r.lognormvariate(math.log((lo + hi) / 2), 0.4))


def _ip(r: random.Random) -> str:
    return f"10.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}"


def _uuid(r: random.Random) -> str:
    a, b, c, d, e = (r.getrandbits(n) for n in (32, 16, 16, 16, 48))
    return f"{a:08x}-{b:04x}-{c:04x}-{d:04x}-{e:012x}"


_fast, _slow = _ms(20, 120), _ms(1500, 4000)

# (service, level, relative weight, message factory)
NORMAL_TEMPLATES: list[tuple[str, str, float, MessageFn]] = [
    ("api-gateway", "INFO", 40, lambda r: f"GET /api/v1/{r.choice(_ENDPOINTS)} -> 200 in {_fast(r)} ms"),
    ("api-gateway", "INFO", 8, lambda r: f"POST /api/v1/orders -> 201 in {_fast(r)} ms"),
    ("api-gateway", "WARN", 2, lambda r: f"GET /api/v1/{r.choice(_ENDPOINTS)} -> 404 in {_fast(r)} ms"),
    ("api-gateway", "WARN", 0.5, lambda r: f"Rate limit reached for client {_ip(r)}"),
    ("api-gateway", "ERROR", 0.3, lambda r: f"Upstream request timed out after {_slow(r)} ms"),
    ("auth-service", "INFO", 12, lambda r: f"User {r.randint(1000, 99999)} authenticated successfully"),
    ("auth-service", "INFO", 6, lambda r: f"Token refreshed for session {_uuid(r)}"),
    ("auth-service", "WARN", 1, lambda r: f"Invalid password attempt for user {r.randint(1000, 99999)} from {_ip(r)}"),
    ("auth-service", "WARN", 0.1, lambda r: f"Retrying token validation (attempt {r.randint(1, 2)}/5)"),
    ("payment-service", "INFO", 6, lambda r: f"Payment {r.randint(10000, 999999)} processed in {_fast(r)} ms"),
    ("payment-service", "INFO", 4, lambda r: f"Payment {r.randint(10000, 999999)} authorised by provider"),
    ("payment-service", "ERROR", 0.4, lambda r: f"Payment {r.randint(10000, 999999)} declined by provider: insufficient funds"),
    ("inventory-service", "INFO", 10, lambda r: f"Stock level for SKU {r.randint(100, 9999)} updated to {r.randint(0, 500)}"),
    ("inventory-service", "INFO", 5, lambda r: f"Reservation {_uuid(r)} created for order {r.randint(10000, 999999)}"),
    ("inventory-service", "WARN", 0.6, lambda r: f"Stock for SKU {r.randint(100, 9999)} below reorder threshold"),
    ("database", "INFO", 12, lambda r: f"Query executed in {_fast(r)} ms on table {r.choice(_TABLES)}"),
    ("database", "INFO", 2, lambda r: f"Connection pool usage {r.randint(5, 40)}/50"),
    ("database", "WARN", 0.5, lambda r: f"Slow query detected: {_slow(r)} ms on table {r.choice(_TABLES)}"),
]

# (service, level, lines per second at intensity 1.0, message factory)
INCIDENT_TEMPLATES: dict[str, list[tuple[str, str, float, MessageFn]]] = {
    "db_outage": [
        ("database", "ERROR", 0.8, lambda r: f"Connection refused to primary replica {_ip(r)}"),
        ("payment-service", "ERROR", 0.4, lambda r: f"Payment {r.randint(10000, 999999)} failed: database unavailable"),
        ("api-gateway", "ERROR", 0.5, lambda r: f"GET /api/v1/orders -> 503 in {_slow(r)} ms"),
    ],
    "memory_leak": [
        ("payment-service", "ERROR", 0.05, lambda r: f"java.lang.OutOfMemoryError: Java heap space in worker {r.randint(1, 16)}"),
        ("payment-service", "WARN", 0.15, lambda r: f"GC pause of {_slow(r)} ms exceeded budget"),
    ],
    "latency_degradation": [
        ("database", "WARN", 0.6, lambda r: f"Slow query detected: {_slow(r)} ms on table {r.choice(_TABLES)}"),
        ("api-gateway", "ERROR", 0.15, lambda r: f"Upstream request timed out after {_slow(r)} ms"),
    ],
    "service_silence": [
        ("api-gateway", "ERROR", 0.04, lambda r: "Upstream inventory-service unavailable, serving cached response"),
    ],
    "retry_storm": [
        ("auth-service", "WARN", 1.2, lambda r: f"Retrying token validation (attempt {r.randint(1, 5)}/5)"),
    ],
    "credential_stuffing": [
        ("auth-service", "WARN", 1.0, lambda r: f"Invalid password attempt for user {r.randint(1000, 99999)} from {_ip(r)}"),
    ],
}
INCIDENT_TYPES = list(INCIDENT_TEMPLATES)
_SILENCED_SERVICE = "inventory-service"


@dataclass(frozen=True, slots=True)
class Incident:
    kind: str
    start: datetime
    end: datetime
    intensity: float

    def overlaps(self, start: datetime, end: datetime) -> bool:
        return self.start < end and start < self.end


@dataclass(frozen=True, slots=True)
class BenignEvent:
    kind: str  # "deployment" | "nightly_job"
    service: str
    start: datetime
    end: datetime


@dataclass
class SyntheticDataset:
    records: list[LogRecord]
    incidents: list[Incident]
    benign_events: list[BenignEvent]
    start: datetime
    end: datetime

    def lines(self) -> list[str]:
        return [record.raw for record in self.records]


def _diurnal_factor(ts: datetime) -> float:
    """Traffic multiplier: quiet at night (~0.4), peaking mid-afternoon (~1.6)."""
    hour = ts.hour + ts.minute / 60
    return 1.0 + 0.6 * math.sin((hour - 9) / 24 * 2 * math.pi)


def _schedule_incidents(
    rnd: random.Random, start: datetime, end: datetime, per_day: float
) -> list[Incident]:
    span = (end - start).total_seconds()
    count = int(round(per_day * span / 86400))
    incidents: list[Incident] = []
    attempts = 0
    while len(incidents) < count and attempts < count * 50:
        attempts += 1
        kind = INCIDENT_TYPES[len(incidents) % len(INCIDENT_TYPES)]
        duration = timedelta(seconds=rnd.randint(4 * 60, 15 * 60))
        offset = int(rnd.uniform(1800, max(1801.0, span - duration.total_seconds() - 600)))
        inc_start = start + timedelta(seconds=offset)
        candidate = Incident(kind, inc_start, inc_start + duration, rnd.uniform(0.4, 1.4))
        margin = timedelta(minutes=20)
        if all(not candidate.overlaps(i.start - margin, i.end + margin) for i in incidents):
            incidents.append(candidate)
    return sorted(incidents, key=lambda i: i.start)


def _schedule_benign(rnd: random.Random, start: datetime, end: datetime) -> list[BenignEvent]:
    events: list[BenignEvent] = []
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < end:
        job_start = day + timedelta(hours=2)
        events.append(BenignEvent("nightly_job", "inventory-service", job_start, job_start + timedelta(minutes=12)))
        for _ in range(rnd.randint(1, 2)):
            dep_start = day + timedelta(hours=rnd.uniform(9, 17))
            events.append(
                BenignEvent("deployment", rnd.choice(SERVICES[:4]), dep_start, dep_start + timedelta(minutes=2))
            )
        day += timedelta(days=1)
    return [e for e in events if e.end > start and e.start < end]


def _benign_lines(rnd: random.Random, event: BenignEvent, second: datetime) -> list[tuple[str, str, str]]:
    elapsed = (second - event.start).total_seconds()
    if event.kind == "nightly_job":
        out = []
        for _ in range(np.random.default_rng(rnd.getrandbits(32)).poisson(1.5)):
            out.append((event.service, "INFO", f"Nightly report job processed {rnd.randint(100, 5000)} records"))
        return out
    # deployment: shutdown, restart and cache warm-up messages
    if elapsed == 0:
        return [(event.service, "INFO", "Received SIGTERM, shutting down gracefully")]
    if elapsed == 30:
        return [
            (event.service, "INFO", f"Starting {event.service} version 2.{rnd.randint(0, 30)}.{rnd.randint(0, 9)}"),
            (event.service, "INFO", "Loaded configuration from /etc/app/config.yaml"),
        ]
    if elapsed > 30 and rnd.random() < 0.8:
        return [(event.service, "INFO", f"Warming up cache: {rnd.randint(1000, 90000)} entries loaded")]
    return []


def generate(
    start: datetime | None = None,
    hours: float = 24,
    lines_per_second: float = 2.0,
    incidents_per_day: float = 8,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate a labelled synthetic log dataset.

    Args:
        start: first timestamp (UTC). Defaults to midnight today.
        hours: length of the generated period.
        lines_per_second: average application log volume (heartbeats come on top).
        incidents_per_day: how many incidents to inject; use 0 for a healthy period.
        seed: makes the output fully reproducible.
    """
    rnd = random.Random(seed)
    nprng = np.random.default_rng(seed)
    if start is None:
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = start.astimezone(timezone.utc).replace(microsecond=0)
    end = start + timedelta(hours=hours)

    incidents = _schedule_incidents(rnd, start, end, incidents_per_day) if incidents_per_day > 0 else []
    benign = _schedule_benign(rnd, start, end)
    weights = [t[2] for t in NORMAL_TEMPLATES]
    total_seconds = int((end - start).total_seconds())

    records: list[LogRecord] = []
    for s in range(total_seconds):
        second = start + timedelta(seconds=s)
        active = [i for i in incidents if i.start <= second < i.end]
        silenced = {_SILENCED_SERVICE} if any(i.kind == "service_silence" for i in active) else set()
        deploying = {
            e.service for e in benign
            if e.kind == "deployment" and e.start + timedelta(seconds=1) <= second < e.start + timedelta(seconds=30)
        }
        quiet = silenced | deploying

        batch: list[tuple[str, str, str, int]] = []
        n = nprng.poisson(lines_per_second * _diurnal_factor(second))
        for service, level, _, make in rnd.choices(NORMAL_TEMPLATES, weights=weights, k=int(n)):
            batch.append((service, level, make(rnd), 0))
        if s % 10 == 0:
            batch.extend((svc, "INFO", "Health check OK", 0) for svc in SERVICES)
        for event in benign:
            if event.start <= second < event.end:
                batch.extend((*line, 0) for line in _benign_lines(rnd, event, second))
        for incident in active:
            for service, level, rate, make in INCIDENT_TEMPLATES[incident.kind]:
                for _ in range(nprng.poisson(rate * incident.intensity)):
                    batch.append((service, level, make(rnd), 1))

        offsets = sorted(rnd.random() for _ in batch)
        for (service, level, message, label), frac in zip(batch, offsets):
            if service in quiet:
                continue
            ts = second + timedelta(seconds=frac)
            records.append(
                LogRecord(
                    timestamp=ts,
                    level=level,
                    service=service,
                    message=message,
                    template=extract_template(message),
                    raw=format_app_line(ts, level, service, message),
                    label=label,
                )
            )
    return SyntheticDataset(records=records, incidents=incidents, benign_events=benign, start=start, end=end)
