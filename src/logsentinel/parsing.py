"""Parse raw log lines into structured records and extract log templates.

Two formats are supported:

* ``app``  - the microservice log format produced by the synthetic generator::

      2026-09-23T10:15:02.123Z INFO [payment-service] Payment 48213 processed in 132 ms

* ``bgl``  - the BlueGene/L supercomputer format from the Loghub benchmark.
  The first token is the ground-truth label (``-`` means normal).

Templates are extracted by masking variable parts of a message (numbers, IPs,
UUIDs, hex values). Two messages that differ only in those parts share a
template, e.g. ``Payment <NUM> processed in <NUM> ms``. This is the same idea
as the preprocessing step of the Drain log parser, kept deliberately simple so
it is deterministic and needs no state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

ERROR_LEVELS = frozenset({"ERROR", "FATAL", "FAILURE", "SEVERE", "CRITICAL"})
WARN_LEVELS = frozenset({"WARN", "WARNING"})

_MASKS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<UUID>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\b0x[0-9a-f]+\b", re.I), "<HEX>"),
    (re.compile(r"\b[0-9a-f]{12,}\b", re.I), "<HEX>"),
    (re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?"), "<NUM>"),
]
_WHITESPACE = re.compile(r"\s+")

_APP_PATTERN = re.compile(
    r"^(?P<ts>\S+)\s+(?P<level>[A-Z]+)\s+\[(?P<service>[^\]]+)\]\s+(?P<message>.*)$"
)


@dataclass(slots=True)
class LogRecord:
    """A single parsed log line."""

    timestamp: datetime
    level: str
    service: str
    message: str
    template: str
    raw: str
    label: int = 0  # 1 = known anomalous (only available for labelled datasets)

    @property
    def is_error(self) -> bool:
        return self.level in ERROR_LEVELS

    @property
    def is_warning(self) -> bool:
        return self.level in WARN_LEVELS


def extract_template(message: str) -> str:
    """Replace the variable parts of a message with placeholders."""
    template = message
    for pattern, placeholder in _MASKS:
        template = pattern.sub(placeholder, template)
    return _WHITESPACE.sub(" ", template).strip()


def format_app_line(timestamp: datetime, level: str, service: str, message: str) -> str:
    ts = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return f"{ts} {level} [{service}] {message}"


def parse_app_line(line: str) -> LogRecord | None:
    """Parse a line in the synthetic microservice format. Returns None if malformed."""
    line = line.rstrip("\r\n")
    match = _APP_PATTERN.match(line)
    if not match:
        return None
    try:
        timestamp = datetime.fromisoformat(match["ts"].replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    message = match["message"]
    return LogRecord(
        timestamp=timestamp,
        level=match["level"],
        service=match["service"],
        message=message,
        template=extract_template(message),
        raw=line,
    )


def parse_bgl_line(line: str) -> LogRecord | None:
    """Parse a BlueGene/L line from Loghub.

    Layout: ``label unix_ts date node time node_repeat type component level content``
    """
    line = line.rstrip("\r\n")
    parts = line.split(maxsplit=9)
    if len(parts) < 9:
        return None
    label_token, unix_ts = parts[0], parts[1]
    component, level = parts[7], parts[8]
    message = parts[9] if len(parts) > 9 else ""
    try:
        timestamp = datetime.fromtimestamp(int(unix_ts), tz=timezone.utc)
    except (ValueError, OverflowError):
        return None
    return LogRecord(
        timestamp=timestamp,
        level=level.upper(),
        service=component,
        message=message,
        template=extract_template(message),
        raw=line,
        label=0 if label_token == "-" else 1,
    )


PARSERS = {"app": parse_app_line, "bgl": parse_bgl_line}


def parse_line(line: str, fmt: str = "app") -> LogRecord | None:
    try:
        parser = PARSERS[fmt]
    except KeyError:
        raise ValueError(f"Unknown log format {fmt!r}; expected one of {sorted(PARSERS)}") from None
    return parser(line)


def parse_lines(lines, fmt: str = "app") -> list[LogRecord]:
    """Parse an iterable of lines, silently skipping malformed ones."""
    records = []
    for line in lines:
        if not line.strip():
            continue
        record = parse_line(line, fmt)
        if record is not None:
            records.append(record)
    return records
