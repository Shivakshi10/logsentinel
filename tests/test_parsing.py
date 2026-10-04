from datetime import datetime, timezone

import pytest

from logsentinel.parsing import (
    extract_template,
    format_app_line,
    parse_app_line,
    parse_bgl_line,
    parse_line,
    parse_lines,
)


def test_app_line_round_trip():
    ts = datetime(2026, 9, 23, 10, 15, 2, 123000, tzinfo=timezone.utc)
    line = format_app_line(ts, "INFO", "payment-service", "Payment 48213 processed in 132 ms")
    assert line == "2026-09-23T10:15:02.123Z INFO [payment-service] Payment 48213 processed in 132 ms"
    record = parse_app_line(line)
    assert record.timestamp == ts
    assert record.level == "INFO"
    assert record.service == "payment-service"
    assert record.template == "Payment <NUM> processed in <NUM> ms"
    assert not record.is_error


@pytest.mark.parametrize(
    "message, template",
    [
        ("Connection refused to primary replica 10.2.3.4", "Connection refused to primary replica <IP>"),
        ("Token refreshed for session 96c8da19-da71-08d6-7af0-be653e2434e3", "Token refreshed for session <UUID>"),
        ("bad address 0x00ff1a", "bad address <HEX>"),
        ("Connection pool usage 12/50", "Connection pool usage <NUM>/<NUM>"),
        ("node R02-M1-N0 failed", "node R02-M1-N0 failed"),  # identifiers with letters are kept
        ("  extra    spaces  ", "extra spaces"),
    ],
)
def test_template_masking(message, template):
    assert extract_template(message) == template


def test_malformed_lines_are_skipped():
    assert parse_app_line("not a log line") is None
    assert parse_app_line("yesterday INFO [svc] hello") is None
    assert parse_lines(["", "garbage", "2026-01-01T00:00:00.000Z WARN [svc] ok"]) [0].level == "WARN"


def test_bgl_line_with_label():
    normal = ("- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 "
              "R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected")
    alert = ("KERNDTLB 1118536327 2005.06.11 R30-M0-N9-C:J16-U01 2005-06-11-17.32.07.581048 "
             "R30-M0-N9-C:J16-U01 RAS KERNEL FATAL data TLB error interrupt")
    n, a = parse_bgl_line(normal), parse_bgl_line(alert)
    assert (n.label, a.label) == (0, 1)
    assert a.is_error and a.service == "KERNEL"
    assert a.template == "data TLB error interrupt"


def test_unknown_format():
    with pytest.raises(ValueError):
        parse_line("x", "json")
