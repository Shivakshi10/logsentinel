from collections import Counter

from logsentinel import synthetic
from logsentinel.parsing import parse_app_line

from .conftest import START


def test_generation_is_reproducible():
    a = synthetic.generate(START, hours=0.5, seed=5)
    b = synthetic.generate(START, hours=0.5, seed=5)
    assert a.lines() == b.lines()


def test_healthy_period_has_no_labels(healthy):
    assert healthy.incidents == []
    assert all(r.label == 0 for r in healthy.records)
    assert healthy.records == sorted(healthy.records, key=lambda r: r.timestamp)


def test_incidents_are_labelled_and_varied(with_incidents):
    kinds = Counter(i.kind for i in with_incidents.incidents)
    assert len(kinds) == len(synthetic.INCIDENT_TYPES)
    assert any(r.label == 1 for r in with_incidents.records)


def test_service_silence_removes_logs(with_incidents):
    silence = next(i for i in with_incidents.incidents if i.kind == "service_silence")
    during = [r for r in with_incidents.records if silence.start <= r.timestamp < silence.end]
    assert during, "other services keep logging"
    assert not any(r.service == "inventory-service" for r in during)


def test_generated_lines_parse_back(healthy):
    for record in healthy.records[:500]:
        parsed = parse_app_line(record.raw)
        assert parsed is not None
        assert (parsed.service, parsed.level, parsed.template) == (record.service, record.level, record.template)
