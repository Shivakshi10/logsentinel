import pytest
from fastapi.testclient import TestClient

from logsentinel.api import create_app
from logsentinel.pipeline import Pipeline
from logsentinel.storage import InMemoryRepository


@pytest.fixture(scope="module")
def client(bundle, with_incidents, tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "model.pkl"
    bundle.save(path)
    repo = InMemoryRepository()
    pipeline = Pipeline(bundle, repo)
    for line in with_incidents.lines()[:30000]:
        pipeline.handle_line(line)
    pipeline.flush()
    return TestClient(create_app(repo, str(path)))


def test_health_and_model(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/v1/model").json()
    assert info["detector"] == "isolation_forest" and info["window_seconds"] == 60


def test_score_normal_and_broken_window(client, healthy):
    normal = client.post("/v1/score", json={"lines": healthy.lines()[:120]}).json()
    assert normal["parsed_lines"] == 120 and not normal["is_anomaly"]
    broken_lines = [line.replace("INFO", "ERROR").replace("Health check OK", "Segfault in worker 1")
                    for line in healthy.lines()[:120]]
    broken = client.post("/v1/score", json={"lines": broken_lines}).json()
    assert broken["is_anomaly"] and broken["score"] > normal["score"]


def test_score_validation(client):
    assert client.post("/v1/score", json={"lines": []}).status_code == 422
    assert client.post("/v1/score", json={"lines": ["garbage"]}).status_code == 422


def test_incidents_and_windows(client):
    incidents = client.get("/v1/incidents", params={"limit": 5}).json()
    assert 0 < len(incidents) <= 5
    one = client.get(f"/v1/incidents/{incidents[0]['id']}").json()
    assert one["id"] == incidents[0]["id"]
    assert client.get("/v1/incidents/999999").status_code == 404
    assert len(client.get("/v1/windows", params={"limit": 10}).json()) == 10


def test_metrics_endpoint(client):
    body = client.get("/metrics/").text
    assert "logsentinel_lines_processed_total" in body


def test_no_model_returns_503(tmp_path):
    client = TestClient(create_app(InMemoryRepository(), str(tmp_path / "missing.pkl")))
    assert client.get("/health").json()["model_loaded"] is False
    assert client.post("/v1/score", json={"lines": ["x"]}).status_code == 503
