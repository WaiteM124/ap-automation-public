"""API behaviour: governed routing is exposed, confirmations are validated and idempotent."""

import pytest
from fastapi.testclient import TestClient

import main
from ap_automation.core.config import settings


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


BODY = {"raw_vendor_name": "Meridian Facilities Ltd", "description": "Office cleaning services monthly",
        "amount": 1200, "entity_id": "UK001", "cost_centre": "CC300"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_exposes_both_decisions(client, seeded):
    r = client.post("/predict", json=BODY)
    assert r.status_code == 200
    data = r.json()
    assert {"routing_decision", "final_routing", "final_routing_reason", "agent_flags"} <= data.keys()
    assert data["predicted_gl"] == "6300"


def test_confirm_rejects_unknown_gl(client, seeded):
    pid = client.post("/predict", json=BODY).json()["prediction_id"]
    r = client.post(f"/confirm/{pid}", json={"confirmed_gl": "9999"})
    assert r.status_code == 422


def test_confirm_unknown_prediction_is_404(client):
    r = client.post("/confirm/does-not-exist", json={"confirmed_gl": "6300"})
    assert r.status_code == 404


def test_confirm_is_idempotent(client, seeded):
    pid = client.post("/predict", json=BODY).json()["prediction_id"]
    first = client.post(f"/confirm/{pid}", json={"confirmed_gl": "6300"})
    second = client.post(f"/confirm/{pid}", json={"confirmed_gl": "6300"})
    assert first.status_code == 200 and first.json()["learned"] is True
    assert second.status_code == 409


def test_a_changed_gl_is_recorded_as_a_correction(client, seeded):
    pid = client.post("/predict", json=BODY).json()["prediction_id"]
    r = client.post(f"/confirm/{pid}", json={"confirmed_gl": "6200", "source": "human_confirm"})
    assert r.json()["source"] == "human_correct"


def test_api_key_is_enforced_when_configured(client, seeded, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "s3cret")
    assert client.post("/predict", json=BODY).status_code == 401
    assert client.post("/predict", json=BODY, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200  # health stays open for the platform probe


def test_performance_reports_zero_not_invented_numbers(client):
    r = client.get("/performance")
    assert r.status_code == 200


def test_batch_runs_and_never_learns_from_auto_posts(client, seeded):
    import io
    import time

    csv_text = "vendor_name,description,amount,entity_id,cost_centre,ground_truth_gl\n"
    csv_text += "Meridian Facilities Ltd,Office cleaning services monthly,1200,UK001,,6300\n"
    csv_text += "NovaCorp Services Ltd,Management consulting services,3000,UK001,,6400\n"
    r = client.post("/batch", params={"simulate_reviewer": "true"},
                    files={"file": ("b.csv", io.BytesIO(csv_text.encode()), "text/csv")})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    for _ in range(100):
        status = client.get(f"/batch/{job_id}").json()
        if status["status"] != "running":
            break
        time.sleep(0.1)
    assert status["status"] == "complete" and status["completed"] == 2 and status["errors"] == 0
    by_vendor = {x["vendor"]: x for x in status["results"]}
    assert by_vendor["Meridian Facilities Ltd"]["routing"] == "auto_post"
    assert by_vendor["Meridian Facilities Ltd"]["learned"] is False      # auto-post: not learned
    assert by_vendor["NovaCorp Services Ltd"]["routing"] == "human_review"
    assert by_vendor["NovaCorp Services Ltd"]["learned"] is True         # simulated reviewer: learned
