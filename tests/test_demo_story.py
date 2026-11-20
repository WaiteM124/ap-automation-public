"""
The demo script, replayed end to end through the real pipeline.

If this fails, the demo no longer tells the story it is meant to tell - fix the system or the
script before recording.
"""

import pytest
from fastapi.testclient import TestClient

import main
from ap_automation.core import demo
from ap_automation.core.config import settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "demo_mode", True)
    with TestClient(main.app) as c:
        c.post("/demo/reset")
        yield c


def run(client, scenario):
    r = client.post("/predict", json=scenario["invoice"])
    assert r.status_code == 200, r.text
    return r.json()


def reviewer_confirms(client, scenario, prediction):
    r = client.post(f"/confirm/{prediction['prediction_id']}",
                    json={"confirmed_gl": scenario["reviewer_gl"], "confirmed_cc": scenario["invoice"]["cost_centre"]})
    assert r.status_code == 200, r.text
    return r.json()


def test_every_scenario_does_what_the_script_says(client):
    for sc in demo.SCENARIOS:
        pred = run(client, sc)
        exp = sc["expect"]
        assert pred["final_routing"] == exp["final_routing"], (sc["id"], pred["final_routing_reason"])
        if "gl" in exp:
            assert pred["predicted_gl"] == exp["gl"], sc["id"]
        if "flag" in exp:
            # exactly one flag: each safety scenario must demonstrate one thing, not two
            assert pred["agent_flags"] == [exp["flag"]], (sc["id"], pred["agent_flags"])
        elif exp["final_routing"] == "auto_post":
            assert pred["agent_flags"] == [], (sc["id"], pred["agent_flags"])
        if pred["final_routing"] == "human_review":
            reviewer_confirms(client, sc, pred)   # a human works the queue, as in the demo


def test_the_safety_scenarios_pass_the_rule_gate_but_are_stopped_by_governance(client):
    """The point of the demo: confidence alone would have auto-posted these."""
    for sc_id in ("vat_block", "wrong_cc"):
        sc = next(s for s in demo.SCENARIOS if s["id"] == sc_id)
        pred = run(client, sc)
        assert pred["routing_decision"] == "auto_post"      # rule gate: confident
        assert pred["final_routing"] == "human_review"      # governed decision: stopped


def test_new_vendor_graduates_to_auto_post_only_through_human_confirmations(client):
    steps = [s for s in demo.SCENARIOS if s["group"] == "Learning"]
    outcomes = []
    for sc in steps:
        pred = run(client, sc)
        outcomes.append(pred["final_routing"])
        if pred["final_routing"] == "human_review":
            assert reviewer_confirms(client, sc, pred)["learned"] is True
    assert outcomes[0] == "human_review"
    assert outcomes[-1] == "auto_post"
    first_auto = outcomes.index("auto_post")
    assert 1 <= first_auto <= 4          # after at least one, and few, human decisions


def test_reset_makes_every_take_identical(client):
    steps = [s for s in demo.SCENARIOS if s["group"] == "Learning"]
    first_take = []
    for sc in steps:
        pred = run(client, sc)
        first_take.append(pred["final_routing"])
        if pred["final_routing"] == "human_review":
            reviewer_confirms(client, sc, pred)
    assert client.get("/demo/status").json()["corpus_size"] > 135   # the take changed the corpus

    assert client.post("/demo/reset").json()["status"] == "reset"
    status = client.get("/demo/status").json()
    assert status["corpus_size"] == 135 and status["ready"]
    assert client.get("/performance").status_code == 200

    second_take = []
    for sc in steps:
        pred = run(client, sc)
        second_take.append(pred["final_routing"])
        if pred["final_routing"] == "human_review":
            reviewer_confirms(client, sc, pred)
    assert second_take == first_take


def test_reset_is_refused_outside_demo_mode(client, monkeypatch):
    monkeypatch.setattr(settings, "demo_mode", False)
    assert client.post("/demo/reset").status_code == 403


def test_preflight_reports_what_it_is_running_on(client):
    status = client.get("/demo/status").json()
    assert status["ready"] is True
    assert "offline" in status["embeddings"]          # tests run on the lexical embedder - it says so
    assert {c["name"] for c in status["checks"]} >= {"Corpus", "ERP ledger and audit trail", "LLM gate"}


def test_pending_predictions_survive_a_restart(client):
    """/confirm used to 404 after a restart because predictions lived only in process memory."""
    sc = demo.SCENARIOS[0]
    pred = run(client, sc)
    from ap_automation.core.prediction_store import PredictionStore

    revived = PredictionStore()          # a fresh store object, as after a process restart
    assert revived.get(pred["prediction_id"]) is not None
    assert reviewer_confirms(client, sc, pred)["status"] == "confirmed"
