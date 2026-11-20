"""The ledger, audit trail, corpus explorer and analytics are all backed by real recorded state."""

import pytest
from conftest import make_invoice
from fastapi.testclient import TestClient

import main
from ap_automation.core import demo, retrieval
from ap_automation.core.demo import SCENARIOS


@pytest.fixture
def client(seeded):
    with TestClient(main.app) as c:
        main._store.clear()
        yield c


def _scenario(sid):
    return next(s for s in SCENARIOS if s["id"] == sid)["invoice"]


def _predict(client, sid):
    return client.post("/predict", json=_scenario(sid)).json()


def test_auto_post_posts_to_ledger_and_is_not_learned(client):
    before = retrieval._collection_size()
    pred = _predict(client, "routine")
    assert pred["final_routing"] == "auto_post"
    r = client.post(f"/auto-post/{pred['prediction_id']}")
    assert r.status_code == 200 and r.json()["learned"] is False
    assert retrieval._collection_size() == before  # the system never learns from its own auto-posts

    ledger = client.get("/ledger").json()
    assert ledger["summary"]["count"] == 1
    row = ledger["postings"][0]
    assert row["source"] == "auto_post" and row["vendor_name"] == "Meridian Facilities Ltd"
    assert client.post(f"/auto-post/{pred['prediction_id']}").status_code == 409  # cannot post twice


def test_auto_post_refused_when_governance_blocks(client):
    pred = _predict(client, "vat_block")
    assert pred["final_routing"] == "human_review"
    r = client.post(f"/auto-post/{pred['prediction_id']}")
    assert r.status_code == 403
    assert client.get("/ledger").json()["summary"]["count"] == 0


def test_audit_trail_has_the_full_decision_record(client):
    pred = _predict(client, "wrong_cc")
    client.post(f"/confirm/{pred['prediction_id']}", json={"confirmed_gl": pred["predicted_gl"]})
    records = client.get("/audit").json()["records"]
    assert len(records) == 1
    rec = records[0]
    assert rec["source"] == "human_confirm" and rec["vendor_name"] == "Meridian Facilities Ltd"
    assert "cost_centre_mismatch" in rec["provenance"]["agent_flags"]
    assert rec["provenance"]["retrieved_evidence"]


def test_analytics_are_computed_locally_without_supabase(client, monkeypatch):
    from ap_automation.core import supabase
    monkeypatch.setattr(supabase, "_key", lambda: "")
    assert client.get("/analytics").json()["data_available"] is False
    _predict(client, "routine")
    _predict(client, "vat_block")
    data = client.get("/analytics?days=7").json()
    assert data["data_available"] is True and data["total_predictions"] == 2
    assert data["auto_count"] == 1 and data["automation_rate"] == 50.0
    assert data["agent_flags"]["vat_blocked"] == 1


def test_corpus_explorer_lists_and_removes_points(client):
    vendors = client.get("/corpus/vendors").json()
    meridian = next(v for v in vendors["vendors"] if v["vendor_id"] == "MERIDIAN_FACILITIES")
    assert meridian["points"] == 15 and meridian["top_gl"] == "6300"
    pts = client.get(f"/corpus/vendors/{meridian['vendor_id']}").json()["points"]
    assert len(pts) == 15
    assert client.delete(f"/corpus/points/{pts[0]['point_id']}").status_code == 200
    assert len(client.get(f"/corpus/vendors/{meridian['vendor_id']}").json()["points"]) == 14
    assert client.delete(f"/corpus/points/{pts[0]['point_id']}").status_code == 404


def test_retrieval_is_deterministic(seeded):
    """Same inputs, same evidence, whatever the (random) process state: results must be reproducible."""
    inv = make_invoice()
    from ap_automation.core.models import amount_to_band
    runs = [
        [(e.invoice_id, round(e.rrf_score, 8)) for e in retrieval.retrieve(
            inv.description, "MERIDIAN_FACILITIES", inv.entity_id, amount_to_band(inv.amount), inv.currency)]
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2] and len(runs[0]) > 0


def test_reset_clears_local_events(client, monkeypatch):
    from ap_automation.core import database
    _predict(client, "routine")
    assert database.get_events()
    demo.reset(main._store)
    assert database.get_events() == []
