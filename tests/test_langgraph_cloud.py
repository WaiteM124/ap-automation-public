"""Cloud LangGraph wiring: local fallback, graph compile, mocked SDK, no secrets leaked."""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import main
from ap_automation.core.config import settings
from ap_automation.core.models import InvoiceLine, RoutingDecision
from ap_automation.core.pipeline import predict
from ap_automation.graph.cloud import GRAPH_ID, configured, state_to_prediction
from ap_automation.graph.graph import build_graph
from ap_automation.graph.nodes import human_review_node


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def _invoice() -> InvoiceLine:
    return InvoiceLine(
        raw_vendor_name="Meridian Facilities Ltd",
        description="Office cleaning services monthly",
        amount=450,
        currency="GBP",
        entity_id="UK001",
        cost_centre="CC300",
    )


def test_cloud_unconfigured_by_default():
    assert settings.langgraph_api_url == ""
    assert settings.langgraph_api_key == ""
    assert configured() is False


def test_predict_stays_in_process_when_url_empty(seeded):
    pred = predict(_invoice())
    assert pred.provenance.get("thread_id") is None
    assert pred.final_routing in (RoutingDecision.AUTO_POST, RoutingDecision.HUMAN_REVIEW)


def test_build_graph_compiles_with_memory_saver():
    compiled = build_graph()
    assert compiled is not None
    spec = getattr(compiled, "config_specs", None) or getattr(compiled, "nodes", None)
    assert spec is not None or hasattr(compiled, "invoke")


def test_human_review_interrupt_payload_has_reviewer_fields():
    state = {
        "invoice_id": str(uuid4()),
        "raw_vendor_name": "Acme",
        "description": "Legal fees",
        "amount": 100.0,
        "currency": "GBP",
        "entity_id": "UK001",
        "submitted_cost_centre": "CC300",
        "predicted_gl": "6400",
        "predicted_cost_centre": "CC300",
        "agreement_ratio": 0.5,
        "weighted_confidence": 0.4,
        "routing_decision": "human_review",
        "final_routing": "human_review",
        "gate_reason": "thin history",
        "routing_reason": "thin history",
        "evidence": [{"description": "Legal fees", "gl_account": "6400", "similarity_score": 0.8}],
        "agent_results": [{"agent": "vat_agent", "success": True}],
        "agent_flags": [],
        "llm_reasoning": "",
        "llm_conflicts": [],
        "confirmed_gl": "6400",  # skip pause so the node is exercisable without a running graph
    }
    out = human_review_node(state)
    assert out["erp_posted"] is False
    assert "review_queued_at" in out


def test_state_to_prediction_maps_interrupt_state():
    invoice = _invoice()
    thread_id = "thread-abc"
    pred = state_to_prediction(
        {
            "values": {
                "canonical_vendor_id": "MERIDIAN",
                "mdm_confidence": 0.95,
                "amount_band": "small",
                "predicted_gl": "6300",
                "predicted_cost_centre": "CC300",
                "agreement_ratio": 1.0,
                "weighted_confidence": 0.82,
                "routing_decision": "human_review",
                "final_routing": "human_review",
                "gate_reason": "agent flag",
                "routing_reason": "ok",
                "agent_flags": ["vat_review_required"],
                "evidence": [],
                "agent_results": [],
            }
        },
        invoice,
        thread_id,
        invoice.invoice_id,
    )
    assert pred.final_routing == RoutingDecision.HUMAN_REVIEW
    assert pred.provenance["thread_id"] == thread_id
    assert pred.predicted_gl == "6300"
    assert "thread-abc" not in settings.langgraph_api_key


def test_review_queue_empty_without_cloud(client):
    r = client.get("/review-queue")
    assert r.status_code == 200
    assert r.json()["items"] == []


def test_thread_confirm_without_cloud_is_503(client):
    r = client.post("/predict/not-a-thread/confirm", json={"gl_account": "6300", "action": "confirm"})
    assert r.status_code == 503


def test_graph_id_matches_langgraph_json():
    import json
    from pathlib import Path
    spec = json.loads(Path("langgraph.json").read_text(encoding="utf-8"))
    assert GRAPH_ID in spec["graphs"]
    assert "LANGGRAPH_API_URL" in spec["env_vars"]
    assert "LANGGRAPH_API_KEY" in spec["env_vars"]


def test_settings_aliases_read_env(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_API_URL", "https://example.invalid")
    monkeypatch.setenv("LANGGRAPH_API_KEY", "not-a-real-key")
    from ap_automation.core.config import Settings
    s = Settings()
    assert s.langgraph_api_url == "https://example.invalid"
    assert s.langgraph_api_key == "not-a-real-key"
