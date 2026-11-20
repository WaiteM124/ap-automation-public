"""The rule gate, the governance layer and the end-to-end routing they produce."""

from datetime import datetime, timezone

import pytest
from conftest import make_invoice

from ap_automation.core import governance
from ap_automation.core.confidence import route
from ap_automation.core.models import ConfirmationSource, RetrievedEvidence, RoutingDecision
from ap_automation.core.pipeline import AutoPostNotPermitted, confirm, predict

AUTO, REVIEW = RoutingDecision.AUTO_POST, RoutingDecision.HUMAN_REVIEW


def evidence(gls):
    return [
        RetrievedEvidence(
            invoice_id=str(i), description="x", gl_account=gl, cost_centre="CC300", vendor_id="V",
            similarity_score=0.9, dense_rank=i, sparse_rank=i, rrf_score=0.03,
            confirmation_timestamp=datetime.now(timezone.utc), source=ConfirmationSource.HUMAN_CONFIRM,
        )
        for i, gl in enumerate(gls)
    ]


# ------------------------------------------------------------------ rule gate

def test_route_auto_posts_on_strong_unanimous_evidence():
    decision, _ = route(evidence(["6300"] * 5), "V", 0.9)
    assert decision == AUTO


def test_route_reviews_sparse_evidence():
    decision, reason = route(evidence(["6300"] * 2), "V", 0.99)
    assert decision == REVIEW and "Sparse" in reason


def test_route_reviews_empty_evidence():
    assert route([], "V", 0.0)[0] == REVIEW


def test_route_reviews_uncertain_vendor_identity():
    decision, reason = route(evidence(["6300"] * 5), "V", 0.9, mdm_confidence=0.3)
    assert decision == REVIEW and "identity" in reason


def test_route_reviews_a_slim_plurality():
    # 3 of 5 = 60% agreement: previously enough to auto-post, now not
    decision, reason = route(evidence(["6300"] * 3 + ["6200"] * 2), "V", 0.9)
    assert decision == REVIEW and "disagrees" in reason


def test_route_reviews_low_confidence():
    assert route(evidence(["6300"] * 5), "V", 0.01)[0] == REVIEW


# ------------------------------------------------------------------ governance

def test_agent_flags_force_review_even_when_rule_gate_passes():
    decision, reason = governance.apply_gate(AUTO, "strong", ["vat_blocked"])
    assert decision == REVIEW and "vat_blocked" in reason


def test_governance_never_loosens_a_review():
    assert governance.apply_gate(REVIEW, "weak", [])[0] == REVIEW
    assert governance.never_loosen(REVIEW, "auto_post") == REVIEW


def test_llm_gate_can_tighten_but_not_loosen():
    assert governance.never_loosen(AUTO, "human_review") == REVIEW
    assert governance.never_loosen(AUTO, "garbage") == AUTO


# ------------------------------------------------------------------ end to end

def test_clean_invoice_from_known_vendor_auto_posts(seeded):
    pred = predict(make_invoice())
    assert pred.predicted_gl == "6300"
    assert pred.final_routing == AUTO, pred.final_routing_reason


def test_cold_start_vendor_goes_to_review(seeded):
    pred = predict(make_invoice(vendor="NovaCorp Services Ltd", gl="6400"))
    assert pred.final_routing == REVIEW


def test_unknown_vendor_goes_to_review_and_never_borrows_another_vendors_history(seeded):
    pred = predict(make_invoice(vendor="Meridian East Europe s.r.o."))
    assert pred.final_routing == REVIEW
    assert all(e.vendor_id == pred.resolved_line.canonical_vendor_id for e in pred.evidence)


def test_wrong_entity_is_caught_by_the_agents(seeded):
    pred = predict(make_invoice(entity="EU001"))
    assert pred.final_routing == REVIEW and pred.agent_flags


def test_invalid_cost_centre_is_caught(seeded):
    pred = predict(make_invoice(cost_centre="CC999"))
    assert pred.final_routing == REVIEW and "cost_centre_mismatch" in pred.agent_flags


def test_retrieval_failure_fails_closed(seeded, monkeypatch):
    from ap_automation.core import pipeline

    def boom(*a, **k):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(pipeline, "retrieve", boom)
    pred = predict(make_invoice())
    assert pred.final_routing == REVIEW


def test_confirm_refuses_to_auto_post_a_reviewed_prediction(seeded):
    pred = predict(make_invoice(vendor="NovaCorp Services Ltd", gl="6400"))
    assert pred.final_routing == REVIEW
    with pytest.raises(AutoPostNotPermitted):
        confirm(pred, pred.predicted_gl or "6400", "CC300", "attacker", ConfirmationSource.AUTO_POST)


def test_confirm_refuses_an_auto_post_that_changes_the_gl(seeded):
    pred = predict(make_invoice())
    assert pred.final_routing == AUTO
    with pytest.raises(AutoPostNotPermitted):
        confirm(pred, "7100", "CC300", "system", ConfirmationSource.AUTO_POST)
