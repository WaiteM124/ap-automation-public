"""The write-back policy and the self-learning loop."""

from conftest import make_invoice

from ap_automation.core import database
from ap_automation.core.config import settings
from ap_automation.core.models import ConfirmationSource, RoutingDecision
from ap_automation.core.pipeline import predict
from ap_automation.core.posting import post_outcome
from ap_automation.core.retrieval import get_qdrant


def corpus_size() -> int:
    return get_qdrant().get_collection(settings.qdrant_collection).points_count


def test_auto_posts_are_not_learned_from(seeded):
    pred = predict(make_invoice())
    assert pred.final_routing == RoutingDecision.AUTO_POST
    before = corpus_size()

    res = post_outcome(pred, pred.predicted_gl, pred.predicted_cost_centre, "system",
                       ConfirmationSource.AUTO_POST, mirror_to_sheet=False)

    assert res.erp_posted and res.audited
    assert res.learned is False
    assert corpus_size() == before


def test_human_decisions_are_learned_from(seeded):
    pred = predict(make_invoice(vendor="NovaCorp Services Ltd", gl="6400"))
    before = corpus_size()

    res = post_outcome(pred, "6400", "CC300", "reviewer", ConfirmationSource.HUMAN_CORRECT,
                       mirror_to_sheet=False)

    assert res.learned is True
    assert corpus_size() == before + 1


def test_every_posting_writes_an_audit_record(seeded):
    pred = predict(make_invoice())
    post_outcome(pred, pred.predicted_gl, pred.predicted_cost_centre, "system",
                 ConfirmationSource.AUTO_POST, mirror_to_sheet=False)
    records = database.get_audit_records()
    assert len(records) == 1
    assert records[0]["routing_decision"] == "auto_post"


def test_cold_start_vendor_graduates_to_auto_post_after_human_reviews(seeded):
    """The core claim: review -> human decisions -> the vendor starts to auto-post."""
    vendor = "NovaCorp Services Ltd"
    descriptions = ["Management consulting services", "Legal advisory fees Q1",
                    "Audit and assurance services", "HR consulting project",
                    "Financial advisory retainer", "Management consulting services"]
    routed = []
    for desc in descriptions:
        inv = make_invoice(vendor=vendor, description=desc, gl="6400", amount=3000)
        pred = predict(inv)
        routed.append(pred.final_routing)
        if pred.final_routing != RoutingDecision.AUTO_POST:
            post_outcome(pred, "6400", pred.predicted_cost_centre or "CC300", "reviewer",
                         ConfirmationSource.HUMAN_CORRECT, mirror_to_sheet=False)

    assert routed[0] == RoutingDecision.HUMAN_REVIEW          # no history yet
    assert RoutingDecision.AUTO_POST in routed[3:]            # graduated after a few human decisions
