"""
Demo support: a scripted set of scenarios, a deterministic reset, and a pre-flight check.

The scenarios are not canned output. Each one is a real invoice that goes through the real
pipeline; `expect` records what the system is supposed to do, and tests/test_demo_story.py runs
the whole script so the demo cannot silently drift out of shape.
"""

from __future__ import annotations

from ap_automation.core import chart_of_accounts, database, dataset, embeddings, retrieval
from ap_automation.core.chart_of_accounts import CHART
from ap_automation.core.config import settings
from ap_automation.core.seeding import seed_corpus

_CC = {gl: acct.default_cost_centre for gl, acct in CHART.items()}


def _inv(vendor: str, description: str, amount: float, gl: str, *, cc: str | None = None,
         entity: str = "UK001") -> dict:
    return {
        "raw_vendor_name": vendor, "description": description, "amount": amount, "currency": "GBP",
        "entity_id": entity, "cost_centre": cc or _CC[gl],
    }


_NOVA = "NovaCorp Services Ltd"
_NOVA_LINES = [
    ("Management consulting services", 3200),
    ("Legal advisory fees Q3", 4500),
    ("Audit and assurance services", 5200),
    ("HR consulting project", 2800),
    ("Financial advisory retainer", 3900),
]

SCENARIOS: list[dict] = [
    {
        "id": "routine", "group": "Auto-post", "title": "Routine invoice, known vendor",
        "narrative": "Strong, unanimous history for this vendor and every agent agrees, so it is cleared to auto-post.",
        "invoice": _inv("Meridian Facilities Ltd", "Office cleaning services monthly", 1200, "6300"),
        "expect": {"final_routing": "auto_post", "gl": "6300"},
        "reviewer_gl": "6300",
    },
    {
        "id": "vat_block", "group": "Safety", "title": "Confidence is high, but VAT is blocked",
        "narrative": "The vendor history would auto-post this. The VAT agent spots client entertainment "
                     "(input VAT not recoverable) and overrides it.",
        "invoice": _inv("Meridian Facilities Ltd", "Client hospitality dinner and entertainment", 640, "6300",
                    cc="CC100"),  # blocked VAT => the cost-centre agent expects CC100, so only the VAT flag fires
        "expect": {"final_routing": "human_review", "flag": "vat_blocked"},
        "reviewer_gl": "6300",
    },
    {
        "id": "wrong_cc", "group": "Safety", "title": "Wrong cost centre on the invoice",
        "narrative": "Right vendor, right GL - but the submitted cost centre does not match the chart of accounts.",
        "invoice": _inv("Meridian Facilities Ltd", "Office cleaning services monthly", 1200, "6300", cc="CC500"),
        "expect": {"final_routing": "human_review", "flag": "cost_centre_mismatch"},
        "reviewer_gl": "6300",
    },
    {
        "id": "unknown", "group": "Safety", "title": "Vendor the system has never seen",
        "narrative": "No vendor-master match and no history. The system does not guess and does not borrow "
                     "another vendor's history.",
        "invoice": _inv("Zephyr Global Holdings", "Professional advisory services Q3", 18500, "6400"),
        "expect": {"final_routing": "human_review"},
        "reviewer_gl": "6400",
    },
] + [
    {
        "id": f"learn_{i}", "group": "Learning", "title": f"New vendor - invoice {i} of {len(_NOVA_LINES)}",
        "narrative": ("NovaCorp is in the vendor master but has no coded history yet. Confirm each invoice as a "
                      "reviewer: after a few human decisions the vendor starts to auto-post."
                      if i == 1 else "Confirm this one as the reviewer, then run the next."),
        "invoice": _inv(_NOVA, desc, amt, "6400"),
        "expect": {"final_routing": "human_review" if i <= 3 else "auto_post"},
        "reviewer_gl": "6400",
        "learning_step": i,
    }
    for i, (desc, amt) in enumerate(_NOVA_LINES, start=1)
]


def reset(store, per_vendor: int = 15) -> dict:
    """Return the system to a known starting state: seeded corpus, empty ledger, no pending predictions."""
    # The demo story is written against the built-in chart; drop any client chart first.
    chart_of_accounts.reset_chart(settings.chart_path)
    retrieval.reset_collection()
    seeded = seed_corpus(dataset.generate_history(per_vendor))
    database.clear_all()
    store.clear()
    return {"corpus_size": sum(seeded.values()), "vendors_seeded": len(seeded)}


def status(store) -> dict:
    """Pre-flight: everything a demo depends on, with a plain-language reason when it is not ready."""
    checks: list[dict] = []

    def check(name: str, fn, *, required: bool = True) -> None:
        try:
            ok, detail = fn()
        except Exception as e:  # a failing probe is itself the answer
            ok, detail = False, f"{type(e).__name__}: {str(e)[:120]}"
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "required": required})

    def corpus():
        n = retrieval._collection_size()
        return n > 0, f"{n:,} human-confirmed lines" if n else "corpus is empty - reset or seed it"

    def ledger():
        database.init_db()
        return True, f"{len(database.get_postings(10000))} postings recorded"

    def predictions():
        return True, f"{store.count()} predictions pending"

    check("Corpus", corpus)
    check("ERP ledger and audit trail", ledger)
    check("Prediction store", predictions)
    check("LLM gate", lambda: (bool(settings.anthropic_api_key),
                               "Claude gate active" if settings.anthropic_api_key
                               else "off - the rule gate and agents decide alone (deterministic)"),
          required=False)

    return {
        "ready": all(c["ok"] for c in checks if c["required"]),
        "checks": checks,
        "embeddings": embeddings.backend_name(),
        "demo_mode": settings.demo_mode,
        "corpus_size": retrieval._collection_size(),
    }
