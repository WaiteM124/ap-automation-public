"""
Seed the learning corpus with historical, human-coded invoice lines.
Shared by the Streamlit UI and the offline evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable

from ap_automation.core.mdm import resolve_vendor
from ap_automation.core.models import ConfirmationSource, InvoiceLine, amount_to_band
from ap_automation.core.retrieval import index_outcome, point_id_for


def seed_corpus(history: Iterable[InvoiceLine], timestamp: str = "2025-01-01T00:00:00+00:00") -> dict[str, int]:
    """
    Index historical lines as human-confirmed evidence.
    Returns {canonical_vendor_id: number of lines seeded}.
    """
    counts: dict[str, int] = {}
    for inv in history:
        if not inv.ground_truth_gl:
            continue
        vendor_id, _ = resolve_vendor(inv.raw_vendor_name)
        index_outcome({
            "point_id": point_id_for(str(inv.invoice_id)),
            "invoice_id": str(inv.invoice_id),
            "vendor_id": vendor_id,
            "vendor_name": inv.raw_vendor_name,
            "entity_id": inv.entity_id,
            "amount_band": amount_to_band(inv.amount).value,
            "amount": inv.amount,
            "currency": inv.currency,
            "gl_account": inv.ground_truth_gl,
            "cost_centre": inv.cost_centre or "",
            "tax_code": "STANDARD",
            "description": inv.description,
            "confirming_user_id": "seed",
            "confirmation_timestamp": timestamp,
            "prediction_confidence": 1.0,
            "source": ConfirmationSource.HUMAN_CONFIRM.value,
        })
        counts[vendor_id] = counts.get(vendor_id, 0) + 1
    return counts
