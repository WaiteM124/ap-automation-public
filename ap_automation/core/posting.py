"""
Posting: the one code path that turns a confirmation into an ERP posting, an audit
record and (for human decisions) a learning-corpus entry.

The API, the batch job, the Streamlit UI and the evaluation all call `post_outcome`,
so every route has identical behaviour. Previously each caller re-implemented this,
and nothing ever wrote the audit trail the README promises.

The ERP is pluggable (ERP_BACKEND): "simulated" (default) writes the local ledger / ERP_URL service;
"sap" posts to SAP S/4HANA Cloud through core/sap_poster.py; "xero" creates a bill in Xero through
core/xero_poster.py. A post that the external system does not accept leaves the invoice UNPOSTED: it is audited with
that system's (sanitised) error, it is not learned from and not mirrored, and the caller keeps the invoice in human
review (PostResult.erp_failed).
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from ap_automation.core import clock, database, sap_poster, xero_poster
from ap_automation.core.config import settings
from ap_automation.core.models import (
    ConfirmationSource,
    ConfirmedOutcome,
    ERPPostingPayload,
    Prediction,
)
from ap_automation.core.pipeline import confirm, write_back

_db_ready = False


@dataclass
class PostResult:
    outcome: ConfirmedOutcome
    erp_payload: ERPPostingPayload
    erp_posted: bool
    audited: bool
    learned: bool          # True if the line was added to the learning corpus
    error: str | None = None
    erp_backend: str = "simulated"
    erp_failed: bool = False       # SAP / Xero did not accept the post: the invoice must stay in review
    erp_ref: dict | None = None    # external system's result (document, reference, error...). No secrets.


def _ensure_db() -> None:
    global _db_ready
    if not _db_ready:
        database.init_db()
        _db_ready = True


def _post_to_erp(payload: dict) -> None:
    if settings.erp_url:
        resp = httpx.post(f"{settings.erp_url.rstrip('/')}/post-invoice", json=payload, timeout=5)
        resp.raise_for_status()
    else:
        _ensure_db()
        database.post_to_erp_ledger(payload)


def external_backend() -> str | None:
    """
    "sap" or "xero" when that backend is selected, else None (the simulated ledger). A simulated clock (replay
    experiments) never posts to an external system, whatever ERP_BACKEND says.
    """
    if clock.is_simulated():
        return None
    if sap_poster.backend_selected():
        return "sap"
    if xero_poster.backend_selected():
        return "xero"
    return None


def use_sap() -> bool:
    return external_backend() == "sap"


_LABEL = {"sap": "SAP", "xero": "Xero"}
_FAILED_SOURCE = {"sap": database.SAP_FAILED_SOURCE, "xero": database.XERO_FAILED_SOURCE}


def post_outcome(
    prediction: Prediction,
    confirmed_gl: str,
    confirmed_cc: str,
    confirming_user_id: str,
    source: ConfirmationSource,
    mirror_to_sheet: bool = True,
) -> PostResult:
    """
    Confirm -> post to ERP -> write audit record -> learn (human decisions only).

    Raises AutoPostNotPermitted if source is AUTO_POST but the governed gate did not
    approve the prediction. ERP / audit / Sheets failures are reported on the result,
    not swallowed silently.
    """
    outcome, erp = confirm(prediction, confirmed_gl, confirmed_cc, confirming_user_id, source)
    payload = erp.model_dump(mode="json")
    errors: list[str] = []

    backend = external_backend()
    label = _LABEL.get(backend or "", "")
    ext_result: sap_poster.SapPostResult | xero_poster.XeroPostResult | None = None
    erp_posted = False
    if backend:
        ext_result = (sap_poster if backend == "sap" else xero_poster).post_invoice(payload)
        erp_posted = ext_result.ok
        if erp_posted:
            try:  # the local ledger is the ERP page's data source; one row per invoice, even on an idempotent replay
                _ensure_db()
                if not database.has_posting(payload["invoice_id"]):
                    database.post_to_erp_ledger(payload)
            except Exception as e:
                errors.append(f"ledger: {e}")
                print(f"[Posting] ledger write failed: {e}")
        else:
            errors.append(f"{backend}: {ext_result.error}")
            print(f"[Posting] {label} post not completed ({ext_result.status}): {ext_result.error}")
    else:
        try:
            _post_to_erp(payload)
            erp_posted = True
        except Exception as e:
            errors.append(f"erp: {e}")
            print(f"[Posting] ERP post failed: {e}")

    ext_failed = bool(backend) and not erp_posted

    provenance = dict(prediction.provenance)
    audit_source, routing_decision, routing_reason = (
        source.value, prediction.final_routing.value, prediction.final_routing_reason or prediction.routing_reason)
    if ext_result is not None:
        provenance["erp"] = ext_result.audit_view()
    if ext_failed:
        # Recorded as a failed attempt, not as a posting: the invoice is back in review with the system's error.
        audit_source, routing_decision = _FAILED_SOURCE[backend], "human_review"
        routing_reason = f"{label} post not completed ({ext_result.status}): {ext_result.error}"

    audited = False
    try:
        _ensure_db()
        database.write_audit({
            "prediction_id": str(prediction.prediction_id),
            "invoice_id": str(outcome.invoice_id),
            "vendor_id": outcome.canonical_vendor_id,
            "predicted_gl": prediction.predicted_gl,
            "confirmed_gl": confirmed_gl,
            "confidence": prediction.weighted_confidence,
            "routing_decision": routing_decision,
            "routing_reason": routing_reason,
            "source": audit_source,
            "confirming_user": confirming_user_id,
            "provenance": provenance,
        })
        audited = True
    except Exception as e:
        errors.append(f"audit: {e}")
        print(f"[Posting] audit write failed: {e}")

    learned = False
    if not ext_failed:  # nothing is learned from, or mirrored for, an invoice the external system did not take
        try:
            learned = write_back(outcome)
        except Exception as e:
            errors.append(f"write_back: {e}")
            print(f"[Posting] write-back failed: {e}")

        if mirror_to_sheet:
            try:
                from ap_automation.core.sheets import post_to_sheet
                post_to_sheet(payload)
            except Exception as e:  # Sheets is an optional mirror
                print(f"[Posting] Sheets mirror skipped: {e}")

    return PostResult(outcome, erp, erp_posted, audited, learned, "; ".join(errors) or None,
                      erp_backend=backend or "simulated", erp_failed=ext_failed,
                      erp_ref=ext_result.audit_view() if ext_result is not None else None)
