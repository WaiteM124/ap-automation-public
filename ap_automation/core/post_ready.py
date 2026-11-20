"""
Post-ready output: an ERP-neutral canonical record for every invoice that is cleared to post.

An invoice is *cleared* when the governed gate approved it (final_routing == auto_post) or a human confirmed / corrected
it. Each cleared invoice becomes one flat record that carries everything a downstream ERP needs, plus the decision
trail (routing, confidence, who/what approved it). The same record feeds the CSV / JSON downloads, the UI table and the
"view payload" preview for Xero / SAP.

Two sources, merged and de-duplicated by reference key:

* ledger rows (the ERP posting ledger) - invoices that were confirmed or auto-posted. Whether they have been posted to
  an external system is read from the SAP / Xero posting tables: ONLY a stored external document id makes a record
  "posted". A row that exists solely in the simulated ledger is "ready_to_post": nothing external has it.
* predictions still waiting in the prediction store whose governed decision is auto_post - cleared, not yet posted.

Nothing here posts anything or contacts any external system.
"""

from __future__ import annotations

import csv
import hashlib
import io
from datetime import date, datetime, timezone

from ap_automation.core import chart_of_accounts, database

STATUS_READY = "ready_to_post"
STATUS_POSTED = "posted"
STATUSES = (STATUS_READY, STATUS_POSTED)
MAX_LIMIT = 5000

# Column order of the CSV download and the key order of the JSON records.
FIELDS = (
    "reference_key", "invoice_id", "prediction_id", "vendor_name", "vendor_id", "description", "date", "amount",
    "currency", "gl_account", "gl_name", "cost_centre", "cost_centre_name", "tax_treatment", "entity_id",
    "routing_decision", "routing_reason", "confidence", "approval_source", "approved_by", "approved_by_type",
    "approved_at", "status", "posted_to", "erp_document", "erp_url", "ledger_recorded",
)


def reference_for(invoice_id: object) -> str:
    """
    The idempotency / reference key shared by every ERP backend: "AP" + first 14 hex chars of SHA-256(invoice_id)
    (16 chars, fits SAP's SupplierInvoiceIDByInvcgParty). Deterministic in the invoice id, so the same invoice always
    carries the same key. Limitation: two separate submissions of the same real-world invoice have different invoice
    ids (our data has no supplier invoice number) and therefore different keys.
    """
    return "AP" + hashlib.sha256(str(invoice_id).encode("utf-8")).hexdigest()[:14].upper()


class PostReadyError(ValueError):
    """A bad filter (date format, status)."""


# --------------------------------------------------------------------------- building records

def _day(value: object) -> str:
    """YYYY-MM-DD of an ISO timestamp / datetime / epoch seconds; '' if it cannot be read."""
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, timezone.utc).date().isoformat()
        if isinstance(value, datetime):
            return value.date().isoformat()
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date().isoformat()
    except (TypeError, ValueError, OSError):
        return ""


def _approver_type(source: str, user: str) -> str:
    if source == "auto_post":
        return "system"
    return "human"


def _external(invoice_id: str, sap: dict, xero: dict) -> tuple[str, str, str]:
    """(posted_to, document id, url) - non-empty ONLY when an external system returned a document id."""
    x = xero.get(invoice_id)
    if x and x.get("xero_invoice_id"):
        return "xero", x["xero_invoice_id"], x.get("xero_url") or ""
    s = sap.get(invoice_id)
    if s and s.get("sap_document"):
        return "sap", s["sap_document"], ""
    return "", "", ""


def _record(*, invoice_id: str, prediction_id: str, vendor_name: str, vendor_id: str, description: str, day: str,
            amount: float, currency: str, gl: str, cost_centre: str, tax: str, entity: str, routing: str,
            routing_reason: str, confidence: float | None, approval_source: str, approved_by: str, approved_at: str,
            posted_to: str, erp_document: str, erp_url: str, ledger_recorded: bool) -> dict:
    return {
        "reference_key": reference_for(invoice_id),
        "invoice_id": invoice_id,
        "prediction_id": prediction_id,
        "vendor_name": vendor_name or vendor_id,
        "vendor_id": vendor_id,
        "description": description,
        "date": day,
        "amount": round(float(amount), 2),
        "currency": currency,
        "gl_account": gl,
        "gl_name": chart_of_accounts.GL_NAMES.get(gl, ""),
        "cost_centre": cost_centre,
        "cost_centre_name": chart_of_accounts.COST_CENTRES.get(cost_centre, ""),
        "tax_treatment": tax,
        "entity_id": entity,
        "routing_decision": routing,
        "routing_reason": routing_reason,
        "confidence": None if confidence is None else round(float(confidence), 4),
        "approval_source": approval_source,
        "approved_by": approved_by,
        "approved_by_type": _approver_type(approval_source, approved_by),
        "approved_at": approved_at,
        "status": STATUS_POSTED if erp_document else STATUS_READY,
        "posted_to": posted_to,
        "erp_document": erp_document,
        "erp_url": erp_url,
        "ledger_recorded": ledger_recorded,
        # what the ERP adapters are given (ERPPostingPayload shape). Internal: removed by public().
        "_erp": {
            "invoice_id": invoice_id, "vendor_id": vendor_id, "vendor_name": vendor_name, "entity_id": entity,
            "gl_account": gl, "cost_centre": cost_centre, "tax_code": tax, "amount": float(amount),
            "currency": currency, "description": description, "posted_by": approved_by, "posted_at": approved_at,
            "trace_id": prediction_id,
        },
    }


def public(record: dict) -> dict:
    """The record without internal fields, in the documented key order."""
    return {k: record[k] for k in FIELDS}


def _from_ledger(limit: int) -> list[dict]:
    rows = database.get_ledger(limit)
    audit = {}
    for a in database.get_audit_records(max(limit * 2, 500)):        # newest first: keep the newest SUCCESSFUL one
        if a["source"] not in database.ERP_FAILED_SOURCES:
            audit.setdefault(a["prediction_id"], a)
    sap = {r["invoice_id"]: r for r in database.get_sap_postings(limit=max(limit * 2, 500), status="posted")}
    xero = {r["invoice_id"]: r for r in database.get_xero_postings(limit=max(limit * 2, 500), status="posted")}
    out = []
    for p in rows:
        a = audit.get(p["trace_id"], {})
        source = p.get("source") or a.get("source") or ""
        posted_to, doc, url = _external(p["invoice_id"], sap, xero)
        out.append(_record(
            invoice_id=p["invoice_id"], prediction_id=p["trace_id"], vendor_name=p.get("vendor_name") or "",
            vendor_id=p["vendor_id"], description=p["description"], day=_day(p["posted_at"]), amount=p["amount"],
            currency=p["currency"], gl=p["gl_account"], cost_centre=p["cost_centre"], tax=p["tax_code"],
            entity=p["entity_id"], routing=a.get("routing_decision") or "", routing_reason=a.get("routing_reason") or "",
            confidence=p.get("confidence"), approval_source=source, approved_by=p["posted_by"],
            approved_at=str(p["posted_at"]), posted_to=posted_to, erp_document=doc, erp_url=url, ledger_recorded=True))
    return out


def _from_pending(store, limit: int) -> list[dict]:
    """Predictions the governed gate cleared and nobody has posted yet."""
    out = []
    for pred, created in store.pending_auto_post(limit):
        line = pred.resolved_line.invoice_line
        out.append(_record(
            invoice_id=str(line.invoice_id), prediction_id=str(pred.prediction_id), vendor_name=line.raw_vendor_name,
            vendor_id=pred.resolved_line.canonical_vendor_id, description=line.description, day=_day(created),
            amount=line.amount, currency=line.currency, gl=pred.predicted_gl, cost_centre=pred.predicted_cost_centre,
            tax=pred.provenance.get("vat_treatment") or "UNASSESSED", entity=line.entity_id,
            routing=pred.final_routing.value, routing_reason=pred.final_routing_reason or pred.routing_reason,
            confidence=pred.weighted_confidence, approval_source="auto_post", approved_by="governed gate",
            approved_at=datetime.fromtimestamp(created, timezone.utc).isoformat(), posted_to="", erp_document="",
            erp_url="", ledger_recorded=False))
    return out


def _parse_day(value: str | None, name: str) -> str | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError:
        raise PostReadyError(f"{name} must be a date like 2026-10-27") from None


def build_records(store=None, *, status: str | None = None, date_from: str | None = None, date_to: str | None = None,
                  limit: int = 200) -> list[dict]:
    """
    Cleared invoices, newest first, as internal records (use `public()` for the documented shape).
    `status`: ready_to_post | posted (None / "all" = both). Dates are inclusive and compared to the record `date`.
    """
    status = (status or "").strip().lower() or None
    if status == "all":
        status = None
    if status is not None and status not in STATUSES:
        raise PostReadyError(f"status must be one of: {', '.join(STATUSES)}, all")
    d_from, d_to = _parse_day(date_from, "date_from"), _parse_day(date_to, "date_to")
    if d_from and d_to and d_from > d_to:
        raise PostReadyError("date_from is after date_to")
    limit = min(max(int(limit), 1), MAX_LIMIT)

    records = _from_ledger(MAX_LIMIT)
    seen = {r["reference_key"] for r in records}
    if store is not None:
        records += [r for r in _from_pending(store, MAX_LIMIT) if r["reference_key"] not in seen]

    def keep(r: dict) -> bool:
        return ((status is None or r["status"] == status) and (not d_from or (r["date"] and r["date"] >= d_from))
                and (not d_to or (r["date"] and r["date"] <= d_to)))

    records = [r for r in records if keep(r)]
    records.sort(key=lambda r: (r["approved_at"], r["reference_key"]), reverse=True)
    return records[:limit]


def summarise(records: list[dict]) -> dict:
    totals: dict[str, float] = {}
    for r in records:
        totals[r["currency"]] = round(totals.get(r["currency"], 0.0) + r["amount"], 2)
    return {"count": len(records),
            "ready_to_post": sum(1 for r in records if r["status"] == STATUS_READY),
            "posted": sum(1 for r in records if r["status"] == STATUS_POSTED),
            "total_by_currency": totals}


def find_record(reference_key: str, store=None) -> dict | None:
    for r in build_records(store, limit=MAX_LIMIT):
        if r["reference_key"] == reference_key:
            return r
    return None


# --------------------------------------------------------------------------- downloads

_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _cell(value: object) -> object:
    """Spreadsheet-safe cell: text that would be read as a formula (=, +, -, @) is neutralised with a leading quote."""
    if isinstance(value, str) and value.startswith(_FORMULA_START):
        return "'" + value
    if value is None:
        return ""
    return value


def to_csv(records: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(FIELDS)
    for r in records:
        w.writerow([_cell(r[k]) for k in FIELDS])
    return buf.getvalue()


# --------------------------------------------------------------------------- payload previews

TARGETS = ("neutral", "xero", "sap")


def payload_for(record: dict, target: str) -> dict:
    """
    What would be sent to `target` for this record: {"target", "reference_key", "payload", "error"}.
    A preview only: nothing is sent. A mapping problem (e.g. no SAP supplier configured) is returned as `error` with
    `payload` null rather than guessed around.
    """
    target = (target or "neutral").lower()
    if target not in TARGETS:
        raise PostReadyError(f"target must be one of: {', '.join(TARGETS)}")
    out = {"target": target, "reference_key": record["reference_key"], "payload": None, "error": None}
    if target == "neutral":
        out["payload"] = public(record)
        return out
    try:
        if target == "xero":
            from ap_automation.core import xero_poster

            out["payload"] = xero_poster.build_payload(record["_erp"], demo=xero_poster.is_demo_org())
        else:
            from ap_automation.core import sap_poster

            out["payload"] = sap_poster.build_payload(record["_erp"])
    except Exception as e:   # SapError / XeroError carry a plain-English reason; anything else is shown sanitised
        msg = str(e) if hasattr(e, "kind") else f"{type(e).__name__}: {e}"
        out["error"] = " ".join(msg.split())[:400]
    return out
