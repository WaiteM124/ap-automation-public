"""
SAP S/4HANA Cloud posting backend: non-PO supplier invoices with G/L account lines.

Target: OData v2 service API_SUPPLIERINVOICE_PROCESS_SRV, entity A_SupplierInvoice, with the G/L lines deep-inserted
through the navigation property to_SupplierInvoiceItemGLAcct (entity A_SupplierInvoiceItemGLAcct).

Safety properties (each is covered by tests/test_sap_poster.py):

* OFF unless ERP_BACKEND=sap AND SAP_POSTING_ENABLED=true. Anything else is refused before a single request is made.
* Sandbox only: a host other than sandbox.api.sap.com is refused unless SAP_ALLOW_NON_SANDBOX=true as well.
* Idempotent. Every invoice has one reference key (SupplierInvoiceIDByInvcgParty, derived from the invoice id).
  A local table (sap_postings) remembers what happened to it, and SAP itself is asked "does this reference
  already exist?" before every POST, so a lost response or a wiped local database cannot cause a second document.
* A POST is never blindly retried. Only requests that provably never reached SAP (connection could not be
  established) and idempotent GETs are retried, a bounded number of times. If a POST may have been processed
  (timeout after sending, 5xx) the outcome is reconciled by reading the reference back; if that is inconclusive the
  post is recorded as "unknown" and is re-checked before any further attempt.
* Failures are never raised into the caller and never lost: they come back as a result carrying a sanitised
  message (secrets removed, no raw headers, bounded length) and are persisted in sap_postings.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from urllib.parse import urlparse

import httpx

from ap_automation.core import database
from ap_automation.core.config import settings
from ap_automation.core.post_ready import reference_for

SERVICE_PATH = "/sap/opu/odata/sap/API_SUPPLIERINVOICE_PROCESS_SRV"
SANDBOX_HOSTS = frozenset({"sandbox.api.sap.com"})
GL_NAV = "to_SupplierInvoiceItemGLAcct"

# Fields of the created / found invoice that are kept as the "response summary" (nothing else is stored).
_SUMMARY_FIELDS = ("SupplierInvoice", "FiscalYear", "CompanyCode", "InvoicingParty", "SupplierInvoiceIDByInvcgParty",
                   "DocumentCurrency", "InvoiceGrossAmount", "SupplierInvoiceStatus", "JournalEntry")

# Safe, idempotent requests may be retried on these; a POST never is (see _post_invoice).
_RETRY_STATUS = frozenset({429, 502, 503, 504})
_MAX_ERROR_CHARS = 600

_transport: httpx.BaseTransport | None = None   # tests install httpx.MockTransport here
_sleep = time.sleep                              # tests replace it so retries do not wait


def set_transport(transport: httpx.BaseTransport | None) -> None:
    """Route every SAP request through `transport` (tests). None restores the real network."""
    global _transport
    _transport = transport


# --------------------------------------------------------------------------- errors / results

class SapError(Exception):
    """A post could not be completed. `kind`: config | mapping | rejected | unavailable | unknown_outcome."""

    def __init__(self, message: str, kind: str, http_status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.http_status = http_status


@dataclass
class SapPostResult:
    ok: bool
    status: str                      # posted | already_posted | failed | unknown | in_progress | disabled
    reference: str = ""
    document: str = ""               # SAP supplier invoice number
    fiscal_year: str = ""
    company_code: str = ""
    http_status: int | None = None
    error: str = ""                  # sanitised; empty when ok
    kind: str = ""                   # error kind when not ok
    replayed: bool = False           # True when this call did not create anything new (idempotent replay)
    summary: dict = field(default_factory=dict)

    def audit_view(self) -> dict:
        """What goes into the audit trail / API responses. Contains no secrets by construction."""
        return {"backend": "sap", "status": self.status, "reference": self.reference, "document": self.document,
                "fiscal_year": self.fiscal_year, "company_code": self.company_code, "replayed": self.replayed,
                "http_status": self.http_status, "error": self.error, "summary": self.summary}


# --------------------------------------------------------------------------- sanitising

def _secrets(extra: tuple[str, ...] = ()) -> list[str]:
    vals = [settings.sap_api_key, settings.sap_password, *extra]
    return sorted({v for v in vals if v and len(v) >= 4}, key=len, reverse=True)


def sanitize(text: object, extra_secrets: tuple[str, ...] = ()) -> str:
    """Make `text` safe to store, log and return: configured secrets and CSRF tokens removed, one line, bounded."""
    s = str(text)
    for secret in _secrets(extra_secrets):
        s = s.replace(secret, "***")
    s = re.sub(r"(?i)(apikey|x-csrf-token|authorization|password|cookie)\s*[:=]\s*\S+", r"\1=***", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= _MAX_ERROR_CHARS else s[: _MAX_ERROR_CHARS - 3] + "..."


# --------------------------------------------------------------------------- configuration / status

def _json_map(raw: str, name: str) -> dict[str, str]:
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("not an object")
        return {str(k): str(v) for k, v in data.items()}
    except ValueError:
        raise SapError(f"{name} is not a valid JSON object", "config") from None


def host() -> str:
    return (urlparse(settings.sap_base_url).hostname or "").lower()


def is_sandbox() -> bool:
    return host() in SANDBOX_HOSTS


def backend_selected() -> bool:
    return settings.erp_backend.strip().lower() == "sap"


def has_credentials() -> bool:
    return bool(settings.sap_api_key or (settings.sap_username and settings.sap_password))


def check_ready() -> None:
    """Raise SapError(kind='config') unless posting to SAP is explicitly and safely configured."""
    if not backend_selected():
        raise SapError("ERP_BACKEND is not 'sap'", "config")
    if not settings.sap_posting_enabled:
        raise SapError("SAP posting is disabled (set SAP_POSTING_ENABLED=true to enable it)", "config")
    if urlparse(settings.sap_base_url).scheme != "https" or not host():
        raise SapError("SAP_BASE_URL must be an https URL", "config")
    if not is_sandbox() and not settings.sap_allow_non_sandbox:
        raise SapError(f"Refusing to post to {host()}: only the SAP sandbox is permitted "
                       "(SAP_ALLOW_NON_SANDBOX is not set)", "config")
    if not has_credentials():
        raise SapError("SAP credentials are not configured (SAP_API_KEY, or SAP_USERNAME + SAP_PASSWORD)", "config")


def status() -> dict:
    """Configuration state for GET /sap/status. Host only, never a key, password or full URL."""
    try:
        check_ready()
        ready, reason = True, ""
    except SapError as e:
        ready, reason = False, str(e)
    return {
        "backend": "sap" if backend_selected() else "simulated",
        "configured": has_credentials(),
        "auth": "basic" if settings.sap_username and settings.sap_password
                else "api_key" if settings.sap_api_key else "none",
        "posting_enabled": bool(settings.sap_posting_enabled),
        "ready_to_post": ready,
        "not_ready_reason": reason,
        "host": host(),
        "sandbox": is_sandbox(),
        "non_sandbox_allowed": bool(settings.sap_allow_non_sandbox),
        "default_company_code": settings.sap_company_code,
        "mappings": {name: len(_safe_map(raw)) for name, raw in (
            ("company_code", settings.sap_company_code_map), ("supplier", settings.sap_supplier_map),
            ("gl_account", settings.sap_gl_map), ("cost_centre", settings.sap_cost_centre_map),
            ("tax_code", settings.sap_tax_code_map))},
    }


def _safe_map(raw: str) -> dict:
    try:
        return _json_map(raw, "map")
    except SapError:
        return {}


# --------------------------------------------------------------------------- mapping

# reference_for (imported above) is the key shared by every ERP backend: see core/post_ready.py.


def _odata_date(dt: datetime) -> str:
    midnight = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
    return f"/Date({int(midnight.timestamp() * 1000)})/"


def _posting_datetime(erp: dict) -> datetime:
    raw = erp.get("posted_at")
    try:
        dt = raw if isinstance(raw, datetime) else datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def build_payload(erp: dict) -> dict:
    """
    Map our ERP payload (ERPPostingPayload as a dict) to an A_SupplierInvoice create request with one G/L line.
    Raises SapError(kind='mapping') when something SAP requires cannot be derived.
    """
    company_map = _json_map(settings.sap_company_code_map, "SAP_COMPANY_CODE_MAP")
    supplier_map = _json_map(settings.sap_supplier_map, "SAP_SUPPLIER_MAP")
    gl_map = _json_map(settings.sap_gl_map, "SAP_GL_MAP")
    cc_map = _json_map(settings.sap_cost_centre_map, "SAP_COST_CENTRE_MAP")
    tax_map = _json_map(settings.sap_tax_code_map, "SAP_TAX_CODE_MAP")

    company = company_map.get(str(erp.get("entity_id", ""))) or settings.sap_company_code
    supplier = supplier_map.get(str(erp.get("vendor_id", ""))) or settings.sap_default_supplier
    if not company:
        raise SapError("No SAP company code: set SAP_COMPANY_CODE or SAP_COMPANY_CODE_MAP", "mapping")
    if not supplier:
        raise SapError(f"No SAP supplier mapped for vendor {erp.get('vendor_id')!r} "
                       "(add it to SAP_SUPPLIER_MAP, or set SAP_DEFAULT_SUPPLIER for the sandbox)", "mapping")

    try:
        amount = Decimal(str(erp.get("amount"))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except Exception:
        raise SapError("Invoice amount is not a valid number", "mapping") from None
    if amount <= 0:
        raise SapError("Invoice amount must be greater than zero (credit memos are not supported)", "mapping")

    treatment = str(erp.get("tax_code") or "")
    tax_code = tax_map.get(treatment)
    if tax_code is None:
        if not is_sandbox():  # a real tenant must never get a guessed tax code
            raise SapError(f"No SAP tax code mapped for VAT treatment {treatment!r} (SAP_TAX_CODE_MAP)", "mapping")
        tax_code = settings.sap_default_tax_code

    gl = gl_map.get(str(erp.get("gl_account", ""))) or str(erp.get("gl_account", ""))
    cost_centre = cc_map.get(str(erp.get("cost_centre", ""))) or str(erp.get("cost_centre", ""))
    if not gl:
        raise SapError("No G/L account on the invoice", "mapping")

    currency = (erp.get("currency") or settings.sap_default_currency or "").upper()
    text = " ".join(str(erp.get("description") or "").split())
    when = _odata_date(_posting_datetime(erp))
    amount_s = format(amount, "f")

    item = {
        "SupplierInvoiceItem": "1",
        "CompanyCode": company,
        "GLAccount": gl,
        "DocumentCurrency": currency,
        "SupplierInvoiceItemAmount": amount_s,
        "DebitCreditCode": "S",
        "TaxCode": tax_code,
        "SupplierInvoiceItemText": text[:50].rstrip(),
    }
    if cost_centre:
        item["CostCenter"] = cost_centre
    return {
        "CompanyCode": company,
        "DocumentDate": when,
        "PostingDate": when,
        "SupplierInvoiceIDByInvcgParty": reference_for(erp.get("invoice_id")),
        "InvoicingParty": supplier,
        "DocumentCurrency": currency,
        "InvoiceGrossAmount": amount_s,
        "DocumentHeaderText": text[:25].rstrip(),
        "TaxIsCalculatedAutomatically": bool(settings.sap_tax_calculated_automatically),
        GL_NAV: {"results": [item]},
    }


# --------------------------------------------------------------------------- HTTP

def _sap_message(resp: httpx.Response) -> str:
    """The human-readable reason SAP (or the API gateway in front of it) gave. Bodies are parsed, never echoed raw."""
    try:
        body = resp.json()
    except ValueError:
        return f"HTTP {resp.status_code} (non-JSON response)"
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        err = body["error"]
        msg = err.get("message")
        msg = msg.get("value") if isinstance(msg, dict) else msg
        parts = [f"SAP {err.get('code', '')}: {msg or 'no message'}".replace("SAP : ", "SAP: ")]
        details = (err.get("innererror") or {}).get("errordetails") or []
        extra = [f"{d.get('code', '')}: {d.get('message', '')}".strip(": ") for d in details
                 if isinstance(d, dict) and d.get("message") and d.get("message") != msg]
        if extra:
            parts.append("details: " + " | ".join(extra[:5]))
        return f"HTTP {resp.status_code} " + " ".join(parts)
    if isinstance(body, dict) and isinstance(body.get("fault"), dict):  # API gateway (e.g. invalid API key)
        return f"HTTP {resp.status_code} gateway: {body['fault'].get('faultstring', 'fault')}"
    if isinstance(body, dict) and isinstance(body.get("message"), str):  # API Business Hub sandbox: GET-only
        return f"HTTP {resp.status_code} gateway {body.get('code', '')}: {body['message']}".replace("gateway : ", "gateway: ")
    return f"HTTP {resp.status_code} (unrecognised error body)"


class _Session:
    """One SAP conversation: a cookie-keeping HTTP client plus the CSRF token."""

    def __init__(self) -> None:
        self.base = settings.sap_base_url.rstrip("/") + SERVICE_PATH
        headers = {"Accept": "application/json"}
        if settings.sap_api_key:
            headers["APIKey"] = settings.sap_api_key
        auth = (settings.sap_username, settings.sap_password) if settings.sap_username and settings.sap_password else None
        self.http = httpx.Client(headers=headers, auth=auth, timeout=httpx.Timeout(settings.sap_timeout_seconds),
                                 transport=_transport, follow_redirects=False)
        self.token: str | None = None

    def close(self) -> None:
        self.http.close()

    def _safe_get(self, url: str, **kw) -> httpx.Response:
        """GET with bounded retries on transport errors / 429 / 502 / 503 / 504. Only ever used for idempotent reads."""
        attempts = 1 + max(0, settings.sap_max_retries)
        last: Exception | None = None
        for i in range(attempts):
            try:
                resp = self.http.get(url, **kw)
                if resp.status_code not in _RETRY_STATUS or i == attempts - 1:
                    return resp
            except httpx.TransportError as e:
                last = e
                if i == attempts - 1:
                    raise SapError(f"SAP unreachable: {type(e).__name__}", "unavailable") from None
            _sleep(min(0.5 * 2 ** i, 4.0))
        raise SapError(f"SAP unreachable: {type(last).__name__ if last else 'retries exhausted'}", "unavailable")

    def fetch_csrf(self) -> None:
        """OData v2 CSRF: a GET with 'X-CSRF-Token: Fetch' returns the token; cookies from it must accompany the POST."""
        resp = self._safe_get(self.base + "/", headers={"X-CSRF-Token": "Fetch"})
        if resp.status_code >= 400:
            raise SapError(_sap_message(resp), "rejected" if resp.status_code < 500 else "unavailable",
                           resp.status_code)
        token = resp.headers.get("x-csrf-token")
        self.token = token if token and token.lower() != "required" else None

    def find_by_reference(self, company: str, supplier: str, reference: str) -> dict | None:
        """The invoice already carrying this reference, or None. Raises SapError if SAP cannot be asked."""
        flt = (f"CompanyCode eq '{_q(company)}' and InvoicingParty eq '{_q(supplier)}' "
               f"and SupplierInvoiceIDByInvcgParty eq '{_q(reference)}'")
        resp = self._safe_get(self.base + "/A_SupplierInvoice", params={
            "$filter": flt, "$top": "5", "$format": "json",
            "$select": ",".join(_SUMMARY_FIELDS + ("IsReversed",)),
        })
        if resp.status_code >= 400:
            raise SapError(_sap_message(resp), "rejected" if resp.status_code < 500 else "unavailable",
                           resp.status_code)
        try:
            rows = resp.json()["d"]["results"]
        except (ValueError, KeyError, TypeError):
            raise SapError("Unexpected response while checking for an existing invoice", "unavailable") from None
        live = [r for r in rows if not r.get("IsReversed")]
        if not live and rows:
            raise SapError(f"An invoice with reference {reference} exists but is reversed; manual action required",
                           "rejected")
        return live[0] if live else None

    def create(self, payload: dict) -> httpx.Response:
        """
        POST once. Retried only when the request provably never left this machine (connection not established), and
        once more after a CSRF-token rejection (SAP refuses those before processing anything).
        """
        refreshed = False
        connect_attempts = 1 + max(0, settings.sap_max_retries)
        i = 0
        while True:
            headers = {"Content-Type": "application/json"}
            if self.token:
                headers["X-CSRF-Token"] = self.token
            try:
                resp = self.http.post(self.base + "/A_SupplierInvoice", json=payload, headers=headers)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                i += 1
                if i >= connect_attempts:
                    raise SapError("SAP unreachable: connection could not be established", "unavailable") from None
                _sleep(min(0.5 * 2 ** i, 4.0))
                continue
            csrf_rejected = (resp.status_code == 403 and
                             (resp.headers.get("x-csrf-token", "").lower() == "required" or "csrf" in resp.text.lower()))
            if csrf_rejected and not refreshed:
                refreshed = True
                self.fetch_csrf()
                continue
            return resp


def _q(value: str) -> str:
    return str(value).replace("'", "''")


def _summary(d: dict) -> dict:
    return {k: d[k] for k in _SUMMARY_FIELDS if d.get(k) not in (None, "")}


# --------------------------------------------------------------------------- orchestration

def post_invoice(erp: dict) -> SapPostResult:
    """
    Post one invoice to SAP exactly once. Never raises: the outcome is the returned result, and it is persisted.

    `erp` is ERPPostingPayload as a dict. Calling this again for the same invoice is safe and does not create a
    second document.
    """
    reference = reference_for(erp.get("invoice_id"))
    try:
        check_ready()
    except SapError as e:
        return SapPostResult(False, "disabled", reference, error=sanitize(e), kind=e.kind)

    claimed, row = database.sap_claim(reference, str(erp.get("invoice_id")), str(erp.get("trace_id") or ""))
    if not claimed:
        if row["status"] == "posted":
            return SapPostResult(True, "already_posted", reference, row["sap_document"], row["fiscal_year"],
                                 row["company_code"], replayed=True)
        return SapPostResult(False, "in_progress", reference, kind="in_progress",
                             error="Another post of this invoice is in progress; not posting again")

    session = None
    try:
        payload = build_payload(erp)
        company, supplier = payload["CompanyCode"], payload["InvoicingParty"]
        session = _Session()

        existing = session.find_by_reference(company, supplier, reference)
        if existing is not None:
            return _finish_ok(reference, existing, None, replayed=True)

        session.fetch_csrf()
        try:
            resp = session.create(payload)
        except httpx.TransportError as e:   # request was sent; SAP may or may not have processed it
            return _reconcile(session, reference, company, supplier, f"no response ({type(e).__name__})", None)

        if resp.status_code == 201:
            try:
                created = resp.json()["d"]
            except (ValueError, KeyError, TypeError):
                return _reconcile(session, reference, company, supplier, "201 with an unreadable body", 201)
            return _finish_ok(reference, created, 201, replayed=False)
        if resp.status_code >= 500 or resp.status_code == 408:
            return _reconcile(session, reference, company, supplier, f"HTTP {resp.status_code}", resp.status_code)

        # 4xx: SAP (or its gateway) refused the document. Nothing was created.
        msg = sanitize(_sap_message(resp), (session.token or "",))
        database.sap_finish(reference, "failed", error=msg)
        print(f"[SAP] post rejected for {reference}: {msg}")
        return SapPostResult(False, "failed", reference, http_status=resp.status_code, error=msg, kind="rejected")
    except SapError as e:
        msg = sanitize(e)
        status = "unknown" if e.kind == "unknown_outcome" else "failed"
        database.sap_finish(reference, status, error=msg)
        print(f"[SAP] post not completed for {reference} ({e.kind}): {msg}")
        return SapPostResult(False, status, reference, http_status=e.http_status, error=msg, kind=e.kind)
    except Exception as e:  # defensive: never lose the invoice, never leak a secret through str(e)
        msg = sanitize(f"{type(e).__name__}: {e}")
        database.sap_finish(reference, "unknown", error=msg)
        print(f"[SAP] unexpected error for {reference}: {msg}")
        return SapPostResult(False, "unknown", reference, error=msg, kind="unknown_outcome")
    finally:
        if session is not None:
            session.close()


def _finish_ok(reference: str, doc: dict, http_status: int | None, replayed: bool) -> SapPostResult:
    summary = _summary(doc)
    database.sap_finish(reference, "posted", sap_document=str(doc.get("SupplierInvoice", "")),
                        fiscal_year=str(doc.get("FiscalYear", "")), company_code=str(doc.get("CompanyCode", "")),
                        summary=summary)
    return SapPostResult(True, "already_posted" if replayed else "posted", reference,
                         str(doc.get("SupplierInvoice", "")), str(doc.get("FiscalYear", "")),
                         str(doc.get("CompanyCode", "")), http_status, replayed=replayed, summary=summary)


def _reconcile(session: _Session, reference: str, company: str, supplier: str, why: str,
               http_status: int | None) -> SapPostResult:
    """A POST may or may not have been processed: ask SAP by reference instead of guessing or re-posting."""
    try:
        found = session.find_by_reference(company, supplier, reference)
    except SapError:
        found = None
        checked = False
    else:
        checked = True
    if found is not None:
        print(f"[SAP] {reference}: POST outcome was uncertain ({why}) but SAP has the invoice; not re-posting")
        return _finish_ok(reference, found, http_status, replayed=False)
    msg = sanitize(f"SAP post outcome unconfirmed ({why}); "
                   + ("no invoice with this reference was found yet" if checked else "SAP could not be queried")
                   + ". It was NOT re-posted; it will be checked by reference before any further attempt")
    database.sap_finish(reference, "unknown", error=msg)
    print(f"[SAP] {reference}: {msg}")
    return SapPostResult(False, "unknown", reference, http_status=http_status, error=msg, kind="unknown_outcome")
