"""
Xero posting backend: accounts-payable bills (Accounting API, `Invoices` with Type ACCPAY).

Target: `PUT https://api.xero.com/api.xro/2.0/Invoices` (operation `createInvoices`: create only). `POST` on the same
path is `updateOrCreateInvoices`, which can modify an existing bill; this adapter deliberately uses the create-only
verb so that it can never overwrite a bill. The Contact is given by NAME: Xero matches an existing contact of that name
or creates one, so there is no supplier-number mapping problem as with SAP.

Safety properties (each is covered by tests/test_xero_poster.py), mirroring sap_poster.py:

* OFF unless ERP_BACKEND=xero AND XERO_POSTING_ENABLED=true AND the app has been connected with
  scripts/xero_authorize.py. Anything else is refused before a single request is made.
* Idempotent. Every invoice has one reference key (core/post_ready.reference_for), used as the bill's InvoiceNumber.
  A local table (xero_postings) remembers what happened to it, and Xero itself is asked "is there a bill with this
  InvoiceNumber?" before every create, so a lost response or a wiped local database cannot cause a second bill.
  (Xero does not enforce uniqueness of InvoiceNumber on bills, so this check is what protects us.) The create also
  carries an Idempotency-Key header, which Xero documents (max 128 chars); its server-side behaviour is not
  documented in detail, so it is belt and braces, not the guarantee.
* A bill creation is never blindly retried. Only requests that provably never reached Xero (connection could not be
  established, HTTP 429 which Xero rejects before processing, HTTP 401) and idempotent GETs are retried, a bounded
  number of times. If a create may have been processed (timeout after sending, 5xx) the outcome is reconciled by
  reading the InvoiceNumber back; if that is inconclusive the post is recorded as "unknown" and is re-checked before
  any further attempt.
* Failures are never raised into the caller and never lost: they come back as a result carrying a sanitised message
  (secrets and tokens removed, bounded length) and are persisted in xero_postings.
* Tax is never guessed on a real organisation: a VAT treatment with no entry in XERO_TAX_TYPE_MAP is an error unless
  the connected organisation is the Xero Demo Company.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from urllib.parse import quote

import httpx

from ap_automation.core import chart_of_accounts, database, xero_auth
from ap_automation.core.config import settings
from ap_automation.core.post_ready import reference_for
from ap_automation.core.xero_auth import XeroError, sanitize

API_BASE = "https://api.xero.com/api.xro/2.0"
BILL_STATUSES = ("DRAFT", "SUBMITTED", "AUTHORISED")
LINE_AMOUNT_TYPES = ("Inclusive", "Exclusive", "NoTax")
MAX_RATE_LIMIT_WAIT = 30.0            # seconds we are willing to sleep on a 429 inside a request
_RETRY_STATUS = frozenset({502, 503, 504})

# Our VAT treatment -> Xero UK TaxType for purchases. Used ONLY for the Demo Company (see build_payload); a real
# organisation must configure XERO_TAX_TYPE_MAP itself. These are the UK expense tax types listed in Xero's UK tax
# type tables; they have not been verified against a live organisation by us.
DEFAULT_TAX_MAP = {
    "STANDARD": "INPUT2",          # 20% (VAT on Expenses)
    "REDUCED": "RRINPUT",          # 5% (VAT on Expenses)
    "ZERO": "ZERORATEDINPUT",      # Zero Rated Expenses
    "EXEMPT": "EXEMPTINPUT",       # Exempt Expenses
    "OUTSIDE_SCOPE": "NONE",       # No VAT
}

_SUMMARY_FIELDS = ("InvoiceID", "InvoiceNumber", "Type", "Status", "Date", "DueDate", "CurrencyCode",
                   "LineAmountTypes", "SubTotal", "TotalTax", "Total")

_sleep = time.sleep      # tests replace it so retries do not wait


def set_transport(transport: httpx.BaseTransport | None) -> None:
    """Route every Xero request through `transport` (tests). None restores the real network."""
    xero_auth.set_transport(transport)


# --------------------------------------------------------------------------- results

@dataclass
class XeroPostResult:
    ok: bool
    status: str                      # posted | already_posted | failed | unknown | in_progress | disabled
    reference: str = ""
    xero_invoice_id: str = ""        # Xero's InvoiceID (a GUID): THE proof that Xero has the bill
    invoice_number: str = ""
    url: str = ""                    # deep link to the bill in Xero, when derivable
    bill_status: str = ""            # Xero's own status of the bill: DRAFT / SUBMITTED / AUTHORISED
    http_status: int | None = None
    error: str = ""                  # sanitised; empty when ok
    kind: str = ""
    replayed: bool = False           # True when this call did not create anything new (idempotent replay)
    summary: dict = field(default_factory=dict)

    @property
    def document(self) -> str:
        return self.xero_invoice_id

    def audit_view(self) -> dict:
        """What goes into the audit trail / API responses. Contains no secrets by construction."""
        return {"backend": "xero", "status": self.status, "reference": self.reference,
                "document": self.xero_invoice_id, "xero_invoice_id": self.xero_invoice_id,
                "invoice_number": self.invoice_number, "url": self.url, "bill_status": self.bill_status,
                "replayed": self.replayed, "http_status": self.http_status, "error": self.error,
                "summary": self.summary}


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
        raise XeroError(f"{name} is not a valid JSON object", "config") from None


def _safe_map(raw: str) -> dict:
    try:
        return _json_map(raw, "map")
    except XeroError:
        return {}


def backend_selected() -> bool:
    return settings.erp_backend.strip().lower() == "xero"


def _connection() -> dict:
    """The stored connection details (tenant id / name), never the tokens. {} when not connected or unreadable."""
    try:
        data = xero_auth.load_tokens() or {}
    except XeroError:
        return {}
    return {"tenant_id": data.get("tenant_id", ""), "tenant_name": data.get("tenant_name", ""),
            "short_code": data.get("short_code", "")}


def is_connected() -> bool:
    try:
        return xero_auth.load_tokens() is not None
    except XeroError:
        return False


def is_demo_org() -> bool:
    """True when the connected organisation is a Xero Demo Company (its name starts with 'Demo Company')."""
    conn = _connection()
    if settings.xero_tenant_id and conn.get("tenant_id") and settings.xero_tenant_id != conn["tenant_id"]:
        return False
    return conn.get("tenant_name", "").strip().lower().startswith("demo company")


def check_ready() -> None:
    """Raise XeroError(kind='config') unless posting to Xero is explicitly and safely configured."""
    if not backend_selected():
        raise XeroError("ERP_BACKEND is not 'xero'", "config")
    if not settings.xero_posting_enabled:
        raise XeroError("Xero posting is disabled (set XERO_POSTING_ENABLED=true to enable it)", "config")
    check_connected()
    if settings.xero_bill_status.strip().upper() not in BILL_STATUSES:
        raise XeroError(f"XERO_BILL_STATUS must be one of {', '.join(BILL_STATUSES)}", "config")
    if settings.xero_line_amount_types.strip() not in LINE_AMOUNT_TYPES:
        raise XeroError(f"XERO_LINE_AMOUNT_TYPES must be one of {', '.join(LINE_AMOUNT_TYPES)}", "config")


def check_connected() -> None:
    """Credentials configured and the one-time consent done. (Used by posting and by the chart sync.)"""
    if not xero_auth.has_client_credentials():
        raise XeroError("Xero credentials are not configured (XERO_CLIENT_ID, XERO_CLIENT_SECRET)", "config")
    try:
        connected = xero_auth.load_tokens() is not None
    except XeroError as e:
        raise XeroError(str(e), "config") from None
    if not connected:
        raise XeroError("Xero is not connected: run scripts/xero_authorize.py once", "config")


def status() -> dict:
    """Configuration state for GET /xero/status. No client id, secret, token or tenant id is ever returned."""
    try:
        check_ready()
        ready, reason = True, ""
    except XeroError as e:
        ready, reason = False, str(e)
    conn = _connection()
    return {
        "backend": "xero" if backend_selected() else "simulated",
        "configured": xero_auth.has_client_credentials(),
        "connected": is_connected(),
        "posting_enabled": bool(settings.xero_posting_enabled),
        "ready_to_post": ready,
        "not_ready_reason": reason,
        "org_name": conn.get("tenant_name") or None,
        "demo_company": is_demo_org(),
        "tenant_configured": bool(settings.xero_tenant_id),
        "bill_status": settings.xero_bill_status.strip().upper(),
        "line_amount_types": settings.xero_line_amount_types,
        "scopes": settings.xero_scopes.split(),
        "mappings": {name: len(_safe_map(raw)) for name, raw in (
            ("account", settings.xero_account_map), ("tax_type", settings.xero_tax_type_map),
            ("cost_centre", settings.xero_cost_centre_map))},
    }


# --------------------------------------------------------------------------- mapping

def _posting_datetime(erp: dict) -> datetime:
    raw = erp.get("posted_at")
    try:
        dt = raw if isinstance(raw, datetime) else datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def effective_tax_map(demo: bool) -> dict[str, str]:
    """Demo Company: built-in UK defaults overlaid with XERO_TAX_TYPE_MAP. Any other organisation: only XERO_TAX_TYPE_MAP."""
    configured = _json_map(settings.xero_tax_type_map, "XERO_TAX_TYPE_MAP")
    return {**DEFAULT_TAX_MAP, **configured} if demo else configured


def build_payload(erp: dict, demo: bool = False) -> dict:
    """
    Map our ERP payload (ERPPostingPayload as a dict) to the body of a create-bills request: {"Invoices": [bill]}.
    Raises XeroError(kind='mapping') when something Xero requires cannot be derived. `demo` says whether the connected
    organisation is the Xero Demo Company, which is the only place a default tax type is allowed.
    """
    account_map = _json_map(settings.xero_account_map, "XERO_ACCOUNT_MAP")
    cc_map = _json_map(settings.xero_cost_centre_map, "XERO_COST_CENTRE_MAP")

    contact = " ".join(str(erp.get("vendor_name") or erp.get("vendor_id") or "").split())
    if not contact:
        raise XeroError("No vendor name on the invoice, so no Xero contact can be named", "mapping")

    try:
        amount = Decimal(str(erp.get("amount"))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except Exception:
        raise XeroError("Invoice amount is not a valid number", "mapping") from None
    if amount <= 0:
        raise XeroError("Invoice amount must be greater than zero (credit notes are not supported)", "mapping")

    treatment = str(erp.get("tax_code") or "")
    tax_type = effective_tax_map(demo).get(treatment)
    if tax_type is None:
        if not demo:   # a real organisation must never get a guessed tax type
            raise XeroError(f"No Xero tax type mapped for VAT treatment {treatment!r} (XERO_TAX_TYPE_MAP)", "mapping")
        tax_type = settings.xero_default_tax_type

    gl = str(erp.get("gl_account") or "")
    if not gl:
        raise XeroError("No G/L account on the invoice", "mapping")
    account_code = account_map.get(gl) or gl

    status_ = settings.xero_bill_status.strip().upper()
    if status_ not in BILL_STATUSES:
        raise XeroError(f"XERO_BILL_STATUS must be one of {', '.join(BILL_STATUSES)}", "config")
    amount_types = settings.xero_line_amount_types.strip()
    if amount_types not in LINE_AMOUNT_TYPES:
        raise XeroError(f"XERO_LINE_AMOUNT_TYPES must be one of {', '.join(LINE_AMOUNT_TYPES)}", "config")

    bill_date: date = _posting_datetime(erp).date()
    currency = (erp.get("currency") or "").upper()
    if not currency:
        raise XeroError("No currency on the invoice", "mapping")

    line: dict = {
        "Description": " ".join(str(erp.get("description") or "").split())[:4000] or "Invoice line",
        "Quantity": 1,
        "UnitAmount": float(amount),
        "AccountCode": account_code,
        "TaxType": tax_type,
    }
    cost_centre = str(erp.get("cost_centre") or "")
    if settings.xero_tracking_category and cost_centre:
        line["Tracking"] = [{"Name": settings.xero_tracking_category, "Option": cc_map.get(cost_centre) or cost_centre}]

    bill: dict = {
        "Type": "ACCPAY",
        "Contact": {"Name": contact[:255]},
        "InvoiceNumber": reference_for(erp.get("invoice_id")),
        "Date": bill_date.isoformat(),
        "CurrencyCode": currency,
        "LineAmountTypes": amount_types,
        "Status": status_,
        "LineItems": [line],
    }
    if settings.xero_due_days and settings.xero_due_days > 0:
        bill["DueDate"] = (bill_date + timedelta(days=int(settings.xero_due_days))).isoformat()
    return {"Invoices": [bill]}


# --------------------------------------------------------------------------- HTTP

def _xero_message(resp: httpx.Response) -> str:
    """The human-readable reason Xero gave. Bodies are parsed, never echoed raw."""
    try:
        body = resp.json()
    except ValueError:
        return f"HTTP {resp.status_code} (non-JSON response)"
    if isinstance(body, dict):
        head = body.get("Message") or body.get("Detail") or body.get("Title")
        kind = body.get("Type") or ""
        problems: list[str] = []
        for el in body.get("Elements") or []:
            if isinstance(el, dict):
                problems += [str(v.get("Message")) for v in (el.get("ValidationErrors") or [])
                             if isinstance(v, dict) and v.get("Message")]
        if head or problems:
            text = f"HTTP {resp.status_code} Xero {kind + ': ' if kind else ''}{head or 'error'}"
            if problems:
                text += " | details: " + " | ".join(dict.fromkeys(problems[:5]))
            return text
    return f"HTTP {resp.status_code} (unrecognised error body)"


def _kind_for(status_code: int) -> str:
    return "unavailable" if status_code >= 500 else "auth" if status_code in (401, 403) else "rejected"


class _Session:
    """One Xero conversation: an HTTP client plus the tenant id. Access tokens are fetched (and refreshed) per call."""

    def __init__(self) -> None:
        self.http = httpx.Client(timeout=httpx.Timeout(settings.xero_timeout_seconds),
                                 transport=xero_auth.transport(), follow_redirects=False)
        self.tenant: str = ""

    def close(self) -> None:
        self.http.close()

    def ensure_tenant(self) -> str:
        """XERO_TENANT_ID, else the tenant stored at connect time, else discovered from GET /connections (once)."""
        conn = _connection()
        self.tenant = settings.xero_tenant_id or conn.get("tenant_id", "")
        if self.tenant:
            return self.tenant
        rows = xero_auth.list_connections(xero_auth.access_token())
        if not rows:
            raise XeroError("No Xero organisation is connected to this app; run scripts/xero_authorize.py", "auth")
        if len(rows) > 1:
            raise XeroError("More than one Xero organisation is connected; set XERO_TENANT_ID to choose one",
                            "config")
        self.tenant = rows[0]["tenantId"]
        xero_auth.update_connection(self.tenant, rows[0].get("tenantName") or "")
        return self.tenant

    def send(self, method: str, path: str, *, params: dict | None = None, body: dict | None = None,
             headers: dict | None = None, safe: bool) -> httpx.Response:
        """
        One API call with bounded, careful retries. `safe` = the request is idempotent (a GET), so transport errors and
        502/503/504 may be retried. A non-safe request is only repeated when it provably was not processed: the
        connection could not be established, HTTP 429 (Xero rejects before processing) or HTTP 401 (after a refresh).
        Any other transport error on a non-safe request propagates to the caller, which must reconcile.
        """
        max_retries = max(0, settings.xero_max_retries)
        transient = rate_waits = 0
        force_refresh = refreshed = False
        while True:
            hdrs = {"Authorization": f"Bearer {xero_auth.access_token(force_refresh)}", "Accept": "application/json",
                    "xero-tenant-id": self.tenant, **(headers or {})}
            force_refresh = False
            try:
                resp = self.http.request(method, API_BASE + path, params=params, json=body, headers=hdrs)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                transient += 1
                if transient > max_retries:
                    raise XeroError("Xero unreachable: connection could not be established", "unavailable") from None
                _sleep(min(0.5 * 2 ** transient, 4.0))
                continue
            except httpx.TransportError as e:
                if not safe:
                    raise
                transient += 1
                if transient > max_retries:
                    raise XeroError(f"Xero unreachable: {type(e).__name__}", "unavailable") from None
                _sleep(min(0.5 * 2 ** transient, 4.0))
                continue
            if resp.status_code == 401 and not refreshed:
                refreshed = force_refresh = True      # the access token was refused: not processed; refresh once
                continue
            if resp.status_code == 429:
                try:
                    wait = float(resp.headers.get("retry-after", "5"))
                except ValueError:
                    wait = 5.0
                problem = resp.headers.get("x-rate-limit-problem", "")
                if wait <= MAX_RATE_LIMIT_WAIT and rate_waits < max_retries:
                    rate_waits += 1
                    _sleep(wait)
                    continue
                raise XeroError(f"Xero rate limit reached ({problem or 'limit'}); nothing was sent for this invoice. "
                                f"Retry after {int(wait)}s", "rate_limited", 429)
            if safe and resp.status_code in _RETRY_STATUS and transient < max_retries:
                transient += 1
                _sleep(min(0.5 * 2 ** transient, 4.0))
                continue
            return resp

    def find_by_reference(self, reference: str) -> dict | None:
        """The live bill already carrying this InvoiceNumber, or None. Raises XeroError if Xero cannot be asked."""
        resp = self.send("GET", "/Invoices", params={"InvoiceNumbers": reference}, safe=True)
        if resp.status_code >= 400:
            raise XeroError(_xero_message(resp), _kind_for(resp.status_code), resp.status_code)
        try:
            rows = resp.json()["Invoices"]
        except (ValueError, KeyError, TypeError):
            raise XeroError("Unexpected response while checking for an existing bill", "unavailable") from None
        ours = [r for r in rows if isinstance(r, dict) and r.get("InvoiceNumber") == reference
                and r.get("Type", "ACCPAY") == "ACCPAY"]
        live = [r for r in ours if r.get("Status") not in ("VOIDED", "DELETED")]
        if not live and ours:
            raise XeroError(f"A bill with reference {reference} exists but is {ours[0].get('Status', 'inactive')}; "
                            "manual action required", "rejected")
        return live[0] if live else None

    def create(self, payload: dict, idempotency_key: str) -> httpx.Response:
        return self.send("PUT", "/Invoices", body=payload, headers={"Idempotency-Key": idempotency_key[:128]},
                         safe=False)

    def deep_link(self, invoice_id: str) -> str:
        """Link to the bill in Xero. Best effort: never fails the post. Uses the org short code when it is known."""
        try:
            code = _connection().get("short_code", "")
            if not code:
                resp = self.send("GET", "/Organisation", safe=True)
                org = resp.json()["Organisations"][0] if resp.status_code == 200 else {}
                code = org.get("ShortCode", "") or ""
                if code:
                    xero_auth.update_connection(self.tenant, org.get("Name", "") or _connection().get("tenant_name", ""),
                                                short_code=code)
            path = f"/AccountsPayable/View.aspx?InvoiceID={invoice_id}"
            if code:
                return f"https://go.xero.com/organisationlogin/default.aspx?shortcode={quote(code, safe='!')}&redirecturl={path}"
            return "https://go.xero.com" + path
        except Exception:
            return f"https://go.xero.com/AccountsPayable/View.aspx?InvoiceID={invoice_id}"


def _summary(inv: dict) -> dict:
    return {k: inv[k] for k in _SUMMARY_FIELDS if inv.get(k) not in (None, "")}


# --------------------------------------------------------------------------- orchestration

def post_invoice(erp: dict) -> XeroPostResult:
    """
    Post one invoice to Xero as a bill, exactly once. Never raises: the outcome is the returned result, and it is
    persisted. `erp` is ERPPostingPayload as a dict. Calling this again for the same invoice is safe.
    """
    reference = reference_for(erp.get("invoice_id"))
    try:
        check_ready()
    except XeroError as e:
        return XeroPostResult(False, "disabled", reference, error=sanitize(e), kind=e.kind)

    claimed, row = database.xero_claim(reference, str(erp.get("invoice_id")), str(erp.get("trace_id") or ""))
    if not claimed:
        if row["status"] == "posted":
            return XeroPostResult(True, "already_posted", reference, row["xero_invoice_id"], reference,
                                  row["xero_url"], replayed=True)
        return XeroPostResult(False, "in_progress", reference, kind="in_progress",
                              error="Another post of this invoice is in progress; not posting again")

    session = None
    try:
        session = _Session()
        session.ensure_tenant()                      # first, so the organisation name (demo or real) is known
        payload = build_payload(erp, demo=is_demo_org())

        existing = session.find_by_reference(reference)
        if existing is not None:
            return _finish_ok(session, reference, existing, None, replayed=True)

        try:
            resp = session.create(payload, f"{reference}-{row['attempts']}")
        except httpx.TransportError as e:   # request was sent; Xero may or may not have processed it
            return _reconcile(session, reference, f"no response ({type(e).__name__})", None)

        if resp.status_code in (200, 201):
            try:
                created = resp.json()["Invoices"][0]
                if not created.get("InvoiceID"):
                    raise KeyError("InvoiceID")
            except (ValueError, KeyError, IndexError, TypeError):
                return _reconcile(session, reference, f"HTTP {resp.status_code} with an unreadable body",
                                  resp.status_code)
            return _finish_ok(session, reference, created, resp.status_code, replayed=False,
                              day_limit=resp.headers.get("x-daylimit-remaining"))
        if resp.status_code >= 500 or resp.status_code == 408:
            return _reconcile(session, reference, f"HTTP {resp.status_code}", resp.status_code)

        # 4xx: Xero refused the bill. Nothing was created.
        msg = sanitize(_xero_message(resp))
        database.xero_finish(reference, "failed", error=msg, tenant_id=session.tenant)
        print(f"[Xero] post rejected for {reference}: {msg}")
        return XeroPostResult(False, "failed", reference, http_status=resp.status_code, error=msg,
                              kind=_kind_for(resp.status_code))
    except XeroError as e:
        msg = sanitize(e)
        status_ = "unknown" if e.kind == "unknown_outcome" else "failed"
        database.xero_finish(reference, status_, error=msg, tenant_id=session.tenant if session else "")
        print(f"[Xero] post not completed for {reference} ({e.kind}): {msg}")
        return XeroPostResult(False, status_, reference, http_status=e.http_status, error=msg, kind=e.kind)
    except Exception as e:  # defensive: never lose the invoice, never leak a secret through str(e)
        msg = sanitize(f"{type(e).__name__}: {e}")
        database.xero_finish(reference, "unknown", error=msg)
        print(f"[Xero] unexpected error for {reference}: {msg}")
        return XeroPostResult(False, "unknown", reference, error=msg, kind="unknown_outcome")
    finally:
        if session is not None:
            session.close()


def _finish_ok(session: _Session, reference: str, inv: dict, http_status: int | None, replayed: bool,
               day_limit: str | None = None) -> XeroPostResult:
    invoice_id = str(inv.get("InvoiceID", ""))
    summary = _summary(inv)
    if day_limit:
        summary["day_limit_remaining"] = day_limit
    url = session.deep_link(invoice_id) if invoice_id else ""
    database.xero_finish(reference, "posted", xero_invoice_id=invoice_id, xero_url=url, tenant_id=session.tenant,
                         summary=summary)
    return XeroPostResult(True, "already_posted" if replayed else "posted", reference, invoice_id,
                          str(inv.get("InvoiceNumber", reference)), url, str(inv.get("Status", "")), http_status,
                          replayed=replayed, summary=summary)


def _reconcile(session: _Session, reference: str, why: str, http_status: int | None) -> XeroPostResult:
    """A create may or may not have been processed: ask Xero by InvoiceNumber instead of guessing or re-posting."""
    try:
        found = session.find_by_reference(reference)
    except XeroError:
        found = None
        checked = False
    else:
        checked = True
    if found is not None:
        print(f"[Xero] {reference}: create outcome was uncertain ({why}) but Xero has the bill; not re-posting")
        return _finish_ok(session, reference, found, http_status, replayed=False)
    msg = sanitize(f"Xero post outcome unconfirmed ({why}); "
                   + ("no bill with this reference was found yet" if checked else "Xero could not be queried")
                   + ". It was NOT re-posted; it will be checked by reference before any further attempt")
    database.xero_finish(reference, "unknown", error=msg, tenant_id=session.tenant)
    print(f"[Xero] {reference}: {msg}")
    return XeroPostResult(False, "unknown", reference, http_status=http_status, error=msg, kind="unknown_outcome")


# --------------------------------------------------------------------------- chart of accounts

def chart_types() -> list[str]:
    return [t.strip().upper() for t in settings.xero_chart_account_types.split(",") if t.strip()]


def _vat_treatment_for(tax_type: str | None) -> str | None:
    """Our VAT treatment for a Xero account's default TaxType, or None when we cannot tell."""
    configured = _safe_map(settings.xero_tax_type_map)
    reverse: dict[str, str] = {}
    for treatment, tt in {**DEFAULT_TAX_MAP, **configured}.items():
        reverse.setdefault(tt, treatment)
    return reverse.get(tax_type or "")


def accounts_to_chart(rows: list[dict]) -> tuple[dict[str, chart_of_accounts.Account], dict]:
    """
    Turn Xero /Accounts rows into our chart. Only ACTIVE accounts of the types in XERO_CHART_ACCOUNT_TYPES (default
    EXPENSE, DIRECTCOSTS, OVERHEADS: the types a supplier bill is coded to) that have a code are kept.
    The account's default Xero TaxType becomes our `vat_treatment`; a TaxType we cannot map becomes NOT_ASSESSED,
    which always routes the invoice to human review (never a silent wrong tax).
    """
    wanted = set(chart_types())
    chart: dict[str, chart_of_accounts.Account] = {}
    skipped = {"wrong_type": 0, "inactive": 0, "no_code": 0, "system_account": 0, "duplicate_code": 0}
    unmapped: dict[str, int] = {}
    for r in rows:
        if r.get("Status", "ACTIVE") != "ACTIVE":
            skipped["inactive"] += 1
        elif r.get("Type") not in wanted:
            skipped["wrong_type"] += 1
        elif not str(r.get("Code") or "").strip():
            skipped["no_code"] += 1
        elif r.get("SystemAccount"):
            skipped["system_account"] += 1
        else:
            code = str(r["Code"]).strip()
            if code in chart:
                skipped["duplicate_code"] += 1
                continue
            treatment = _vat_treatment_for(r.get("TaxType"))
            if treatment is None:
                unmapped[str(r.get("TaxType") or "(none)")] = unmapped.get(str(r.get("TaxType") or "(none)"), 0) + 1
                treatment = "NOT_ASSESSED"
            chart[code] = chart_of_accounts.Account(
                code=code, name=" ".join(str(r.get("Name") or code).split())[:150], vat_treatment=treatment,
                default_cost_centre="", vat_note=f"Xero tax type {r.get('TaxType') or 'none'}")
    return chart, {"skipped": skipped, "unmapped_tax_types": unmapped, "types": sorted(wanted)}


def fetch_accounts() -> list[dict]:
    """GET /Accounts for the connected organisation (needs the accounting.settings.read scope)."""
    check_connected()
    session = _Session()
    try:
        session.ensure_tenant()
        resp = session.send("GET", "/Accounts", params={"where": 'Status=="ACTIVE"'}, safe=True)
        if resp.status_code >= 400:
            raise XeroError(sanitize(_xero_message(resp)), _kind_for(resp.status_code), resp.status_code)
        try:
            return [r for r in resp.json()["Accounts"] if isinstance(r, dict)]
        except (ValueError, KeyError, TypeError):
            raise XeroError("Unexpected response while reading the Xero chart of accounts", "unavailable") from None
    finally:
        session.close()


def sync_chart(apply: bool = True) -> dict:
    """
    Pull Xero's chart of accounts into our per-client chart (same effect as POST /chart with a CSV). With apply=False
    nothing is changed. Raises XeroError if Xero cannot be read or no account qualifies (the chart is never emptied).
    """
    rows = fetch_accounts()
    chart, info = accounts_to_chart(rows)
    if not chart:
        raise XeroError("No active Xero accounts of type " + ", ".join(chart_types()) + " were found; the chart was "
                        "not changed", "mapping")
    persisted = False
    if apply:
        chart_of_accounts.set_chart(chart)
        persisted = chart_of_accounts.save_chart(settings.chart_path)
    return {"status": "chart_replaced" if apply else "dry_run", "accounts": len(chart), "persisted": persisted,
            "xero_accounts_read": len(rows), **info,
            "chart": [{"code": a.code, "name": a.name, "vat_treatment": a.vat_treatment} for a in chart.values()]}
