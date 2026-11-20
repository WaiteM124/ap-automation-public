"""
Xero posting backend and OAuth handling. Everything runs against a fake Xero behind httpx.MockTransport: no network,
no real credentials, nothing verified live. The "secrets" used here are made-up strings; the tests check that they never
reach a result, a log line, an audit record, the Xero posting table or the ledger, even when Xero's own error text
echoes them back.
"""

import base64
import json
import os
import threading
import time
import uuid
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from ap_automation.core import chart_of_accounts, database, posting, xero_auth, xero_poster
from ap_automation.core.config import settings
from ap_automation.core.models import ConfirmationSource

CLIENT_ID = "FAKE-CLIENT-ID-0123456789"
CLIENT_SECRET = "FAKE-CLIENT-SECRET-abcdef0123456789"
TENANT = "11111111-2222-3333-4444-555555555555"
SECRETS = [CLIENT_SECRET]            # extended by the fake with every token it hands out


# --------------------------------------------------------------------------- fake Xero

class Crash(BaseException):
    """Stands for the process being killed at an awkward moment (not an Exception: nothing may catch it)."""


class FakeXero:
    """
    Just enough of Xero: the token endpoint (single-use refresh tokens with a grace period for the previous one),
    /connections, GET/PUT /Invoices, /Organisation and /Accounts.
    """

    def __init__(self):
        self.invoices: list[dict] = []
        self.requests: list[httpx.Request] = []
        self.script: list[str] = []            # one entry consumed per PUT /Invoices; "ok" once exhausted
        self.token_script: list[str] = []      # one entry consumed per token request
        self.accounts: list[dict] = []
        self.org_name = "Demo Company (UK)"
        self.transport = httpx.MockTransport(self.handle)
        self._n = 0
        self._lock = threading.Lock()
        self.access_valid: set[str] = set()
        self.refresh_current = ""
        self.refresh_prev = ""                 # the previous refresh token: valid for the grace period
        self.token_requests: list[dict] = []   # form fields of each token request (for assertions)

    # -- helpers
    @property
    def puts(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == "PUT" and r.url.path.endswith("/Invoices")]

    @property
    def token_calls(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.host == "identity.xero.com"]

    def issue(self) -> tuple[str, str]:
        self._n += 1
        access, refresh = f"ACCESS-TOKEN-{self._n}-xyz0123456789", f"REFRESH-TOKEN-{self._n}-xyz0123456789"
        self.access_valid.add(access)
        SECRETS.extend([access, refresh])
        return access, refresh

    def seed_connection(self) -> tuple[str, str]:
        """The tokens a completed consent would have produced."""
        access, refresh = self.issue()
        self.refresh_current = refresh
        return access, refresh

    @staticmethod
    def _err(status: int, message: str, problems: list[str] | None = None) -> httpx.Response:
        return httpx.Response(status, json={
            "ErrorNumber": 10, "Type": "ValidationException", "Message": message,
            "Elements": [{"ValidationErrors": [{"Message": m} for m in (problems or [])]}]})

    def _authorised(self, request: httpx.Request) -> httpx.Response | None:
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        if token not in self.access_valid:
            return httpx.Response(401, json={"Title": "Unauthorized", "Status": 401,
                                             "Detail": "TokenExpired: token expired at 2026-10-27T07:00:00Z"})
        if request.url.path != "/connections" and request.headers.get("xero-tenant-id") != TENANT:
            return httpx.Response(403, json={"Title": "Forbidden", "Status": 403,
                                             "Detail": "AuthenticationUnsuccessful"})
        return None

    # -- the endpoints
    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host, path = request.url.host, request.url.path
        if host == "identity.xero.com" and path == "/connect/token":
            return self._token(request)
        if host != "api.xero.com":
            return httpx.Response(404, json={})
        denied = self._authorised(request)
        if denied is not None:
            return denied
        if path == "/connections":
            return httpx.Response(200, json=[
                {"id": "c1", "tenantId": TENANT, "tenantType": "ORGANISATION", "tenantName": self.org_name,
                 "authEventId": "ev1"},
                {"id": "c2", "tenantId": "pm-1", "tenantType": "PRACTICEMANAGER", "tenantName": None}])
        if path == "/api.xro/2.0/Organisation":
            return httpx.Response(200, json={"Organisations": [{"Name": self.org_name, "ShortCode": "!AbC12"}]})
        if path == "/api.xro/2.0/Accounts":
            return httpx.Response(200, json={"Accounts": self.accounts})
        if path == "/api.xro/2.0/Invoices" and request.method == "GET":
            number = request.url.params["InvoiceNumbers"]
            return httpx.Response(200, json={"Invoices": [i for i in self.invoices if i["InvoiceNumber"] == number]})
        if path == "/api.xro/2.0/Invoices" and request.method == "PUT":
            return self._create(request)
        return httpx.Response(404, json={})

    def _token(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            self.token_requests.append(form)
            expected = "Basic " + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
            if request.headers.get("authorization") != expected:
                return httpx.Response(401, json={"error": "invalid_client"})
            action = self.token_script.pop(0) if self.token_script else "ok"
            if action == "invalid_client":
                return httpx.Response(400, json={"error": "invalid_client"})
            if action == "server_error":
                return httpx.Response(503, text="down")
            if form.get("grant_type") == "authorization_code":
                if form.get("code") != "AUTH-CODE-1" or form.get("redirect_uri") != settings.xero_redirect_uri:
                    return httpx.Response(400, json={"error": "invalid_grant"})
            elif form.get("grant_type") == "refresh_token":
                given = form.get("refresh_token")
                if action == "invalid_grant" or given not in (self.refresh_current, self.refresh_prev):
                    return httpx.Response(400, json={"error": "invalid_grant"})
            else:
                return httpx.Response(400, json={"error": "unsupported_grant_type"})
            old = self.refresh_current
            access, refresh = self.issue()
            self.refresh_prev, self.refresh_current = old, refresh
            if action == "timeout_lost":    # Xero rotated the token, the answer never arrived
                raise httpx.ReadTimeout("read timed out", request=request)
            return httpx.Response(200, json={"access_token": access, "refresh_token": refresh, "expires_in": 1800,
                                             "token_type": "Bearer", "scope": settings.xero_scopes})

    def _create(self, request: httpx.Request) -> httpx.Response:
        action = self.script.pop(0) if self.script else "ok"
        if action == "connect_error":
            raise httpx.ConnectError("name resolution failed", request=request)
        if action == "unauthorized_once":
            self.access_valid.clear()
            return self._authorised(request)
        if action == "429_short":
            return httpx.Response(429, headers={"retry-after": "2", "x-rate-limit-problem": "minute"}, json={})
        if action == "429_long":
            return httpx.Response(429, headers={"retry-after": "3600", "x-rate-limit-problem": "day"}, json={})
        body = json.loads(request.content)["Invoices"][0]
        if action == "reject":
            return self._err(400, "A validation exception occurred",
                             ["Account code '6300' is not a valid code", "Account code '6300' is not a valid code"])
        if action == "reject_echo_secrets":
            echoed = f"token {self._any_token()} and secret {CLIENT_SECRET}"
            return self._err(400, f"A validation exception occurred: {echoed}", [f"Authorization: Bearer {self._any_token()}"])
        if action == "timeout_lost":
            raise httpx.ReadTimeout("read timed out", request=request)
        if action == "boom_with_secret":
            raise RuntimeError(f"unexpected failure with {CLIENT_SECRET} and {self._any_token()}")
        created = self._store(body)
        if action == "timeout_created":
            raise httpx.ReadTimeout("read timed out", request=request)
        if action == "500_created":
            return httpx.Response(500, text="boom")
        if action == "200_unreadable":
            return httpx.Response(200, text="<html>not json</html>")
        return httpx.Response(200, json={"Invoices": [created]}, headers={"x-daylimit-remaining": "998"})

    def _any_token(self) -> str:
        return sorted(self.access_valid)[-1] if self.access_valid else "none"

    def _store(self, body: dict) -> dict:
        created = {**{k: v for k, v in body.items() if k != "Contact"},
                   "InvoiceID": str(uuid.uuid4()), "Contact": {"Name": body["Contact"]["Name"], "ContactID": "c-1"},
                   "Total": body["LineItems"][0]["UnitAmount"], "SecretLookingExtra": "must-not-be-stored"}
        self.invoices.append(created)
        return created


def erp_payload(**over) -> dict:
    base = {"invoice_id": str(uuid.uuid4()), "vendor_id": "VND-ACME", "vendor_name": "Meridian Facilities Ltd",
            "entity_id": "UK001", "gl_account": "6300", "cost_centre": "CC300", "tax_code": "STANDARD",
            "amount": 1200.0, "currency": "GBP", "description": "Office cleaning services monthly",
            "posted_by": "auto_post_service", "posted_at": "2026-10-27T07:45:00+00:00", "trace_id": str(uuid.uuid4())}
    base.update(over)
    return base


def connect(fake: FakeXero, expires_in: float = 1800, tenant_name: str = "Demo Company (UK)") -> tuple[str, str]:
    """Write the token file a completed consent would have written."""
    access, refresh = fake.seed_connection()
    xero_auth.save_tokens({"access_token": access, "refresh_token": refresh, "expires_at": time.time() + expires_in,
                           "scope": settings.xero_scopes, "tenant_id": TENANT, "tenant_name": tenant_name,
                           "short_code": ""})
    return access, refresh


@pytest.fixture
def xero(monkeypatch):
    """Xero backend switched fully on against the fake. Individual tests switch things off again."""
    fake = FakeXero()
    SECRETS.clear()
    SECRETS.append(CLIENT_SECRET)
    for k, v in {"erp_backend": "xero", "xero_posting_enabled": True, "xero_client_id": CLIENT_ID,
                 "xero_client_secret": CLIENT_SECRET, "xero_tenant_id": "", "xero_bill_status": "DRAFT",
                 "xero_line_amount_types": "Inclusive", "xero_due_days": 30, "xero_max_retries": 2,
                 "xero_account_map": "", "xero_tax_type_map": "", "xero_tracking_category": "",
                 "xero_cost_centre_map": "", "sap_posting_enabled": False}.items():
        monkeypatch.setattr(settings, k, v)
    monkeypatch.setattr(xero_poster, "_sleep", lambda s: None)
    monkeypatch.setattr(xero_auth, "_sleep", lambda s: None)
    xero_poster.set_transport(fake.transport)
    connect(fake)
    fake.requests.clear()
    return fake


def _all_stored_text() -> str:
    """Everything persisted by a post: Xero posting table, audit trail, ledger. (Not the token file: that holds tokens.)"""
    return "\n".join([json.dumps(database.get_xero_postings(1000)), json.dumps(database.get_audit_records(1000)),
                      json.dumps(database.get_postings(1000))])


def _leaks(blob: str) -> list[str]:
    return [s for s in SECRETS if s in blob]


# --------------------------------------------------------------------------- mapping

def test_payload_mapping_and_defaults(xero):
    body = xero_poster.build_payload(erp_payload(invoice_id="inv-1"), demo=True)
    bill = body["Invoices"][0]
    assert bill["Type"] == "ACCPAY" and bill["Contact"] == {"Name": "Meridian Facilities Ltd"}
    assert bill["InvoiceNumber"] == xero_poster.reference_for("inv-1")
    assert (bill["Date"], bill["DueDate"]) == ("2026-10-27", "2026-11-26")
    assert (bill["CurrencyCode"], bill["LineAmountTypes"], bill["Status"]) == ("GBP", "Inclusive", "DRAFT")
    assert bill["LineItems"] == [{"Description": "Office cleaning services monthly", "Quantity": 1,
                                  "UnitAmount": 1200.0, "AccountCode": "6300", "TaxType": "INPUT2"}]


def test_mappings_are_configurable(xero, monkeypatch):
    monkeypatch.setattr(settings, "xero_account_map", json.dumps({"6300": "412"}))
    monkeypatch.setattr(settings, "xero_tax_type_map", json.dumps({"STANDARD": "TAX001"}))
    monkeypatch.setattr(settings, "xero_tracking_category", "Cost centre")
    monkeypatch.setattr(settings, "xero_cost_centre_map", json.dumps({"CC300": "Facilities"}))
    monkeypatch.setattr(settings, "xero_bill_status", "authorised")
    monkeypatch.setattr(settings, "xero_due_days", 0)
    bill = xero_poster.build_payload(erp_payload(), demo=False)["Invoices"][0]
    line = bill["LineItems"][0]
    assert (line["AccountCode"], line["TaxType"]) == ("412", "TAX001")
    assert line["Tracking"] == [{"Name": "Cost centre", "Option": "Facilities"}]
    assert bill["Status"] == "AUTHORISED" and "DueDate" not in bill


def test_demo_company_defaults_tax_but_a_real_org_never_guesses(xero):
    # Demo Company: built-in UK map, then XERO_DEFAULT_TAX_TYPE for what is not covered
    line = xero_poster.build_payload(erp_payload(tax_code="REVERSE_CHARGE"), demo=True)["Invoices"][0]["LineItems"][0]
    assert line["TaxType"] == "NONE"
    # any other organisation: only the configured map; nothing mapped = error
    with pytest.raises(xero_auth.XeroError, match="No Xero tax type mapped"):
        xero_poster.build_payload(erp_payload(tax_code="STANDARD"), demo=False)


def test_real_org_post_is_refused_before_any_request_when_tax_is_unmapped(xero):
    xero_auth.update_connection(TENANT, "Acme Trading Ltd")
    assert not xero_poster.is_demo_org()
    res = xero_poster.post_invoice(erp_payload())
    assert not res.ok and res.kind == "mapping" and "No Xero tax type mapped" in res.error
    assert xero.requests == []                                       # nothing sent, not even a read
    assert database.get_xero_postings(10, "failed")[0]["error"] == res.error


@pytest.mark.parametrize("over,expect", [
    ({"amount": 0}, "greater than zero"), ({"amount": -5}, "greater than zero"), ({"gl_account": ""}, "G/L account"),
    ({"vendor_name": "", "vendor_id": ""}, "No vendor name"), ({"currency": ""}, "No currency"),
])
def test_unmappable_invoices_are_refused_without_a_request(xero, over, expect):
    res = xero_poster.post_invoice(erp_payload(**over))
    assert not res.ok and res.kind == "mapping" and expect in res.error
    assert xero.requests == []


def test_contact_falls_back_to_the_vendor_id(xero):
    bill = xero_poster.build_payload(erp_payload(vendor_name=""), demo=True)["Invoices"][0]
    assert bill["Contact"]["Name"] == "VND-ACME"


def test_bad_mapping_json_is_a_config_error_not_a_crash(xero, monkeypatch):
    monkeypatch.setattr(settings, "xero_account_map", "{not json")
    res = xero_poster.post_invoice(erp_payload())
    assert not res.ok and "XERO_ACCOUNT_MAP" in res.error and xero.puts == []


# --------------------------------------------------------------------------- the happy path

def test_success_creates_a_bill_and_records_the_xero_id(xero):
    res = xero_poster.post_invoice(erp_payload())
    assert res.ok and res.status == "posted" and res.http_status == 200
    created = xero.invoices[0]
    assert res.xero_invoice_id == created["InvoiceID"] and res.bill_status == "DRAFT"
    assert res.invoice_number == res.reference and "SecretLookingExtra" not in res.summary
    assert res.summary["Total"] == 1200.0 and res.summary["day_limit_remaining"] == "998"
    assert res.url == ("https://go.xero.com/organisationlogin/default.aspx?shortcode=!AbC12"
                       f"&redirecturl=/AccountsPayable/View.aspx?InvoiceID={created['InvoiceID']}")

    calls = [(r.method, r.url.path.removeprefix("/api.xro/2.0")) for r in xero.requests]
    assert calls == [("GET", "/Invoices"), ("PUT", "/Invoices"), ("GET", "/Organisation")]
    find, put = xero.requests[0], xero.puts[0]
    assert find.url.params["InvoiceNumbers"] == res.reference            # asked before creating
    assert put.headers["xero-tenant-id"] == TENANT and put.headers["authorization"].startswith("Bearer ACCESS-")
    assert put.headers["idempotency-key"].startswith(res.reference) and len(put.headers["idempotency-key"]) <= 128
    assert put.headers["content-type"].startswith("application/json") and put.headers["accept"] == "application/json"
    row = database.get_xero_postings(10, "posted")[0]
    assert row["xero_invoice_id"] == created["InvoiceID"] and row["attempts"] == 1 and row["tenant_id"] == TENANT
    assert xero_auth.load_tokens()["short_code"] == "!AbC12"             # remembered: no second /Organisation call


def test_deep_link_without_a_short_code_falls_back_to_the_plain_url(xero, monkeypatch):
    real = xero.handle
    xero.transport = httpx.MockTransport(lambda r: httpx.Response(500) if r.url.path.endswith("/Organisation")
                                         else real(r))
    xero_poster.set_transport(xero.transport)
    res = xero_poster.post_invoice(erp_payload())
    assert res.ok and res.url == f"https://go.xero.com/AccountsPayable/View.aspx?InvoiceID={res.xero_invoice_id}"


def test_authorised_bills_when_configured(xero, monkeypatch):
    monkeypatch.setattr(settings, "xero_bill_status", "AUTHORISED")
    res = xero_poster.post_invoice(erp_payload())
    assert res.ok and res.bill_status == "AUTHORISED"


def test_tenant_is_discovered_from_connections_and_remembered(xero):
    xero_auth.save_tokens({**xero_auth.load_tokens(), "tenant_id": "", "tenant_name": ""})
    res = xero_poster.post_invoice(erp_payload())
    assert res.ok and any(r.url.path == "/connections" for r in xero.requests)
    assert xero_auth.load_tokens()["tenant_id"] == TENANT and xero_poster.is_demo_org()
    n = len([r for r in xero.requests if r.url.path == "/connections"])
    xero_poster.post_invoice(erp_payload())
    assert len([r for r in xero.requests if r.url.path == "/connections"]) == n     # not asked again


def test_explicit_tenant_id_wins(xero, monkeypatch):
    monkeypatch.setattr(settings, "xero_tenant_id", "99999999-0000-0000-0000-000000000000")
    monkeypatch.setattr(settings, "xero_tax_type_map", json.dumps({"STANDARD": "INPUT2"}))   # not the demo org now
    res = xero_poster.post_invoice(erp_payload())
    assert not res.ok and res.http_status == 403 and xero.puts == []        # sent to the tenant the user chose
    assert xero.requests[0].headers["xero-tenant-id"] == "99999999-0000-0000-0000-000000000000"


# --------------------------------------------------------------------------- Xero says no

def test_xero_validation_error_is_recorded_and_not_lost(xero):
    xero.script = ["reject"]
    erp = erp_payload()
    res = xero_poster.post_invoice(erp)
    assert not res.ok and res.status == "failed" and res.kind == "rejected" and res.http_status == 400
    assert "A validation exception occurred" in res.error and "Account code '6300' is not a valid code" in res.error
    assert res.error.count("Account code") == 1                            # de-duplicated
    assert xero.invoices == [] and len(xero.puts) == 1                     # a refusal is final: not retried
    assert database.get_xero_postings(10, "failed")[0]["error"] == res.error

    xero.script = []                                                       # chart fixed -> retry posts once
    again = xero_poster.post_invoice(erp)
    assert again.ok and len(xero.invoices) == 1 and again.reference == res.reference
    assert database.get_xero_postings(10)[0]["attempts"] == 2


def test_expired_access_token_is_refreshed_once_and_nothing_is_duplicated(xero):
    xero.script = ["unauthorized_once"]
    res = xero_poster.post_invoice(erp_payload())
    assert res.ok and len(xero.invoices) == 1 and len(xero.puts) == 2      # 401 = not processed: safe to repeat
    assert len(xero.token_calls) == 1


def test_a_persistent_401_gives_up_after_one_refresh(xero):
    real = xero.handle
    xero.transport = httpx.MockTransport(
        lambda r: httpx.Response(401, json={"Title": "Unauthorized", "Detail": "AuthenticationUnsuccessful"})
        if r.url.path.endswith("/Invoices") and r.method == "PUT" else real(r))
    xero_poster.set_transport(xero.transport)
    res = xero_poster.post_invoice(erp_payload())
    assert not res.ok and res.kind == "auth" and res.http_status == 401 and len(xero.token_calls) == 1


# --------------------------------------------------------------------------- rate limits and network trouble

def test_short_rate_limit_is_waited_out_and_nothing_is_duplicated(xero):
    xero.script = ["429_short"]
    res = xero_poster.post_invoice(erp_payload())
    assert res.ok and len(xero.invoices) == 1 and len(xero.puts) == 2


def test_exhausted_daily_limit_is_a_visible_failure_not_a_wait(xero):
    xero.script = ["429_long"]
    res = xero_poster.post_invoice(erp_payload())
    assert not res.ok and res.status == "failed" and res.kind == "rate_limited" and res.http_status == 429
    assert "day" in res.error and "nothing was sent" in res.error and xero.invoices == [] and len(xero.puts) == 1
    assert database.get_xero_postings(10, "failed")


def test_connection_failure_is_retried_a_bounded_number_of_times(xero):
    xero.script = ["connect_error"] * 10
    res = xero_poster.post_invoice(erp_payload())
    assert not res.ok and res.kind == "unavailable" and xero.invoices == []
    assert len(xero.puts) == 1 + settings.xero_max_retries                 # bounded: 3 attempts, then give up


def test_connection_failure_then_recovery_posts_exactly_once(xero):
    xero.script = ["connect_error", "connect_error"]
    res = xero_poster.post_invoice(erp_payload())
    assert res.ok and len(xero.invoices) == 1 and len(xero.puts) == 3


def test_timeout_after_sending_is_not_retried_and_is_reconciled_by_invoice_number(xero):
    xero.script = ["timeout_created"]
    res = xero_poster.post_invoice(erp_payload())
    assert res.ok and res.status == "posted" and res.xero_invoice_id == xero.invoices[0]["InvoiceID"]
    assert len(xero.puts) == 1 and len(xero.invoices) == 1


def test_server_error_that_actually_created_the_bill_is_reconciled(xero):
    xero.script = ["500_created"]
    res = xero_poster.post_invoice(erp_payload())
    assert res.ok and len(xero.puts) == 1 and len(xero.invoices) == 1


def test_unreadable_success_body_is_reconciled_not_reposted(xero):
    xero.script = ["200_unreadable"]
    res = xero_poster.post_invoice(erp_payload())
    assert res.ok and len(xero.puts) == 1 and len(xero.invoices) == 1


def test_unconfirmed_outcome_stays_unknown_and_is_checked_before_any_retry(xero):
    xero.script = ["timeout_lost"]
    erp = erp_payload()
    res = xero_poster.post_invoice(erp)
    assert not res.ok and res.status == "unknown" and res.kind == "unknown_outcome"
    assert "NOT re-posted" in res.error and len(xero.puts) == 1 and database.get_xero_postings(10, "unknown")
    again = xero_poster.post_invoice(erp)                                  # asks Xero by number first, then posts once
    assert again.ok and len(xero.invoices) == 1 and len(xero.puts) == 2


def test_unknown_outcome_that_was_in_fact_created_is_never_double_posted(xero):
    xero.script = ["timeout_created"]
    erp = erp_payload()
    real, state = xero.handle, {"fail_reads": 4}

    def flaky(request):
        if request.method == "GET" and request.url.path.endswith("/Invoices") and xero.puts and state["fail_reads"]:
            state["fail_reads"] -= 1
            raise httpx.ConnectError("down", request=request)
        return real(request)

    xero_poster.set_transport(httpx.MockTransport(flaky))
    first = xero_poster.post_invoice(erp)
    assert first.status == "unknown" and len(xero.invoices) == 1
    second = xero_poster.post_invoice(erp)                                 # reads work again: it finds the bill
    assert second.ok and second.replayed and second.xero_invoice_id == xero.invoices[0]["InvoiceID"]
    assert len(xero.puts) == 1 and len(xero.invoices) == 1


# --------------------------------------------------------------------------- idempotency

def test_second_call_does_not_create_a_second_bill(xero):
    erp = erp_payload()
    first = xero_poster.post_invoice(erp)
    n = len(xero.requests)
    second = xero_poster.post_invoice(erp)
    assert first.status == "posted" and second.status == "already_posted" and second.replayed
    assert second.xero_invoice_id == first.xero_invoice_id and second.url == first.url
    assert len(xero.invoices) == 1 and len(xero.puts) == 1 and len(xero.requests) == n   # answered locally


def test_lost_local_state_is_covered_by_asking_xero_for_the_invoice_number(xero, tmp_path, monkeypatch):
    erp = erp_payload()
    first = xero_poster.post_invoice(erp)
    monkeypatch.setattr(settings, "erp_db_path", str(tmp_path / "fresh_after_redeploy.db"))   # ephemeral disk wiped
    second = xero_poster.post_invoice(erp)
    assert second.ok and second.replayed and second.xero_invoice_id == first.xero_invoice_id
    assert len(xero.invoices) == 1 and len(xero.puts) == 1


def test_concurrent_post_of_the_same_invoice_is_refused_not_duplicated(xero):
    erp = erp_payload()
    claimed, _ = database.xero_claim(xero_poster.reference_for(erp["invoice_id"]), erp["invoice_id"], "p")
    assert claimed
    res = xero_poster.post_invoice(erp)
    assert not res.ok and res.status == "in_progress" and xero.requests == []


def test_a_voided_bill_is_never_silently_recreated(xero):
    erp = erp_payload()
    xero_poster.post_invoice(erp)
    xero.invoices[0]["Status"] = "VOIDED"
    database.xero_finish(xero_poster.reference_for(erp["invoice_id"]), "failed", error="x")   # local state forgotten
    res = xero_poster.post_invoice(erp)
    assert not res.ok and "VOIDED" in res.error and "manual action" in res.error and len(xero.puts) == 1


def test_different_invoices_get_different_bills(xero):
    a, b = xero_poster.post_invoice(erp_payload()), xero_poster.post_invoice(erp_payload())
    assert a.ok and b.ok and a.xero_invoice_id != b.xero_invoice_id and len(xero.invoices) == 2


def test_the_reference_key_is_shared_with_sap_and_the_post_ready_output(xero):
    from ap_automation.core import post_ready, sap_poster
    assert xero_poster.reference_for("x") == sap_poster.reference_for("x") == post_ready.reference_for("x")


# --------------------------------------------------------------------------- off by default / guard rails

def test_defaults_are_off():
    fresh = type(settings)(_env_file=None)
    assert fresh.erp_backend == "simulated" and fresh.xero_posting_enabled is False
    assert fresh.xero_bill_status == "DRAFT" and fresh.xero_client_id == "" and fresh.xero_client_secret == ""
    assert fresh.xero_token_path == "data/xero_tokens.json"


def test_simulated_backend_never_touches_xero(xero, monkeypatch, seeded):
    monkeypatch.setattr(settings, "erp_backend", "simulated")
    pred = _auto_post_prediction()
    res = posting.post_outcome(pred, pred.predicted_gl, pred.predicted_cost_centre, "system",
                               ConfirmationSource.AUTO_POST)
    assert res.erp_posted and res.erp_backend == "simulated" and xero.requests == []
    assert len(database.get_postings(10)) == 1


def test_backend_xero_without_the_enable_flag_posts_nothing_anywhere(xero, monkeypatch, seeded):
    monkeypatch.setattr(settings, "xero_posting_enabled", False)
    pred = _auto_post_prediction()
    res = posting.post_outcome(pred, pred.predicted_gl, pred.predicted_cost_centre, "system",
                               ConfirmationSource.AUTO_POST)
    assert not res.erp_posted and res.erp_failed and "XERO_POSTING_ENABLED" in res.error
    assert xero.requests == [] and database.get_postings(10) == []         # not silently diverted to the ledger


def test_not_connected_or_no_credentials_is_refused(xero, monkeypatch):
    monkeypatch.setattr(settings, "xero_client_secret", "")
    res = xero_poster.post_invoice(erp_payload())
    assert not res.ok and res.status == "disabled" and "credentials are not configured" in res.error
    monkeypatch.setattr(settings, "xero_client_secret", CLIENT_SECRET)
    xero_auth.token_path().unlink()
    res = xero_poster.post_invoice(erp_payload())
    assert not res.ok and "not connected" in res.error and xero.requests == []


@pytest.mark.parametrize("name,value", [("xero_bill_status", "PAID"), ("xero_line_amount_types", "Gross")])
def test_invalid_settings_are_refused(xero, monkeypatch, name, value):
    monkeypatch.setattr(settings, name, value)
    res = xero_poster.post_invoice(erp_payload())
    assert not res.ok and res.kind == "config" and xero.requests == []


def test_a_simulated_clock_never_posts_to_xero_whatever_the_backend(xero):
    from datetime import datetime, timezone

    from ap_automation.core import clock
    clock.set_clock(lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))
    try:
        assert posting.external_backend() is None
    finally:
        clock.set_clock(None)
    assert posting.external_backend() == "xero"


def test_evaluation_forces_the_simulated_backend(xero):
    from ap_automation.core import evaluation
    out = evaluation.run_evaluation(evaluation.EvalConfig(n_routine=6, n_near_edge=2, n_hard_edge=2,
                                                          history_per_vendor=4, window=5))
    assert out["with_learning"] and xero.requests == []
    assert settings.erp_backend == "xero"                                  # restored afterwards


def test_scripts_force_the_simulated_backend():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for name in ("run_demo.py", "replay_public_data.py"):
        text = (root / "scripts" / name).read_text(encoding="utf-8")
        assert 'settings.erp_backend = "simulated"' in text, name
    assert 'settings.erp_backend = "simulated"' in (root / "ap_automation/core/evaluation.py").read_text("utf-8")


# --------------------------------------------------------------------------- OAuth: authorize, refresh, rotation

def test_authorize_url_and_code_exchange(xero, tmp_path):
    url = xero_auth.authorize_url("STATE123")
    assert url.startswith("https://login.xero.com/identity/connect/authorize?")
    assert "state=STATE123" in url and "response_type=code" in url and f"client_id={CLIENT_ID}" in url
    assert "offline_access" in url and "accounting.invoices" in url and "accounting.settings.read" in url
    assert "accounting.transactions" not in url and CLIENT_SECRET not in url          # granular scopes, no secret

    settings.xero_token_path = str(tmp_path / "fresh" / "tokens.json")
    tokens = xero_auth.exchange_code("AUTH-CODE-1")
    form = xero.token_requests[-1]
    assert form["grant_type"] == "authorization_code" and form["redirect_uri"] == settings.xero_redirect_uri
    stored = json.loads(xero_auth.token_path().read_text("utf-8"))
    assert stored["refresh_token"] == tokens["refresh_token"] and stored["expires_at"] > time.time()


def test_a_wrong_authorization_code_is_a_clear_error(xero):
    with pytest.raises(xero_auth.XeroError, match="expired or was revoked"):
        xero_auth.exchange_code("WRONG")


def test_refresh_rotates_and_persists_the_new_refresh_token(xero):
    access, refresh = connect(xero, expires_in=-10)                         # access token has expired
    res = xero_poster.post_invoice(erp_payload())
    assert res.ok and len(xero.token_calls) == 1
    form = xero.token_requests[-1]
    assert form["grant_type"] == "refresh_token" and form["refresh_token"] == refresh
    stored = json.loads(xero_auth.token_path().read_text("utf-8"))
    assert stored["refresh_token"] != refresh and stored["refresh_token"] == xero.refresh_current   # the NEW one
    assert stored["access_token"] != access and stored["tenant_id"] == TENANT                       # tenant kept
    xero_poster.post_invoice(erp_payload())
    assert len(xero.token_calls) == 1                                       # token still valid: no second refresh


def test_the_refresh_token_is_replaced_on_every_rotation(xero):
    connect(xero, expires_in=-10)
    seen = []
    for _ in range(3):
        xero_auth.access_token()
        seen.append(xero_auth.load_tokens()["refresh_token"])
        xero_auth.save_tokens({**xero_auth.load_tokens(), "expires_at": time.time() - 1})
    assert len(set(seen)) == 3 and seen[-1] == xero.refresh_current


def test_a_lost_refresh_response_is_retried_with_the_same_old_token(xero):
    _, old = connect(xero, expires_in=-10)
    xero.token_script = ["timeout_lost"]                                    # rotated at Xero, answer never arrived
    token = xero_auth.access_token()
    used = [f["refresh_token"] for f in xero.token_requests]
    assert used == [old, old]                                               # same token, inside Xero's grace period
    assert token and xero_auth.load_tokens()["refresh_token"] == xero.refresh_current


def test_crash_before_the_new_token_is_saved_loses_nothing(xero, monkeypatch):
    """The process dies after Xero answered but before we wrote the new refresh token."""
    _, old = connect(xero, expires_in=-10)
    real_save = xero_auth.save_tokens

    def die(data):
        raise Crash()

    monkeypatch.setattr(xero_auth, "save_tokens", die)
    with pytest.raises(Crash):
        xero_auth.access_token()
    monkeypatch.setattr(xero_auth, "save_tokens", real_save)

    assert json.loads(xero_auth.token_path().read_text("utf-8"))["refresh_token"] == old     # old set intact on disk
    xero_auth.reset_state()                                                 # "restart the process"
    res = xero_poster.post_invoice(erp_payload())                           # old token still works (grace period)
    assert res.ok
    assert json.loads(xero_auth.token_path().read_text("utf-8"))["refresh_token"] == xero.refresh_current


def test_crash_in_the_middle_of_writing_never_leaves_a_corrupt_file(xero, monkeypatch):
    connect(xero, expires_in=-10)
    before = xero_auth.token_path().read_text("utf-8")
    monkeypatch.setattr(xero_auth.os, "replace", lambda *a: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        xero_auth.save_tokens({**json.loads(before), "refresh_token": "NEW-TOKEN-NEVER-COMMITTED"})
    assert xero_auth.token_path().read_text("utf-8") == before              # untouched, still valid JSON
    assert [p.name for p in xero_auth.token_path().parent.iterdir() if p.name.endswith(".tmp")] == []   # tmp cleaned


def test_a_full_or_read_only_disk_keeps_the_new_tokens_in_memory_and_reports_it(xero, monkeypatch, capsys):
    _, old = connect(xero, expires_in=-10)
    real_replace = os.replace
    monkeypatch.setattr(xero_auth.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    token = xero_auth.access_token()                                        # works for this process
    assert token and xero_auth.load_tokens()["refresh_token"] == xero.refresh_current
    assert json.loads(xero_auth.token_path().read_text("utf-8"))["refresh_token"] == old     # disk still the old set
    assert "could not write the token file" in capsys.readouterr().out
    assert xero_poster.post_invoice(erp_payload()).ok and len(xero.token_calls) == 1          # no second rotation

    monkeypatch.setattr(xero_auth.os, "replace", real_replace)              # disk recovers: next use flushes
    xero_auth.load_tokens()
    assert json.loads(xero_auth.token_path().read_text("utf-8"))["refresh_token"] == xero.refresh_current


def test_revoked_or_expired_connection_fails_visibly_and_keeps_the_invoice_unposted(xero):
    connect(xero, expires_in=-10)
    xero.token_script = ["invalid_grant"]
    res = xero_poster.post_invoice(erp_payload())
    assert not res.ok and res.kind == "auth" and "scripts/xero_authorize.py" in res.error and xero.puts == []
    assert database.get_xero_postings(10, "failed")


def test_invalid_client_is_a_config_error(xero):
    connect(xero, expires_in=-10)
    xero.token_script = ["invalid_client"]
    res = xero_poster.post_invoice(erp_payload())
    assert not res.ok and res.kind == "config" and "XERO_CLIENT_ID" in res.error


def test_simultaneous_requests_cause_exactly_one_refresh(xero):
    connect(xero, expires_in=-10)
    results, errors = [], []

    def worker():
        try:
            results.append(xero_auth.access_token())
        except Exception as e:      # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(set(results)) == 1 and len(xero.token_calls) == 1


def test_a_stale_lock_file_from_a_dead_process_does_not_block(xero):
    connect(xero, expires_in=-10)
    lock = xero_auth.token_path().with_name(xero_auth.token_path().name + ".lock")
    lock.write_text("")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    assert xero_auth.access_token() and not lock.exists()


def test_unreadable_token_file_is_a_clear_error(xero):
    xero_auth.token_path().write_text("{not json", encoding="utf-8")
    xero_auth.reset_state()
    res = xero_poster.post_invoice(erp_payload())
    assert not res.ok and "unreadable" in res.error and xero.requests == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_token_file_is_owner_only_on_posix(xero):
    connect(xero)
    assert (xero_auth.token_path().stat().st_mode & 0o777) == 0o600


@pytest.mark.skipif(os.name != "nt", reason="Windows ACLs")
def test_token_file_acl_is_restricted_to_the_current_user_on_windows(xero, monkeypatch):
    import subprocess
    monkeypatch.setattr(xero_auth, "HARDEN_WINDOWS_ACL", True)
    connect(xero)
    out = subprocess.run(["icacls", str(xero_auth.token_path())], capture_output=True, text=True).stdout
    assert os.environ["USERNAME"].lower() in out.lower()
    acl = out.replace(str(xero_auth.token_path()), "")
    assert "Administrators" not in acl and "BUILTIN" not in acl and "Everyone" not in acl and "SYSTEM" not in acl
    assert acl.count("(F)") == 1                      # exactly one entry: the current user


# --------------------------------------------------------------------------- secrets never leak

def test_secrets_never_appear_in_results_logs_or_stored_records(xero, capsys, caplog):
    erp = erp_payload()
    xero.script = ["reject_echo_secrets", "boom_with_secret", "ok"]
    results = [xero_poster.post_invoice(erp)]                               # Xero echoes a token and the secret
    results.append(xero_poster.post_invoice(erp))                           # an unexpected exception mentions them
    results.append(xero_poster.post_invoice(erp))
    out = capsys.readouterr()
    blob = "\n".join([out.out, out.err, caplog.text, _all_stored_text(), json.dumps([r.audit_view() for r in results])])
    assert _leaks(blob) == [] and "***" in results[0].error and not results[0].ok
    assert results[1].status == "unknown" and results[2].ok


def test_oauth_failures_do_not_leak_tokens_or_the_client_secret(xero, capsys):
    _, old = connect(xero, expires_in=-10)
    xero.token_script = ["invalid_grant"]
    res = xero_poster.post_invoice(erp_payload())
    blob = "\n".join([capsys.readouterr().out, _all_stored_text(), json.dumps(res.audit_view())])
    assert _leaks(blob) == [] and old not in blob and CLIENT_ID not in json.dumps(res.audit_view())


def test_sanitize_removes_secrets_tokens_headers_and_bounds_length(xero):
    jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxMjM0NTYifQ.c2lnbmF0dXJl"
    text = (f'Authorization: Bearer {jwt}\n"refresh_token":"abcdEFGH1234" access_token={jwt} {CLIENT_SECRET} '
            "Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ== " + "y" * 5000)
    clean = xero_auth.sanitize(text)
    for bad in (jwt, "abcdEFGH1234", CLIENT_SECRET, "QWxhZGRpbjpvcGVuIHNlc2FtZQ=="):
        assert bad not in clean
    assert "\n" not in clean and len(clean) <= 600


def test_status_reports_state_without_secrets(xero):
    s = xero_poster.status()
    assert s["configured"] and s["connected"] and s["posting_enabled"] and s["ready_to_post"] and s["demo_company"]
    assert s["org_name"] == "Demo Company (UK)" and s["backend"] == "xero" and s["bill_status"] == "DRAFT"
    blob = json.dumps(s)
    assert _leaks(blob) == [] and CLIENT_ID not in blob and TENANT not in blob


# --------------------------------------------------------------------------- chart of accounts

ACCOUNTS = [
    {"Code": "400", "Name": "Advertising", "Type": "EXPENSE", "Status": "ACTIVE", "TaxType": "INPUT2"},
    {"Code": "461", "Name": "Printing & Stationery", "Type": "OVERHEADS", "Status": "ACTIVE", "TaxType": "RRINPUT"},
    {"Code": "310", "Name": "Cost of Goods Sold", "Type": "DIRECTCOSTS", "Status": "ACTIVE", "TaxType": "ZERORATEDINPUT"},
    {"Code": "477", "Name": "Wages", "Type": "EXPENSE", "Status": "ACTIVE", "TaxType": "WEIRDTAX"},
    {"Code": "200", "Name": "Sales", "Type": "REVENUE", "Status": "ACTIVE", "TaxType": "OUTPUT2"},
    {"Code": "090", "Name": "Business Bank Account", "Type": "BANK", "Status": "ACTIVE", "TaxType": "NONE"},
    {"Code": "425", "Name": "Old stuff", "Type": "EXPENSE", "Status": "ARCHIVED", "TaxType": "INPUT2"},
    {"Name": "No code", "Type": "EXPENSE", "Status": "ACTIVE", "TaxType": "INPUT2"},
]


def test_accounts_become_our_chart_with_only_expense_type_accounts(xero):
    chart, info = xero_poster.accounts_to_chart(ACCOUNTS)
    assert sorted(chart) == ["310", "400", "461", "477"]
    assert (chart["400"].vat_treatment, chart["461"].vat_treatment, chart["310"].vat_treatment) == (
        "STANDARD", "REDUCED", "ZERO")
    assert chart["477"].vat_treatment == "NOT_ASSESSED" and info["unmapped_tax_types"] == {"WEIRDTAX": 1}
    assert info["skipped"]["wrong_type"] == 2 and info["skipped"]["inactive"] == 1 and info["skipped"]["no_code"] == 1
    assert chart["400"].name == "Advertising" and chart["400"].default_cost_centre == ""


def test_chart_types_are_configurable(xero, monkeypatch):
    monkeypatch.setattr(settings, "xero_chart_account_types", "EXPENSE")
    chart, _ = xero_poster.accounts_to_chart(ACCOUNTS)
    assert sorted(chart) == ["400", "477"]


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def test_sync_chart_endpoint_replaces_the_chart_and_makes_codes_valid_for_confirmation(client, xero, seeded):
    xero.accounts = ACCOUNTS
    dry = client.post("/xero/sync-chart", params={"dry_run": "true"}).json()
    assert dry["status"] == "dry_run" and dry["accounts"] == 4 and not chart_of_accounts.is_custom()

    r = client.post("/xero/sync-chart")
    assert r.status_code == 200 and r.json()["status"] == "chart_replaced" and r.json()["persisted"]
    assert chart_of_accounts.is_custom() and sorted(chart_of_accounts.CHART) == ["310", "400", "461", "477"]
    assert client.get("/chart").json()["accounts"] == 4
    assert [q.url.params.get("where") for q in xero.requests if q.url.path.endswith("/Accounts")][0] == 'Status=="ACTIVE"'
    p = client.post("/predict", json={"raw_vendor_name": "Meridian Facilities Ltd", "description": "ads",
                                      "amount": 100}).json()
    assert client.post(f"/confirm/{p['prediction_id']}", json={"confirmed_gl": "6300"}).status_code == 422   # not in Xero
    ok = client.post(f"/confirm/{p['prediction_id']}", json={"confirmed_gl": "400"})
    assert ok.status_code == 200 and xero.invoices[0]["LineItems"][0]["AccountCode"] == "400"


def test_sync_chart_never_empties_the_chart(client, xero):
    xero.accounts = [a for a in ACCOUNTS if a.get("Type") == "BANK"]
    r = client.post("/xero/sync-chart")
    assert r.status_code == 502 and "was not changed" in r.json()["detail"]["message"]
    assert not chart_of_accounts.is_custom()


def test_sync_chart_when_not_connected_is_a_clear_409(client, xero):
    xero_auth.token_path().unlink()
    r = client.post("/xero/sync-chart")
    assert r.status_code == 409 and "not connected" in r.json()["detail"]["message"]


# --------------------------------------------------------------------------- through the API

BODY = {"raw_vendor_name": "Meridian Facilities Ltd", "description": "Office cleaning services monthly",
        "amount": 1200, "entity_id": "UK001", "cost_centre": "CC300"}


def _auto_post_prediction():
    from ap_automation.core.pipeline import predict
    from tests.conftest import make_invoice

    pred = predict(make_invoice(), batch_mode=True)
    assert pred.final_routing.value == "auto_post"
    return pred


def _predict(client) -> dict:
    return client.post("/predict", json=BODY).json()


def test_status_endpoint(client, xero):
    r = client.get("/xero/status")
    assert r.status_code == 200 and r.json()["ready_to_post"] is True and r.json()["org_name"] == "Demo Company (UK)"
    assert _leaks(r.text) == [] and CLIENT_ID not in r.text


def test_status_endpoint_when_off_by_default(client, monkeypatch):
    body = client.get("/xero/status").json()
    assert body["backend"] == "simulated" and body["posting_enabled"] is False and body["ready_to_post"] is False
    assert body["configured"] is False and body["connected"] is False and body["org_name"] is None


def test_auto_post_still_403_for_human_review_even_with_xero_on(client, xero, seeded):
    p = client.post("/predict", json={**BODY, "raw_vendor_name": "Totally Unknown Vendor GmbH"}).json()
    assert p["final_routing"] == "human_review"
    r = client.post(f"/auto-post/{p['prediction_id']}")
    assert r.status_code == 403 and xero.requests == [] and database.get_postings(10) == []


def test_auto_post_to_xero_succeeds_once_and_shows_in_ledger_audit_and_postings(client, xero, seeded):
    p = _predict(client)
    assert p["final_routing"] == "auto_post"
    r = client.post(f"/auto-post/{p['prediction_id']}")
    assert r.status_code == 200, r.text
    body = r.json()
    created = xero.invoices[0]
    assert body["erp_posted"] is True and body["erp_backend"] == "xero" and body["learned"] is False
    assert body["erp_ref"]["xero_invoice_id"] == created["InvoiceID"] and "go.xero.com" in body["erp_ref"]["url"]
    assert created["Contact"]["Name"] == "Meridian Facilities Ltd" and created["Status"] == "DRAFT"

    ledger = client.get("/ledger").json()["postings"]
    assert len(ledger) == 1 and ledger[0]["erp_backend"] == "xero" and ledger[0]["erp_document"] == created["InvoiceID"]
    assert ledger[0]["xero_invoice_id"] == created["InvoiceID"] and ledger[0]["source"] == "auto_post"
    rec = client.get("/audit").json()["records"][0]
    assert rec["source"] == "auto_post" and rec["provenance"]["erp"]["xero_invoice_id"] == created["InvoiceID"]
    listed = client.get("/xero/postings", params={"status": "posted"}).json()["postings"]
    assert len(listed) == 1 and listed[0]["xero_invoice_id"] == created["InvoiceID"]
    assert _leaks(_all_stored_text() + r.text) == []

    assert client.post(f"/auto-post/{p['prediction_id']}").status_code == 409        # no second post
    assert len(xero.invoices) == 1 and len(xero.puts) == 1


def test_xero_rejection_keeps_the_invoice_in_review_with_the_error(client, xero, seeded):
    xero.script = ["reject"]
    p = _predict(client)
    r = client.post(f"/auto-post/{p['prediction_id']}")
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail["status"] == "xero_post_failed" and "not a valid code" in detail["xero"]["error"]
    assert "Xero did not accept" in detail["message"]

    stored = main._store.get(p["prediction_id"])
    assert stored["posted"] is False and stored["pred"].final_routing.value == "human_review"
    assert "Xero post not completed" in stored["pred"].final_routing_reason
    assert client.get("/ledger").json()["postings"] == []
    rec = client.get("/audit").json()["records"][0]
    assert rec["source"] == "xero_post_failed" and rec["routing_decision"] == "human_review"
    assert "not a valid code" in rec["routing_reason"]
    failed = client.get("/xero/postings", params={"status": "failed"}).json()["postings"]
    assert len(failed) == 1 and "not a valid code" in failed[0]["error"]

    # no longer auto-postable, but a human can confirm it once Xero is fixed; still one bill
    assert client.post(f"/auto-post/{p['prediction_id']}").status_code == 403
    r2 = client.post(f"/confirm/{p['prediction_id']}", json={"confirmed_gl": p["predicted_gl"]})
    assert r2.status_code == 200 and r2.json()["erp_posted"] is True and r2.json()["learned"] is True
    assert len(xero.invoices) == 1
    ledger = client.get("/ledger").json()["postings"]
    assert len(ledger) == 1 and ledger[0]["source"] == "human_confirm" and ledger[0]["xero_invoice_id"]


def test_human_confirm_failure_is_not_learned_and_can_be_retried(client, xero, seeded):
    xero.script = ["reject"]
    p = _predict(client)
    r = client.post(f"/confirm/{p['prediction_id']}", json={"confirmed_gl": "6300"})
    assert r.status_code == 502 and r.json()["detail"]["status"] == "xero_post_failed"
    assert client.get("/corpus/vendors").json()["corpus_size"] == main._collection_size()
    assert main._store.get(p["prediction_id"])["posted"] is False
    ok = client.post(f"/confirm/{p['prediction_id']}", json={"confirmed_gl": "6300"})
    assert ok.status_code == 200 and len(xero.invoices) == 1


def test_batch_marks_a_xero_failure_as_review_not_auto_confirmed(client, xero, seeded):
    import io
    xero.script = ["reject"]
    csv_text = ("vendor_name,description,amount,entity_id,cost_centre\n"
                "Meridian Facilities Ltd,Office cleaning services monthly,1200,UK001,\n")
    job = client.post("/batch", files={"file": ("b.csv", io.BytesIO(csv_text.encode()), "text/csv")}).json()["job_id"]
    for _ in range(100):
        status = client.get(f"/batch/{job}").json()
        if status["status"] != "running":
            break
        time.sleep(0.1)
    row = status["results"][0]
    assert status["auto_confirmed"] == 0 and status["human_review"] == 1
    assert row["routing"] == "human_review" and row["erp_posted"] is False and "Xero post not completed" in row["reason"]
    assert xero.invoices == []


def test_simulated_default_still_works_end_to_end(client, seeded):
    p = _predict(client)
    r = client.post(f"/auto-post/{p['prediction_id']}")
    assert r.status_code == 200 and r.json()["erp_backend"] == "simulated" and r.json()["erp_posted"] is True
    led = client.get("/ledger").json()["postings"][0]
    assert led["erp_backend"] == "simulated" and led["erp_document"] == ""


def test_the_api_key_protects_the_new_endpoints(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "FAKE-API-KEY-12345")
    for method, path in (("get", "/xero/status"), ("get", "/xero/postings"), ("post", "/xero/sync-chart")):
        assert getattr(client, method)(path).status_code == 401
    assert client.get("/xero/status", headers={"X-API-Key": "FAKE-API-KEY-12345"}).status_code == 200
