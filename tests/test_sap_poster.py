"""
SAP posting backend. Everything runs against a fake SAP behind httpx.MockTransport: no network, no real key.

The "key" used here is a made-up string; the tests check that it - and CSRF tokens - never reach a result, a log
line, an audit record or the SAP posting table, even when SAP's own error text echoes it back.
"""

import io
import json
import re
import uuid
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from ap_automation.core import database, posting, sap_poster
from ap_automation.core.config import settings
from ap_automation.core.models import ConfirmationSource

FAKE_KEY = "FAKE-SANDBOX-KEY-0123456789abcdef"
CSRF = "csrf-token-XYZ-98765"
SERVICE = "/s4hanacloud" + sap_poster.SERVICE_PATH   # base URL path + service path
METADATA = json.loads((Path(__file__).parent / "fixtures" / "sap_supplierinvoice_metadata.json").read_text("utf-8"))


# --------------------------------------------------------------------------- fake SAP

class FakeSap:
    """Just enough of API_SUPPLIERINVOICE_PROCESS_SRV: CSRF fetch, read by reference, create."""

    def __init__(self):
        self.invoices: list[dict] = []
        self.requests: list[httpx.Request] = []
        self.script: list[str] = []          # one entry consumed per POST; "ok" once exhausted
        self.token_valid = True
        self.transport = httpx.MockTransport(self.handle)
        self._n = 5105600000

    # -- helpers
    @property
    def posts(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == "POST"]

    def _create(self, body: dict) -> dict:
        self._n += 1
        doc = {k: v for k, v in body.items() if not k.startswith("to_")}
        doc.update({"SupplierInvoice": str(self._n), "FiscalYear": "2026", "JournalEntry": str(self._n),
                    "SupplierInvoiceStatus": "5", "IsReversed": False,
                    "SecretLookingExtra": "must-not-be-stored"})
        self.invoices.append(doc)
        return doc

    @staticmethod
    def _err(status: int, message: str, code: str = "FIN_SUPPLIER_INV/001") -> httpx.Response:
        return httpx.Response(status, json={"error": {"code": code, "message": {"lang": "en", "value": message},
                                                       "innererror": {"errordetails": [
                                                           {"code": "FIN/002", "message": "Supplier is blocked"}]}}})

    # -- the endpoint
    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path.removeprefix("/s4hanacloud")   # the sandbox base URL has a path prefix, a tenant none
        if request.method == "GET" and path == sap_poster.SERVICE_PATH + "/":
            if request.headers.get("x-csrf-token") == "Fetch":
                return httpx.Response(200, json={"d": {"EntitySets": []}},
                                      headers={"x-csrf-token": CSRF, "set-cookie": "sap-XSRF=abc123; Path=/"})
            return httpx.Response(200, json={})
        if request.method == "GET" and path == sap_poster.SERVICE_PATH + "/A_SupplierInvoice":
            m = re.search(r"SupplierInvoiceIDByInvcgParty eq '([^']*)'", request.url.params["$filter"])
            rows = [i for i in self.invoices if i.get("SupplierInvoiceIDByInvcgParty") == m.group(1)]
            return httpx.Response(200, json={"d": {"results": rows}})
        if request.method == "POST" and path == sap_poster.SERVICE_PATH + "/A_SupplierInvoice":
            action = self.script.pop(0) if self.script else "ok"
            if action == "connect_error":
                raise httpx.ConnectError("name resolution failed", request=request)
            if action == "csrf_stale" or request.headers.get("x-csrf-token") != CSRF or not self.token_valid:
                return httpx.Response(403, text="CSRF token validation failed",
                                      headers={"x-csrf-token": "Required"})
            body = json.loads(request.content)
            if action == "reject":
                return self._err(400, "Supplier 11300001 is blocked for posting")
            if action == "reject_echo_key":
                return self._err(400, f"Authentication {FAKE_KEY} refused; APIKey: {FAKE_KEY}; token {CSRF}")
            if action == "gateway_405":
                return httpx.Response(405, json={"code": "OPERATION_NOT_SUPPORTED", "message": "GET only"})
            if action == "timeout_created":     # SAP processed it, the answer never arrived
                self._create(body)
                raise httpx.ReadTimeout("read timed out", request=request)
            if action == "timeout_lost":        # the request never took effect
                raise httpx.ReadTimeout("read timed out", request=request)
            if action == "500_created":
                self._create(body)
                return httpx.Response(500, text="boom")
            if action == "boom_with_key":
                raise RuntimeError(f"unexpected failure talking to SAP with {FAKE_KEY}")
            return httpx.Response(201, json={"d": self._create(body)})
        return httpx.Response(404, json={})


def erp_payload(**over) -> dict:
    base = {"invoice_id": str(uuid.uuid4()), "vendor_id": "VND-ACME", "entity_id": "UK001", "gl_account": "6300",
            "cost_centre": "CC300", "tax_code": "STANDARD", "amount": 1200.0, "currency": "GBP",
            "description": "Office cleaning services monthly", "posted_by": "auto_post_service",
            "posted_at": "2026-10-27T07:45:00+00:00", "trace_id": str(uuid.uuid4())}
    base.update(over)
    return base


@pytest.fixture
def sap(monkeypatch):
    """SAP backend switched fully on against the fake. Individual tests switch things off again."""
    fake = FakeSap()
    for k, v in {"erp_backend": "sap", "sap_posting_enabled": True, "sap_api_key": FAKE_KEY,
                 "sap_base_url": "https://sandbox.api.sap.com/s4hanacloud", "sap_default_supplier": "11300001",
                 "sap_company_code": "1110", "sap_max_retries": 2, "sap_allow_non_sandbox": False,
                 "sap_username": "", "sap_password": ""}.items():
        monkeypatch.setattr(settings, k, v)
    monkeypatch.setattr(sap_poster, "_sleep", lambda s: None)
    sap_poster.set_transport(fake.transport)
    return fake


def _all_stored_text() -> str:
    """Everything persisted by a post: SAP posting table, audit trail, ledger."""
    parts = [json.dumps(database.get_sap_postings(1000)), json.dumps(database.get_audit_records(1000)),
             json.dumps(database.get_postings(1000))]
    return "\n".join(parts)


# --------------------------------------------------------------------------- mapping

def test_reference_key_is_deterministic_and_fits_the_sap_field():
    inv = str(uuid.uuid4())
    ref = sap_poster.reference_for(inv)
    assert ref == sap_poster.reference_for(inv) and ref != sap_poster.reference_for(str(uuid.uuid4()))
    assert len(ref) <= METADATA["A_SupplierInvoiceType"]["SupplierInvoiceIDByInvcgParty"]["max"] == 16


def test_payload_mapping_and_defaults(sap):
    p = sap_poster.build_payload(erp_payload(invoice_id="inv-1"))
    item = p["to_SupplierInvoiceItemGLAcct"]["results"][0]
    assert p["CompanyCode"] == "1110" and p["InvoicingParty"] == "11300001"        # defaults
    assert p["DocumentCurrency"] == "GBP" and p["InvoiceGrossAmount"] == "1200.00"
    assert p["SupplierInvoiceIDByInvcgParty"] == sap_poster.reference_for("inv-1")
    assert p["DocumentHeaderText"] == "Office cleaning services" and len(p["DocumentHeaderText"]) <= 25
    assert p["PostingDate"] == p["DocumentDate"] == "/Date(1793059200000)/"       # 2026-10-27 00:00 UTC
    assert p["TaxIsCalculatedAutomatically"] is True
    assert item == {"SupplierInvoiceItem": "1", "CompanyCode": "1110", "GLAccount": "6300", "CostCenter": "CC300",
                    "DocumentCurrency": "GBP", "SupplierInvoiceItemAmount": "1200.00", "DebitCreditCode": "S",
                    "TaxCode": "V0", "SupplierInvoiceItemText": "Office cleaning services monthly"}


def test_mappings_are_configurable(sap, monkeypatch):
    monkeypatch.setattr(settings, "sap_company_code_map", json.dumps({"UK001": "2000"}))
    monkeypatch.setattr(settings, "sap_supplier_map", json.dumps({"VND-ACME": "SUP-77"}))
    monkeypatch.setattr(settings, "sap_gl_map", json.dumps({"6300": "61061000"}))
    monkeypatch.setattr(settings, "sap_cost_centre_map", json.dumps({"CC300": "17101201"}))
    monkeypatch.setattr(settings, "sap_tax_code_map", json.dumps({"STANDARD": "I1"}))
    p = sap_poster.build_payload(erp_payload())
    item = p["to_SupplierInvoiceItemGLAcct"]["results"][0]
    assert (p["CompanyCode"], p["InvoicingParty"]) == ("2000", "SUP-77")
    assert (item["CompanyCode"], item["GLAccount"], item["CostCenter"], item["TaxCode"]) == (
        "2000", "61061000", "17101201", "I1")


def test_payload_fields_exist_in_the_sap_metadata_and_fit(sap):
    p = sap_poster.build_payload(erp_payload(description="x" * 200))
    for entity, body in (("A_SupplierInvoiceType", {k: v for k, v in p.items() if k != sap_poster.GL_NAV}),
                         ("A_SupplierInvoiceItemGLAcctType", p[sap_poster.GL_NAV]["results"][0])):
        meta = METADATA[entity]
        for name, value in body.items():
            assert name in meta, f"{name} is not a field of {entity}"
            assert meta[name]["creatable"], f"{name} is not creatable"
            if meta[name]["max"] and isinstance(value, str):
                assert len(value) <= meta[name]["max"], f"{name} longer than {meta[name]['max']}"
    assert sap_poster.GL_NAV in {"to_SupplierInvoiceItemGLAcct"}


@pytest.mark.parametrize("over,expect", [
    ({"amount": 0}, "greater than zero"),
    ({"amount": -5}, "greater than zero"),
    ({"gl_account": ""}, "G/L account"),
])
def test_unmappable_invoices_are_refused_without_a_request(sap, over, expect):
    res = sap_poster.post_invoice(erp_payload(**over))
    assert not res.ok and res.kind == "mapping" and expect in res.error
    assert sap.requests == []
    assert database.get_sap_postings(10, "failed")[0]["error"] == res.error     # visible, not lost


def test_unmapped_supplier_is_a_clear_error_not_a_guess(sap, monkeypatch):
    monkeypatch.setattr(settings, "sap_default_supplier", "")
    res = sap_poster.post_invoice(erp_payload())
    assert not res.ok and "No SAP supplier mapped" in res.error and sap.posts == []


# --------------------------------------------------------------------------- the happy path and the CSRF flow

def test_success_uses_the_csrf_flow_and_records_the_document(sap):
    res = sap_poster.post_invoice(erp_payload())
    assert res.ok and res.status == "posted" and res.http_status == 201
    assert (res.document, res.fiscal_year, res.company_code) == ("5105600001", "2026", "1110")
    assert res.summary["InvoiceGrossAmount"] == "1200.00" and "SecretLookingExtra" not in res.summary

    methods = [(r.method, r.url.path.removeprefix(SERVICE)) for r in sap.requests]
    assert methods == [("GET", "/A_SupplierInvoice"), ("GET", "/"), ("POST", "/A_SupplierInvoice")]
    fetch, post = sap.requests[1], sap.posts[0]
    assert fetch.headers["x-csrf-token"] == "Fetch" and fetch.headers["accept"] == "application/json"
    assert post.headers["x-csrf-token"] == CSRF                       # the fetched token is used...
    assert "sap-XSRF=abc123" in post.headers["cookie"]                # ...with the cookie that came with it
    assert post.headers["content-type"].startswith("application/json") and post.headers["accept"] == "application/json"
    assert post.headers["apikey"] == FAKE_KEY                         # sent to SAP, and only there
    row = database.get_sap_postings(10, "posted")[0]
    assert row["sap_document"] == "5105600001" and row["reference"] == res.reference and row["attempts"] == 1


def test_basic_auth_for_a_real_tenant(sap, monkeypatch):
    monkeypatch.setattr(settings, "sap_api_key", "")
    monkeypatch.setattr(settings, "sap_username", "COMM_USER")
    monkeypatch.setattr(settings, "sap_password", "pa55-word-xyz")
    assert sap_poster.post_invoice(erp_payload()).ok
    assert sap.posts[0].headers["authorization"].startswith("Basic ") and "apikey" not in sap.posts[0].headers


def test_stale_csrf_token_is_refreshed_once_and_nothing_is_duplicated(sap):
    sap.script = ["csrf_stale"]
    res = sap_poster.post_invoice(erp_payload())
    assert res.ok and len(sap.invoices) == 1 and len(sap.posts) == 2
    assert sum(1 for r in sap.requests if r.headers.get("x-csrf-token") == "Fetch") == 2


# --------------------------------------------------------------------------- SAP says no

def test_sap_validation_error_is_recorded_and_not_lost(sap):
    sap.script = ["reject"]
    erp = erp_payload()
    res = sap_poster.post_invoice(erp)
    assert not res.ok and res.status == "failed" and res.kind == "rejected" and res.http_status == 400
    assert "Supplier 11300001 is blocked for posting" in res.error and "FIN/002: Supplier is blocked" in res.error
    assert sap.invoices == [] and len(sap.posts) == 1                  # a refusal is final: not retried
    assert database.get_sap_postings(10, "failed")[0]["error"] == res.error

    sap.script = []                                                     # master data fixed -> retry posts once
    again = sap_poster.post_invoice(erp)
    assert again.ok and len(sap.invoices) == 1 and again.reference == res.reference
    assert database.get_sap_postings(10)[0]["attempts"] == 2


def test_gateway_get_only_response_is_reported_plainly(sap):
    sap.script = ["gateway_405"]
    res = sap_poster.post_invoice(erp_payload())
    assert not res.ok and res.http_status == 405 and "OPERATION_NOT_SUPPORTED" in res.error and "GET only" in res.error


# --------------------------------------------------------------------------- network trouble

def test_connection_failure_is_retried_a_bounded_number_of_times(sap):
    sap.script = ["connect_error"] * 10
    res = sap_poster.post_invoice(erp_payload())
    assert not res.ok and res.kind == "unavailable" and sap.invoices == []
    assert len(sap.posts) == 1 + settings.sap_max_retries               # bounded: 3 attempts, then give up


def test_connection_failure_then_recovery_posts_exactly_once(sap):
    sap.script = ["connect_error", "connect_error"]
    res = sap_poster.post_invoice(erp_payload())
    assert res.ok and len(sap.invoices) == 1 and len(sap.posts) == 3    # only the attempts that never connected


def test_timeout_after_sending_is_not_retried_and_is_reconciled_by_reference(sap):
    sap.script = ["timeout_created"]
    res = sap_poster.post_invoice(erp_payload())
    assert res.ok and res.status == "posted" and res.document == "5105600001"
    assert len(sap.posts) == 1 and len(sap.invoices) == 1              # found by reference, never re-posted


def test_server_error_that_actually_created_the_invoice_is_reconciled(sap):
    sap.script = ["500_created"]
    res = sap_poster.post_invoice(erp_payload())
    assert res.ok and len(sap.posts) == 1 and len(sap.invoices) == 1


def test_unconfirmed_outcome_stays_unknown_and_is_checked_before_any_retry(sap):
    sap.script = ["timeout_lost"]
    erp = erp_payload()
    res = sap_poster.post_invoice(erp)
    assert not res.ok and res.status == "unknown" and res.kind == "unknown_outcome"
    assert "NOT re-posted" in res.error and len(sap.posts) == 1
    assert database.get_sap_postings(10, "unknown")

    # the next attempt first asks SAP for the reference (none), then posts once
    again = sap_poster.post_invoice(erp)
    assert again.ok and len(sap.invoices) == 1 and len(sap.posts) == 2


def test_unknown_outcome_that_was_in_fact_created_is_never_double_posted(sap):
    sap.script = ["timeout_created"]
    erp = erp_payload()
    sap.invoices.clear()
    # the reconcile read is itself unreachable the first time -> unknown
    real_handle = sap.handle
    state = {"fail_reads": 3}

    def flaky(request):
        if request.method == "GET" and request.url.path.endswith("/A_SupplierInvoice") and sap.posts and state["fail_reads"]:
            state["fail_reads"] -= 1
            raise httpx.ConnectError("down", request=request)
        return real_handle(request)

    sap_poster.set_transport(httpx.MockTransport(flaky))
    first = sap_poster.post_invoice(erp)
    assert first.status == "unknown" and len(sap.invoices) == 1
    second = sap_poster.post_invoice(erp)                              # reads work again: it finds the invoice
    assert second.ok and second.replayed and second.document == sap.invoices[0]["SupplierInvoice"]
    assert len(sap.posts) == 1 and len(sap.invoices) == 1


# --------------------------------------------------------------------------- idempotency

def test_second_call_does_not_create_a_second_document(sap):
    erp = erp_payload()
    first = sap_poster.post_invoice(erp)
    n_requests = len(sap.requests)
    second = sap_poster.post_invoice(erp)
    assert first.status == "posted" and second.status == "already_posted" and second.replayed
    assert second.document == first.document and len(sap.invoices) == 1 and len(sap.posts) == 1
    assert len(sap.requests) == n_requests                              # answered from the local record, no SAP call


def test_lost_local_state_is_covered_by_asking_sap_for_the_reference(sap, tmp_path, monkeypatch):
    erp = erp_payload()
    first = sap_poster.post_invoice(erp)
    monkeypatch.setattr(settings, "erp_db_path", str(tmp_path / "fresh_after_redeploy.db"))   # ephemeral disk wiped
    second = sap_poster.post_invoice(erp)
    assert second.ok and second.replayed and second.document == first.document
    assert len(sap.invoices) == 1 and len(sap.posts) == 1


def test_concurrent_post_of_the_same_invoice_is_refused_not_duplicated(sap):
    erp = erp_payload()
    claimed, _ = database.sap_claim(sap_poster.reference_for(erp["invoice_id"]), erp["invoice_id"], "p")
    assert claimed                                                      # another worker is mid-post
    res = sap_poster.post_invoice(erp)
    assert not res.ok and res.status == "in_progress" and sap.requests == []


def test_different_invoices_get_different_documents(sap):
    a, b = sap_poster.post_invoice(erp_payload()), sap_poster.post_invoice(erp_payload())
    assert a.ok and b.ok and a.document != b.document and len(sap.invoices) == 2


# --------------------------------------------------------------------------- off by default / guard rails

def test_defaults_are_off(monkeypatch):
    fresh = type(settings)(_env_file=None)
    assert fresh.erp_backend == "simulated" and fresh.sap_posting_enabled is False
    assert fresh.sap_base_url == "https://sandbox.api.sap.com/s4hanacloud" and fresh.sap_allow_non_sandbox is False


def test_simulated_backend_never_touches_sap(sap, monkeypatch, seeded):
    monkeypatch.setattr(settings, "erp_backend", "simulated")
    pred = _auto_post_prediction()
    res = posting.post_outcome(pred, pred.predicted_gl, pred.predicted_cost_centre, "system",
                               ConfirmationSource.AUTO_POST)
    assert res.erp_posted and res.erp_backend == "simulated" and sap.requests == []
    assert len(database.get_postings(10)) == 1


def test_backend_sap_without_the_enable_flag_posts_nothing_anywhere(sap, monkeypatch, seeded):
    monkeypatch.setattr(settings, "sap_posting_enabled", False)
    pred = _auto_post_prediction()
    res = posting.post_outcome(pred, pred.predicted_gl, pred.predicted_cost_centre, "system",
                               ConfirmationSource.AUTO_POST)
    assert not res.erp_posted and res.erp_failed and "SAP_POSTING_ENABLED" in res.error
    assert sap.requests == [] and database.get_postings(10) == []       # not silently diverted to the simulated ledger


def test_non_sandbox_host_is_refused_unless_explicitly_allowed(sap, monkeypatch):
    monkeypatch.setattr(settings, "sap_base_url", "https://my300001-api.s4hana.cloud.sap")
    res = sap_poster.post_invoice(erp_payload())
    assert not res.ok and res.status == "disabled" and "only the SAP sandbox" in res.error and sap.requests == []
    monkeypatch.setattr(settings, "sap_allow_non_sandbox", True)
    monkeypatch.setattr(settings, "sap_tax_code_map", json.dumps({"STANDARD": "I1"}))
    assert sap_poster.post_invoice(erp_payload()).ok


def test_real_tenant_never_gets_a_guessed_tax_code(sap, monkeypatch):
    monkeypatch.setattr(settings, "sap_base_url", "https://my300001-api.s4hana.cloud.sap")
    monkeypatch.setattr(settings, "sap_allow_non_sandbox", True)
    res = sap_poster.post_invoice(erp_payload())
    assert not res.ok and "No SAP tax code mapped" in res.error and sap.posts == []


@pytest.mark.parametrize("url", ["http://sandbox.api.sap.com/s4hanacloud", "not a url", ""])
def test_https_is_required(sap, monkeypatch, url):
    monkeypatch.setattr(settings, "sap_base_url", url)
    assert not sap_poster.post_invoice(erp_payload()).ok and sap.requests == []


def test_missing_credentials_are_refused(sap, monkeypatch):
    monkeypatch.setattr(settings, "sap_api_key", "")
    res = sap_poster.post_invoice(erp_payload())
    assert not res.ok and "credentials are not configured" in res.error and sap.requests == []


def test_bad_mapping_json_is_a_config_error_not_a_crash(sap, monkeypatch):
    monkeypatch.setattr(settings, "sap_supplier_map", "{not json")
    res = sap_poster.post_invoice(erp_payload())
    assert not res.ok and "SAP_SUPPLIER_MAP" in res.error and sap.posts == []


# --------------------------------------------------------------------------- secrets never leak

def test_key_never_appears_in_results_logs_or_stored_records(sap, capsys, caplog):
    erp = erp_payload()
    sap.script = ["reject_echo_key", "boom_with_key", "ok"]
    results = [sap_poster.post_invoice(erp)]                            # SAP echoes the key and a CSRF token in its error
    results.append(sap_poster.post_invoice(erp))                        # an unexpected exception mentions the key
    results.append(sap_poster.post_invoice(erp))
    out = capsys.readouterr()
    blob = "\n".join([out.out, out.err, caplog.text, _all_stored_text(), json.dumps([r.audit_view() for r in results])])
    assert FAKE_KEY not in blob and CSRF not in blob
    assert "***" in results[0].error and not results[0].ok
    assert results[1].status == "unknown" and results[2].ok


def test_sanitize_removes_secrets_headers_and_bounds_length(sap):
    text = f"APIKey: {FAKE_KEY}\nX-CSRF-Token: {CSRF} Authorization: Basic abc " + "y" * 5000
    clean = sap_poster.sanitize(text)
    assert FAKE_KEY not in clean and CSRF not in clean and "Basic abc" not in clean and "\n" not in clean
    assert len(clean) <= 600


def test_status_reports_state_without_secrets(sap, monkeypatch):
    monkeypatch.setattr(settings, "sap_base_url", "https://sandbox.api.sap.com/s4hanacloud")
    monkeypatch.setattr(settings, "sap_supplier_map", json.dumps({"A": "1", "B": "2"}))
    s = sap_poster.status()
    assert s["configured"] and s["posting_enabled"] and s["ready_to_post"] and s["sandbox"]
    assert s["host"] == "sandbox.api.sap.com" and s["auth"] == "api_key" and s["mappings"]["supplier"] == 2
    assert FAKE_KEY not in json.dumps(s) and "/sap/opu" not in json.dumps(s)


# --------------------------------------------------------------------------- through the API

BODY = {"raw_vendor_name": "Meridian Facilities Ltd", "description": "Office cleaning services monthly",
        "amount": 1200, "entity_id": "UK001", "cost_centre": "CC300"}


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def _auto_post_prediction():
    from ap_automation.core.pipeline import predict
    from tests.conftest import make_invoice

    pred = predict(make_invoice(), batch_mode=True)
    assert pred.final_routing.value == "auto_post"
    return pred


def _predict(client) -> dict:
    return client.post("/predict", json=BODY).json()


def test_status_endpoint(client, sap):
    r = client.get("/sap/status")
    assert r.status_code == 200 and r.json()["ready_to_post"] is True
    assert FAKE_KEY not in r.text


def test_status_endpoint_when_off_by_default(client, monkeypatch):
    monkeypatch.setattr(settings, "erp_backend", "simulated")
    monkeypatch.setattr(settings, "sap_api_key", "")
    body = client.get("/sap/status").json()
    assert body["backend"] == "simulated" and body["posting_enabled"] is False and body["ready_to_post"] is False
    assert body["configured"] is False


def test_auto_post_still_403_for_human_review_even_with_sap_on(client, sap, seeded):
    p = client.post("/predict", json={**BODY, "raw_vendor_name": "Totally Unknown Vendor GmbH"}).json()
    assert p["final_routing"] == "human_review"
    r = client.post(f"/auto-post/{p['prediction_id']}")
    assert r.status_code == 403 and sap.requests == [] and database.get_postings(10) == []


def test_auto_post_to_sap_succeeds_once_and_shows_in_ledger_and_audit(client, sap, seeded):
    p = _predict(client)
    assert p["final_routing"] == "auto_post"
    r = client.post(f"/auto-post/{p['prediction_id']}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["erp_posted"] is True and body["erp_backend"] == "sap" and body["learned"] is False
    assert body["erp_ref"]["document"] == "5105600001" and body["erp_ref"]["fiscal_year"] == "2026"

    ledger = client.get("/ledger").json()["postings"]
    assert len(ledger) == 1 and ledger[0]["sap_document"] == "5105600001" and ledger[0]["erp_backend"] == "sap"
    assert ledger[0]["source"] == "auto_post"
    rec = client.get("/audit").json()["records"][0]
    assert rec["source"] == "auto_post" and rec["provenance"]["erp"]["document"] == "5105600001"
    assert FAKE_KEY not in _all_stored_text()

    assert client.post(f"/auto-post/{p['prediction_id']}").status_code == 409     # no second post
    assert len(sap.invoices) == 1 and len(sap.posts) == 1


def test_sap_rejection_keeps_the_invoice_in_review_with_the_error(client, sap, seeded):
    sap.script = ["reject"]
    p = _predict(client)
    r = client.post(f"/auto-post/{p['prediction_id']}")
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail["status"] == "sap_post_failed" and "Supplier 11300001 is blocked" in detail["sap"]["error"]

    stored = main._store.get(p["prediction_id"])
    assert stored["posted"] is False                                    # claim released
    assert stored["pred"].final_routing.value == "human_review"         # stricter, never looser
    assert "SAP post not completed" in stored["pred"].final_routing_reason
    assert client.get("/ledger").json()["postings"] == []               # nothing recorded as posted
    rec = client.get("/audit").json()["records"][0]
    assert rec["source"] == "sap_post_failed" and rec["routing_decision"] == "human_review"
    assert "Supplier 11300001 is blocked" in rec["routing_reason"]
    failed = client.get("/sap/postings", params={"status": "failed"}).json()["postings"]
    assert len(failed) == 1 and "Supplier 11300001 is blocked" in failed[0]["error"]
    assert client.get("/performance").status_code == 200

    # it can no longer auto-post (it is a review item now) but a human can confirm it once SAP is fixed
    assert client.post(f"/auto-post/{p['prediction_id']}").status_code == 403
    r2 = client.post(f"/confirm/{p['prediction_id']}", json={"confirmed_gl": p["predicted_gl"]})
    assert r2.status_code == 200 and r2.json()["erp_posted"] is True and r2.json()["learned"] is True
    assert len(sap.invoices) == 1
    ledger = client.get("/ledger").json()["postings"]
    assert len(ledger) == 1 and ledger[0]["source"] == "human_confirm" and ledger[0]["sap_document"]


def test_human_confirm_failure_is_not_learned_and_can_be_retried(client, sap, seeded):
    sap.script = ["reject"]
    p = _predict(client)
    r = client.post(f"/confirm/{p['prediction_id']}", json={"confirmed_gl": "6300"})
    assert r.status_code == 502
    assert client.get("/corpus/vendors").json()["corpus_size"] == main._collection_size()
    stored = main._store.get(p["prediction_id"])
    assert stored["posted"] is False
    ok = client.post(f"/confirm/{p['prediction_id']}", json={"confirmed_gl": "6300"})
    assert ok.status_code == 200 and len(sap.invoices) == 1


def test_batch_marks_a_sap_failure_as_review_not_auto_confirmed(client, sap, seeded):
    sap.script = ["reject"]
    csv_text = "vendor_name,description,amount,entity_id,cost_centre\n"
    csv_text += "Meridian Facilities Ltd,Office cleaning services monthly,1200,UK001,\n"
    r = client.post("/batch", files={"file": ("b.csv", io.BytesIO(csv_text.encode()), "text/csv")})
    job_id = r.json()["job_id"]
    import time
    for _ in range(100):
        status = client.get(f"/batch/{job_id}").json()
        if status["status"] != "running":
            break
        time.sleep(0.1)
    row = status["results"][0]
    assert status["auto_confirmed"] == 0 and status["human_review"] == 1
    assert row["routing"] == "human_review" and row["erp_posted"] is False and "Supplier 11300001" in row["warning"]
    assert sap.invoices == []


def test_simulated_default_still_works_end_to_end(client, seeded):
    p = _predict(client)
    r = client.post(f"/auto-post/{p['prediction_id']}")
    assert r.status_code == 200 and r.json()["erp_backend"] == "simulated" and r.json()["erp_posted"] is True
    assert client.get("/ledger").json()["postings"][0]["erp_backend"] == "simulated"
