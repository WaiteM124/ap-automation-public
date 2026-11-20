"""
Post-ready output: the ERP-neutral record of every cleared invoice, its CSV / JSON downloads, filters, the
"ready to post" vs "posted" labelling and the per-target payload preview.

Nothing here contacts SAP or Xero. A "posted" status is only ever produced by writing an external document id into
the SAP / Xero posting tables, exactly as the real adapters do after the external system answered.
"""

import csv
import io

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from ap_automation.core import database, post_ready, xero_auth
from ap_automation.core.config import settings

BODY = {"raw_vendor_name": "Meridian Facilities Ltd", "description": "Office cleaning services monthly",
        "amount": 1200, "entity_id": "UK001", "cost_centre": "CC300"}


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def _predict(client, **over) -> dict:
    return client.post("/predict", json={**BODY, **over}).json()


def _csv_rows(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def _records(client, **params) -> list[dict]:
    r = client.get("/post-ready", params=params)
    assert r.status_code == 200, r.text
    return r.json()["records"]


def _mark_xero_posted(invoice_id: str, doc: str = "11111111-2222-3333-4444-555555555555") -> None:
    ref = post_ready.reference_for(invoice_id)
    database.xero_claim(ref, invoice_id, "p")
    database.xero_finish(ref, "posted", xero_invoice_id=doc, xero_url=f"https://go.xero.com/x/{doc}",
                         tenant_id="t")


# --------------------------------------------------------------------------- record content and labelling

def test_the_record_schema_is_the_documented_one(client, seeded):
    _predict(client)
    rec = _records(client)[0]
    assert tuple(rec) == post_ready.FIELDS
    assert not any(k.startswith("_") for k in rec)
    for k in ("reference_key", "vendor_name", "description", "date", "amount", "currency", "gl_account", "gl_name",
              "cost_centre", "tax_treatment", "entity_id", "routing_decision", "confidence", "approved_by"):
        assert rec[k] not in ("", None), k
    assert rec["reference_key"] == post_ready.reference_for(rec["invoice_id"]) and len(rec["reference_key"]) == 16
    assert rec["vendor_name"] == "Meridian Facilities Ltd" and rec["amount"] == 1200.0 and rec["currency"] == "GBP"
    assert rec["gl_account"] == "6300" and rec["gl_name"] and rec["cost_centre"] == "CC300"


def test_a_cleared_but_unposted_invoice_is_ready_to_post_and_never_labelled_posted(client, seeded):
    p = _predict(client)
    assert p["final_routing"] == "auto_post"
    rec = _records(client)[0]
    assert rec["status"] == "ready_to_post" and rec["posted_to"] == "" and rec["erp_document"] == ""
    assert rec["routing_decision"] == "auto_post" and rec["approval_source"] == "auto_post"
    assert rec["approved_by"] == "governed gate" and rec["approved_by_type"] == "system"
    assert rec["ledger_recorded"] is False and rec["prediction_id"] == p["prediction_id"]


def test_an_invoice_needing_review_is_not_post_ready(client, seeded):
    p = _predict(client, raw_vendor_name="Totally New Vendor GmbH", description="Quarterly consulting retainer")
    assert p["final_routing"] == "human_review"
    assert _records(client) == []
    c = client.post(f"/confirm/{p['prediction_id']}", json={"confirmed_gl": "6300"})
    assert c.status_code == 200, c.text
    rec = _records(client)[0]                       # now a human cleared it
    assert rec["approval_source"] == "human_confirm" and rec["approved_by_type"] == "human"
    assert rec["status"] == "ready_to_post"


def test_a_simulated_ledger_row_is_ready_to_post_not_posted(client, seeded):
    p = _predict(client)
    r = client.post(f"/auto-post/{p['prediction_id']}")
    assert r.status_code == 200 and r.json()["erp_backend"] == "simulated"
    recs = _records(client)
    assert len(recs) == 1                            # the pending prediction and the ledger row are one record
    assert recs[0]["status"] == "ready_to_post" and recs[0]["posted_to"] == "" and recs[0]["erp_document"] == ""
    assert recs[0]["ledger_recorded"] is True and recs[0]["routing_decision"] == "auto_post"


def test_posted_only_when_an_external_system_stored_a_document_id(client, seeded):
    p = _predict(client)
    client.post(f"/auto-post/{p['prediction_id']}")
    inv = _records(client)[0]["invoice_id"]
    ref = post_ready.reference_for(inv)

    database.xero_claim(ref, inv, "p")
    database.xero_finish(ref, "failed", error="rejected")           # attempted, refused: still not posted
    assert _records(client)[0]["status"] == "ready_to_post"
    database.xero_finish(ref, "unknown", error="timeout")           # outcome unknown: not "posted" either
    assert _records(client)[0]["status"] == "ready_to_post"

    database.xero_finish(ref, "posted", xero_invoice_id="")         # a 'posted' row WITHOUT a document id is not posted
    assert _records(client)[0]["status"] == "ready_to_post"

    database.xero_finish(ref, "posted", xero_invoice_id="abc-123", xero_url="https://go.xero.com/x")
    rec = _records(client)[0]
    assert rec["status"] == "posted" and rec["posted_to"] == "xero" and rec["erp_document"] == "abc-123"
    assert rec["erp_url"] == "https://go.xero.com/x"


def test_sap_document_marks_posted_to_sap(client, seeded):
    p = _predict(client)
    client.post(f"/auto-post/{p['prediction_id']}")
    inv = _records(client)[0]["invoice_id"]
    ref = post_ready.reference_for(inv)
    database.sap_claim(ref, inv, "p")
    database.sap_finish(ref, "failed", error="refused")
    assert _records(client)[0]["status"] == "ready_to_post"
    database.sap_finish(ref, "posted", sap_document="5105600123", fiscal_year="2026", company_code="1110")
    rec = _records(client)[0]
    assert rec["status"] == "posted" and rec["posted_to"] == "sap" and rec["erp_document"] == "5105600123"


# --------------------------------------------------------------------------- filters

def test_status_filter_and_summary(client, seeded):
    a = _predict(client)
    b = _predict(client)
    client.post(f"/auto-post/{a['prediction_id']}")
    inv_a = next(r["invoice_id"] for r in _records(client) if r["prediction_id"] == a["prediction_id"])
    _mark_xero_posted(inv_a)

    posted = _records(client, status="posted")
    ready = _records(client, status="ready_to_post")
    assert [r["prediction_id"] for r in posted] == [a["prediction_id"]]
    assert [r["prediction_id"] for r in ready] == [b["prediction_id"]]
    body = client.get("/post-ready").json()
    assert body["summary"]["count"] == 2 and body["summary"]["posted"] == 1 and body["summary"]["ready_to_post"] == 1
    assert body["summary"]["total_by_currency"] == {"GBP": 2400.0}
    assert body["filters"]["status"] == "all" and len(_records(client, status="all")) == 2


def test_date_filters_are_inclusive(client, seeded):
    _predict(client)
    day = _records(client)[0]["date"]
    assert len(_records(client, date_from=day, date_to=day)) == 1
    assert _records(client, date_from="2999-01-01") == []
    assert _records(client, date_to="2000-01-01") == []
    assert len(_records(client, date_from="2000-01-01", date_to="2999-12-31")) == 1


def test_limit_applies_newest_first(client, seeded):
    ids = [_predict(client)["prediction_id"] for _ in range(3)]
    assert len(_records(client)) == 3
    two = client.get("/post-ready", params={"limit": 2}).json()
    assert len(two["records"]) == 2 and two["summary"]["count"] == 2 and two["filters"]["limit"] == 2
    approved = [r["approved_at"] for r in _records(client)]
    assert approved == sorted(approved, reverse=True) and {r["prediction_id"] for r in _records(client)} == set(ids)
    assert len(_records(client, limit=0)) == 1                       # clamped to at least one


@pytest.mark.parametrize("params", [{"status": "bogus"}, {"date_from": "27/10/2026"}, {"date_to": "not-a-date"},
                                    {"date_from": "2026-10-28", "date_to": "2026-10-01"}, {"format": "xml"}])
def test_bad_filters_are_422(client, params):
    assert client.get("/post-ready", params=params).status_code == 422


# --------------------------------------------------------------------------- downloads

def test_json_and_csv_downloads_match(client, seeded):
    _predict(client)
    _predict(client, description="Window cleaning, lobby")
    as_json = _records(client)
    r = client.get("/post-ready", params={"format": "csv"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert 'attachment; filename="post-ready.csv"' in r.headers["content-disposition"]
    assert r.text.splitlines()[0] == ",".join(post_ready.FIELDS)
    rows = _csv_rows(r.text)
    assert len(rows) == len(as_json) == 2
    for csv_row, rec in zip(rows, as_json):
        assert tuple(csv_row) == post_ready.FIELDS
        assert csv_row["reference_key"] == rec["reference_key"] and float(csv_row["amount"]) == rec["amount"]
        assert csv_row["gl_account"] == rec["gl_account"] and csv_row["status"] == "ready_to_post"
    assert _csv_rows(client.get("/post-ready", params={"format": "csv", "status": "posted"}).text) == []


def test_csv_neutralises_spreadsheet_formulas(client, seeded):
    p = _predict(client, raw_vendor_name="=HYPERLINK(\"http://evil\")", description="@SUM(1+1) cleaning services")
    client.post(f"/confirm/{p['prediction_id']}", json={"confirmed_gl": "6300"})
    text = client.get("/post-ready", params={"format": "csv"}).text
    rows = _csv_rows(text)
    assert rows and all(not c.startswith(("=", "@", "+")) for row in rows for c in row.values() if c)
    assert any(row["description"].startswith("'@SUM") for row in rows)
    unit = post_ready.to_csv([{**{k: "" for k in post_ready.FIELDS}, "description": "=1+1", "vendor_name": "-2",
                               "routing_reason": "\tx", "amount": -5.0}])
    row = _csv_rows(unit)[0]
    assert row["description"] == "'=1+1" and row["vendor_name"] == "'-2" and row["routing_reason"] == "'\tx"
    assert row["amount"] == "-5.0"                                    # a number stays a number


def test_empty_state(client, seeded):
    body = client.get("/post-ready").json()
    assert body["records"] == [] and body["summary"]["count"] == 0 and body["summary"]["total_by_currency"] == {}
    assert client.get("/post-ready", params={"format": "csv"}).text == ",".join(post_ready.FIELDS) + "\r\n"


# --------------------------------------------------------------------------- payload preview

def test_neutral_payload_is_the_record(client, seeded):
    rec = _predict(client) and _records(client)[0]
    r = client.get(f"/post-ready/{rec['reference_key']}/payload")
    assert r.status_code == 200
    assert r.json() == {"target": "neutral", "reference_key": rec["reference_key"], "payload": rec, "error": None}


def test_xero_preview_shows_a_mapping_problem_instead_of_guessing(client, seeded):
    rec = _predict(client) and _records(client)[0]
    r = client.get(f"/post-ready/{rec['reference_key']}/payload", params={"target": "xero"}).json()
    assert r["payload"] is None and "XERO_TAX_TYPE_MAP" in r["error"]       # not the demo company: no guessed tax


def test_xero_preview_for_the_demo_company_and_with_a_map(client, seeded, monkeypatch):
    rec = _predict(client) and _records(client)[0]
    xero_auth.save_tokens({"access_token": "a", "refresh_token": "r", "expires_at": 0, "tenant_id": "t",
                           "tenant_name": "Demo Company (UK)", "short_code": ""})
    r = client.get(f"/post-ready/{rec['reference_key']}/payload", params={"target": "xero"}).json()
    bill = r["payload"]["Invoices"][0]
    assert r["error"] is None and bill["Type"] == "ACCPAY" and bill["InvoiceNumber"] == rec["reference_key"]
    assert bill["Contact"]["Name"] == "Meridian Facilities Ltd" and bill["Status"] == "DRAFT"

    xero_auth.save_tokens({"access_token": "a", "refresh_token": "r", "expires_at": 0, "tenant_id": "t",
                           "tenant_name": "Acme Trading Ltd", "short_code": ""})
    monkeypatch.setattr(settings, "xero_tax_type_map", '{"STANDARD": "INPUT2"}')
    r = client.get(f"/post-ready/{rec['reference_key']}/payload", params={"target": "xero"}).json()
    assert r["error"] is None and r["payload"]["Invoices"][0]["LineItems"][0]["TaxType"] == "INPUT2"


def test_sap_preview_needs_a_supplier_then_maps(client, seeded, monkeypatch):
    rec = _predict(client) and _records(client)[0]
    url = f"/post-ready/{rec['reference_key']}/payload"
    r = client.get(url, params={"target": "sap"}).json()
    assert r["payload"] is None and "SAP_SUPPLIER_MAP" in r["error"]
    monkeypatch.setattr(settings, "sap_default_supplier", "17300001")
    r = client.get(url, params={"target": "sap"}).json()
    assert r["error"] is None and r["payload"]["SupplierInvoiceIDByInvcgParty"] == rec["reference_key"]


def test_preview_never_calls_an_external_system(client, seeded, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("preview must not touch the network")

    monkeypatch.setattr(httpx.Client, "send", boom)
    rec = _predict(client) and _records(client)[0]
    for target in ("neutral", "xero", "sap"):
        assert client.get(f"/post-ready/{rec['reference_key']}/payload", params={"target": target}).status_code == 200


def test_unknown_reference_and_bad_target(client, seeded):
    rec = _predict(client) and _records(client)[0]
    assert client.get("/post-ready/APDOESNOTEXIST000/payload").status_code == 404
    assert client.get(f"/post-ready/{rec['reference_key']}/payload", params={"target": "oracle"}).status_code == 422


# --------------------------------------------------------------------------- safety

def test_auto_post_still_403_for_an_invoice_that_needs_review(client, seeded):
    p = _predict(client, raw_vendor_name="Totally New Vendor GmbH", description="Quarterly consulting retainer")
    assert p["final_routing"] == "human_review"
    assert client.post(f"/auto-post/{p['prediction_id']}").status_code == 403
    assert _records(client) == []                                     # and listing records did not change that
    assert main._store.get(p["prediction_id"])["posted"] is False


def test_the_endpoints_are_protected_by_the_api_key(client, monkeypatch, seeded):
    rec = _predict(client) and _records(client)[0]
    monkeypatch.setattr(settings, "api_key", "k-test-123")
    assert client.get("/post-ready").status_code == 401
    assert client.get("/post-ready", params={"format": "csv"}).status_code == 401
    assert client.get(f"/post-ready/{rec['reference_key']}/payload").status_code == 401
    ok = client.get("/post-ready", headers={"X-API-Key": "k-test-123"})
    assert ok.status_code == 200 and len(ok.json()["records"]) == 1


def test_records_carry_no_secrets_or_tokens(client, seeded, monkeypatch):
    monkeypatch.setattr(settings, "xero_client_secret", "SECRET-VALUE-XYZ")
    monkeypatch.setattr(settings, "sap_api_key", "SAP-KEY-VALUE-XYZ")
    _predict(client)
    text = client.get("/post-ready").text + client.get("/post-ready", params={"format": "csv"}).text
    rec = _records(client)[0]
    for t in ("xero", "sap", "neutral"):
        text += client.get(f"/post-ready/{rec['reference_key']}/payload", params={"target": t}).text
    assert "SECRET-VALUE-XYZ" not in text and "SAP-KEY-VALUE-XYZ" not in text
