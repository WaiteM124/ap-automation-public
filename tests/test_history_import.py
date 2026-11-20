"""Per-client chart of accounts and bulk history import: validation, idempotency, and the end-to-end path."""

import io

import pytest
from fastapi.testclient import TestClient

import main
from ap_automation.core import chart_of_accounts, demo, history_import, retrieval
from ap_automation.core.chart_of_accounts import CHART, ChartError, parse_chart_csv
from ap_automation.core.config import settings

CHART_CSV = "code,name,default_cost_centre,vat_treatment\n5000,Professional fees,,NOT_APPLICABLE\n5100,Supplies,,\n"

HISTORY_CSV = (
    "vendor_name,description,amount,date,gl\n"
    + "".join(f"Northwind Traders,Monthly retainer fee invoice {i},{1000 + i},2026-0{1 + i % 5}-1{i % 9},5000\n"
             for i in range(6))
)


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def _upload(client, path, text, **params):
    return client.post(path, params=params, files={"file": ("f.csv", io.BytesIO(text.encode()), "text/csv")})


# --------------------------------------------------------------------------- chart

def test_parse_chart_csv_reads_codes_names_and_defaults():
    accounts = parse_chart_csv(CHART_CSV)
    assert set(accounts) == {"5000", "5100"}
    assert accounts["5000"].vat_treatment == "NOT_APPLICABLE"
    assert accounts["5100"].vat_treatment == "STANDARD"       # blank -> default
    assert accounts["5100"].default_cost_centre == ""


def test_parse_chart_csv_accepts_alternative_headers():
    accounts = parse_chart_csv("Account_Code,Account_Name\n003090,Vehicle parts\n")
    assert accounts["003090"].name == "Vehicle parts"


@pytest.mark.parametrize("text,fragment", [
    ("name\nX\n", "no account-code column"),
    ("code\n5000\n", "no name column"),
    ("code,name\n5000,A\n5000,B\n", "more than once"),
    ("code,name,vat_treatment\n5000,A,WEIRD\n", "unknown vat_treatment"),
    ("code,name\n,A\n", "missing account code"),
    ("code,name\n", "no accounts"),
])
def test_parse_chart_csv_rejects_bad_files_with_a_reason(text, fragment):
    with pytest.raises(ChartError) as e:
        parse_chart_csv(text)
    assert fragment in str(e.value)


def test_set_chart_mutates_in_place_so_existing_imports_see_it():
    ref = CHART                                   # the object other modules imported
    chart_of_accounts.set_chart(parse_chart_csv(CHART_CSV))
    assert ref is CHART and set(ref) == {"5000", "5100"} and chart_of_accounts.is_custom()
    chart_of_accounts.reset_chart()
    assert "6300" in ref and not chart_of_accounts.is_custom()


def test_chart_persists_and_reloads(tmp_path):
    path = str(tmp_path / "chart.json")
    chart_of_accounts.set_chart(parse_chart_csv(CHART_CSV))
    assert chart_of_accounts.save_chart(path)
    chart_of_accounts.reset_chart()
    assert chart_of_accounts.load_saved_chart(path)
    assert set(CHART) == {"5000", "5100"}


def test_chart_endpoints_replace_validate_and_reset(client):
    assert client.get("/chart").json()["custom"] is False
    bad = _upload(client, "/chart", "code,name\n5000,A\n5000,B\n")
    assert bad.status_code == 422 and "more than once" in str(bad.json()["detail"]["problems"])
    ok = _upload(client, "/chart", CHART_CSV)
    assert ok.status_code == 200 and ok.json()["accounts"] == 2
    assert client.get("/chart").json()["custom"] is True
    assert client.delete("/chart").json()["accounts"] == 10
    assert client.get("/chart").json()["custom"] is False


def test_confirm_validates_against_the_active_chart(client):
    _upload(client, "/chart", CHART_CSV)
    history_import.import_history(HISTORY_CSV)
    body = {"raw_vendor_name": "Northwind Traders", "description": "Monthly retainer fee invoice",
            "amount": 1000, "entity_id": "UK001", "cost_centre": "X"}
    pid = client.post("/predict", json=body).json()["prediction_id"]
    assert client.post(f"/confirm/{pid}", json={"confirmed_gl": "6300"}).status_code == 422   # built-in code
    assert client.post(f"/confirm/{pid}", json={"confirmed_gl": "5000"}).status_code == 200   # client code


def test_cost_centre_agent_does_not_flag_when_the_chart_sets_no_default():
    from ap_automation.agents.cost_centre_agent import CostCentreAgent

    chart_of_accounts.set_chart(parse_chart_csv(CHART_CSV))
    r = CostCentreAgent().run({"predicted_gl": "5000", "submitted_cost_centre": "540"})
    assert r.data["cost_centre_mismatch"] is False


def test_demo_reset_restores_the_built_in_chart():
    chart_of_accounts.set_chart(parse_chart_csv(CHART_CSV))
    demo.reset(main._store)
    assert "6300" in CHART and not chart_of_accounts.is_custom()


# --------------------------------------------------------------------------- import

def test_import_loads_validated_lines_as_human_confirmed_evidence():
    chart_of_accounts.set_chart(parse_chart_csv(CHART_CSV))
    before = retrieval._collection_size()
    out = history_import.import_history(HISTORY_CSV)
    assert out["imported"] == 6 and out["rejected_count"] == 0 and out["vendors"] == 1
    assert retrieval._collection_size() == before + 6
    assert out["top_gl"][0] == {"gl": "5000", "name": "Professional fees", "lines": 6}
    vendor_id = retrieval.corpus_vendors()[0]["vendor_id"]
    pts = retrieval.corpus_points(vendor_id)
    assert {p["confirmed_by"] for p in pts} == {"history_import"} and {p["source"] for p in pts} == {"human_confirm"}


def test_import_rejects_unknown_gl_missing_fields_bad_amount_and_future_dates_with_reasons():
    chart_of_accounts.set_chart(parse_chart_csv(CHART_CSV))
    text = (
        "vendor_name,description,amount,date,gl\n"
        "A Ltd,fine line,10,2026-01-01,5000\n"
        "A Ltd,wrong code,10,2026-01-02,9999\n"
        ",no vendor,10,2026-01-03,5000\n"
        "A Ltd,,10,2026-01-04,5000\n"
        "A Ltd,bad amount,abc,2026-01-05,5000\n"
        "A Ltd,future,10,2999-01-01,5000\n"
        "A Ltd,bad date,10,yesterday,5000\n"
    )
    out = history_import.import_history(text)
    assert out["imported"] == 1 and out["rejected_count"] == 6
    reasons = " | ".join(r["reason"] for r in out["rejected"])
    for fragment in ("GL 9999 is not in the active chart", "missing vendor", "missing description",
                     "amount", "in the future", "not ISO"):
        assert fragment in reasons
    assert [r["line"] for r in out["rejected"]] == [3, 4, 5, 6, 7, 8]


def test_import_is_idempotent_and_skips_duplicates_inside_the_file():
    chart_of_accounts.set_chart(parse_chart_csv(CHART_CSV))
    text = HISTORY_CSV + HISTORY_CSV.split("\n", 1)[1].split("\n")[0] + "\n"      # repeat the first data line
    first = history_import.import_history(text)
    assert first["imported"] == 6 and first["duplicates_in_file"] == 1
    size = retrieval._collection_size()
    again = history_import.import_history(text)
    assert again["imported"] == 0 and again["already_present"] == 6
    assert retrieval._collection_size() == size


def test_dry_run_validates_without_writing():
    chart_of_accounts.set_chart(parse_chart_csv(CHART_CSV))
    before = retrieval._collection_size()
    out = history_import.import_history(HISTORY_CSV, dry_run=True)
    assert out["dry_run"] is True and out["would_import"] == 6 and out["imported"] == 0
    assert retrieval._collection_size() == before


def test_import_refuses_files_that_cannot_work():
    with pytest.raises(history_import.HistoryImportError, match="no vendor column"):
        history_import.import_history("description,gl\nx,6300\n")
    with pytest.raises(history_import.HistoryImportError, match="no data rows"):
        history_import.import_history("vendor_name,description,gl\n")


def test_import_enforces_the_row_cap(monkeypatch):
    monkeypatch.setattr(history_import, "MAX_ROWS", 3)
    text = "vendor_name,description,gl\n" + "".join(f"V,d{i},6300\n" for i in range(4))
    with pytest.raises(history_import.HistoryImportError, match="split the file"):
        history_import.import_history(text)


def test_import_endpoint_reports_and_validates(client):
    _upload(client, "/chart", CHART_CSV)
    dry = _upload(client, "/corpus/import", HISTORY_CSV, dry_run="true")
    assert dry.status_code == 200 and dry.json()["would_import"] == 6
    done = _upload(client, "/corpus/import", HISTORY_CSV)
    assert done.json()["imported"] == 6
    bad = _upload(client, "/corpus/import", "x,y\n1,2\n")
    assert bad.status_code == 422


def test_import_then_predict_end_to_end_with_a_client_chart(client):
    """The workflow the UI exposes: upload the chart, import history, then a new invoice gets the learned code."""
    _upload(client, "/chart", CHART_CSV)
    assert _upload(client, "/corpus/import", HISTORY_CSV).json()["imported"] == 6
    r = client.post("/predict", json={"raw_vendor_name": "Northwind Traders",
                                      "description": "Monthly retainer fee invoice", "amount": 1005,
                                      "entity_id": "UK001", "cost_centre": "X"})
    data = r.json()
    assert data["predicted_gl"] == "5000"
    assert data["routing_decision"] == "auto_post"          # the rule gate is satisfied by six agreeing lines
    assert len(data["evidence"]) >= 3


def test_import_endpoint_requires_the_api_key_when_one_is_set(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret")
    assert _upload(client, "/corpus/import", HISTORY_CSV).status_code == 401
    assert client.get("/chart").status_code == 401
    assert client.delete("/chart").status_code == 401
    assert client.post("/chart", files={"file": ("f.csv", b"code,name\n1,A\n")},
                       headers={"X-API-Key": "secret"}).status_code == 200
