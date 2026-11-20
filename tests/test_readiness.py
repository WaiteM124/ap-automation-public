"""Data-readiness report: scoring, isolation, endpoint caps, and no ERP contact."""

import io
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import main
from ap_automation.core import (
    chart_of_accounts,
    database,
    history_import,
    mdm,
    posting,
    readiness,
    retrieval,
    sap_poster,
    xero_poster,
)
from ap_automation.core.config import settings
from ap_automation.core.history_import import parse_rows

NOW = datetime(2026, 10, 27, tzinfo=timezone.utc)


def _csv(header: str, lines: list[str]) -> str:
    return header + "\n" + "\n".join(lines) + "\n"


def _clean_lines(n: int = 24, vendors: int = 8) -> list[str]:
    names = [f"Northwind Traders {i}" for i in range(vendors)]
    gls = ["6100", "6200", "6300", "6400", "6500", "6600", "6700", "6800"]
    rows = []
    for i in range(n):
        v = names[i % vendors]
        gl = gls[i % vendors]
        month = 1 + (i % 9)
        rows.append(
            f"{v},Monthly office supplies order batch {i},{100 + i},2026-0{month}-15,{gl},CC100"
        )
    return rows


CLEAN_HEADER = "vendor_name,description,amount,date,gl,cost_centre"


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def _upload(client, path, text, **params):
    return client.post(path, params=params, files={"file": ("f.csv", io.BytesIO(text.encode()), "text/csv")})


def _by_id(report):
    return {c["id"]: c for c in report["categories"]}


# --------------------------------------------------------------------------- scoring


def test_clean_history_scores_high():
    text = _csv(CLEAN_HEADER, _clean_lines(240, 8))
    report = readiness.assess(text, now=NOW)
    cats = _by_id(report)
    assert report["score"] >= 80
    assert report["thresholds_provisional"] is True
    assert cats["vendor_consistency"]["status"] == "ok"
    assert cats["gl_coding_consistency"]["status"] == "ok"
    assert cats["description_quality"]["status"] == "ok"
    assert cats["volume_sufficiency"]["status"] == "ok"
    assert cats["chart_coverage"]["status"] == "ok"
    assert cats["cost_centre_consistency"]["assessed"] is True
    assert cats["vat_completeness"]["status"] == "not_provided"
    assert cats["recency"]["assessed"] is True
    assert cats["replay"]["assessed"] is True
    assert report["replay"]["status"] != "insufficient"


def test_typo_ridden_vendors_lower_vendor_consistency():
    clean = readiness.assess(_csv(CLEAN_HEADER, _clean_lines(40, 4)), now=NOW)
    messy_lines = []
    for i in range(40):
        variants = ["Acme Ltd", "ACME LIMITED", "acme, ltd.", "Acme Ltd."]
        messy_lines.append(
            f"{variants[i % 4]},Monthly office supplies order batch {i},10,2026-01-15,6100,CC100"
        )
    messy = readiness.assess(_csv(CLEAN_HEADER, messy_lines), now=NOW)
    assert _by_id(messy)["vendor_consistency"]["score"] < _by_id(clean)["vendor_consistency"]["score"]
    assert _by_id(messy)["vendor_consistency"]["status"] != "ok"
    assert "exact-only" in _by_id(messy)["vendor_consistency"]["finding"]


def test_generic_descriptions_lower_description_quality():
    good = readiness.assess(_csv(CLEAN_HEADER, _clean_lines(40, 4)), now=NOW)
    bad_lines = [
        f"Vendor {i % 4},Invoice,{10 + i},2026-01-15,6100,CC100" for i in range(40)
    ]
    bad = readiness.assess(_csv(CLEAN_HEADER, bad_lines), now=NOW)
    assert _by_id(bad)["description_quality"]["score"] < _by_id(good)["description_quality"]["score"]
    assert _by_id(bad)["description_quality"]["status"] == "fail"
    phrases = {p["phrase"] for p in _by_id(bad)["description_quality"]["metrics"]["phrases"]}
    assert "invoice" in phrases


def test_inconsistent_gl_lowers_coding_consistency():
    good = readiness.assess(_csv(CLEAN_HEADER, _clean_lines(40, 4)), now=NOW)
    lines = []
    for i in range(40):
        gl = "6100" if i % 2 == 0 else "6200"
        lines.append(f"Same Vendor Ltd,Identical widget purchase line,{10 + i},2026-01-15,{gl},CC100")
    bad = readiness.assess(_csv(CLEAN_HEADER, lines), now=NOW)
    assert _by_id(bad)["gl_coding_consistency"]["score"] < _by_id(good)["gl_coding_consistency"]["score"]
    assert _by_id(bad)["gl_coding_consistency"]["status"] == "fail"


def test_off_chart_codes_are_reported_not_fatal():
    text = _csv(
        CLEAN_HEADER,
        [f"Vendor A,Monthly office supplies order {i},10,2026-01-15,9999,CC100" for i in range(12)]
        + [f"Vendor B,Monthly office supplies order {i},10,2026-01-15,6100,CC100" for i in range(12)],
    )
    report = readiness.assess(text, now=NOW)
    cats = _by_id(report)
    assert report["rows"]["valid"] == 24
    assert report["rows"]["off_chart"] == 12
    assert cats["chart_coverage"]["status"] in {"warn", "fail"}
    assert "9999" in cats["chart_coverage"]["metrics"]["off_chart_codes"]


def test_too_few_lines_lower_volume_and_skip_replay():
    text = _csv(CLEAN_HEADER, _clean_lines(8, 2))
    report = readiness.assess(text, now=NOW)
    cats = _by_id(report)
    assert cats["volume_sufficiency"]["status"] == "fail"
    assert cats["replay"]["assessed"] is False
    assert report["replay"]["status"] == "insufficient"
    assert "insufficient data" in report["replay"]["reason"]
    assert report["mode"] == "not ready"


def test_not_provided_columns_are_excluded_not_failed():
    text = _csv(
        "vendor_name,description,gl\n",
        [f"Vendor {i % 3},Monthly office supplies order {i},6100" for i in range(20)],
    )
    report = readiness.assess(text, now=NOW)
    cats = _by_id(report)
    for key in ("cost_centre_consistency", "vat_completeness", "recency"):
        assert cats[key]["status"] == "not_provided"
        assert cats[key]["assessed"] is False
        assert cats[key]["score"] is None
    assert "cost_centre_consistency" not in report["weights_used"]
    assert "vat_completeness" not in report["weights_used"]
    assert "recency" not in report["weights_used"]
    assert abs(sum(report["weights_used"].values()) - 1.0) < 0.02


def test_vat_column_is_scored_when_present():
    header = "vendor_name,description,gl,tax_code"
    text = _csv(header, [f"Vendor A,Monthly office supplies order {i},6100," + ("S" if i < 2 else "")
                         for i in range(10)])
    report = readiness.assess(text, now=NOW)
    vat = _by_id(report)["vat_completeness"]
    assert vat["assessed"] is True
    assert vat["status"] == "fail"
    assert vat["metrics"]["filled"] == 2


def test_assess_has_no_side_effects():
    history_import.import_history(_csv(CLEAN_HEADER, _clean_lines(12, 3)))
    before_corpus = retrieval._collection_size()
    before_vendors = dict(mdm.VENDOR_MAP)
    before_chart = {code: (a.name, a.vat_treatment, a.default_cost_centre)
                    for code, a in chart_of_accounts.CHART.items()}
    before_custom = chart_of_accounts.is_custom()
    before_events = len(database.get_events())
    before_ledger = len(database.get_ledger())

    readiness.assess(_csv(CLEAN_HEADER, _clean_lines(40, 4)), now=NOW)

    assert retrieval._collection_size() == before_corpus
    assert dict(mdm.VENDOR_MAP) == before_vendors
    assert {code: (a.name, a.vat_treatment, a.default_cost_centre)
            for code, a in chart_of_accounts.CHART.items()} == before_chart
    assert chart_of_accounts.is_custom() is before_custom
    assert len(database.get_events()) == before_events
    assert len(database.get_ledger()) == before_ledger


def test_erp_is_never_called(monkeypatch):
    assert settings.erp_backend == "simulated"
    assert settings.xero_posting_enabled is False
    assert settings.sap_posting_enabled is False

    def boom(*_a, **_k):
        raise AssertionError("ERP post was called during readiness")

    monkeypatch.setattr(sap_poster, "post_invoice", boom)
    monkeypatch.setattr(xero_poster, "post_invoice", boom)
    monkeypatch.setattr(posting, "post_outcome", boom)
    readiness.assess(_csv(CLEAN_HEADER, _clean_lines(220, 8)), now=NOW)


def test_parse_rows_keeps_off_chart_when_allowed():
    text = "vendor_name,description,gl\nA Ltd,Monthly office supplies order,9999\n"
    valid, rejected, _, _ = parse_rows(text, NOW, allow_unknown_gl=True)
    assert len(valid) == 1 and valid[0]["in_chart"] is False and not rejected
    valid2, rejected2, _, _ = parse_rows(text, NOW)
    assert not valid2 and rejected2 and "not in the active chart" in rejected2[0]["reason"]


# --------------------------------------------------------------------------- endpoint


def test_endpoint_requires_the_api_key_when_one_is_set(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret")
    assert _upload(client, "/readiness", _csv(CLEAN_HEADER, _clean_lines(8, 2))).status_code == 401
    ok = client.post(
        "/readiness",
        headers={"X-API-Key": "secret"},
        files={"file": ("f.csv", io.BytesIO(_csv(CLEAN_HEADER, _clean_lines(8, 2)).encode()), "text/csv")},
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["thresholds_provisional"] is True
    assert body["replay"]["status"] == "insufficient"


def test_endpoint_enforces_row_and_size_caps(client, monkeypatch):
    monkeypatch.setattr(history_import, "MAX_ROWS", 3)
    text = "vendor_name,description,gl\n" + "".join(f"V,d{i},6300\n" for i in range(4))
    r = _upload(client, "/readiness", text)
    assert r.status_code == 422
    assert "split the file" in r.json()["detail"]

    huge = b"v" * (main.MAX_UPLOAD_BYTES + 2)
    fat = client.post("/readiness", files={"file": ("f.csv", io.BytesIO(huge), "text/csv")})
    assert fat.status_code == 413


def test_endpoint_rejects_a_file_that_cannot_be_read(client):
    r = _upload(client, "/readiness", "x,y\n1,2\n")
    assert r.status_code == 422


def test_usable_but_inaccurate_replay_is_suggest_only_not_not_ready():
    """Below the auto-post bar is suggest-only; only a poor replay (<80% accuracy) is not-ready."""
    cats = [
        {"id": "volume_sufficiency", "status": "ok", "weight": 14, "assessed": True, "score": 90, "action": None},
        {"id": "chart_coverage", "status": "ok", "weight": 14, "assessed": True, "score": 90,
         "metrics": {"off_chart_rate": 0.0}, "action": None},
        {"id": "replay", "status": "warn", "weight": 17, "assessed": True, "score": 80, "action": None,
         "metrics": {"automation_rate": 0.60, "auto_post_accuracy": 0.87}},
    ]
    rows = {"total": 400, "valid": 400, "invalid": 0, "off_chart": 0}
    mode, _ = readiness._verdict(80, cats, rows)
    assert mode == "suggest-only"
    cats[2]["metrics"] = {"automation_rate": 0.40, "auto_post_accuracy": 0.70}
    cats[2]["status"] = "fail"
    mode, reason = readiness._verdict(70, cats, rows)
    assert mode == "not ready" and "poor" in reason


def test_findings_truncate_examples():
    lines = [f"Vendor {i},Invoice,{i},2026-01-15,6100,CC100" for i in range(20)]
    report = readiness.assess(_csv(CLEAN_HEADER, lines), now=NOW)
    desc = _by_id(report)["description_quality"]
    assert len(desc["metrics"]["examples"]) <= 5
    assert all(len(e) <= 40 for e in desc["metrics"]["examples"])


def _settings_fingerprint():
    return {name: getattr(settings, name) for name in type(settings).model_fields}


# --------------------------------------------------------------------------- gate profiles


def test_default_profile_is_standard():
    report = readiness.assess(_csv(CLEAN_HEADER, _clean_lines(8, 2)), now=NOW)
    assert report["gate_profile"] == "standard"
    assert report["gate_profile_used"] == "standard"
    assert report["replay"]["profile"] == "standard"
    assert {r["profile"] for r in report["replays"]} >= {"standard"}
    assert "recalibration" in report["replay_note"].lower() or "recalibrat" in report["replay_note"]
    assert "context" in report["replay_note"].lower()


def test_invalid_profile_is_422(client):
    r = _upload(client, "/readiness", _csv(CLEAN_HEADER, _clean_lines(8, 2)), gate_profile="nope")
    assert r.status_code == 422


def test_tuned_returns_its_own_replay_entry():
    text = _csv(CLEAN_HEADER, _clean_lines(220, 8))
    report = readiness.assess(text, now=NOW, gate_profile="tuned")
    assert report["gate_profile"] == "tuned"
    assert report["gate_profile_used"] == "tuned"
    assert report["replay"]["profile"] == "tuned"
    by_profile = {r["profile"]: r for r in report["replays"]}
    assert "standard" in by_profile and "tuned" in by_profile
    assert by_profile["tuned"]["status"] != "insufficient"
    assert "lines" in by_profile["tuned"]
    assert "automation_rate" in by_profile["tuned"]
    assert "auto_post_accuracy" in by_profile["tuned"]
    assert "wrong_auto_posts" in by_profile["tuned"]


def test_tuned_context_unavailable_without_department_and_fund():
    # cost_centre only — fund without department is not enough for dept+fund
    text = _csv(CLEAN_HEADER, _clean_lines(220, 8))
    report = readiness.assess(text, now=NOW, gate_profile="tuned_context")
    assert report["gate_profile"] == "tuned_context"
    assert report["gate_profile_used"] == "tuned"
    assert report["context_available"] is False
    by_profile = {r["profile"]: r for r in report["replays"]}
    assert by_profile["tuned_context"]["status"] == "not_available"
    assert "no context columns" in by_profile["tuned_context"]["reason"]
    assert report["replay"]["profile"] == "tuned"


def test_tuned_context_available_with_department_and_fund():
    header = "vendor_name,description,amount,date,gl,cost_centre,department_name"
    lines = []
    depts = ["PARKS", "SANITATION", "TRANSPORT"]
    funds = ["100", "200", "300"]
    names = [f"Northwind Traders {i}" for i in range(8)]
    gls = ["6100", "6200", "6300", "6400", "6500", "6600", "6700", "6800"]
    for i in range(220):
        lines.append(
            f"{names[i % 8]},Monthly office supplies order batch {i},{100 + i},"
            f"2026-0{1 + (i % 9)}-15,{gls[i % 8]},{funds[i % 3]},{depts[i % 3]}"
        )
    report = readiness.assess(_csv(header, lines), now=NOW, gate_profile="tuned_context")
    assert report["context_available"] is True
    assert report["gate_profile_used"] == "tuned_context"
    assert report["replay"]["profile"] == "tuned_context"
    assert not any(r.get("status") == "not_available" for r in report["replays"])


def test_settings_restored_after_success_and_failure(monkeypatch):
    text = _csv(CLEAN_HEADER, _clean_lines(220, 8))
    before = _settings_fingerprint()
    readiness.assess(text, now=NOW, gate_profile="tuned")
    assert _settings_fingerprint() == before

    def boom(*_a, **_k):
        raise RuntimeError("predict failed during replay")

    monkeypatch.setattr(readiness, "predict", boom)
    with pytest.raises(RuntimeError, match="predict failed"):
        readiness.assess(text, now=NOW, gate_profile="tuned")
    assert _settings_fingerprint() == before


def test_erp_is_never_called_on_tuned_profiles(monkeypatch):
    assert settings.erp_backend == "simulated"
    assert settings.xero_posting_enabled is False
    assert settings.sap_posting_enabled is False

    def boom(*_a, **_k):
        raise AssertionError("ERP post was called during readiness")

    monkeypatch.setattr(sap_poster, "post_invoice", boom)
    monkeypatch.setattr(xero_poster, "post_invoice", boom)
    monkeypatch.setattr(posting, "post_outcome", boom)
    header = "vendor_name,description,amount,date,gl,cost_centre,department_name"
    gls = ["6100", "6200", "6300", "6400", "6500", "6600", "6700", "6800"]
    lines = [
        f"Northwind Traders {i % 8},Monthly office supplies order batch {i},{100 + i},"
        f"2026-0{1 + (i % 9)}-15,{gls[i % 8]},CC100,PARKS"
        for i in range(220)
    ]
    readiness.assess(_csv(header, lines), now=NOW, gate_profile="tuned_context")
    readiness.assess(_csv(CLEAN_HEADER, _clean_lines(220, 8)), now=NOW, gate_profile="tuned")
