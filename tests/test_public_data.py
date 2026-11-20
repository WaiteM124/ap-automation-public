"""
Public-data replay: mapping, leak guard, normalisation, sampling, calibration and the token cap.
No network, no OpenAI traffic, no real data files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import public_data_import as pdi  # noqa: E402
import replay_public_data as rpd  # noqa: E402
from inproc_vector_store import InProcessVectorStore  # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402

from ap_automation.core import mdm, pipeline, retrieval  # noqa: E402
from ap_automation.core.config import settings  # noqa: E402
from ap_automation.core.models import ConfirmationSource, RoutingDecision  # noqa: E402
from ap_automation.core.pipeline import predict  # noqa: E402
from ap_automation.core.seeding import seed_corpus  # noqa: E402

SENTINEL = "LEAKSENTINEL"


def raw_row(i: int = 1, vendor_id: str = "000001", vendor: str = "ACME SUPPLY CO", desc: str = "TONER CARTRIDGE BLACK",
            code: str = "003040", date: str = "2025-08-04T00:00:00.000", amount: str = "125.50", **extra) -> dict:
    row = {
        ":id": f"row-{i}", "fiscal_year": "2026", "department_name": "GENERAL SERVICES", "vendor_name": vendor,
        "vendor_id": vendor_id, "transaction_date": date, "dollar_amount": amount, "fund_name": "GENERAL FUND",
        "fund": "100", "account_code": code, "detailed_item_description": desc, "inv_num": f"INV{i}",
        "inv_date": "2025-07-30T00:00:00.000", "po_num": "PO1", "currency": "USD",
    }
    row.update(extra)
    return row


class RecordingRow(dict):
    """A row that remembers which keys the mapping function touched."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.accessed: set[str] = set()

    def __getitem__(self, key):
        self.accessed.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.accessed.add(key)
        return super().get(key, default)


# --------------------------------------------------------------------------- leak guard

def test_leak_columns_are_never_downloaded():
    assert set(pdi.LEAK_COLUMNS) == {"account_name", "expenditure_type", "item_code", "item_code_name"}
    assert not set(pdi.LEAK_COLUMNS) & set(pdi.RAW_COLUMNS)


def test_mapping_cannot_emit_leak_columns_into_the_invoice():
    leaks = {c: f"{SENTINEL}_{c}" for c in pdi.LEAK_COLUMNS}
    clean, _ = pdi.normalise_rows([raw_row(**leaks)])
    assert not set(pdi.LEAK_COLUMNS) & set(clean[0])          # normalisation drops them outright

    row = RecordingRow(clean[0] | leaks)                       # ...and even if one were present, it is not read
    inv = pdi.row_to_invoice(row)
    assert not set(pdi.LEAK_COLUMNS) & row.accessed
    assert SENTINEL not in inv.model_dump_json()
    for field in ("raw_vendor_name", "description", "entity_id", "cost_centre", "currency"):
        assert SENTINEL not in str(getattr(inv, field))


def test_mapping_reads_only_the_documented_columns():
    row = RecordingRow(pdi.normalise_rows([raw_row()])[0][0])
    pdi.row_to_invoice(row)
    assert row.accessed <= {":id", "vendor_name", "detailed_item_description", "dollar_amount", "department_name",
                            "fund", "account_code"}


def test_mapping_fields():
    inv = pdi.row_to_invoice(pdi.normalise_rows([raw_row(code="40000A")])[0][0])
    assert inv.raw_vendor_name == "ACME SUPPLY CO"
    assert inv.description == "TONER CARTRIDGE BLACK"
    assert inv.amount == 125.5 and inv.currency == "USD"
    assert inv.entity_id == "GENERAL SERVICES" and inv.cost_centre == "100"
    assert inv.ground_truth_gl == "40000A"                     # label only; non-numeric codes survive untouched


def test_invoice_ids_are_stable_and_unique():
    rows = pdi.normalise_rows([raw_row(1), raw_row(2, desc="OTHER")])[0]
    a, b = (pdi.row_to_invoice(r).invoice_id for r in rows)
    assert a != b and a == pdi.row_to_invoice(rows[0]).invoice_id


# --------------------------------------------------------------------------- normalisation

def test_normalise_drops_bad_rows_and_dedupes():
    rows = [
        raw_row(1, desc="  TONER   CARTRIDGE  "),
        raw_row(2, desc="TONER CARTRIDGE", amount="-5"),               # negative
        raw_row(3, amount="0"),                                        # zero
        raw_row(4, desc="   "),                                        # blank
        raw_row(5, desc="123456789012"),                               # nothing left after preprocessing
        raw_row(6, vendor="PRIVACY-POLICE", vendor_id="PRIVACY-POLICE"),
        raw_row(7, amount="not-a-number"),
        raw_row(8, desc="TONER CARTRIDGE", code=""),
        raw_row(1, desc="toner cartridge"),                            # duplicate of row 1 (case-insensitive)
    ]
    clean, stats = pdi.normalise_rows(rows)
    assert [r["detailed_item_description"] for r in clean] == ["TONER CARTRIDGE"]
    assert stats["dropped"]["duplicate_line"] == 1
    assert stats["dropped"]["privacy_masked_payee"] == 1
    assert stats["dropped"]["non_positive_amount"] == 2
    assert clean[0]["transaction_date"] == "2025-08-04" and clean[0]["dollar_amount"] == "125.50"


def test_account_codes_stay_strings():
    clean, _ = pdi.normalise_rows([raw_row(1, code="003040"), raw_row(2, desc="X Y", code="94A427")])
    assert [r["account_code"] for r in clean] == ["003040", "94A427"]


# --------------------------------------------------------------------------- sampling + split

def _vendor_rows(vendor: str, n: int, start_day: int = 1) -> list[dict]:
    out = []
    for i in range(n):
        month = 7 + (i * 6) // n
        out.append({
            ":id": f"{vendor}-{i:04d}", "vendor_id": vendor, "vendor_name": f"VENDOR {vendor} INC",
            "department_name": "GENERAL SERVICES", "fund": "100", "fund_name": "GF", "account_code": "003040",
            "dollar_amount": "10.00", "transaction_date": f"2025-{month:02d}-{(i % 27) + 1:02d}",
            "detailed_item_description": f"item {i}", "inv_num": "", "inv_date": "", "po_num": "", "currency": "USD",
            "fiscal_year": "2026",
        })
    return out


def test_select_sample_keeps_busiest_vendors_within_cap_in_date_order():
    rows = _vendor_rows("A", 400) + _vendor_rows("B", 300) + _vendor_rows("C", 120) + _vendor_rows("D", 10)
    sample, params = pdi.select_sample(rows, max_rows=500, max_per_vendor=200, min_vendor_rows=50)
    assert len(sample) <= 500
    by = {}
    for r in sample:
        by.setdefault(r["vendor_id"], []).append(r)
    assert set(by) == {"A", "B"}                                   # D is too small, C no longer fits
    assert all(len(v) == 200 for v in by.values())                 # thinned evenly to the per-vendor cap
    keys = [(r["transaction_date"], r[":id"]) for r in sample]
    assert keys == sorted(keys)
    assert min(r["transaction_date"] for r in by["A"]) <= "2025-07-05"        # thinning keeps the date span
    assert max(r["transaction_date"] for r in by["A"]) >= "2025-12-01"
    assert params["max_rows"] == 500


def test_sample_stats_label_noise_indicators():
    rows = _vendor_rows("A", 10) + _vendor_rows("B", 10)
    for r in rows[:5]:
        r["account_code"] = "94A427"
    s = pdi.sample_stats(rows)
    assert s["rows"] == 20 and s["vendors"] == 2
    assert s["vendors_with_gt1_account_code"] == 1
    assert s["non_numeric_account_codes"] == 1
    assert s["date_min"] <= s["date_max"]


def test_split_seed_replay():
    rows = _vendor_rows("A", 60) + _vendor_rows("B", 60) + _vendor_rows("C", 60)
    split = rpd.split_seed_replay(rows, seed_months=2, holdout_vendors=1, seed=7)
    assert split["cutoff"] == "2025-09-01"
    assert all(r["transaction_date"] < "2025-09-01" for r in split["seed"])
    assert all(r["transaction_date"] >= "2025-09-01" for r in split["replay"])
    keys = [(r["transaction_date"], r[":id"]) for r in split["replay"]]
    assert keys == sorted(keys)
    held = split["held_out_vendors"][0]
    assert held not in {r["vendor_id"] for r in split["seed"]}     # held-out vendor arrives with no history
    assert held in {r["vendor_id"] for r in split["replay"]}
    assert split["dropped_held_out_seed_lines"] > 0
    assert rpd.split_seed_replay(rows, 2, 1, 7)["held_out_vendors"] == split["held_out_vendors"]  # deterministic


def test_month_arithmetic():
    assert rpd.month_start_plus("2025-11-17", 3) == "2026-02-01"
    assert rpd.month_start_plus("2025-07-01", 12) == "2026-07-01"


def test_vendor_master_is_keyed_on_vendor_id_and_collisions_are_detected():
    rows = _vendor_rows("A", 3) + _vendor_rows("B", 3)
    master = rpd.build_vendor_rows(rows)
    assert {m["canonical_id"] for m in master} == {"A", "B"}
    assert rpd.name_collisions(master) == []
    clash = master + [{"raw_name": "Vendor A Ltd.", "canonical_id": "Z"}]   # normalises to the same key as "VENDOR A INC"
    assert rpd.name_collisions(clash)


# --------------------------------------------------------------------------- pipeline handles LA-shaped data

def _la_history(vendor: str, vendor_id: str, code: str, n: int = 6):
    rows = pdi.normalise_rows([
        raw_row(i, vendor_id=vendor_id, vendor=vendor, desc=f"TONER CARTRIDGE BLACK MODEL {i}", code=code,
                date=f"2025-08-{i + 1:02d}T00:00:00.000") for i in range(n)])[0]
    return rows


def test_non_numeric_account_codes_flow_through_the_real_pipeline(monkeypatch):
    rows = _la_history("ACME SUPPLY CO", "000001", "40000A")
    mdm.set_vendor_map(rpd.build_vendor_rows(rows))
    seed_corpus([pdi.row_to_invoice(r) for r in rows])
    probe = pdi.row_to_invoice(pdi.normalise_rows([raw_row(99, desc="TONER CARTRIDGE BLACK MODEL 3", code="40000A")])[0][0])

    monkeypatch.setattr(pipeline, "_orchestrator", rpd.build_orchestrator("neutral", {"GENERAL SERVICES"}))
    pred = predict(probe, batch_mode=True)
    assert pred.predicted_gl == "40000A"
    assert pred.agent_flags == [] and pred.final_routing == RoutingDecision.AUTO_POST
    assert pred.resolved_line.canonical_vendor_id == "000001"       # vendor_id is the stable identity

    monkeypatch.setattr(pipeline, "_orchestrator", rpd.build_orchestrator("neutral", {"PARKS"}))
    assert "entity_validation_failed" in predict(probe, batch_mode=True).agent_flags  # unknown department is flagged

    monkeypatch.setattr(pipeline, "_orchestrator", rpd.build_orchestrator("shipped", set()))
    shipped = predict(probe, batch_mode=True)                       # the shipped UK VAT config blocks every LA line
    assert "vat_not_assessed" in shipped.agent_flags and shipped.final_routing == RoutingDecision.HUMAN_REVIEW


# --------------------------------------------------------------------------- metrics

def _rec(i, conf, auto, correct, history=5, learned=False, date="2025-10-05", vendor="V"):
    return rpd.ReplayRecord(
        index=i, vendor_id=vendor, category="public", anomaly=None, history_before=history,
        predicted_gl="A" if correct else "B", truth_gl="A",
        rule_routing=("auto_post" if auto else "human_review"), final_routing=("auto_post" if auto else "human_review"),
        flags=[], reason="" if auto else "Evidence disagrees: agreement 50% below 80%", learned=learned,
        date=date, confidence=conf, evidence_n=10, baseline_all_pred="A", baseline_corpus_pred="A")


def test_calibration_bins_and_ece():
    recs = [_rec(0, 0.3, False, False), _rec(1, 0.6, False, True), _rec(2, 0.6, False, False),
            _rec(3, 0.9, True, True), _rec(4, 0.97, True, True), _rec(5, 0.99, True, False)]
    cal = rpd.calibration(recs)
    bands = {b["band"]: b for b in cal["bins"]}
    assert [b["band"] for b in cal["bins"]] == ["<0.5", "0.5-0.7", "0.7-0.85", "0.85-0.95", "0.95+"]
    assert bands["<0.5"]["n"] == 1 and bands["<0.5"]["accuracy"] == 0.0
    assert bands["0.5-0.7"]["n"] == 2 and bands["0.5-0.7"]["accuracy"] == 0.5
    assert bands["0.7-0.85"]["n"] == 0 and bands["0.7-0.85"]["accuracy"] is None
    assert bands["0.95+"]["n"] == 2 and bands["0.95+"]["auto_post_accuracy"] == 0.5
    assert 0 <= cal["ece_all_predictions"] <= 1


def test_summary_metrics_match_the_harness_definitions():
    recs = [_rec(0, 0.9, True, True, learned=False), _rec(1, 0.9, True, False), _rec(2, 0.4, False, True, learned=True),
            _rec(3, 0.4, False, False, learned=True, vendor="NEW", history=0, date="2025-11-02")]
    s = rpd.summarise_replay(recs, seeded_vendors={"V"})
    assert s["invoices"] == 4 and s["automation_rate"] == 0.5
    assert s["auto_post_accuracy"] == 0.5 and s["false_positives"] == 1
    assert s["gl_accuracy_all_predictions"] == 0.5
    assert s["lines_learned_from_auto_posts"] == 0
    assert s["new_to_corpus_vendors"]["vendors"] == 1 and s["new_to_corpus_vendors"]["automation_rate"] == 0.0
    assert [m["month"] for m in s["learning_curve_by_month"]] == ["2025-10", "2025-11"]
    assert s["review_reasons"] == {"evidence_disagrees": 2}
    assert s["share_of_reviews_that_were_unnecessary"] == 0.5


# --------------------------------------------------------------------------- live-embedding cost cap

def test_token_cap_refuses_before_any_api_call(monkeypatch, tmp_path):
    monkeypatch.setattr(rpd, "CACHE_DIR", tmp_path)

    def boom():
        raise AssertionError("the API must not be called when the cap is exceeded")

    monkeypatch.setattr(rpd.embeddings, "get_client", boom)
    texts = {f"description number {i} with a few extra words" for i in range(500)}
    with pytest.raises(rpd.TokenCapExceeded):
        rpd.precompute_live_embeddings(texts, max_tokens=100, max_usd=1.0)
    with pytest.raises(rpd.TokenCapExceeded):
        rpd.precompute_live_embeddings(texts, max_tokens=10**9, max_usd=1e-9)


def test_token_counter_and_spend_report_with_a_fake_client(monkeypatch, tmp_path):
    monkeypatch.setattr(rpd, "CACHE_DIR", tmp_path)
    calls = []

    class Usage:
        def __init__(self, n):
            self.total_tokens = n

    class FakeClient:
        class embeddings:  # noqa: N801
            @staticmethod
            def create(model, input):
                calls.append(list(input))
                data = [type("D", (), {"index": i, "embedding": [float(i + 1)] * settings.embedding_dimensions})()
                        for i, _ in enumerate(input)]
                return type("R", (), {"data": data, "usage": Usage(7 * len(input))})()

    monkeypatch.setattr(rpd.embeddings, "get_client", lambda: FakeClient)
    texts = {f"text {i}" for i in range(600)}
    rep = rpd.precompute_live_embeddings(texts, max_tokens=10**6, max_usd=1.0)
    assert rep["actual_tokens"] == 7 * 600 and rep["api_requests"] == 3
    assert rep["actual_cost_usd"] == round(7 * 600 * 0.02 / 1e6, 5)
    again = rpd.precompute_live_embeddings(texts, max_tokens=10**6, max_usd=1.0)   # served from the disk cache
    assert again["to_embed"] == 0 and len(calls) == 3
    emb = rpd.CachedLiveEmbedder(settings.embedding_model)
    assert len(emb("text 5")) == settings.embedding_dimensions
    with pytest.raises(KeyError):
        emb("never embedded")                                        # no silent fallback


# --------------------------------------------------------------------------- vector-store stand-in

def test_inprocess_store_matches_qdrant_local():
    rng = np.random.default_rng(0)
    dim = settings.embedding_dimensions
    cfg = models.VectorParams(size=dim, distance=models.Distance.COSINE)
    ref, fast = QdrantClient(":memory:"), InProcessVectorStore()
    for c in (ref, fast):
        c.create_collection(collection_name="c", vectors_config=cfg)
    pts = []
    for i in range(60):
        pts.append(models.PointStruct(id=retrieval.point_id_for(f"i{i}"), vector=rng.normal(size=dim).tolist(),
                                      payload={"vendor_id": "AB"[i % 2], "invoice_id": f"i{i}", "n": i}))
    for c in (ref, fast):
        c.upsert(collection_name="c", points=pts)
    q = rng.normal(size=dim).tolist()
    flt = models.Filter(must=[models.FieldCondition(key="vendor_id", match=models.MatchValue(value="A"))])
    a = ref.query_points(collection_name="c", query=q, query_filter=flt, limit=8).points
    b = fast.query_points(collection_name="c", query=q, query_filter=flt, limit=8).points
    assert [p.id for p in a] == [p.id for p in b]
    assert [round(p.score, 4) for p in a] == [round(p.score, 4) for p in b]
    ids = [p.id for p in a[:3]]
    by_id = fast.query_points(collection_name="c", query=q, limit=3,
                              query_filter=models.Filter(must=[models.HasIdCondition(has_id=ids)])).points
    assert {p.id for p in by_id} == set(ids)
    seen, off = [], None
    while True:
        batch, off = fast.scroll(collection_name="c", scroll_filter=flt, limit=7, offset=off)
        seen += [r.payload["invoice_id"] for r in batch]
        if off is None:
            break
    assert sorted(seen) == sorted(f"i{i}" for i in range(0, 60, 2))
    assert fast.get_collection("c").points_count == 60
    fast.upsert(collection_name="c", points=[pts[0]])                  # upsert by id replaces, never duplicates
    assert fast.get_collection("c").points_count == 60


# --------------------------------------------------------------------------- end-to-end smoke

def test_replay_arm_end_to_end_small(monkeypatch):
    rows = []
    for v, code in (("A", "003040"), ("B", "40000A"), ("C", "94A427")):
        rows += _vendor_rows(v, 70)
        for r in rows:
            if r["vendor_id"] == v:
                r["account_code"] = code
    cfg = {"seed_months": 2, "holdout_vendors": 1, "seed": 3, "agents": "neutral", "live_embeddings": False,
           "store": "inproc"}
    with_learning = rpd.run_arm({"name": "w", "learn": True, "cfg": cfg, "rows": rows, "limit": None})
    control = rpd.run_arm({"name": "c", "learn": False, "cfg": cfg, "rows": rows, "limit": None})
    w, c = with_learning["summary"], control["summary"]
    assert w["invoices"] == c["invoices"] > 0
    assert w["lines_learned_from_auto_posts"] == 0 and c["lines_learned"] == 0
    assert w["enforcement"]["forced_auto_post_attempts"] == w["enforcement"]["refused"]
    assert w["automation_rate"] >= c["automation_rate"]
    assert w["new_to_corpus_vendors"]["vendors"] == 1
    assert c["new_to_corpus_vendors"]["automation_rate"] == 0.0           # no history ever, so always reviewed
    assert w["auto_post_accuracy"] == 1.0                                 # one account per vendor in this toy data
    assert ConfirmationSource.AUTO_POST.value == "auto_post"
