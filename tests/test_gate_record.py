"""
The offline recorder (scripts/gate_record.py) must see exactly what the real pipeline sees.

With a gate that sends every line to review, the closed-loop replay learns from every line (a human codes all of
them) - the same corpus the shadow-mode recorder builds. The per-line evidence statistics must then agree.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gate_record import record_stream  # noqa: E402
from replay_public_data import run_arm  # noqa: E402

from ap_automation.core.gate_analysis import vote_arrays  # noqa: E402


def _rows(n_per_vendor=40, seed=7):
    rng = random.Random(seed)
    rows = []
    topics = {"A": ["paper towels", "toner cartridges"], "B": ["pump repair", "valve repair"]}
    k = 0
    for vendor, name in (("V1", "ACME SUPPLY"), ("V2", "BETA SERVICES")):
        for i in range(n_per_vendor):
            dept = rng.choice(["PUBLIC WORKS", "LIBRARY"])
            fund = "F1" if dept == "PUBLIC WORKS" else "F2"
            code = ("A" if dept == "PUBLIC WORKS" else "B") if vendor == "V1" else rng.choice(["A", "B"])
            day = 1 + (i * 85) // n_per_vendor
            date = f"2025-{7 + (day - 1) // 30:02d}-{(day - 1) % 30 + 1:02d}"
            k += 1
            rows.append({
                ":id": f"r{k:04d}", "vendor_id": vendor, "vendor_name": name, "department_name": dept, "fund": fund,
                "fund_name": "x", "account_code": code, "dollar_amount": str(round(rng.uniform(20, 5000), 2)),
                "transaction_date": date, "detailed_item_description": f"{rng.choice(topics[code])} order {k}",
                "inv_num": "", "inv_date": date, "po_num": "", "currency": "USD", "fiscal_year": "2026",
            })
    return sorted(rows, key=lambda r: (r["transaction_date"], r[":id"]))


@pytest.mark.parametrize("overrides", [
    {"context_signals": "off", "top_k": 10},
    {"context_signals": "subset", "context_min_lines": 3, "top_k": 12},
    {"context_signals": "boost", "context_boost": 0.02, "top_k": 12},
])
def test_recorder_matches_pipeline_evidence(overrides):
    rows = _rows()
    k = overrides["top_k"]
    cfg = {"seed_months": 1, "holdout_vendors": 0, "seed": 1, "agents": "neutral", "live_embeddings": False,
           "store": "inproc", "sim_clock": True, "clerk_cc": True,
           "settings": {**overrides, "min_evidence_lines": 10**6}}  # nothing can auto-post -> every line is learned
    arm = run_arm({"name": "t", "cfg": cfg, "rows": rows, "learn": True})
    recs = arm["records"]
    assert all(r["learned"] for r in recs)

    rec = record_stream(rows, 1, 0, 1, k, overrides, live=False)
    v = vote_arrays(rec["codes"], rec["sims"], rec["ages"], rec["valid"], k, 0.0, 0.0, 3)
    names = rec["code_names"]
    assert len(recs) == len(v.n)
    for i, r in enumerate(recs):
        assert r["evidence_n"] == v.n[i], i
        if v.n[i]:
            assert r["predicted_gl"] == names[v.predicted_code[i]], i
            assert r["agreement"] == pytest.approx(v.agreement[i], abs=1e-4), i
            assert r["confidence"] == pytest.approx(v.weighted_confidence[i], abs=1e-3), i
        assert np.isfinite(v.agreement[i])


def test_audit_sample_is_deterministic_and_close_to_its_rate():
    from replay_public_data import _audit_pick

    picks = [_audit_pick(42, i, 0.3) for i in range(5000)]
    assert picks == [_audit_pick(42, i, 0.3) for i in range(5000)]
    assert 0.27 < sum(picks) / len(picks) < 0.33
    assert not any(_audit_pick(42, i, 0.0) for i in range(500))
    assert all(_audit_pick(42, i, 1.0) for i in range(500))


def test_arm_names_carry_an_optional_audit_rate():
    from tuned_replay import split_arm

    assert split_arm("combined/tuned_99") == ("combined/tuned_99", 0.0)
    assert split_arm("combined/tuned_99.5@a30") == ("combined/tuned_99.5", pytest.approx(0.30))
