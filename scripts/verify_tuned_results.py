"""Independent re-check of the tuned shadow-mode headline (live embeddings, combined, target 99.5).

Run:  python scripts/verify_tuned_results.py   (needs data/public from scripts/public_data_import.py and the
gate records from scripts/gate_record.py; makes one request to the City of Los Angeles open-data API).

1. Labels in the recorded evidence file vs the source sample (row for row).
2. Leakage probes on the recorded evidence (future-dated or self-matching evidence).
3. Re-score the TEST window with the SHIPPED scalar gate (confidence.vote_features / route_variant) using the
   exact settings stored in the results file; compare with the saved numbers.
4. Spot-check random auto-posted lines against the City's live API.
"""
import csv
import json
import random
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from ap_automation.core.confidence import GateParams, route_variant, vote_features  # noqa: E402
from ap_automation.core.config import settings  # noqa: E402
from ap_automation.core.gate_analysis import CONTEXT_LEVELS  # noqa: E402
from ap_automation.core.models import ConfirmationSource, RetrievedEvidence, RoutingDecision  # noqa: E402

SPLIT = "2026-03-01"
saved = json.load(open(REPO / "docs" / "public_replay_tuned_live.json"))["shadow_mode_sweep"]["combined"]["tuned_99.5"]
rec = dict(np.load(REPO / "data" / "public" / "gate_records" / f"live_{saved['record']}.npz", allow_pickle=False))
N = len(rec["truth"])
names = rec["code_names"]
print(f"config: record={saved['record']} k={saved['k']} recency={saved['recency']} gate={saved['gate']}")
print(f"saved TEST: auto={saved['test']['auto_posted']} wrong={saved['test']['wrong_auto_posts']} "
      f"acc={saved['test']['auto_post_accuracy']} cov={saved['test']['automation_rate']}\n")

# ---------------------------------------------------------------- 1. labels vs source
with (REPO / "data/public/la_checkbook_fy2026_sample_30000_pv250.csv").open(encoding="utf-8-sig", newline="") as f:
    sample = list(csv.DictReader(f))
replay = sorted((r for r in sample if r["transaction_date"] >= "2025-10-01"), key=lambda r: (r["transaction_date"], r[":id"]))
assert len(replay) == N, (len(replay), N)
bad = sum(1 for i in range(N)
          if (rec["date"][i], rec["vendor"][i], names[rec["truth"][i]]) != (replay[i]["transaction_date"], replay[i]["vendor_id"], replay[i]["account_code"]))
print(f"[1] evidence-file rows disagreeing with the source sample on date/vendor/true code: {bad} of {N}")

# ---------------------------------------------------------------- 2. leakage probes
valid, ages, sims = rec["valid"], rec["ages"], rec["sims"]
print(f"[2] valid evidence entries: {int(valid.sum()):,}")
print(f"    future-dated evidence (age < 0): {int((valid & (ages < 0)).sum())}")
print(f"    same-day evidence (age == 0)   : {int((valid & (ages == 0)).sum())}")
print(f"    exact-duplicate-looking (sim>=0.9999 and age==0): {int((valid & (sims >= 0.9999) & (ages == 0)).sum())}")
held = set(rec["held_out_vendors"].tolist())
held_rows = np.isin(rec["vendor"], list(held)) & (rec["date"] < "2025-10-01")
print(f"    held-out vendors that have seed-period lines: {int(held_rows.sum())}")

# ---------------------------------------------------------------- 3. shipped scalar gate on the TEST window
for k, v in saved["settings"].items():
    if hasattr(settings, k):
        setattr(settings, k if k != "top_k" else "top_k", v)
settings.min_mdm_confidence = 0.0
params = GateParams.from_settings()
print(f"[3] scalar gate params: {params}")
print(f"    settings: min_evidence_lines={settings.min_evidence_lines} agree={settings.min_agreement_ratio} "
      f"wc={settings.default_confidence_threshold}")
K = saved["k"]
test_idx = [i for i in range(N) if rec["date"][i] >= SPLIT]
auto_idx, wrong_idx = [], []
for i in test_idx:
    d = datetime.fromisoformat(rec["date"][i]).replace(tzinfo=timezone.utc)
    ev = []
    for j in range(K):
        if not valid[i, j]:
            continue
        ev.append(RetrievedEvidence(
            invoice_id=str(j), description="x", gl_account=str(names[rec["codes"][i, j]]), cost_centre="",
            vendor_id=str(rec["vendor"][i]), similarity_score=float(sims[i, j]), dense_rank=j, sparse_rank=j, rrf_score=0.0,
            confirmation_timestamp=d - timedelta(days=float(ages[i, j])), source=ConfirmationSource.HUMAN_CONFIRM))
    f = vote_features(ev, d, params, 3)
    level = CONTEXT_LEVELS[int(rec["level"][i])]
    dec, _ = route_variant(f, str(rec["vendor"][i]), params, 1.0, int(rec["counts"][i, 1]), int(rec["counts"][i, 3]), level)
    if dec == RoutingDecision.AUTO_POST:
        auto_idx.append(i)
        if f.predicted_gl != str(names[rec["truth"][i]]):
            wrong_idx.append(i)
n = len(test_idx)
print(f"    TEST lines: {n}  auto-posted: {len(auto_idx)} ({len(auto_idx)/n:.1%})  wrong: {len(wrong_idx)}  "
      f"accuracy: {1-len(wrong_idx)/len(auto_idx):.2%}")
match = (len(auto_idx) == saved["test"]["auto_posted"] and len(wrong_idx) == saved["test"]["wrong_auto_posts"])
print(f"    MATCHES SAVED RESULT EXACTLY: {match}")

# ---------------------------------------------------------------- 4. spot-check vs the City's API
rng = random.Random(11)
picks = sorted(rng.sample(auto_idx, 8) + rng.sample(wrong_idx, 2))
ids = [replay[i][":id"] for i in picks]
q = urllib.parse.urlencode({
    "$select": ":id,vendor_id,vendor_name,account_code,dollar_amount,transaction_date,detailed_item_description,department_name,fund",
    "$where": ":id in (" + ",".join(f"'{x}'" for x in ids) + ")"}, quote_via=urllib.parse.quote)
with urllib.request.urlopen("https://controllerdata.lacity.org/resource/pggv-e4fn.json?" + q, timeout=60) as r:
    live = {row[":id"]: row for row in json.load(r)}
print(f"\n[4] City API returned {len(live)} of {len(ids)} rows (8 random auto-posts + 2 random wrong auto-posts)")
all_ok = True
for i in picks:
    s, c = replay[i], live.get(replay[i][":id"])
    if not c:
        all_ok = False
        print(f"  [{i}] NOT FOUND")
        continue
    ok = c["account_code"] == s["account_code"] and c["vendor_id"] == s["vendor_id"] and \
        c["detailed_item_description"].strip() == s["detailed_item_description"].strip()
    all_ok &= ok
    tag = "wrong-auto-post" if i in wrong_idx else "auto-post ok   "
    print(f"  [{i}] {c['transaction_date'][:10]} {c['vendor_name'][:26]:26s} {c['department_name'][:16]:16s} fund {c['fund']:4s} "
          f"City={c['account_code']:7s} {tag} source==City: {ok}")
print("\nSPOT-CHECK:", "ALL MATCH THE CITY'S LIVE DATA" if all_ok else "MISMATCH")

