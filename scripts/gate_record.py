"""
Record, once per replayed line, the evidence the retriever returns - so gate variants can be swept offline.

    python scripts/gate_record.py --embedder offline --configs vendor_k10,subset_m3 --workers 4
    python scripts/gate_record.py --embedder live    --configs vendor_k10

"Shadow-mode" recording: the product starts in shadow mode, where a human codes EVERY invoice. So here each replayed
line is retrieved against the corpus built from the seed plus every earlier line (all written back with the true
account code, dated by transaction date), and the evidence is stored. A gate variant is then just a function of
those stored evidence sets. Closed-loop behaviour (the system learning only from the lines the gate sends to
review) is NOT modelled here - scripts/tuned_replay.py confirms the chosen gates in the real pipeline for that.

Recordings go to data/public/gate_records/ (gitignored). Same leak guard as the replay: only vendor, description,
amount, department, fund and date are inputs; the account code is only the label.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from public_data_import import ensure_sample, row_to_invoice  # noqa: E402
from replay_public_data import (  # noqa: E402
    CACHE_DIR,
    build_vendor_rows,
    name_collisions,
    split_seed_replay,
    use_store,
)

from ap_automation.core import mdm, offline, retrieval  # noqa: E402
from ap_automation.core.config import settings  # noqa: E402
from ap_automation.core.gate_analysis import CONTEXT_LEVELS  # noqa: E402
from ap_automation.core.models import amount_to_band  # noqa: E402
from ap_automation.core.seeding import seed_corpus  # noqa: E402

RECORD_DIR = CACHE_DIR / "gate_records"
K_MAX = 30

# Retrieval configurations that can be recorded. `k` is how many evidence lines are stored per line; for the
# context modes the evidence is one ranking, so any smaller k is an exact truncation of the recording.
CONFIGS: dict[str, dict] = {
    "vendor_k10": {"k": 10, "settings": {"context_signals": "off"}},
    "vendor_k20": {"k": 20, "settings": {"context_signals": "off"}},
    "vendor_k30": {"k": 30, "settings": {"context_signals": "off"}},
    "subset_m3": {"k": K_MAX, "settings": {"context_signals": "subset", "context_min_lines": 3}},
    "subset_m5": {"k": K_MAX, "settings": {"context_signals": "subset", "context_min_lines": 5}},
    "boost": {"k": K_MAX, "settings": {"context_signals": "boost", "context_boost": 0.02}},
    "boost_amt": {"k": K_MAX, "settings": {"context_signals": "boost", "context_boost": 0.02,
                                           "context_use_amount": True}},
}


def _day(date_iso: str) -> datetime:
    return datetime.fromisoformat(date_iso).replace(tzinfo=timezone.utc)


def record_stream(rows: list[dict], seed_months: int, holdout_vendors: int, seed: int, k: int,
                  overrides: dict, live: bool, limit: int | None = None) -> dict:
    """Run the shadow-mode stream and return the recordings as numpy arrays (plus the split metadata)."""
    split = split_seed_replay(rows, seed_months, holdout_vendors, seed)
    seed_rows, replay_rows = split["seed"], split["replay"]
    if limit:
        replay_rows = replay_rows[:limit]
    vendor_rows = build_vendor_rows(rows)
    if name_collisions(vendor_rows):
        raise RuntimeError("vendor name collisions")
    code_ids = {c: i for i, c in enumerate(sorted({r["account_code"] for r in rows}))}

    saved = {key: getattr(settings, key) for key in overrides}
    settings.telemetry_enabled = False
    try:
        for key, val in overrides.items():
            setattr(settings, key, val)
        mdm.set_vendor_map(vendor_rows)
        use_store("inproc", live=live)
        by_day: dict[str, list[dict]] = defaultdict(list)
        for r in seed_rows:
            by_day[r["transaction_date"]].append(r)
        for day in sorted(by_day):
            seed_corpus([row_to_invoice(r) for r in by_day[day]], timestamp=f"{day}T00:00:00+00:00")

        n = len(replay_rows)
        codes = np.full((n, k), -1, dtype=np.int32)
        sims = np.zeros((n, k), dtype=np.float32)
        ages = np.zeros((n, k), dtype=np.float32)
        valid = np.zeros((n, k), dtype=bool)
        truth = np.zeros(n, dtype=np.int32)
        level = np.full(n, CONTEXT_LEVELS.index("vendor"), dtype=np.int8)
        counts = np.zeros((n, 7), dtype=np.int32)  # vendor_lines, vendor_distinct, subset_lines, subset_distinct, dept+fund, fund, dept
        t0 = time.time()
        for i, row in enumerate(replay_rows):
            inv = row_to_invoice(row)
            as_of = _day(row["transaction_date"]).replace(hour=12)
            vendor, _ = mdm.resolve_vendor(inv.raw_vendor_name)
            info: dict = {}
            ev = retrieval.retrieve(
                description=inv.description, vendor_id=vendor, entity_id=inv.entity_id,
                amount_band=amount_to_band(inv.amount), currency="USD", top_k=k,
                fund=inv.cost_centre, amount=inv.amount, info=info,
            )
            for j, e in enumerate(ev[:k]):
                codes[i, j] = code_ids[e.gl_account]
                sims[i, j] = e.similarity_score
                ages[i, j] = (as_of - e.confirmation_timestamp).total_seconds() / 86400.0
                valid[i, j] = True
            truth[i] = code_ids[row["account_code"]]
            if info:
                level[i] = CONTEXT_LEVELS.index(info.get("context_level", "vendor"))
                counts[i] = [info.get("vendor_lines", 0), info.get("vendor_distinct_codes", 0),
                             info.get("subset_lines", 0), info.get("subset_distinct_codes", 0),
                             info.get("n_dept_fund", 0), info.get("n_fund", 0), info.get("n_dept", 0)]
            # shadow mode: a human codes every line; it joins the corpus with the true code, dated by its transaction
            seed_corpus([inv], timestamp=f"{row['transaction_date']}T00:00:00+00:00")
            if (i + 1) % 4000 == 0:
                print(f"  recorded {i + 1:,}/{n:,}  {time.time() - t0:.0f}s", flush=True)
    finally:
        for key, val in saved.items():
            setattr(settings, key, val)
        offline.reset_backend()
        retrieval.set_client(None)

    return {
        "codes": codes, "sims": sims, "ages": ages, "valid": valid, "truth": truth, "level": level, "counts": counts,
        "date": np.array([r["transaction_date"] for r in replay_rows]),
        "vendor": np.array([r["vendor_id"] for r in replay_rows]),
        "dept": np.array([r["department_name"] for r in replay_rows]),
        "amount": np.array([float(r["dollar_amount"]) for r in replay_rows]),
        "code_names": np.array(sorted(code_ids, key=code_ids.get)),
        "cutoff": np.array(split["cutoff"]),
        "held_out_vendors": np.array(split["held_out_vendors"]),
        "seed_vendors": np.array(sorted({r["vendor_id"] for r in seed_rows})),
        "seed_lines": np.array(len(seed_rows)),
    }


def record_path(embedder: str, name: str) -> Path:
    return RECORD_DIR / f"{embedder}_{name}.npz"


def _job(args: tuple) -> str:
    name, embedder, limit = args
    cfg = CONFIGS[name]
    live = embedder == "live"
    rows, _ = ensure_sample(2026, 30_000, 250, 50, CACHE_DIR)
    t0 = time.time()
    print(f"[{embedder}/{name}] start", flush=True)
    rec = record_stream(rows, 3, 10, 42, cfg["k"], cfg["settings"], live, limit)
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    out = record_path(embedder, name)
    np.savez_compressed(out, **rec)
    msg = f"[{embedder}/{name}] done in {time.time() - t0:.0f}s -> {out.name}"
    print(msg, flush=True)
    return msg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=("offline", "live"), default="offline")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: replay only the first N lines")
    args = ap.parse_args()
    names = [n for n in args.configs.split(",") if n]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; choose from {list(CONFIGS)}")
    jobs = [(n, args.embedder, args.limit or None) for n in names]
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            list(pool.map(_job, jobs))
    else:
        for j in jobs:
            _job(j)
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
