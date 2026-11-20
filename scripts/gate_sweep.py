"""
Sweep gate variants offline over recorded evidence (scripts/gate_record.py), tuning on an early window and
reporting on a later, untouched one.

    python scripts/gate_sweep.py --embedder offline

Windows (by transaction date): TUNE = Oct-Feb (everything before --split), TEST = Mar-Jun (from --split on).
Every threshold is chosen on TUNE only; TEST is then read once for the chosen configurations. A second, clearly
labelled "oracle" series picks thresholds on TEST itself - that shows what the data could support at best, and is
never quoted as held-out.

Families (what varies; all search the gate grid on the TUNE window):
    baseline  vendor-scoped evidence (k=10), shipped thresholds (3 lines / 0.80 agreement / 0.50 confidence)
    tight     baseline retrieval, only the shipped knobs tightened (min lines, agreement, confidence)
    gate      baseline retrieval (k 10/20/30) + the extra gate ideas: top-account margin, Wilson lower bound on
              agreement, evidence requirement that scales with the vendor's code diversity
    context   department / fund / amount-aware retrieval (subset / boost), shipped knobs tightened
    recency   baseline retrieval with recency-weighted votes (time decay or recent window), shipped knobs tightened
    combined  context retrieval + recency + the full gate grid, incl. "require department+fund evidence"

Operating point for a target accuracy T: the config with the largest TUNE coverage whose TUNE auto-post accuracy
is >= T (with at least MIN_POSTS auto-posts). Output: data/public/gate_sweep_<embedder>.json (+ docs/gate_curve.json).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_record import CONFIGS, record_path  # noqa: E402

from ap_automation.core.gate_analysis import (  # noqa: E402
    CONTEXT_LEVELS,
    SweepGate,
    calibration_table,
    gate_mask,
    pareto_front,
    risk_coverage_curve,
    vote_arrays,
    wilson_interval,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = REPO_ROOT / "data" / "public"
MIN_POSTS = 200
TARGETS = (0.95, 0.98, 0.99, 0.995)
CURVE_TARGETS = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995]
SPLIT = "2026-03-01"
CAL_BINS = [(0.0, 0.5, "<0.5"), (0.5, 0.7, "0.5-0.7"), (0.7, 0.85, "0.7-0.85"), (0.85, 0.95, "0.85-0.95"),
            (0.95, 1.01, "0.95+")]
RECENCY = [("none", 0.0, 0.0), ("hl30", 30.0, 0.0), ("hl90", 90.0, 0.0), ("hl180", 180.0, 0.0),
           ("win60", 0.0, 60.0), ("win120", 0.0, 120.0), ("win180", 0.0, 180.0)]
CTX_RECORDS = ("subset_m3", "subset_m5", "boost", "boost_amt")
SHIPPED = SweepGate()

FAMILIES: dict[str, dict] = {
    "tight": {"records": [("vendor_k10", [10])], "recency": ["none"], "grid": "prod"},
    "gate": {"records": [("vendor_k10", [10]), ("vendor_k20", [20]), ("vendor_k30", [30])], "recency": ["none"],
             "grid": "ext"},
    "context": {"records": [(r, [10, 20, 30]) for r in CTX_RECORDS], "recency": ["none"], "grid": "prod"},
    "recency": {"records": [("vendor_k10", [10])], "recency": [r[0] for r in RECENCY[1:]], "grid": "prod"},
    "context_gate": {"records": [(r, [10, 20, 30]) for r in CTX_RECORDS], "recency": ["none"], "grid": "ext"},
    "recency_gate": {"records": [("vendor_k10", [10]), ("vendor_k20", [20]), ("vendor_k30", [30])],
                     "recency": [r[0] for r in RECENCY], "grid": "ext"},
    "combined": {"records": [(r, [10, 20, 30]) for r in CTX_RECORDS], "recency": [r[0] for r in RECENCY],
                 "grid": "ext"},
}


def grid_gates(kind: str, k: int, has_context: bool) -> list[SweepGate]:
    ns = [n for n in (3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30) if n <= k]
    if kind == "prod":
        return [SweepGate(min_n=n, min_agreement=a, min_wc=w)
                for n, a, w in itertools.product(ns, (0.8, 0.85, 0.9, 0.95, 1.0), (0.5, 0.6, 0.7, 0.8, 0.9))]
    ns = [n for n in (3, 5, 8, 10, 15, 20) if n <= k]
    divs = [(0.0, "vendor")] + [(d, s) for d in (0.5, 1.0) for s in (("vendor", "subset") if has_context else ("vendor",))]
    levels = [(), ("dept+fund",), ("dept+fund", "fund")] if has_context else [()]
    return [SweepGate(min_n=n, min_agreement=a, min_wc=w, min_margin=m, min_lb=lb, z=1.0, div_per_code=d,
                      div_scope=s, div_cap=10, required_levels=rl)
            for n, a, w, m, lb, (d, s), rl in itertools.product(
                ns, (0.8, 0.9, 0.95, 1.0), (0.5, 0.7, 0.85), (0.0, 0.6, 0.8), (0.0, 0.6, 0.8, 0.9), divs, levels)]


def load_record(embedder: str, name: str) -> dict:
    return dict(np.load(record_path(embedder, name), allow_pickle=False))


def windows(rec: dict, split: str = SPLIT) -> tuple[np.ndarray, np.ndarray]:
    tune = rec["date"] < split
    return tune, ~tune


def gate_to_dict(g: SweepGate) -> dict:
    return {"min_n": g.min_n, "min_agreement": g.min_agreement, "min_wc": g.min_wc, "min_margin": g.min_margin,
            "min_lb": g.min_lb, "z": g.z, "div_per_code": g.div_per_code, "div_scope": g.div_scope,
            "div_cap": g.div_cap, "required_levels": list(g.required_levels)}


def gate_from_dict(d: dict) -> SweepGate:
    return SweepGate(**{**d, "required_levels": tuple(d.get("required_levels", ()))})


def to_settings(record: str, k: int, rec_name: str, g: SweepGate) -> dict:
    """The core.config.settings overrides that make the real pipeline run this configuration."""
    _, hl, win = next(r for r in RECENCY if r[0] == rec_name)
    s = dict(CONFIGS[record]["settings"])
    s.update(top_k=k, min_evidence_lines=g.min_n, min_agreement_ratio=g.min_agreement,
             default_confidence_threshold=g.min_wc, gate_min_margin=g.min_margin, gate_min_agreement_lb=g.min_lb,
             gate_wilson_z=g.z, gate_diversity_per_code=g.div_per_code, gate_diversity_scope=g.div_scope,
             gate_diversity_cap=g.div_cap, gate_recency_halflife_days=hl, gate_recency_window_days=win,
             gate_required_context=",".join(g.required_levels))
    return s


def _feature_set(task: tuple):
    embedder, record, k, rec_name = task
    rec = load_record(embedder, record)
    _, hl, win = next(r for r in RECENCY if r[0] == rec_name)
    v = vote_arrays(rec["codes"], rec["sims"], rec["ages"], rec["valid"], k, hl, win, 3)
    return rec, v


def sweep_task(task: tuple) -> list[dict]:
    """All grid points of one feature set; returns rows on the TUNE or TEST Pareto front (plus nothing else)."""
    embedder, record, k, rec_name, grid_kind, split = task
    rec, v = _feature_set((embedder, record, k, rec_name))
    tune, test = windows(rec, split)
    correct = v.predicted_code == rec["truth"]
    wrong = ~correct
    cache: dict = {}
    vd, sd, lvl = rec["counts"][:, 1], rec["counts"][:, 3], rec["level"]
    n_tune, n_test = int(tune.sum()), int(test.sum())
    rows = []
    for g in grid_gates(grid_kind, k, record in CTX_RECORDS):
        m = gate_mask(v, g, vd, sd, lvl, cache=cache)
        mt, ms = m & tune, m & test
        rows.append((g, int(mt.sum()), int((mt & wrong).sum()), int(ms.sum()), int((ms & wrong).sum())))
    out: list[dict] = []
    for side, (ni, wi, nn) in (("tune", (1, 2, n_tune)), ("test", (3, 4, n_test))):
        cand = [r for r in rows if r[ni] >= MIN_POSTS]
        pts = [(r[ni] / nn, 1 - r[wi] / r[ni]) for r in cand]
        out += [{"idx": id(cand[i][0]), "front": side, "gate": cand[i][0], "tune_n": cand[i][1], "tune_wrong": cand[i][2],
                 "test_n": cand[i][3], "test_wrong": cand[i][4]} for i in pareto_front(pts)]
    seen, uniq = set(), []
    for r in out:
        key = (r["idx"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return [{"record": record, "k": k, "recency": rec_name, "grid": grid_kind,
             "gate": gate_to_dict(r["gate"]), "front": r["front"], "tune_n": r["tune_n"], "tune_wrong": r["tune_wrong"],
             "test_n": r["test_n"], "test_wrong": r["test_wrong"], "n_tune": n_tune, "n_test": n_test} for r in uniq]


def tkey(t: float) -> str:
    return f"tuned_{t * 100:g}"


def point(row: dict, window: str) -> tuple[float, float, int, int]:
    n, w, tot = row[f"{window}_n"], row[f"{window}_wrong"], row[f"n_{window}"]
    return n / tot, (1 - w / n) if n else float("nan"), n, w


def pick(rows: list[dict], target: float, window: str = "tune") -> dict | None:
    """Largest-coverage row whose accuracy on `window` is >= target (ties: fewer wrong, then simpler gate)."""
    ok = [r for r in rows if r[f"{window}_n"] >= MIN_POSTS and point(r, window)[1] >= target]
    if not ok:
        return None
    return max(ok, key=lambda r: (r[f"{window}_n"], -r[f"{window}_wrong"], -r["gate"]["min_n"]))


# --------------------------------------------------------------------------- detail for a chosen configuration

def vendor_diversity(rec: dict, truth_codes: np.ndarray) -> np.ndarray:
    """True if the vendor uses more than one account code anywhere in the replayed lines (reporting split only)."""
    multi = {}
    for ven in np.unique(rec["vendor"]):
        multi[ven] = len(np.unique(truth_codes[rec["vendor"] == ven])) > 1
    return np.array([multi[x] for x in rec["vendor"]])


def detail(embedder: str, record: str, k: int, rec_name: str, g: SweepGate, split: str = SPLIT) -> dict:
    """Metrics of one configuration on tune / test / all, by month, by vendor diversity, with calibration."""
    rec, v = _feature_set((embedder, record, k, rec_name))
    tune, test = windows(rec, split)
    correct = v.predicted_code == rec["truth"]
    cache: dict = {}
    auto = gate_mask(v, g, rec["counts"][:, 1], rec["counts"][:, 3], rec["level"], cache=cache)
    months = np.array([d[:7] for d in rec["date"]])
    multi = vendor_diversity(rec, rec["truth"])

    def block(mask: np.ndarray) -> dict:
        n, a = int(mask.sum()), int((mask & auto).sum())
        ok = int((mask & auto & correct).sum())
        lo, hi = wilson_interval(ok, a)
        return {"lines": n, "auto_posted": a, "automation_rate": round(a / n, 4) if n else None,
                "auto_post_accuracy": round(ok / a, 4) if a else None, "wrong_auto_posts": a - ok,
                "accuracy_ci95": [round(lo, 4), round(hi, 4)] if a else None,
                "false_positive_rate_of_all_lines": round((a - ok) / n, 4) if n else None,
                "gl_accuracy_all_predictions": round(float(correct[mask].mean()), 4) if n else None}

    out = {"record": record, "k": k, "recency": rec_name, "gate": gate_to_dict(g),
           "tune": block(tune), "test": block(test), "all": block(np.ones_like(tune)),
           "by_month": [{"month": m, **block(months == m)} for m in sorted(set(months))],
           "by_vendor_code_diversity": {
               "test_single_code_vendors": block(test & ~multi), "test_multi_code_vendors": block(test & multi),
               "vendors_single": int(len({x for x, mu in zip(rec["vendor"], multi, strict=True) if not mu})),
               "vendors_multi": int(len({x for x, mu in zip(rec["vendor"], multi, strict=True) if mu}))},
           }
    for name, mask in (("test", test), ("all", np.ones_like(tune))):
        out[f"calibration_{name}"] = calibration_table(
            v.weighted_confidence[mask], correct[mask], auto[mask], CAL_BINS)
    return out


# --------------------------------------------------------------------------- driver

def git_info() -> dict:
    def run(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD"), "working_tree_dirty": bool(run("status", "--porcelain"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=("offline", "live"), default="offline")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--split", default=SPLIT, help="first date of the TEST window")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    t0 = time.time()
    fams = [f for f in args.families.split(",") if f]

    tasks, owner = [], []
    for fam in fams:
        spec = FAMILIES[fam]
        for record, ks in spec["records"]:
            if not record_path(args.embedder, record).exists():
                print(f"missing recording {record} for {args.embedder}; skipping", flush=True)
                continue
            for k in ks:
                for rn in spec["recency"]:
                    tasks.append((args.embedder, record, k, rn, spec["grid"], args.split))
                    owner.append(fam)
    print(f"{len(tasks)} feature sets", flush=True)
    results: dict[str, list[dict]] = {f: [] for f in fams}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fam, rows in zip(owner, pool.map(sweep_task, tasks, chunksize=1), strict=True):
            results[fam].extend(rows)
    print(f"swept in {time.time() - t0:.0f}s: " + ", ".join(f"{f}={len(r)} front rows" for f, r in results.items()),
          flush=True)

    # shipped baseline on the vendor_k10 recording
    selected: dict = {"baseline": {"shipped": detail(args.embedder, "vendor_k10", 10, "none", SHIPPED, args.split)}}
    selected["baseline"]["shipped"]["settings"] = to_settings("vendor_k10", 10, "none", SHIPPED)
    curves: dict = {}
    for fam, rows in results.items():
        if not rows:
            continue
        selected[fam] = {}
        # "honest" curve: thresholds chosen on TUNE for each target, evaluated on TEST
        honest, oracle = [], []
        for t in CURVE_TARGETS:
            r = pick(rows, t, "tune")
            if r:
                cov, acc, n, w = point(r, "test")
                honest.append({"target_accuracy": t, "test_coverage": cov, "test_accuracy": acc, "test_n_auto": n,
                               "test_wrong": w, "tune_coverage": point(r, "tune")[0],
                               "tune_accuracy": point(r, "tune")[1], "config": {k: r[k] for k in ("record", "k", "recency")}
                               | {"gate": r["gate"]}})
            ro = pick(rows, t, "test")
            if ro:
                cov, acc, n, w = point(ro, "test")
                oracle.append({"target_accuracy": t, "test_coverage": cov, "test_accuracy": acc, "test_n_auto": n,
                               "test_wrong": w})
        curves[fam] = {"honest_tuned_on_tune_window": honest, "oracle_tuned_on_test_window": oracle}
        for t in TARGETS:
            r = pick(rows, t, "tune")
            if r is None:
                selected[fam][tkey(t)] = None
                continue
            g = gate_from_dict(r["gate"])
            d = detail(args.embedder, r["record"], r["k"], r["recency"], g, args.split)
            d["settings"] = to_settings(r["record"], r["k"], r["recency"], g)
            d["target_accuracy_on_tune"] = t
            selected[fam][tkey(t)] = d
        if fam in ("context", "recency", "combined", "context_gate", "recency_gate"):  # shipped thresholds on the feature set chosen for the 98% target
            ref = selected[fam].get("tuned_98")
            if ref:
                d = detail(args.embedder, ref["record"], ref["k"], ref["recency"], SHIPPED, args.split)
                d["settings"] = to_settings(ref["record"], ref["k"], ref["recency"], SHIPPED)
                selected[fam]["shipped_thresholds"] = d
    # single-score reference curve: rank by the production weighted confidence
    rec, v = _feature_set((args.embedder, "vendor_k10", 10, "none"))
    tune, test = windows(rec, args.split)
    correct = v.predicted_code == rec["truth"]
    for nm, mask in (("test", test), ("all", np.ones_like(test))):
        cur = risk_coverage_curve(v.weighted_confidence[mask], correct[mask], MIN_POSTS)
        step = max(1, len(cur) // 400)
        curves[f"baseline_confidence_ranked_{nm}"] = cur[::step]

    out = {
        "embedder": args.embedder, "split_test_from": args.split, "min_posts": MIN_POSTS, "targets": list(TARGETS),
        "windows": {"tune": f"before {args.split}", "test": f"{args.split} onwards"},
        "recencies": [{"name": r[0], "halflife_days": r[1], "window_days": r[2]} for r in RECENCY],
        "families": {f: {"records": [r for r, _ in FAMILIES[f]["records"]], "grid": FAMILIES[f]["grid"]} for f in fams},
        "selected": selected, "curves": curves, "git": git_info(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (SWEEP_DIR / f"gate_sweep_{args.embedder}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote data/public/gate_sweep_{args.embedder}.json  ({time.time() - t0:.0f}s)")
    for fam, d in selected.items():
        for name, e in d.items():
            if e is None:
                print(f"{fam:9s} {name:18s} -> not reachable")
                continue
            t, s = e["tune"], e["test"]
            print(f"{fam:9s} {name:18s} {e['record']}/k{e['k']}/{e['recency']:6s} "
                  f"TUNE cov {t['automation_rate']:.3f} acc {t['auto_post_accuracy']:.4f} | "
                  f"TEST cov {s['automation_rate']:.3f} acc {s['auto_post_accuracy']:.4f} "
                  f"CI {s['accuracy_ci95']} n={s['auto_posted']}")
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONUTF8", "1")
    _ = CONTEXT_LEVELS
    raise SystemExit(main())
