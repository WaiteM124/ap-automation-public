"""
Closed-loop confirmation of tuned gate configurations on the Checkbook L.A. replay, through the REAL pipeline.

    python scripts/gate_record.py --embedder offline
    python scripts/gate_sweep.py  --embedder offline          # chooses configurations on the TUNE window (Oct-Feb)
    python scripts/tuned_replay.py --embedder offline         # -> docs/public_replay_tuned_offline.json, docs/gate_curve.json
    python scripts/tuned_replay.py --embedder live            # OpenAI text-embedding-3-small (cached vectors), -> ..._live.json

Each arm replays the same stream as scripts/replay_public_data.py (seed = first 3 months, 10 vendors held out as
new-to-corpus, strictly date-ordered, the clerk's true code written back ONLY for lines routed to human review,
auto-posts never learned) with the gate configuration switched on via core.config.settings overrides. The control
arm replays the same configuration with write-back off. Metrics are reported separately for the TUNE window the
thresholds were chosen on and the later TEST window they were not. Nothing is committed; the raw data stays in the
gitignored data/public/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_sweep import CAL_BINS, SPLIT, SWEEP_DIR  # noqa: E402
from public_data_import import (  # noqa: E402
    ATTRIBUTION,
    DATASET_ID,
    DATASET_NAME,
    DATASET_URL,
    LEAK_COLUMNS,
    LICENCE,
    ensure_sample,
)
from replay_public_data import (  # noqa: E402
    CACHE_DIR,
    ReplayRecord,
    calibration,
    git_info,
    precompute_live_embeddings,
    run_arm,
    split_seed_replay,
)

from ap_automation.core.config import settings  # noqa: E402
from ap_automation.core.gate_analysis import wilson_interval  # noqa: E402
from ap_automation.core.preprocessing import preprocess_description  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARMS = [
    "baseline/shipped",
    "context/shipped_thresholds",
    "gate/tuned_99",
    "recency/shipped_thresholds",
    "context_gate/tuned_99",
    "combined/tuned_98",
    "combined/tuned_99",
    "combined/tuned_99.5",
    "context_gate/tuned_99.5",
]
CONTROL_ARMS = ["baseline/shipped", "combined/tuned_99", "context_gate/tuned_99"]


def _rate(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def block(records: list[ReplayRecord]) -> dict:
    n = len(records)
    auto = [r for r in records if r.auto]
    ok = sum(r.correct for r in auto)
    lo, hi = wilson_interval(ok, len(auto))
    return {
        "lines": n, "auto_posted": len(auto), "automation_rate": _rate(len(auto), n),
        "auto_post_accuracy": _rate(ok, len(auto)),
        "auto_post_accuracy_ci95": [round(lo, 4), round(hi, 4)] if auto else None,
        "false_positives": len(auto) - ok, "false_positive_rate_of_all_lines": _rate(len(auto) - ok, n),
        "gl_accuracy_all_predictions": _rate(sum(r.correct for r in records), n),
        "review_rows": n - len(auto),
        "review_rows_that_were_correct": sum(r.correct for r in records if not r.auto),
    }


def arm_metrics(res: dict, split_date: str) -> dict:
    recs = [ReplayRecord(**r) for r in res["records"]]
    tune = [r for r in recs if r.date < split_date]
    test = [r for r in recs if r.date >= split_date]
    first: dict[str, ReplayRecord] = {}
    for r in recs:
        first.setdefault(r.vendor_id, r)
    new_v = {v for v, r in first.items() if r.history_before == 0}
    codes_by_vendor: dict[str, set] = defaultdict(set)
    for r in recs:
        codes_by_vendor[r.vendor_id].add(r.truth_gl)
    multi = {v for v, c in codes_by_vendor.items() if len(c) > 1}
    months: dict[str, list[ReplayRecord]] = defaultdict(list)
    for r in recs:
        months[r.date[:7]].append(r)
    first_auto: dict[str, int] = {}
    seen: Counter[str] = Counter()
    for r in recs:
        if r.vendor_id in new_v:
            seen[r.vendor_id] += 1
            if r.auto and r.vendor_id not in first_auto:
                first_auto[r.vendor_id] = seen[r.vendor_id] - 1
    ctx_levels = Counter(r.context_level for r in recs if r.context_level)
    return {
        "all": block(recs), "tune_window": block(tune), "test_window": block(test),
        "learning_curve_by_month": [{"month": m, **block(rs)} for m, rs in sorted(months.items())],
        "by_vendor_code_diversity_test_window": {
            "single_code_vendors": block([r for r in test if r.vendor_id not in multi]),
            "multi_code_vendors": block([r for r in test if r.vendor_id in multi]),
            "vendors_single": len(codes_by_vendor) - len(multi), "vendors_multi": len(multi)},
        "new_to_corpus_vendors": {
            "vendors": len(new_v), "all": block([r for r in recs if r.vendor_id in new_v]),
            "test_window": block([r for r in test if r.vendor_id in new_v]),
            "vendors_that_ever_auto_posted": len(first_auto),
            "reviews_before_first_auto_post_median": (
                sorted(first_auto.values())[len(first_auto) // 2] if first_auto else None)},
        "calibration_all": calibration(recs), "calibration_test_window": calibration(test),
        "evidence_context_levels": dict(ctx_levels),
        "lines_learned": sum(r.learned for r in recs),
        "audited_auto_posts": sum(r.audited for r in recs),
        "audit_share_of_all_lines": _rate(sum(r.audited for r in recs), len(recs)),
        "review_reasons": dict(Counter(r.review_code for r in recs if not r.auto).most_common()),
        "seconds": res["summary"]["seconds"],
    }


def _arm_job(payload: dict) -> dict:
    return run_arm(payload)


def split_arm(name: str) -> tuple[str, float]:
    """'combined/tuned_99@a30' -> ('combined/tuned_99', 0.30): the suffix is the QA audit-sampling rate (percent)."""
    base, _, audit = name.partition("@a")
    return base, (float(audit) / 100.0 if audit else 0.0)


def resolve_arm(sweep: dict, name: str) -> dict:
    fam, key = split_arm(name)[0].split("/")
    entry = sweep["selected"][fam][key]
    if entry is None:
        raise SystemExit(f"{name}: no configuration reached its target on the tune window")
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=("offline", "live"), default="offline")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--controls", default=",".join(CONTROL_ARMS))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit-replay", type=int, default=0)
    ap.add_argument("--max-live-usd", type=float, default=0.50)
    ap.add_argument("--max-live-tokens", type=int, default=20_000_000)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--merge", action="store_true", help="add these arms to the existing output file")
    ap.add_argument("--curves-only", action="store_true", help="rebuild docs/gate_curve.json from existing outputs")
    args = ap.parse_args()
    live = args.embedder == "live"
    settings.telemetry_enabled = False

    sweep_file = SWEEP_DIR / f"gate_sweep_{args.embedder}.json"
    sweep = json.loads(sweep_file.read_text())
    split_date = sweep["split_test_from"]
    if args.curves_only:  # rebuild docs/gate_curve.json from an existing replay file, no replay
        out_json = Path(args.json) if args.json else REPO_ROOT / "docs" / f"public_replay_tuned_{args.embedder}.json"
        out = json.loads(out_json.read_text())
        for arm in out["arms"].values():  # arms replayed before audit logging existed: no audits by construction
            for kind in ("with_learning", "no_learning_control"):
                if kind in arm:
                    arm[kind].setdefault("audited_auto_posts", 0)
                    arm[kind].setdefault("audit_share_of_all_lines", 0.0)
        out["meta"]["sweep"] = {k: sweep[k] for k in ("families", "recencies", "min_posts", "targets", "windows")}
        out_json.write_text(json.dumps(out, indent=1, default=float))
        write_curves(args.embedder, out, sweep)
        print(f"rebuilt docs/gate_curve.json[{args.embedder}] from {out_json.name}")
        return 0
    arms = [a for a in args.arms.split(",") if a]
    controls = set(c for c in args.controls.split(",") if c)

    sample, report = ensure_sample(2026, 30_000, 250, 50, CACHE_DIR)
    split = split_seed_replay(sample, 3, 10, 42)
    embed_info: dict
    if live:
        texts = {preprocess_description(r["detailed_item_description"]) for r in sample}
        usage = precompute_live_embeddings(texts, args.max_live_tokens, args.max_live_usd)
        embed_info = {"kind": "openai", "model": settings.embedding_model, "dimensions": usage["dimensions"],
                      "usage": usage}
    else:
        embed_info = {"kind": "offline-hashing", "model": "hashing embedder (crc32 unigrams+bigrams, lexical only)",
                      "dimensions": settings.embedding_dimensions}

    base_cfg = {"seed_months": 3, "holdout_vendors": 10, "seed": 42, "agents": "neutral", "live_embeddings": live,
                "store": "inproc", "sim_clock": True, "clerk_cc": True}
    tasks, meta_arms = [], {}
    for a in arms:
        entry = resolve_arm(sweep, a)
        audit = split_arm(a)[1]
        meta_arms[a] = {k: entry[k] for k in ("record", "k", "recency", "gate", "settings")} | {"audit_rate": audit}
        for learn in (True, False):
            if not learn and a not in controls:
                continue
            tasks.append({"name": f"{a}|{'learn' if learn else 'control'}", "learn": learn,
                          "cfg": {**base_cfg, "settings": entry["settings"], "audit_rate": audit if learn else 0.0},
                          "rows": sample, "limit": args.limit_replay or None})
    print(f"{len(tasks)} replay arms on {args.workers} workers ({args.embedder})", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        results = list(pool.map(_arm_job, tasks, chunksize=1))
    elapsed = round(time.time() - t0, 1)

    out_arms: dict = {}
    for res in results:
        a, kind = res["name"].split("|")
        m = arm_metrics(res, split_date)
        out_arms.setdefault(a, {"config": meta_arms[a]})["with_learning" if kind == "learn" else "no_learning_control"] = m

    meta = {
        "dataset": DATASET_NAME, "dataset_id": DATASET_ID, "url": DATASET_URL, "licence": LICENCE,
        "attribution": ATTRIBUTION, "fiscal_year": 2026, "sample_params": report["sample_params"],
        "sample_stats": report["sample_stats"], "embedder": embed_info,
        "replay_params": {
            "seed_months": 3, "seed_cutoff_date": split["cutoff"], "seed_lines": len(split["seed"]),
            "replay_lines": len(split["replay"]), "held_out_vendor_ids": split["held_out_vendors"],
            "limit_replay": args.limit_replay or None,
            "write_back_policy": "true account_code applied as clerk confirmation (with the invoice's own fund) ONLY "
                                 "for lines routed to human review; auto-posts never written back; control never "
                                 "writes back",
            "clock": "simulated: evidence is dated by transaction date, the pipeline clock follows the replayed line",
            "llm_gate": "off (batch_mode=True)", "vector_store": "inproc vendor-indexed stand-in (see replay_public_data)"},
        "windows": {
            "tune": f"transaction_date < {split_date} (Oct-Feb): thresholds were chosen here, on the shadow-mode "
                    "recording (scripts/gate_sweep.py)",
            "test": f"transaction_date >= {split_date} (Mar-Jun): not used to choose anything"},
        "leak_guard_columns": list(LEAK_COLUMNS),
        "inputs_used": ["vendor_name", "detailed_item_description", "dollar_amount", "department_name (entity)",
                        "fund (cost centre)", "transaction_date (ordering / evidence age)"],
        "seed": 42, "git": git_info(), "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "wall_seconds_arms": elapsed,
        "sweep_file": f"data/public/gate_sweep_{args.embedder}.json (gitignored; summarised below)",
        "sweep_wall_seconds": sweep["wall_seconds"], "targets": sweep["targets"],
        "sweep": {k: sweep[k] for k in ("families", "recencies", "min_posts", "targets", "windows")},
    }
    shadow = {}
    for fam, d in sweep["selected"].items():
        shadow[fam] = {}
        for key, e in d.items():
            shadow[fam][key] = None if e is None else {
                k: e[k] for k in ("record", "k", "recency", "gate", "tune", "test", "by_month",
                                  "by_vendor_code_diversity", "calibration_test", "calibration_all", "settings")
                if k in e}
    out = {"meta": meta, "arms": out_arms, "shadow_mode_sweep": shadow}
    out_json = Path(args.json) if args.json else REPO_ROOT / "docs" / f"public_replay_tuned_{args.embedder}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if args.merge and out_json.exists():  # add these arms to an earlier run's file, keep its meta
        prev = json.loads(out_json.read_text())
        prev["arms"].update(out_arms)
        prev["meta"].setdefault("merged_runs", []).append(
            {"timestamp_utc": meta["timestamp_utc"], "arms": list(out_arms), "git": meta["git"],
             "wall_seconds_arms": elapsed})
        out = prev
    out_json.write_text(json.dumps(out, indent=1, default=float))

    if args.limit_replay:  # debug run: do not touch the shared curve file
        print(f"wrote {out_json} (debug, limited replay)")
        return 0
    write_curves(args.embedder, out, sweep)

    print(f"\nwrote {out_json} ({elapsed}s)")
    for a, v in out_arms.items():
        w = v["with_learning"]
        t = w["test_window"]
        c = v.get("no_learning_control", {}).get("test_window", {})
        print(f"{a:28s} TEST auto {t['automation_rate']:.3f} acc {t['auto_post_accuracy']} "
              f"CI {t['auto_post_accuracy_ci95']} fp {t['false_positives']} "
              f"(control auto {c.get('automation_rate')})")
    return 0


def write_curves(embedder: str, out: dict, sweep: dict) -> None:
    """Merge this embedder's coverage-vs-accuracy curves into docs/gate_curve.json."""
    meta = out["meta"]
    curve_file = REPO_ROOT / "docs" / "gate_curve.json"
    curves = json.loads(curve_file.read_text()) if curve_file.exists() else {}
    curves["meta"] = {k: meta[k] for k in ("dataset", "dataset_id", "url", "licence", "attribution", "sample_params",
                                           "seed", "git", "windows")}
    curves["meta"]["note"] = (
        "Two kinds of curve. SHADOW MODE (every earlier line coded by a human; scripts/gate_record.py): "
        "'honest_tuned_on_tune_window' = for each target accuracy, the config with the largest TUNE (Oct-Feb) coverage "
        "reaching it, measured on TEST (Mar-Jun); 'oracle_tuned_on_test_window' picks on TEST itself (a ceiling, not "
        "a held-out result). CLOSED LOOP (the real pipeline, learning only from reviewed lines, optional QA audit "
        "sampling of auto-posts): 'closed_loop_operating_points'. Families are defined in scripts/gate_sweep.py.")
    points = []
    for name, arm in out["arms"].items():
        w = arm["with_learning"]
        t = w["test_window"]
        points.append({
            "arm": name, "audit_rate": arm["config"].get("audit_rate", 0.0), "record": arm["config"]["record"],
            "k": arm["config"]["k"], "recency": arm["config"]["recency"], "gate": arm["config"]["gate"],
            "test_automation": t["automation_rate"], "test_auto_post_accuracy": t["auto_post_accuracy"],
            "test_auto_post_accuracy_ci95": t["auto_post_accuracy_ci95"], "test_auto_posted": t["auto_posted"],
            "test_false_positives": t["false_positives"],
            "tune_automation": w["tune_window"]["automation_rate"],
            "tune_auto_post_accuracy": w["tune_window"]["auto_post_accuracy"],
            "audited_share_of_all_lines": w.get("audit_share_of_all_lines", 0.0) or 0.0})
    base_curve = sweep["curves"].get("baseline_confidence_ranked_test", [])
    ranked = {}
    for tgt in (0.95, 0.98, 0.99):
        ok = [r for r in base_curve if r["accuracy"] >= tgt and r["n_auto"] >= sweep["min_posts"]]
        best = max(ok, key=lambda r: r["coverage"]) if ok else None
        ranked[f"acc_ge_{int(tgt * 100)}"] = None if best is None else {
            "coverage": round(best["coverage"], 4), "accuracy": round(best["accuracy"], 4),
            "threshold": round(best["threshold"], 4)}
    curves[embedder] = {
        "embedder": meta["embedder"].get("model"), "timestamp_utc": meta["timestamp_utc"],
        "split_test_from": sweep["split_test_from"], "min_posts": sweep["min_posts"], "targets": sweep["targets"],
        "families": sweep["families"], "recency_options": sweep["recencies"], "sweep_git": sweep["git"],
        "sweep_timestamp_utc": sweep["timestamp_utc"], "curves": sweep["curves"],
        "coverage_at_accuracy_shadow_mode": _coverage_table(sweep["curves"], sweep["split_test_from"]),
        "baseline_confidence_ranked_test_window_oracle": ranked,
        "closed_loop_operating_points": points}
    curve_file.write_text(json.dumps(curves, indent=1, default=float))


def _coverage_table(curves: dict, split_date: str) -> dict:
    """Test-window coverage at which each family's tune-chosen operating points reach 95 / 98 / 99 % accuracy."""
    out: dict = {}
    for fam, c in curves.items():
        if not isinstance(c, dict) or "honest_tuned_on_tune_window" not in c:
            continue
        for kind in ("honest_tuned_on_tune_window", "oracle_tuned_on_test_window"):
            rows = c[kind]
            res = {}
            for t in (0.95, 0.98, 0.99):
                ok = [r for r in rows if r["test_accuracy"] >= t]
                best = max(ok, key=lambda r: r["test_coverage"]) if ok else None
                res[f"acc_ge_{int(t * 100)}"] = None if best is None else {
                    "test_coverage": round(best["test_coverage"], 4), "test_accuracy": round(best["test_accuracy"], 4),
                    "target_used": best["target_accuracy"]}
            out.setdefault(fam, {})[kind] = res
    return out


if __name__ == "__main__":
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONUTF8", "1")
    _ = CAL_BINS, SPLIT
    raise SystemExit(main())
