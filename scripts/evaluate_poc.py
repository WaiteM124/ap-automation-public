"""
Run the POC evaluation and print measured results.

    python scripts/evaluate_poc.py                       # offline: no network, no cost
    python scripts/evaluate_poc.py --live-embeddings     # real OpenAI embeddings (small cost)
    python scripts/evaluate_poc.py --json results.json   # also write full results

Exit code is 1 if any claim fails, so this can gate CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ap_automation.core.evaluation import EvalConfig, run_evaluation, verdicts  # noqa: E402


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routine", type=int, default=600)
    ap.add_argument("--near-edge", type=int, default=150)
    ap.add_argument("--hard-edge", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--history-per-vendor", type=int, default=15)
    ap.add_argument("--live-embeddings", action="store_true")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    cfg = EvalConfig(args.routine, args.near_edge, args.hard_edge, args.seed,
                     args.history_per_vendor, args.window, args.live_embeddings)
    result = run_evaluation(cfg)
    w, c = result["with_learning"], result["no_learning_control"]

    print(f"\nEmbeddings: {result['config']['embeddings']}   invoices: {w['invoices']}   seed: {cfg.seed}\n")
    print(f"{'window':<12}{'with learning':>16}{'no-learning control':>22}")
    for a, b in zip(w["windows"], c["windows"]):
        print(f"{a['invoices']:<12}{pct(a['automation_rate']):>16}{pct(b['automation_rate']):>22}")

    print(f"\nAutomation rate          {pct(w['automation_rate'])}  (control {pct(c['automation_rate'])})")
    print(f"Auto-post accuracy       {pct(w['auto_post_accuracy'])}  "
          f"({w['false_positives']} wrong of {w['auto_posted']} auto-posts)")
    print(f"False positives / all    {pct(w['false_positive_rate_of_all_invoices'])}")
    print("By category              " + "  ".join(
        f"{k}: {pct(v['automation_rate'])} auto" for k, v in w["by_category"].items()))
    print(f"Anomalies caught         {w['governance']['caught']}/{w['governance']['injected_anomalies']}  "
          + "  ".join(f"{k} {v['routed_to_review']}/{v['n']}" for k, v in w["governance"]["by_type"].items()))
    print(f"Reviews before 1st auto  {w['new_vendor_invoices_reviewed_before_first_auto_post']}")
    print(f"New-to-corpus vendors    {pct(w['new_to_corpus_vendors']['automation_rate'])} auto  "
          f"(control {pct(c['new_to_corpus_vendors']['automation_rate'])})")
    print("Why invoices go to review: " + ", ".join(f"{k} {v}" for k, v in w["review_reasons"].items()))

    print("\nClaims")
    failed = 0
    all_verdicts = verdicts(result)
    for v in all_verdicts:
        failed += not v.passed
        print(f"  [{'PASS' if v.passed else 'FAIL'}] {v.claim}: {v.detail}")
    result["claims"] = [{"claim": v.claim, "passed": v.passed, "detail": v.detail} for v in all_verdicts]

    if args.json:
        for k in ("with_learning", "no_learning_control"):
            result[k].pop("_records", None)
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"\nFull results written to {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
