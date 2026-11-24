"""
CI eval gate: run the offline POC evaluation and compare to docs/evaluation.json.

    python scripts/check_eval_gate.py

Uses the same defaults as scripts/evaluate_poc.py (seed 42, lexical embedder,
in-memory Qdrant). No API keys. Exit 0 only if:

  1. All built-in eval claims pass (same as evaluate_poc.py).
  2. with_learning auto_post_accuracy does not fall more than ACCURACY_TOLERANCE
     below the baseline in docs/evaluation.json.
  3. Automation rate does not rise (by more than AUTOMATION_TOLERANCE) while
     accuracy falls below baseline (any drop past zero within float noise).

Tolerance (documented for CI and AGENTS.md)
------------------------------------------
Observed offline runs can differ from the checked-in baseline by ~1 auto-post
false positive on 589 posts (~0.0017 absolute accuracy) while automation_rate
stays identical. ACCURACY_TOLERANCE = 0.002 (0.2 percentage points) absorbs that
drift without allowing a material regression. AUTOMATION_TOLERANCE = 0.01 (1 pp)
defines “rises” for the dual condition. Do not loosen these to green CI — change
the baseline via an intentional eval PR and ADR if the product bar moves.

Runtime is ~10–20s on GitHub-hosted runners, so this job runs on every PR/push
alongside lint and pytest (not path-filtered).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ap_automation.core.evaluation import EvalConfig, run_evaluation, verdicts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "docs" / "evaluation.json"

# Absolute tolerances on rates in [0, 1]. See module docstring.
ACCURACY_TOLERANCE = 0.002
AUTOMATION_TOLERANCE = 0.01


def _load_baseline() -> dict:
    if not BASELINE_PATH.is_file():
        raise SystemExit(f"Missing baseline: {BASELINE_PATH}")
    return json.loads(BASELINE_PATH.read_text())


def compare_to_baseline(result: dict, baseline: dict) -> list[str]:
    """Return human-readable failure reasons (empty = pass)."""
    failures: list[str] = []
    cur = result["with_learning"]
    base = baseline["with_learning"]

    cur_acc = float(cur["auto_post_accuracy"])
    base_acc = float(base["auto_post_accuracy"])
    cur_auto = float(cur["automation_rate"])
    base_auto = float(base["automation_rate"])

    if cur_acc < base_acc - ACCURACY_TOLERANCE:
        failures.append(
            f"auto_post_accuracy {cur_acc:.4f} is below baseline {base_acc:.4f} "
            f"by more than ACCURACY_TOLERANCE={ACCURACY_TOLERANCE}"
        )

    accuracy_fell = cur_acc < base_acc - 1e-12
    automation_rose = cur_auto > base_auto + AUTOMATION_TOLERANCE
    if automation_rose and accuracy_fell:
        failures.append(
            f"automation_rate rose ({base_auto:.4f} -> {cur_auto:.4f}, "
            f"tol={AUTOMATION_TOLERANCE}) while auto_post_accuracy fell "
            f"({base_acc:.4f} -> {cur_acc:.4f})"
        )

    return failures


def main() -> int:
    baseline = _load_baseline()
    cfg = baseline.get("config") or {}
    eval_cfg = EvalConfig(
        n_routine=int(cfg.get("n_routine", 600)),
        n_near_edge=int(cfg.get("n_near_edge", 150)),
        n_hard_edge=int(cfg.get("n_hard_edge", 50)),
        seed=int(cfg.get("seed", 42)),
        history_per_vendor=int(cfg.get("history_per_vendor", 15)),
        window=int(cfg.get("window", 100)),
        live_embeddings=False,
    )
    if cfg.get("live_embeddings"):
        print("NOTE: baseline recorded live_embeddings=true; gate still forces offline.")

    print(
        f"Eval gate: seed={eval_cfg.seed} offline lexical embedder, "
        f"invoices={eval_cfg.n_routine + eval_cfg.n_near_edge + eval_cfg.n_hard_edge}, "
        f"baseline={BASELINE_PATH.relative_to(ROOT)}"
    )
    print(
        f"Tolerances: ACCURACY_TOLERANCE={ACCURACY_TOLERANCE} "
        f"AUTOMATION_TOLERANCE={AUTOMATION_TOLERANCE}"
    )

    result = run_evaluation(eval_cfg)
    w = result["with_learning"]
    b = baseline["with_learning"]
    print(
        f"Result   automation={w['automation_rate']:.4f} "
        f"auto_post_accuracy={w['auto_post_accuracy']:.4f} "
        f"false_positives={w['false_positives']}/{w['auto_posted']}"
    )
    print(
        f"Baseline automation={b['automation_rate']:.4f} "
        f"auto_post_accuracy={b['auto_post_accuracy']:.4f} "
        f"false_positives={b['false_positives']}/{b['auto_posted']}"
    )

    failed = 0
    print("\nClaims")
    for v in verdicts(result):
        failed += not v.passed
        print(f"  [{'PASS' if v.passed else 'FAIL'}] {v.claim}: {v.detail}")

    print("\nBaseline comparison")
    for reason in compare_to_baseline(result, baseline):
        failed += 1
        print(f"  [FAIL] {reason}")
    if failed == 0:
        print("  [PASS] accuracy within tolerance; no automation↑/accuracy↓ regression")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
