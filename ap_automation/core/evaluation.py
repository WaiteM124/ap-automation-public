"""
POC evaluation: measures - rather than asserts - what the system claims.

Claims under test
  1. Automation rate improves through write-back alone  (with-learning vs no-learning control)
  2. Novelty is handled      (cold-start and unknown-vendor invoices are routed to review)
  3. Governance is enforced  (injected VAT / entity / cost-centre anomalies are caught, and
                              nothing can be auto-posted that the governed gate rejected)
  4. Auto-posts are accurate (false-positive rate among auto-posted invoices is MEASURED)

Method: a seeded corpus of history for the known vendors, then a stream of invoices in
random order. Invoices routed to review are resolved by a SIMULATED reviewer that applies
the ground-truth GL (confirm if the prediction was right, correct if not) - the only source
of learning. Auto-posted invoices are never written back. The control run is identical but
never learns.

Offline mode uses an in-memory Qdrant and a lexical hashing embedder: it validates the
mechanics of the loop, not the semantic quality of the real embedding model.
"""

from __future__ import annotations

import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ap_automation.core import dataset, mdm, offline, pipeline, posting
from ap_automation.core.config import settings
from ap_automation.core.models import (
    ConfirmationSource,
    InvoiceLine,
    RoutingDecision,
)
from ap_automation.core.pipeline import AutoPostNotPermitted, predict
from ap_automation.core.seeding import seed_corpus

SIMULATED_REVIEWER = "simulated_reviewer"


@dataclass
class EvalConfig:
    n_routine: int = 600
    n_near_edge: int = 150
    n_hard_edge: int = 50
    seed: int = 42
    history_per_vendor: int = 15
    window: int = 100
    live_embeddings: bool = False  # use OpenAI embeddings (needs OPENAI_API_KEY) instead of hashing


@dataclass
class Record:
    index: int
    vendor_id: str
    category: str
    anomaly: str | None
    history_before: int           # human-coded lines the corpus held for this vendor
    predicted_gl: str
    truth_gl: str
    rule_routing: str
    final_routing: str
    flags: list[str]
    reason: str
    learned: bool = False
    bypass_blocked: bool | None = None  # for review-routed rows: was a forced auto-post refused?

    @property
    def review_code(self) -> str:
        """Primary reason an invoice went to review ('' if it was auto-posted)."""
        if self.auto:
            return ""
        if self.rule_routing == RoutingDecision.AUTO_POST.value:   # rule gate passed; agents objected
            return self.flags[0].split(":")[0] if self.flags else "llm_escalation"
        text = self.reason
        for needle, code in (
            ("Sparse retrieval", "no_or_thin_history"),
            ("Retrieval unavailable", "retrieval_unavailable"),
            ("Vendor identity uncertain", "vendor_identity_uncertain"),
            ("Evidence disagrees", "evidence_disagrees"),
            ("below threshold", "low_confidence"),
        ):
            if needle in text:
                return code
        return "other"

    @property
    def auto(self) -> bool:
        return self.final_routing == RoutingDecision.AUTO_POST.value

    @property
    def correct(self) -> bool:
        return self.predicted_gl == self.truth_gl


def run_stream(invoices: list[InvoiceLine], history: list[InvoiceLine], vendor_rows: list[dict],
               learn: bool, cfg: EvalConfig) -> list[Record]:
    """Run one pass over the stream against a fresh in-memory corpus."""
    mdm.set_vendor_map(vendor_rows)
    offline.use_offline_backend(hashing=not cfg.live_embeddings)
    history_counts: Counter[str] = Counter(seed_corpus(history))
    records: list[Record] = []

    for i, inv in enumerate(invoices):
        pred = predict(inv, batch_mode=True)
        vendor = pred.resolved_line.canonical_vendor_id
        rec = Record(
            index=i, vendor_id=vendor, category=inv.category.value, anomaly=inv.anomaly,
            history_before=history_counts[vendor], predicted_gl=pred.predicted_gl,
            truth_gl=inv.ground_truth_gl or "", rule_routing=pred.routing_decision.value,
            final_routing=pred.final_routing.value, flags=list(pred.agent_flags),
            reason=pred.final_routing_reason,
        )

        if pred.final_routing == RoutingDecision.AUTO_POST:
            res = posting.post_outcome(
                pred, pred.predicted_gl, pred.predicted_cost_centre,
                settings.auto_post_user_id, ConfirmationSource.AUTO_POST, mirror_to_sheet=False,
            )
            rec.learned = res.learned  # always False: the system does not learn from itself
        else:
            # Adversarial check: a caller trying to force-post a rejected prediction must be refused.
            try:
                posting.post_outcome(pred, pred.predicted_gl, pred.predicted_cost_centre,
                                     "bypass_attempt", ConfirmationSource.AUTO_POST, mirror_to_sheet=False)
                rec.bypass_blocked = False
            except AutoPostNotPermitted:
                rec.bypass_blocked = True

            if learn and inv.ground_truth_gl:
                source = (ConfirmationSource.HUMAN_CONFIRM if pred.predicted_gl == inv.ground_truth_gl
                          else ConfirmationSource.HUMAN_CORRECT)
                res = posting.post_outcome(
                    pred, inv.ground_truth_gl, pred.predicted_cost_centre or (inv.cost_centre or ""),
                    SIMULATED_REVIEWER, source, mirror_to_sheet=False,
                )
                rec.learned = res.learned
                if res.learned:
                    history_counts[vendor] += 1
        records.append(rec)
    return records


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def summarise(records: list[Record], window: int) -> dict:
    auto = [r for r in records if r.auto]
    wrong_auto = [r for r in auto if not r.correct]

    windows = []
    for start in range(0, len(records), window):
        chunk = records[start:start + window]
        a = [r for r in chunk if r.auto]
        windows.append({
            "invoices": f"{start + 1}-{start + len(chunk)}",
            "automation_rate": _rate(len(a), len(chunk)),
            "auto_post_accuracy": _rate(sum(r.correct for r in a), len(a)),
        })

    cold = [r for r in records if r.history_before == 0]
    by_category: dict[str, dict] = {}
    for cat in sorted({r.category for r in records}):
        rows = [r for r in records if r.category == cat]
        by_category[cat] = {
            "n": len(rows),
            "automation_rate": _rate(sum(r.auto for r in rows), len(rows)),
        }

    anomalies: dict[str, dict] = {}
    for kind in sorted({r.anomaly for r in records if r.anomaly}):
        rows = [r for r in records if r.anomaly == kind]
        anomalies[kind] = {"n": len(rows), "routed_to_review": sum(not r.auto for r in rows)}
    anomaly_rows = [r for r in records if r.anomaly]

    # Per new-to-corpus vendor: how many invoices were reviewed before the first auto-post?
    seen: dict[str, int] = defaultdict(int)
    first_auto: dict[str, int] = {}
    for r in records:
        if r.category == "near_edge" and r.vendor_id in {v.canonical_id for v in dataset.NEW_TO_CORPUS_VENDORS}:
            seen[r.vendor_id] += 1
            if r.auto and r.vendor_id not in first_auto:
                first_auto[r.vendor_id] = seen[r.vendor_id]

    bypass = [r for r in records if r.bypass_blocked is not None]

    # The subset where write-back is the ONLY thing that can change the outcome:
    # vendors that are in the vendor master but began with no history.
    learnable_ids = {v.canonical_id for v in dataset.NEW_TO_CORPUS_VENDORS}
    learnable = [r for r in records if r.vendor_id in learnable_ids]

    return {
        "review_reasons": dict(Counter(r.review_code for r in records if not r.auto).most_common()),
        "new_to_corpus_vendors": {
            "n": len(learnable),
            "automation_rate": _rate(sum(r.auto for r in learnable), len(learnable)),
        },
        "invoices": len(records),
        "automation_rate": _rate(len(auto), len(records)),
        "auto_posted": len(auto),
        "auto_post_accuracy": _rate(len(auto) - len(wrong_auto), len(auto)),
        "false_positives": len(wrong_auto),
        "false_positive_rate_of_auto_posts": _rate(len(wrong_auto), len(auto)),
        "false_positive_rate_of_all_invoices": _rate(len(wrong_auto), len(records)),
        "gl_accuracy_all_predictions": _rate(sum(r.correct for r in records), len(records)),
        "lines_learned_from_auto_posts": sum(1 for r in auto if r.learned),
        "cold_start": {
            "n": len(cold), "routed_to_review": sum(not r.auto for r in cold),
            "review_rate": _rate(sum(not r.auto for r in cold), len(cold)),
        },
        "by_category": by_category,
        "governance": {
            "injected_anomalies": len(anomaly_rows),
            "caught": sum(not r.auto for r in anomaly_rows),
            "catch_rate": _rate(sum(not r.auto for r in anomaly_rows), len(anomaly_rows)),
            "by_type": anomalies,
        },
        "enforcement": {
            "forced_auto_post_attempts": len(bypass),
            "refused": sum(1 for r in bypass if r.bypass_blocked),
        },
        "new_vendor_invoices_reviewed_before_first_auto_post": first_auto,
        "windows": windows,
    }


def run_evaluation(cfg: EvalConfig | None = None) -> dict:
    """Run the with-learning pass and the no-learning control; return both summaries."""
    cfg = cfg or EvalConfig()
    settings.telemetry_enabled = False  # never write evaluation traffic to Langfuse / Supabase
    pipeline._lf_client, pipeline._lf_init_done = None, False

    invoices, vendor_rows = dataset.generate_invoices(
        cfg.n_routine, cfg.n_near_edge, cfg.n_hard_edge, cfg.seed)
    history = dataset.generate_history(cfg.history_per_vendor)

    saved = (settings.erp_db_path, settings.audit_db_path, settings.erp_url)
    saved_backend = settings.erp_backend
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            out: dict = {"config": cfg.__dict__ | {
                "embeddings": "openai" if cfg.live_embeddings else "offline-hashing",
                # added later than the first result files, which only said "openai": record the exact model
                "embedding_model": (settings.embedding_model if cfg.live_embeddings
                                    else "hashing (crc32 unigrams+bigrams, lexical only)"),
                "embedding_dimensions": settings.embedding_dimensions,
            }}
            for name, learn in (("with_learning", True), ("no_learning_control", False)):
                settings.erp_db_path = str(Path(tmp) / f"{name}_erp.db")
                settings.audit_db_path = str(Path(tmp) / f"{name}_audit.db")
                settings.erp_url = ""
                settings.erp_backend = "simulated"  # an evaluation never posts to SAP or Xero or Xero
                posting._db_ready = False
                recs = run_stream(invoices, history, vendor_rows, learn, cfg)
                out[name] = summarise(recs, cfg.window)
                out[name]["_records"] = recs
    finally:
        settings.erp_db_path, settings.audit_db_path, settings.erp_url = saved
        settings.erp_backend = saved_backend
        posting._db_ready = False
        offline.reset_backend()
    return out


@dataclass
class Verdict:
    claim: str
    passed: bool
    detail: str


def verdicts(result: dict) -> list[Verdict]:
    """Turn the measurements into pass/fail statements about the README claims."""
    w, c = result["with_learning"], result["no_learning_control"]
    wl, cl = w["new_to_corpus_vendors"], c["new_to_corpus_vendors"]
    out = [
        Verdict(
            "Automation improves through write-back alone",
            (w["automation_rate"] or 0) > (c["automation_rate"] or 0)
            and (wl["automation_rate"] or 0) > (cl["automation_rate"] or 0),
            f"overall {w['automation_rate']:.1%} vs {c['automation_rate']:.1%} for the no-learning control; "
            f"vendors that began with no history: {wl['automation_rate']:.1%} vs {cl['automation_rate']:.1%}",
        ),
        Verdict(
            "Cold-start invoices are routed to review",
            w["cold_start"]["review_rate"] == 1.0,
            f"{w['cold_start']['routed_to_review']}/{w['cold_start']['n']} vendors-with-no-history routed to review",
        ),
        Verdict(
            "Unknown-vendor (hard-edge) invoices are routed to review",
            w["by_category"].get("hard_edge", {}).get("automation_rate") == 0.0,
            f"hard-edge automation rate {w['by_category'].get('hard_edge', {}).get('automation_rate'):.0%}",
        ),
        Verdict(
            "Injected governance anomalies are caught",
            w["governance"]["catch_rate"] == 1.0,
            f"{w['governance']['caught']}/{w['governance']['injected_anomalies']} routed to review",
        ),
        Verdict(
            "A rejected prediction cannot be force-posted",
            w["enforcement"]["forced_auto_post_attempts"] == w["enforcement"]["refused"],
            f"{w['enforcement']['refused']}/{w['enforcement']['forced_auto_post_attempts']} forced attempts refused",
        ),
        Verdict(
            "The system never learns from its own auto-posts",
            w["lines_learned_from_auto_posts"] == 0,
            f"{w['lines_learned_from_auto_posts']} auto-posted lines were written back",
        ),
    ]
    return out
