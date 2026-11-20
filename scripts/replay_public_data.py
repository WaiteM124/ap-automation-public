"""
Replay evaluation on real public data: Checkbook L.A. (City of Los Angeles), see scripts/public_data_import.py.

    python scripts/replay_public_data.py                         # offline hashing embedder -> docs/public_replay_offline.json
    python scripts/replay_public_data.py --live-embeddings       # OpenAI text-embedding-3-small -> docs/public_replay_live.json

It drives the REAL pipeline (core.pipeline.predict, core.posting.post_outcome, core.seeding.seed_corpus) exactly as
the synthetic harness (core/evaluation.py) does:

  * the earliest --seed-months of the sample are indexed as confirmed history (source human_confirm);
  * the remainder is replayed strictly in transaction_date order;
  * after each replayed line the TRUE account_code is applied as the clerk's confirmation, but ONLY for lines
    the governed gate routed to human review - auto-posts are never written back (the control arm never learns);
  * a no-learning control replays the same stream against the same seed with write-back off.

Only the label (account_code) is ever ground truth. Leak guard: account_name / expenditure_type / item_code /
item_code_name never reach the invoice (see public_data_import.row_to_invoice and tests/test_public_data.py).

LLM gate is OFF (batch_mode=True). Telemetry is off. Nothing here is committed or pushed; raw data stays under
data/public/ (gitignored).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from public_data_import import (  # noqa: E402
    ATTRIBUTION,
    DATASET_ID,
    DATASET_NAME,
    DATASET_URL,
    LEAK_COLUMNS,
    LICENCE,
    ensure_sample,
    row_to_invoice,
)

from ap_automation.agents.base import A2AAgent, AgentResult  # noqa: E402
from ap_automation.agents.cost_centre_agent import CostCentreAgent  # noqa: E402
from ap_automation.agents.orchestrator import AgentOrchestrator  # noqa: E402
from ap_automation.core import clock, embeddings, mdm, offline, pipeline, posting, retrieval  # noqa: E402
from ap_automation.core.config import settings  # noqa: E402
from ap_automation.core.evaluation import SIMULATED_REVIEWER, Record, _rate  # noqa: E402
from ap_automation.core.models import ConfirmationSource, RoutingDecision  # noqa: E402
from ap_automation.core.pipeline import AutoPostNotPermitted, predict  # noqa: E402
from ap_automation.core.preprocessing import preprocess_description  # noqa: E402
from ap_automation.core.seeding import seed_corpus  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "public"
CAL_BINS: list[tuple[float, float, str]] = [
    (0.0, 0.5, "<0.5"), (0.5, 0.7, "0.5-0.7"), (0.7, 0.85, "0.7-0.85"), (0.85, 0.95, "0.85-0.95"), (0.95, 1.01, "0.95+"),
]
HISTORY_BINS: list[tuple[int, int, str]] = [
    (0, 1, "0"), (1, 3, "1-2"), (3, 10, "3-9"), (10, 50, "10-49"), (50, 10**9, "50+"),
]
EMBED_PRICE_PER_M_TOKENS = 0.02  # USD, text-embedding-3-small


# --------------------------------------------------------------------------- neutral agent configuration

class DepartmentEntityAgent(A2AAgent):
    """Entity = department. Valid iff the department is in the department master list (no vendor/entity map)."""

    def __init__(self, departments: set[str]):
        self._departments = {d.upper() for d in departments}

    @property
    def name(self) -> str:
        return "entity_agent"

    @property
    def description(self) -> str:
        return "Checks the posting entity (a City department) against the department master list."

    def can_handle(self, context: dict) -> bool:
        return True

    def run(self, context: dict) -> AgentResult:
        entity = (context.get("entity_id") or "").upper()
        valid = entity in self._departments
        return AgentResult(
            agent_name=self.name, success=True, confidence=0.9,
            reasoning=f"Department {entity or '(none)'} {'is' if valid else 'is NOT'} in the department master list.",
            data={"entity_valid": valid, "provided_entity": entity, "expected_entity": "department master",
                  "entity_sensitive": False, "entity_note": ""},
        )


class NotApplicableVATAgent(A2AAgent):
    """The City of Los Angeles pays no VAT: treatment is NOT_APPLICABLE and never requires review."""

    @property
    def name(self) -> str:
        return "vat_agent"

    @property
    def description(self) -> str:
        return "No VAT applies to this data set (US municipal fund accounting)."

    def can_handle(self, context: dict) -> bool:
        return True

    def run(self, context: dict) -> AgentResult:
        return AgentResult(
            agent_name=self.name, success=True, confidence=1.0, reasoning="VAT not applicable (USD, US public sector).",
            data={"vat_treatment": "NOT_APPLICABLE", "vat_rate": "N/A", "vat_flag": False,
                  "review_required": False, "vat_note": "not applicable", "gl_code": context.get("predicted_gl", "")},
        )


def build_orchestrator(kind: str, departments: set[str]) -> AgentOrchestrator:
    orch = AgentOrchestrator()
    if kind == "neutral":
        orch._pipeline = [DepartmentEntityAgent(departments), NotApplicableVATAgent(), CostCentreAgent()]
    elif kind != "shipped":
        raise ValueError(f"unknown agent configuration {kind!r}")
    return orch


# --------------------------------------------------------------------------- pure helpers (unit-tested)

def _audit_pick(seed: int, index: int, rate: float) -> bool:
    """Deterministic QA sample: True for about `rate` of the line indexes (hash-based, reproducible)."""
    return int(hashlib.sha1(f"{seed}:{index}".encode()).hexdigest()[:8], 16) / 2**32 < rate


def month_start_plus(date_iso: str, months: int) -> str:
    y, m = int(date_iso[:4]), int(date_iso[5:7])
    idx = (y * 12 + (m - 1)) + months
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}-01"


def split_seed_replay(rows: list[dict], seed_months: int, holdout_vendors: int, seed: int) -> dict:
    """
    Seed = lines dated before the cutoff (first-of-month + seed_months). Replay = every line on/after the cutoff,
    in date order. `holdout_vendors` vendors that have seed-period lines are removed from the seed (and their
    seed-period lines are dropped), so they genuinely arrive at the cutoff with no history: a new-vendor
    cohort that only write-back can teach.
    """
    ordered = sorted(rows, key=lambda r: (r["transaction_date"], r[":id"]))
    cutoff = month_start_plus(ordered[0]["transaction_date"], seed_months)
    seed_rows = [r for r in ordered if r["transaction_date"] < cutoff]
    seed_vendors = sorted({r["vendor_id"] for r in seed_rows})
    held = sorted(random.Random(seed).sample(seed_vendors, min(holdout_vendors, len(seed_vendors))))
    held_set = set(held)
    dropped = [r for r in seed_rows if r["vendor_id"] in held_set]
    seed_rows = [r for r in seed_rows if r["vendor_id"] not in held_set]
    replay_rows = [r for r in ordered if r["transaction_date"] >= cutoff]  # strictly date-ordered, after the cutoff
    return {"cutoff": cutoff, "seed": seed_rows, "replay": replay_rows, "held_out_vendors": held,
            "dropped_held_out_seed_lines": len(dropped)}


def build_vendor_rows(rows: list[dict]) -> list[dict]:
    """Vendor master for MDM: every vendor_name seen -> its vendor_id (the stable key)."""
    pairs = sorted({(r["vendor_name"], r["vendor_id"]) for r in rows})
    return [{"raw_name": n, "canonical_id": v} for n, v in pairs]


def name_collisions(vendor_rows: list[dict]) -> list[str]:
    """Normalised vendor names that would resolve to more than one vendor_id (MDM keys on the normalised name)."""
    by_key: dict[str, set[str]] = defaultdict(set)
    for r in vendor_rows:
        by_key[mdm._normalise(r["raw_name"])].add(r["canonical_id"])
    return sorted(k for k, ids in by_key.items() if len(ids) > 1)


@dataclass
class ReplayRecord(Record):
    date: str = ""
    confidence: float = 0.0
    evidence_n: int = 0
    baseline_all_pred: str | None = None      # vendor's most common TRUE account over all earlier lines
    baseline_corpus_pred: str | None = None   # same, but only over lines the system's corpus actually holds
    # diagnostics used by the tuned-gate experiments (scripts/tuned_replay.py); unused by the baseline summary
    context_level: str = ""
    agreement: float = 0.0
    n_agree: int = 0
    vendor_distinct: int = 0
    subset_lines: int = 0
    audited: bool = False  # an auto-post that a human re-coded (QA sample) and whose human code was written back


def _bin_of(value: float) -> str:
    for lo, hi, label in CAL_BINS:
        if lo <= value < hi:
            return label
    return CAL_BINS[-1][2]


def calibration(records: list[ReplayRecord]) -> dict:
    """Accuracy by confidence band over every prediction (and over auto-posts only), plus ECE."""
    rows_out, ece_num = [], 0.0
    predicted = [r for r in records if r.predicted_gl]
    for lo, hi, label in CAL_BINS:
        inb = [r for r in predicted if lo <= r.confidence < hi]
        auto = [r for r in inb if r.auto]
        acc = _rate(sum(r.correct for r in inb), len(inb))
        mean_conf = round(sum(r.confidence for r in inb) / len(inb), 4) if inb else None
        if inb:
            ece_num += len(inb) * abs((acc or 0) - (mean_conf or 0))
        rows_out.append({
            "band": label, "n": len(inb), "accuracy": acc, "mean_confidence": mean_conf,
            "n_auto_posted": len(auto), "auto_post_accuracy": _rate(sum(r.correct for r in auto), len(auto)),
        })
    return {"bins": rows_out, "ece_all_predictions": round(ece_num / len(predicted), 4) if predicted else None,
            "note": "confidence = 0.7*agreement + 0.3*mean similarity of agreeing lines; a heuristic score, "
                    "not a trained probability"}


def by_history(records: list[ReplayRecord]) -> list[dict]:
    out = []
    for lo, hi, label in HISTORY_BINS:
        rs = [r for r in records if lo <= r.history_before < hi]
        a = [r for r in rs if r.auto]
        out.append({
            "vendor_history_lines": label, "n": len(rs), "automation_rate": _rate(len(a), len(rs)),
            "auto_post_accuracy": _rate(sum(r.correct for r in a), len(a)),
            "gl_accuracy": _rate(sum(r.correct for r in rs), len(rs)),
        })
    return out


def summarise_replay(records: list[ReplayRecord], seeded_vendors: set[str]) -> dict:
    """Metric definitions follow core/evaluation.summarise; extras: calibration, monthly curve, baselines."""
    n = len(records)
    auto = [r for r in records if r.auto]
    wrong_auto = [r for r in auto if not r.correct]
    reviewed = [r for r in records if not r.auto]

    months: dict[str, list[ReplayRecord]] = defaultdict(list)
    for r in records:
        months[r.date[:7]].append(r)
    curve = []
    for m in sorted(months):
        rs = months[m]
        a = [r for r in rs if r.auto]
        curve.append({
            "month": m, "invoices": len(rs), "automation_rate": _rate(len(a), len(rs)),
            "auto_post_accuracy": _rate(sum(r.correct for r in a), len(a)),
            "gl_accuracy_all": _rate(sum(r.correct for r in rs), len(rs)),
        })

    # New-to-corpus vendors: no seed history. Write-back is the only way they can ever auto-post.
    new_rows = [r for r in records if r.vendor_id not in seeded_vendors]
    new_vendors = sorted({r.vendor_id for r in new_rows})
    seen_count: Counter[str] = Counter()
    first_auto: dict[str, int] = {}
    for r in new_rows:
        seen_count[r.vendor_id] += 1
        if r.auto and r.vendor_id not in first_auto:
            first_auto[r.vendor_id] = seen_count[r.vendor_id]
    reviews_before_first_auto = sorted(v - 1 for v in first_auto.values())
    new_auto = [r for r in new_rows if r.auto]

    # Vendor-most-common baseline on previously seen vendors.
    seen_rows = [r for r in records if r.baseline_all_pred is not None]
    corpus_rows = [r for r in records if r.baseline_corpus_pred is not None]
    bypass = [r for r in records if r.bypass_blocked is not None]

    return {
        "invoices": n,
        "automation_rate": _rate(len(auto), n),
        "auto_posted": len(auto),
        "auto_post_accuracy": _rate(len(auto) - len(wrong_auto), len(auto)),
        "false_positives": len(wrong_auto),
        "false_positive_rate_of_auto_posts": _rate(len(wrong_auto), len(auto)),
        "false_positive_rate_of_all_invoices": _rate(len(wrong_auto), n),
        "gl_accuracy_all_predictions": _rate(sum(r.correct for r in records), n),
        "gl_accuracy_rows_with_a_prediction": _rate(
            sum(r.correct for r in records if r.predicted_gl), sum(1 for r in records if r.predicted_gl)),
        "review_rows": len(reviewed),
        "review_rows_where_prediction_was_correct": sum(r.correct for r in reviewed),
        "share_of_reviews_that_were_unnecessary": _rate(sum(r.correct for r in reviewed), len(reviewed)),
        "lines_learned": sum(1 for r in records if r.learned),
        "lines_learned_from_auto_posts": sum(1 for r in auto if r.learned),
        "review_reasons": dict(Counter(r.review_code for r in reviewed).most_common()),
        "agent_flag_counts": dict(Counter(f.split(":")[0] for r in records for f in r.flags).most_common()),
        "new_to_corpus_vendors": {
            "vendors": len(new_vendors), "invoices": len(new_rows),
            "automation_rate": _rate(len(new_auto), len(new_rows)),
            "auto_post_accuracy": _rate(sum(r.correct for r in new_auto), len(new_auto)),
            "gl_accuracy_all": _rate(sum(r.correct for r in new_rows), len(new_rows)),
            "vendors_that_ever_auto_posted": len(first_auto),
            "reviews_before_first_auto_post_median": (
                reviews_before_first_auto[len(reviews_before_first_auto) // 2] if reviews_before_first_auto else None),
        },
        "by_vendor_history_lines": by_history(records),
        "calibration": calibration(records),
        "learning_curve_by_month": curve,
        "vendor_most_common_baseline": {
            "definition": "predict the vendor's most common TRUE account over all earlier lines (seed + every "
                          "earlier replayed line, as if all were human-coded), on lines whose vendor was seen before",
            "rows_with_vendor_seen_before": len(seen_rows),
            "baseline_accuracy": _rate(sum(r.baseline_all_pred == r.truth_gl for r in seen_rows), len(seen_rows)),
            "system_gl_accuracy_same_rows": _rate(sum(r.correct for r in seen_rows), len(seen_rows)),
            "system_auto_post_accuracy_same_rows": _rate(
                sum(r.correct for r in seen_rows if r.auto), sum(1 for r in seen_rows if r.auto)),
            "baseline_accuracy_on_the_rows_the_system_auto_posted": _rate(
                sum(r.baseline_all_pred == r.truth_gl for r in seen_rows if r.auto),
                sum(1 for r in seen_rows if r.auto)),
            "same_history_variant": {
                "definition": "baseline restricted to the lines this arm's corpus holds (seed + write-backs)",
                "rows": len(corpus_rows),
                "baseline_accuracy": _rate(
                    sum(r.baseline_corpus_pred == r.truth_gl for r in corpus_rows), len(corpus_rows)),
            },
        },
        "enforcement": {"forced_auto_post_attempts": len(bypass),
                        "refused": sum(1 for r in bypass if r.bypass_blocked)},
    }


# --------------------------------------------------------------------------- live embeddings (token-capped)

class TokenCapExceeded(RuntimeError):
    pass


def _count_tokens(texts: list[str]) -> tuple[int, str]:
    """Token count for the embedding model's tokenizer (cl100k_base); falls back to a conservative estimate."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return sum(len(enc.encode(t)) for t in texts), "tiktoken cl100k_base"
    except Exception:
        return sum(max(1, math.ceil(len(t) / 3)) for t in texts), "estimate: ceil(chars/3)"


def _emb_cache_file(model: str) -> Path:
    return CACHE_DIR / f"embeddings_{model}.npz"


def load_embedding_cache(model: str) -> dict[str, "object"]:
    import numpy as np
    f = _emb_cache_file(model)
    if not f.exists():
        return {}
    d = np.load(f, allow_pickle=False)
    return dict(zip(d["keys"].tolist(), d["vectors"]))


def save_embedding_cache(model: str, cache: dict) -> None:
    import numpy as np
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    keys = sorted(cache)
    np.savez(_emb_cache_file(model), keys=np.array(keys), vectors=np.stack([cache[k] for k in keys]))


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def precompute_live_embeddings(texts: set[str], max_tokens: int, max_usd: float, batch_size: int = 256) -> dict:
    """
    Embed every distinct cleaned description once with the real OpenAI model, in batches, with a hard cap.
    The cap is checked BEFORE any spend (tokenizer count of the texts still to embed) and again before each batch
    against the usage the API actually reported. Returns a usage report; the vectors go to the on-disk cache.
    """
    import numpy as np

    model = settings.embedding_model
    cache = load_embedding_cache(model)
    todo = sorted(t for t in texts if _key(t) not in cache)
    est_tokens, how = _count_tokens(todo)
    cap_tokens = min(max_tokens, int(max_usd / EMBED_PRICE_PER_M_TOKENS * 1_000_000))
    report = {
        "model": model, "distinct_texts": len(texts), "already_cached": len(texts) - len(todo),
        "to_embed": len(todo), "estimated_tokens": est_tokens, "token_count_method": how,
        "estimated_cost_usd": round(est_tokens * EMBED_PRICE_PER_M_TOKENS / 1e6, 5),
        "token_cap": cap_tokens, "usd_cap": max_usd, "price_per_million_tokens_usd": EMBED_PRICE_PER_M_TOKENS,
        "actual_tokens": 0, "actual_cost_usd": 0.0, "api_requests": 0,
    }
    if est_tokens > cap_tokens:
        raise TokenCapExceeded(
            f"{len(todo)} texts need ~{est_tokens:,} tokens (~${report['estimated_cost_usd']:.4f}); "
            f"cap is {cap_tokens:,} tokens (${max_usd:.2f}). Nothing was spent - raise the cap or shrink the sample.")
    print(f"[live] embedding {len(todo):,} distinct texts, ~{est_tokens:,} tokens "
          f"(~${report['estimated_cost_usd']:.4f}); cap {cap_tokens:,} tokens", flush=True)

    client = embeddings.get_client()
    used = 0
    t0 = time.time()
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        batch_est, _ = _count_tokens(batch)
        if used + batch_est > cap_tokens:
            save_embedding_cache(model, cache)
            raise TokenCapExceeded(f"cap of {cap_tokens:,} tokens would be exceeded after {used:,} tokens used")
        resp = client.embeddings.create(model=model, input=batch)
        report["api_requests"] += 1
        used += resp.usage.total_tokens if resp.usage else batch_est
        for text, item in zip(batch, sorted(resp.data, key=lambda d: d.index)):
            cache[_key(text)] = np.asarray(item.embedding, dtype=np.float32)
        if (i // batch_size) % 10 == 0:
            print(f"[live]   {i + len(batch):,}/{len(todo):,} texts, {used:,} tokens, {time.time() - t0:.0f}s", flush=True)
    if todo:
        save_embedding_cache(model, cache)
    report["actual_tokens"] = used
    report["actual_cost_usd"] = round(used * EMBED_PRICE_PER_M_TOKENS / 1e6, 5)
    report["seconds"] = round(time.time() - t0, 1)
    report["dimensions"] = int(len(next(iter(cache.values())))) if cache else 0
    return report


class CachedLiveEmbedder:
    """Looks a CLEANED description up in the precomputed OpenAI vectors. Never calls the network, never guesses."""

    def __init__(self, model: str):
        self._cache = load_embedding_cache(model)

    def __call__(self, cleaned_text: str) -> list[float]:
        try:
            return self._cache[_key(cleaned_text)].tolist()
        except KeyError:
            raise KeyError(f"no precomputed embedding for {cleaned_text[:60]!r}") from None


# --------------------------------------------------------------------------- vector store + one arm

def use_store(kind: str, live: bool) -> None:
    """Point retrieval at a fresh store and install the embedder. Never touches the configured (remote) Qdrant."""
    if kind == "qdrant-local":  # qdrant-client local mode: reference implementation, O(N) filter scans (slow)
        offline.use_offline_backend(hashing=not live)
    elif kind == "inproc":      # vendor-indexed stand-in, see inproc_vector_store.py
        from inproc_vector_store import InProcessVectorStore
        retrieval.set_client(InProcessVectorStore())
        if not live:
            embeddings.set_embedder(offline.HashingEmbedder())
        retrieval.ensure_collection()
    else:
        raise ValueError(f"unknown store {kind!r}")
    if live:
        embeddings.set_embedder(CachedLiveEmbedder(settings.embedding_model))


def run_arm(payload: dict) -> dict:
    """Run one replay arm in this process. Returns plain dicts (picklable across processes)."""
    cfg = payload["cfg"]
    rows = payload["rows"]
    learn = payload["learn"]
    limit = payload.get("limit")
    t0 = time.time()

    settings.telemetry_enabled = False
    pipeline._lf_client, pipeline._lf_init_done = None, False
    # Optional experiment switches (all absent/False in the baseline replay, which stays exactly as it was):
    #   settings   - overrides applied to core.config.settings for this arm only (restored afterwards)
    #   sim_clock  - evidence is dated by transaction date and the pipeline clock follows the replayed line
    #   clerk_cc   - write-back stores the invoice's own fund (the clerk-confirmed cost centre) instead of the prediction
    overrides = cfg.get("settings") or {}
    saved_settings = {key: getattr(settings, key) for key in overrides}
    for key, val in overrides.items():
        setattr(settings, key, val)
    sim_clock, clerk_cc = bool(cfg.get("sim_clock")), bool(cfg.get("clerk_cc"))
    audit_rate = float(cfg.get("audit_rate") or 0.0)  # share of auto-posts a human re-codes (QA sampling); 0 = none
    split = split_seed_replay(rows, cfg["seed_months"], cfg["holdout_vendors"], cfg["seed"])
    seed_rows, replay_rows = split["seed"], split["replay"]
    if limit:
        replay_rows = replay_rows[:limit]
    vendor_rows = build_vendor_rows(rows)
    collisions = name_collisions(vendor_rows)
    if collisions:
        raise RuntimeError(f"vendor name collisions would corrupt MDM identity: {collisions[:5]}")
    departments = {r["department_name"] for r in rows}

    saved = (settings.erp_db_path, settings.audit_db_path, settings.erp_url)
    saved_backend = settings.erp_backend
    saved_orch = pipeline._orchestrator
    records: list[ReplayRecord] = []
    agents_run: Counter[str] = Counter()
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            settings.erp_db_path = str(Path(tmp) / "erp.db")
            settings.audit_db_path = str(Path(tmp) / "audit.db")
            settings.erp_url = ""
            settings.erp_backend = "simulated"  # a replay never posts to SAP or Xero
            posting._db_ready = False
            pipeline._orchestrator = build_orchestrator(cfg["agents"], departments)
            mdm.set_vendor_map(vendor_rows)
            use_store(cfg.get("store", "inproc"), live=cfg["live_embeddings"])

            if sim_clock:  # each seed line is dated by its own transaction date
                history_counts: Counter[str] = Counter()
                by_day: dict[str, list[dict]] = defaultdict(list)
                for r in seed_rows:
                    by_day[r["transaction_date"]].append(r)
                for day in sorted(by_day):
                    history_counts.update(seed_corpus([row_to_invoice(r) for r in by_day[day]],
                                                      timestamp=f"{day}T00:00:00+00:00"))
            else:
                seed_ts = f"{split['cutoff']}T00:00:00+00:00"
                history_counts = Counter(seed_corpus([row_to_invoice(r) for r in seed_rows], timestamp=seed_ts))
            seeded_vendors = set(history_counts)
            all_truth: dict[str, Counter] = defaultdict(Counter)     # every earlier line, as if all were coded
            corpus_truth: dict[str, Counter] = defaultdict(Counter)  # only what this arm's corpus holds
            for r in seed_rows:
                all_truth[r["vendor_id"]][r["account_code"]] += 1
                corpus_truth[r["vendor_id"]][r["account_code"]] += 1
            print(f"[{payload['name']}] seeded {len(seed_rows):,} lines ({len(seeded_vendors)} vendors), "
                  f"replaying {len(replay_rows):,} in {time.time() - t0:.0f}s", flush=True)

            for i, row in enumerate(replay_rows):
                inv = row_to_invoice(row)
                if sim_clock:
                    when = datetime.fromisoformat(row["transaction_date"]).replace(tzinfo=timezone.utc, hour=12)
                    clock.set_clock(lambda when=when: when)
                pred = predict(inv, batch_mode=True)
                vendor = pred.resolved_line.canonical_vendor_id
                truth = inv.ground_truth_gl or ""
                ctx = pred.provenance.get("retrieval_context") or {}
                rec = ReplayRecord(
                    index=i, vendor_id=vendor, category="public", anomaly=None,
                    history_before=history_counts[vendor], predicted_gl=pred.predicted_gl, truth_gl=truth,
                    rule_routing=pred.routing_decision.value, final_routing=pred.final_routing.value,
                    flags=list(pred.agent_flags), reason=pred.final_routing_reason,
                    date=row["transaction_date"], confidence=round(pred.weighted_confidence, 4),
                    evidence_n=len(pred.evidence),
                    baseline_all_pred=all_truth[vendor].most_common(1)[0][0] if all_truth[vendor] else None,
                    baseline_corpus_pred=corpus_truth[vendor].most_common(1)[0][0] if corpus_truth[vendor] else None,
                    context_level=ctx.get("context_level", ""), agreement=round(pred.agreement_ratio, 4),
                    n_agree=sum(1 for e in pred.evidence if e.gl_account == pred.predicted_gl),
                    vendor_distinct=ctx.get("vendor_distinct_codes", 0), subset_lines=ctx.get("subset_lines", 0),
                )
                if pred.final_routing == RoutingDecision.AUTO_POST:
                    res = posting.post_outcome(
                        pred, pred.predicted_gl, pred.predicted_cost_centre,
                        settings.auto_post_user_id, ConfirmationSource.AUTO_POST, mirror_to_sheet=False)
                    rec.learned = res.learned  # always False: the system does not learn from itself
                    if audit_rate > 0 and truth and _audit_pick(cfg["seed"], i, audit_rate):
                        # QA sampling: a human re-codes this auto-posted line; the HUMAN's code is written back
                        # (never the system's own guess). The auto-post decision is still scored as made.
                        src = (ConfirmationSource.HUMAN_CONFIRM if pred.predicted_gl == truth
                               else ConfirmationSource.HUMAN_CORRECT)
                        cc = (inv.cost_centre or pred.predicted_cost_centre or "") if clerk_cc else \
                            (pred.predicted_cost_centre or (inv.cost_centre or ""))
                        ares = posting.post_outcome(pred, truth, cc, "audit_reviewer", src, mirror_to_sheet=False)
                        if ares.learned:
                            rec.audited = True
                            history_counts[vendor] += 1
                            corpus_truth[vendor][truth] += 1
                else:
                    try:  # adversarial: forcing a rejected prediction to post must be refused
                        posting.post_outcome(pred, pred.predicted_gl, pred.predicted_cost_centre,
                                             "bypass_attempt", ConfirmationSource.AUTO_POST, mirror_to_sheet=False)
                        rec.bypass_blocked = False
                    except AutoPostNotPermitted:
                        rec.bypass_blocked = True
                    if learn and truth:
                        source = (ConfirmationSource.HUMAN_CONFIRM if pred.predicted_gl == truth
                                  else ConfirmationSource.HUMAN_CORRECT)
                        cc = ((inv.cost_centre or pred.predicted_cost_centre or "") if clerk_cc
                              else (pred.predicted_cost_centre or (inv.cost_centre or "")))
                        res = posting.post_outcome(pred, truth, cc, SIMULATED_REVIEWER, source, mirror_to_sheet=False)
                        rec.learned = res.learned
                        if res.learned:
                            history_counts[vendor] += 1
                            corpus_truth[vendor][truth] += 1
                for a in pred.provenance.get("a2a_agents", {}).get("results", []):
                    agents_run[a["agent"]] += 1
                all_truth[row["vendor_id"]][truth] += 1  # baseline sees every line's true label eventually
                records.append(rec)
                if (i + 1) % 2000 == 0:
                    done = records
                    a = sum(r.auto for r in done)
                    print(f"[{payload['name']}] {i + 1:,}/{len(replay_rows):,}  auto {a / len(done):.1%}  "
                          f"{time.time() - t0:.0f}s", flush=True)
    finally:
        settings.erp_db_path, settings.audit_db_path, settings.erp_url = saved
        settings.erp_backend = saved_backend
        for key, val in saved_settings.items():
            setattr(settings, key, val)
        if sim_clock:
            clock.set_clock(None)
        pipeline._orchestrator = saved_orch
        posting._db_ready = False
        offline.reset_backend()

    summary = summarise_replay(records, seeded_vendors)
    summary["agents_run_counts"] = dict(agents_run)
    summary["seed_lines"] = len(seed_rows)
    summary["seeded_vendors"] = len(seeded_vendors)
    summary["seconds"] = round(time.time() - t0, 1)
    summary["agents_configuration"] = cfg["agents"]
    summary["replay_cutoff"] = split["cutoff"]
    return {"name": payload["name"], "summary": summary, "records": [asdict(r) for r in records],
            "split": {k: split[k] for k in ("cutoff", "held_out_vendors")},
            "replay_lines": len(replay_rows)}


# --------------------------------------------------------------------------- main

def git_info() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD"), "working_tree_dirty": bool(run("status", "--porcelain"))}


def write_records_csv(path: Path, records: list[dict]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["index", "date", "vendor_id", "history_before", "predicted_gl", "truth_gl", "confidence", "evidence_n",
            "rule_routing", "final_routing", "learned", "baseline_all_pred", "baseline_corpus_pred", "reason"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r.get(c) for c in cols} for r in records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiscal-year", type=int, default=2026)
    ap.add_argument("--max-rows", type=int, default=30_000, help="sample size cap (rows)")
    ap.add_argument("--max-per-vendor", type=int, default=250)
    ap.add_argument("--min-vendor-rows", type=int, default=50)
    ap.add_argument("--seed-months", type=int, default=3, help="months of earliest history indexed as confirmed")
    ap.add_argument("--holdout-vendors", type=int, default=10,
                    help="vendors withheld from the seed so they arrive with no history (new-vendor cohort)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--store", choices=("inproc", "qdrant-local"), default="inproc",
                    help="vector store: vendor-indexed in-process stand-in (fast) or qdrant-client local mode (slow)")
    ap.add_argument("--agents", choices=("neutral", "shipped"), default="neutral")
    ap.add_argument("--shipped-probe-rows", type=int, default=3000,
                    help="also replay this many lines with the agents AS SHIPPED (UK VAT config) to show their effect; 0=skip")
    ap.add_argument("--live-embeddings", action="store_true")
    ap.add_argument("--max-live-usd", type=float, default=0.50, help="hard cap on embedding spend")
    ap.add_argument("--max-live-tokens", type=int, default=20_000_000)
    ap.add_argument("--limit-replay", type=int, default=0, help="debug: replay only the first N lines")
    ap.add_argument("--workers", type=int, default=3, help="processes to run the arms in (1 = sequential)")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    live = args.live_embeddings
    out_json = Path(args.json) if args.json else REPO_ROOT / "docs" / ("public_replay_live.json" if live
                                                                       else "public_replay_offline.json")
    settings.telemetry_enabled = False
    sample, report = ensure_sample(args.fiscal_year, args.max_rows, args.max_per_vendor, args.min_vendor_rows, CACHE_DIR)
    split = split_seed_replay(sample, args.seed_months, args.holdout_vendors, args.seed)
    print(f"sample {len(sample):,} lines / {len({r['vendor_id'] for r in sample})} vendors; seed {len(split['seed']):,} "
          f"(before {split['cutoff']}), replay {len(split['replay']):,}; held-out vendors {len(split['held_out_vendors'])}",
          flush=True)

    embed_info: dict
    if live:
        texts = {preprocess_description(r["detailed_item_description"]) for r in sample}
        usage = precompute_live_embeddings(texts, args.max_live_tokens, args.max_live_usd)
        embed_info = {
            "kind": "openai", "model": settings.embedding_model, "dimensions": usage["dimensions"],
            "method": "distinct cleaned descriptions embedded once in batches; the pipeline's embedder is a lookup "
                      "into those vectors (identical to calling the API per line, just faster)",
            "usage": usage,
        }
    else:
        embed_info = {"kind": "offline-hashing", "model": "hashing embedder (crc32 unigrams+bigrams, lexical only)",
                      "dimensions": settings.embedding_dimensions}

    cfg = {"seed_months": args.seed_months, "holdout_vendors": args.holdout_vendors, "seed": args.seed,
           "agents": args.agents, "live_embeddings": live, "store": args.store}
    tasks = [
        {"name": "with_learning", "learn": True, "cfg": cfg, "rows": sample, "limit": args.limit_replay or None},
        {"name": "no_learning_control", "learn": False, "cfg": cfg, "rows": sample, "limit": args.limit_replay or None},
    ]
    if args.shipped_probe_rows and args.agents == "neutral":
        n = min(args.shipped_probe_rows, args.limit_replay or args.shipped_probe_rows)
        tasks.append({"name": "agents_as_shipped_probe", "learn": True, "cfg": {**cfg, "agents": "shipped"},
                      "rows": sample, "limit": n})

    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            results = list(pool.map(run_arm, tasks))
    else:
        results = [run_arm(t) for t in tasks]
    elapsed = round(time.time() - t0, 1)
    by_name = {r["name"]: r for r in results}

    out = {
        "meta": {
            "dataset": DATASET_NAME, "dataset_id": DATASET_ID, "url": DATASET_URL, "licence": LICENCE,
            "attribution": ATTRIBUTION, "fiscal_year": args.fiscal_year,
            "population_filter": report["where"], "normalisation": report["normalisation"],
            "sample_params": report["sample_params"], "sample_stats": report["sample_stats"],
            "population_stats": report["population_stats"],
            "replay_params": {
                "seed_months": args.seed_months, "seed_cutoff_date": split["cutoff"],
                "seed_lines": len(split["seed"]), "replay_lines": len(split["replay"]),
                "held_out_vendor_ids": split["held_out_vendors"],
                "dropped_held_out_seed_period_lines": split["dropped_held_out_seed_lines"], "limit_replay": args.limit_replay or None,
                "write_back_policy": "true account_code applied as clerk confirmation ONLY for lines routed to human "
                                     "review; auto-posts never written back; control never writes back",
                "llm_gate": "off (batch_mode=True)",
                "vector_store": {
                    "kind": args.store,
                    "note": "inproc = vendor-indexed in-process stand-in for qdrant-client's local mode (scripts/"
                            "inproc_vector_store.py); retrieval/BM25/RRF/confidence/gating code is the shipped code. "
                            "On a 1,200-line check it matched qdrant-client local mode on 95% of lines (differences "
                            "are tie-breaking between identical-description lines) and on aggregate metrics to ~2 pts.",
                },
            },
            "mapping": {"vendor": "vendor_name (MDM canonical id = vendor_id)", "description": "detailed_item_description",
                        "amount": "dollar_amount", "currency": "USD", "date": "transaction_date",
                        "entity": "department_name", "cost_centre": "fund", "label_gl_code": "account_code"},
            "leak_guard_columns": list(LEAK_COLUMNS),
            "embedder": embed_info,
            "agents": {
                "main_runs": args.agents,
                "neutral_configuration": {
                    "entity_agent": "valid iff department is in the department master (all sample departments)",
                    "vat_agent": "NOT_APPLICABLE, never requires review (no VAT on this data)",
                    "cost_centre_agent": "shipped agent; skips itself because LA account codes are not in the "
                                         "synthetic chart of accounts",
                    "unchanged": "retrieval, confidence, rule gate, governance gate, write-back are the shipped code",
                },
            },
            "seed": args.seed, "git": git_info(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0], "wall_seconds_arms": elapsed,
        },
        "with_learning": by_name["with_learning"]["summary"],
        "no_learning_control": by_name["no_learning_control"]["summary"],
    }
    if "agents_as_shipped_probe" in by_name:
        p = by_name["agents_as_shipped_probe"]["summary"]
        out["agents_as_shipped_probe"] = {
            "note": "first N replay lines, with-learning arm, agents exactly as shipped (UK VAT / entity config)",
            "replay_lines": by_name["agents_as_shipped_probe"]["replay_lines"],
            **{k: p[k] for k in ("invoices", "automation_rate", "auto_posted", "review_reasons",
                                 "agent_flag_counts", "gl_accuracy_all_predictions")},
        }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    tag = "live" if live else "offline"
    for name in ("with_learning", "no_learning_control"):
        write_records_csv(CACHE_DIR / f"replay_{tag}_{name}_records.csv", by_name[name]["records"])

    w, c = out["with_learning"], out["no_learning_control"]
    print(f"\nwrote {out_json}  ({elapsed}s)")
    print(f"automation {w['automation_rate']:.1%} (control {c['automation_rate']:.1%}); "
          f"auto-post accuracy {w['auto_post_accuracy']}; GL accuracy {w['gl_accuracy_all_predictions']}")
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
