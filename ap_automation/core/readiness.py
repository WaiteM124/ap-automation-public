"""
Data-readiness report: a deterministic score of how ready a client's coded history is.

`assess(text, now=None, gate_profile="standard")` parses the same CSV as `POST /corpus/import`,
runs the checks below, and optionally estimates automation by replaying the later ~40% of lines
through the live decision path (retrieval + rule gate + governed routing + configured scalar gate)
on an isolated in-process store. It never writes to the live corpus, vendor registry, chart,
database, audit trail, or any ERP. Live gate settings are restored after the call.

Category weights (re-normalised when a check is not assessed):

    vendor_consistency        10
    gl_coding_consistency     16
    chart_coverage            14
    description_quality       10
    volume_sufficiency        14
    cost_centre_consistency    6   (excluded if no cost-centre column)
    vat_completeness           5   (excluded if no VAT/tax column)
    recency                    8   (excluded if no date column)
    replay                    17   (excluded if too few lines)

Verdict thresholds are PROVISIONAL and should be recalibrated against real client data:

    auto-post ready  replay assessed, automation >= 30%, auto-post accuracy >= 97%, no hard fails
    suggest-only     usable history but below that bar, or data-quality warnings
    not ready        too few lines/vendors, most rows invalid or off-chart, or a poor replay
                     (auto-post accuracy below 80%, or almost no auto-posts)

Hard fails: volume fail; >50% rows invalid; >50% rows off-chart; replay accuracy < 80%.
A replay that is merely below the auto-post bar (e.g. 60% automation at 87% accuracy) is
suggest-only, not not-ready.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import re
import tempfile
import threading
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ap_automation.core import chart_of_accounts, embeddings, history_import, mdm, offline, retrieval
from ap_automation.core.config import settings
from ap_automation.core.history_import import HistoryImportError, parse_rows
from ap_automation.core.mdm import _normalise as normalise_vendor
from ap_automation.core.models import (
    ConfirmationSource,
    ConfirmedOutcome,
    InvoiceLine,
    RoutingDecision,
)
from ap_automation.core.pipeline import predict, write_back

# --------------------------------------------------------------------------- documented knobs (PROVISIONAL)

WEIGHTS: dict[str, int] = {
    "vendor_consistency": 10,
    "gl_coding_consistency": 16,
    "chart_coverage": 14,
    "description_quality": 10,
    "volume_sufficiency": 14,
    "cost_centre_consistency": 6,
    "vat_completeness": 5,
    "recency": 8,
    "replay": 17,
}

AUTO_POST_MIN_AUTOMATION = 0.30
AUTO_POST_MIN_ACCURACY = 0.97
NOT_READY_MAX_SCORE = 45
NOT_READY_OFF_CHART = 0.50
NOT_READY_INVALID = 0.50

MIN_LINES_FOR_REPLAY = 200
MIN_REPLAYABLE = 10
MAX_REPLAY_LINES = 2_000
SEED_FRACTION = 0.60
MAX_REPLAYS = 2

# gate_profile values. "standard" is today's live/default gate. "tuned" is the gate-only
# winner at a 99% target on public LA data (no extra columns). "tuned_context" adds the
# LA department+fund requirement and is only used when those columns are in the file.
PROFILES = ("standard", "tuned", "tuned_context")

# docs/public_replay_tuned_live.json → shadow_mode_sweep.gate.tuned_99
# TEST window: 34.9% automated at 98.84% accuracy (3,880 / 11,118 auto-posts, 45 wrong).
# Closed-loop confirmation of the same settings: gate/tuned_99.settings.
# Extra gate knobs stayed off; the lift is min_agreement_ratio=1.0 and top_k=20.
TUNED_OVERRIDES: dict = {
    "top_k": 20,
    "min_evidence_lines": 3,
    "min_agreement_ratio": 1.0,
    "default_confidence_threshold": 0.5,
    "context_signals": "off",
    "gate_min_margin": 0.0,
    "gate_min_agreement_lb": 0.0,
    "gate_wilson_z": 1.0,
    "gate_diversity_per_code": 0.0,
    "gate_diversity_scope": "vendor",
    "gate_diversity_cap": 10,
    "gate_recency_halflife_days": 0.0,
    "gate_recency_window_days": 0.0,
    "gate_required_context": "",
}

# docs/public_replay_tuned_live.json → shadow_mode_sweep.combined.tuned_99.5
# TEST window: 71.5% automated at 99.47% accuracy. Needs department + fund.
TUNED_CONTEXT_OVERRIDES: dict = {
    "top_k": 30,
    "min_evidence_lines": 3,
    "min_agreement_ratio": 1.0,
    "default_confidence_threshold": 0.5,
    "context_signals": "subset",
    "context_min_lines": 5,
    "gate_min_margin": 0.0,
    "gate_min_agreement_lb": 0.0,
    "gate_wilson_z": 1.0,
    "gate_diversity_per_code": 0.5,
    "gate_diversity_scope": "vendor",
    "gate_diversity_cap": 10,
    "gate_recency_halflife_days": 0.0,
    "gate_recency_window_days": 120.0,
    "gate_required_context": "dept+fund",
}

PROFILE_OVERRIDES: dict[str, dict] = {
    "standard": {},
    "tuned": TUNED_OVERRIDES,
    "tuned_context": TUNED_CONTEXT_OVERRIDES,
}

REPLAY_NOTE = (
    "These figures are estimates on this file's own coded history, using the offline hashing "
    "embedder — not a live ERP post. Tuned profiles use gate settings chosen on public City of "
    "Los Angeles payments data (Checkbook L.A.) and need recalibration on each client's data. "
    "Context-assisted results need department and fund columns (entity_id / department_name and "
    "cost_centre / fund); the 71.5% automation at 99.47% accuracy LA headline used those "
    "LA-specific fields. Without them the gate-only LA result was about 34.9% automated at "
    "98.84%. A pair of replays (each at most 2,000 lines) typically finishes in well under two "
    "minutes. This check does not change the live gate (those settings are server-wide "
    "environment variables)."
)

_ISOLATE_KEYS = (
    "audit_db_path", "erp_db_path", "predictions_db_path",
    "erp_backend", "erp_url", "xero_posting_enabled", "sap_posting_enabled", "telemetry_enabled",
)

GENERIC_PHRASES = frozenset({
    "invoice", "invoices", "services", "service", "payment", "payments",
    "bill", "billing", "misc", "miscellaneous", "various", "other",
    "fee", "fees", "charge", "charges", "item", "items",
})

_ASSESS_LOCK = threading.Lock()
_DESC_NORM = re.compile(r"[^a-z0-9\s]+")
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _inproc_store():
    """Load the replay stand-in without putting `scripts/` on sys.path permanently."""
    spec = importlib.util.spec_from_file_location("inproc_vector_store", _SCRIPTS / "inproc_vector_store.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/inproc_vector_store.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.InProcessVectorStore()


def _headers(text: str) -> list[str]:
    reader = csv.DictReader(io.StringIO(text))
    return [(h or "").strip().lower() for h in (reader.fieldnames or [])]


def _has_col(headers: list[str], names: tuple[str, ...]) -> bool:
    return any(c in headers for c in names)


def _clip(text: str, n: int = 40) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _examples(items: list[str], n: int = 5, width: int = 40) -> list[str]:
    out, seen = [], set()
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(_clip(item, width))
        if len(out) >= n:
            break
    return out


def _norm_desc(text: str) -> str:
    return re.sub(r"\s+", " ", _DESC_NORM.sub(" ", text.lower())).strip()


def _status(score: float, ok: float, warn: float) -> str:
    if score >= ok:
        return "ok"
    if score >= warn:
        return "warn"
    return "fail"


def _cat(id_: str, name: str, weight: int, *, assessed: bool, status: str, score: float | None,
         metrics: dict, finding: str, action: str | None = None) -> dict:
    return {
        "id": id_, "name": name, "weight": weight, "assessed": assessed, "status": status,
        "score": None if score is None else round(score),
        "metrics": metrics, "finding": finding, "action": action,
    }


def _not_provided(id_: str, name: str, weight: int, finding: str) -> dict:
    return _cat(id_, name, weight, assessed=False, status="not_provided", score=None,
                metrics={}, finding=finding, action=None)


def _has_context_columns(headers: list[str]) -> bool:
    """Department (entity) + fund (cost centre) — the two columns dept+fund retrieval needs."""
    return _has_col(headers, history_import._ENTITY) and _has_col(headers, history_import._CC)


def _settings_snapshot() -> dict:
    return {name: getattr(settings, name) for name in type(settings).model_fields}


def _restore_settings(snap: dict) -> None:
    for name, val in snap.items():
        setattr(settings, name, val)


def _validate_profile(gate_profile: str) -> str:
    if gate_profile not in PROFILES:
        raise HistoryImportError(
            "gate_profile must be one of: standard, tuned, tuned_context "
            f"(got {gate_profile!r})"
        )
    return gate_profile


def _insufficient_replay(rows_n: int, reason: str | None = None) -> dict:
    w = WEIGHTS["replay"]
    text = reason or (
        f"insufficient data for a replay estimate (need at least {MIN_LINES_FOR_REPLAY} usable lines "
        f"and {MIN_REPLAYABLE} replayable lines)"
    )
    return _cat("replay", "Replay estimate", w, assessed=False, status="not_provided", score=None,
                metrics={"reason": text, "lines": rows_n}, finding=text,
                action="Import more coded history and run this check again.")


def _replay_payload(cat: dict, profile: str) -> dict:
    if cat.get("assessed"):
        metrics = cat["metrics"]
        return {
            **metrics,
            "profile": profile,
            "lines": metrics.get("lines_replayed"),
            "finding": cat["finding"],
            "status": cat["status"],
        }
    return {
        "profile": profile,
        "status": "insufficient" if cat.get("status") != "not_available" else "not_available",
        "reason": (cat.get("metrics") or {}).get("reason")
        or cat.get("finding")
        or "insufficient data for a replay estimate",
        "lines": (cat.get("metrics") or {}).get("lines"),
    }


# --------------------------------------------------------------------------- checks


def _vendor_consistency(rows: list[dict]) -> dict:
    w = WEIGHTS["vendor_consistency"]
    clusters: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        clusters[normalise_vendor(r["vendor"])].add(r["vendor"])
    n_raw = sum(len(v) for v in clusters.values())
    n_norm = len(clusters)
    extra = n_raw - n_norm
    extra_ratio = extra / n_raw if n_raw else 0.0
    score = 100.0 * (1.0 - extra_ratio)
    multi = sorted(((len(names), key, sorted(names)) for key, names in clusters.items() if len(names) > 1),
                   reverse=True)
    examples = []
    for _, key, names in multi[:5]:
        examples.append({"normalised": key, "variants": _examples(names)})
    status = _status(score, 92, 75)
    finding = (
        f"Matching is exact-only after simple normalisation (case, punctuation, legal suffixes such as Ltd/Inc). "
        f"{n_raw} raw vendor name{'' if n_raw == 1 else 's'} collapse to {n_norm} normalised vendor"
        f"{'' if n_norm == 1 else 's'}"
    )
    if extra:
        finding += f" ({extra} extra spelling variant{'' if extra == 1 else 's'})."
    else:
        finding += "; no spelling variants."
    action = None
    if status != "ok" and examples:
        bits = [f"{', '.join(e['variants'])} -> {e['normalised'] or '(empty)'}" for e in examples]
        action = "Merge these vendor-name variants so history is not split: " + "; ".join(bits) + "."
    return _cat("vendor_consistency", "Vendor consistency", w, assessed=True, status=status, score=score,
                metrics={"raw_names": n_raw, "normalised_vendors": n_norm, "extra_variants": extra,
                         "extra_ratio": round(extra_ratio, 4), "clusters": examples},
                finding=finding, action=action)


def _gl_consistency(rows: list[dict]) -> dict:
    w = WEIGHTS["gl_coding_consistency"]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(normalise_vendor(r["vendor"]), _norm_desc(r["description"]))].append(r)
    inconsistent_lines = 0
    vendor_bad: Counter[str] = Counter()
    vendor_examples: dict[str, list[str]] = defaultdict(list)
    for (_, desc), items in groups.items():
        gls = {i["gl"] for i in items}
        if len(gls) > 1:
            inconsistent_lines += len(items)
            vendor = items[0]["vendor"]
            vendor_bad[vendor] += len(items)
            if len(vendor_examples[vendor]) < 5:
                vendor_examples[vendor].append(
                    f"{_clip(items[0]['description'])} -> {', '.join(sorted(gls)[:4])}"
                )
    n = len(rows)
    rate = 1.0 - (inconsistent_lines / n) if n else 1.0
    score = 100.0 * rate
    offenders = []
    for vendor, count in vendor_bad.most_common(5):
        offenders.append({"vendor": _clip(vendor), "inconsistent_lines": count,
                          "examples": vendor_examples[vendor][:5]})
    status = _status(score, 90, 70)
    finding = (
        f"{rate:.0%} of lines share a GL with every other line that has the same vendor and a "
        f"near-identical description ({inconsistent_lines} of {n} sit in a mixed-GL group)."
    )
    action = None
    if status != "ok" and offenders:
        names = ", ".join(o["vendor"] for o in offenders)
        action = f"Review coding for the most inconsistent vendors: {names}."
    return _cat("gl_coding_consistency", "GL coding consistency", w, assessed=True, status=status, score=score,
                metrics={"consistency_rate": round(rate, 4), "inconsistent_lines": inconsistent_lines,
                         "worst_vendors": offenders},
                finding=finding, action=action)


def _chart_coverage(rows: list[dict]) -> dict:
    w = WEIGHTS["chart_coverage"]
    chart = chart_of_accounts.CHART
    n = len(rows)
    off = [r for r in rows if not r.get("in_chart")]
    off_rate = len(off) / n if n else 0.0
    used = {r["gl"] for r in rows if r.get("in_chart")}
    coverage = (len(used) / len(chart)) if chart else 0.0
    # Coverage of a large chart by a short file is not a failure; off-chart rows are.
    score = 100.0 * (0.85 * (1.0 - off_rate) + 0.15 * min(coverage / 0.15, 1.0))
    if off_rate > 0.25:
        status = "fail"
    elif off_rate > 0.05:
        status = "warn"
    else:
        status = "ok"
    missing = sorted({r["gl"] for r in off})
    finding = (
        f"{coverage:.0%} of the active chart ({len(used)} of {len(chart)} codes) appears in this file. "
        f"{off_rate:.0%} of history rows ({len(off)} of {n}) use a GL that is not in the chart."
    )
    action = None
    if missing:
        shown = ", ".join(missing[:12])
        more = f" (+{len(missing) - 12} more)" if len(missing) > 12 else ""
        action = f"Add these GL codes to the chart, or recode the rows: {shown}{more}."
    return _cat("chart_coverage", "Chart coverage", w, assessed=True, status=status, score=score,
                metrics={"chart_codes": len(chart), "codes_seen": len(used), "coverage": round(coverage, 4),
                         "off_chart_rows": len(off), "off_chart_rate": round(off_rate, 4),
                         "off_chart_codes": missing[:20]},
                finding=finding, action=action)


def _description_quality(rows: list[dict]) -> dict:
    w = WEIGHTS["description_quality"]
    generic: Counter[str] = Counter()
    flagged = 0
    samples: list[str] = []
    for r in rows:
        desc = r["description"].strip()
        words = [t for t in re.split(r"\s+", desc) if t]
        letters = sum(ch.isalpha() for ch in desc)
        digits = sum(ch.isdigit() for ch in desc)
        norm = _norm_desc(desc)
        phrase = norm if norm in GENERIC_PHRASES else ""
        short = len(desc) < 3 or len(words) < 3
        mostly_digits = len(desc) >= 3 and digits >= 0.6 * max(len(desc), 1) and digits > letters
        if phrase or short or mostly_digits:
            flagged += 1
            key = phrase or ("(mostly digits)" if mostly_digits else "(too short)")
            generic[key] += 1
            if len(samples) < 5:
                samples.append(_clip(desc))
    n = len(rows)
    rate = flagged / n if n else 0.0
    score = 100.0 * (1.0 - rate)
    status = _status(score, 90, 70)
    phrases = [{"phrase": p, "count": c} for p, c in generic.most_common(8)]
    found = ", ".join(f"{p['phrase']} ×{p['count']}" for p in phrases) or "none"
    finding = (
        f"{rate:.0%} of descriptions are generic, very short (under 3 characters or words), or mostly digits "
        f"(phrases found: {found})."
    )
    action = None
    if status != "ok":
        action = (
            "Replace generic line text with what was actually bought (noun + qualifier). "
            f"Examples from this file: {', '.join(samples) if samples else found}."
        )
    return _cat("description_quality", "Description quality", w, assessed=True, status=status, score=score,
                metrics={"generic_rate": round(rate, 4), "generic_lines": flagged, "phrases": phrases,
                         "examples": samples},
                finding=finding, action=action)


def _volume(rows: list[dict]) -> dict:
    w = WEIGHTS["volume_sufficiency"]
    min_n = settings.min_evidence_lines  # live gate; GateParams / shipped default is 3
    by_vendor: Counter[str] = Counter(normalise_vendor(r["vendor"]) for r in rows)
    qualifying = {v for v, n in by_vendor.items() if n >= min_n}
    n = len(rows)
    line_share = (sum(c for v, c in by_vendor.items() if v in qualifying) / n) if n else 0.0
    vendor_part = min(len(qualifying) / 10.0, 1.0)
    score = 100.0 * (0.7 * line_share + 0.3 * vendor_part)
    if n < 50 or len(qualifying) < 3 or line_share < 0.40:
        status = "fail"
    elif line_share < 0.75 or len(qualifying) < 8:
        status = "warn"
    else:
        status = "ok"
    need_vendors = max(0, 8 - len(qualifying))
    need_lines = max(0, 200 - n)
    finding = (
        f"{len(qualifying)} of {len(by_vendor)} vendors have at least {min_n} lines "
        f"(the live gate's min_n). {n} lines in total; {line_share:.0%} of lines belong to vendors that meet it."
    )
    action = None
    if status != "ok":
        parts = []
        if need_lines:
            parts.append(f"about {need_lines} more coded lines (aim for 200+)")
        if need_vendors:
            parts.append(f"{need_vendors} more vendors with at least {min_n} lines each")
        action = "Add " + " and ".join(parts) + " before relying on auto-post."
    return _cat("volume_sufficiency", "Volume sufficiency", w, assessed=True, status=status, score=score,
                metrics={"total_lines": n, "vendors": len(by_vendor), "min_n": min_n,
                         "vendors_meeting_min_n": len(qualifying), "line_share_meeting_min_n": round(line_share, 4)},
                finding=finding, action=action)


def _cost_centre(rows: list[dict], present: bool) -> dict:
    w = WEIGHTS["cost_centre_consistency"]
    if not present:
        return _not_provided("cost_centre_consistency", "Cost-centre consistency", w,
                             "No cost-centre column was provided, so this check was not assessed.")
    by_gl: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        by_gl[r["gl"]][r.get("cost_centre") or ""] += 1
    weighted = 0.0
    n = 0
    mixed = []
    for gl, counts in by_gl.items():
        total = sum(counts.values())
        filled = Counter({k: v for k, v in counts.items() if k})
        if not filled:
            majority = 0.0
        else:
            majority = filled.most_common(1)[0][1] / total
        weighted += majority * total
        n += total
        if filled and majority < 0.8 and len(filled) > 1:
            top = filled.most_common(3)
            mixed.append({"gl": gl, "majority": round(majority, 3),
                          "values": [{"cost_centre": c, "n": k} for c, k in top]})
    mixed.sort(key=lambda x: x["majority"])
    score = 100.0 * (weighted / n) if n else 0.0
    empty = sum(1 for r in rows if not r.get("cost_centre"))
    if empty == len(rows):
        status = "warn"
        score = min(score, 40.0)
        finding = "A cost-centre column is present but every value is blank."
        action = "Fill the cost-centre column; the system cannot learn a consistent fund/CC from empty values."
    else:
        status = _status(score, 80, 55)
        finding = (
            f"Across each GL, the most common cost centre covers {score:.0f}% of that GL's lines on average. "
            f"{empty} rows have no cost centre."
        )
        action = None
        if status != "ok" and mixed:
            codes = ", ".join(m["gl"] for m in mixed[:5])
            action = f"Standardise the cost centre on these mixed GLs: {codes}."
    return _cat("cost_centre_consistency", "Cost-centre consistency", w, assessed=True, status=status, score=score,
                metrics={"weighted_majority": round(score / 100.0, 4), "empty_rows": empty,
                         "mixed_gls": mixed[:5]},
                finding=finding, action=action)


def _vat(rows: list[dict], present: bool) -> dict:
    w = WEIGHTS["vat_completeness"]
    if not present:
        return _not_provided(
            "vat_completeness", "VAT completeness", w,
            "No VAT/tax-code column was provided. Imported history is otherwise UNASSESSED, "
            "so this check was not assessed rather than failed.",
        )
    filled = sum(1 for r in rows if (r.get("tax_code") or "").strip())
    n = len(rows)
    rate = filled / n if n else 0.0
    score = 100.0 * rate
    status = _status(score, 90, 50)
    finding = f"{rate:.0%} of rows have a VAT/tax code ({filled} of {n})."
    action = None if status == "ok" else "Add a vat / tax_code column and fill it for every line."
    return _cat("vat_completeness", "VAT completeness", w, assessed=True, status=status, score=score,
                metrics={"completeness": round(rate, 4), "filled": filled},
                finding=finding, action=action)


def _recency(rows: list[dict], present: bool, now: datetime) -> dict:
    w = WEIGHTS["recency"]
    if not present:
        return _not_provided("recency", "Recency", w,
                             "No date column was provided, so this check was not assessed.")
    newest = max(r["when"] for r in rows)
    age_days = max(0, (now - newest).days)
    last_12 = sum(1 for r in rows if (now - r["when"]).days <= 365)
    n = len(rows)
    share = last_12 / n if n else 0.0
    newest_part = 1.0 if age_days <= 90 else max(0.0, 1.0 - (age_days - 90) / 275.0)
    score = 100.0 * (0.6 * share + 0.4 * newest_part)
    if share >= 0.40 and age_days <= 180:
        status = "ok"
    elif share >= 0.15 or age_days <= 365:
        status = "warn"
    else:
        status = "fail"
    finding = (
        f"{share:.0%} of lines fall in the last 12 months. The newest line is {age_days} day"
        f"{'' if age_days == 1 else 's'} old ({newest.date().isoformat()})."
    )
    action = None
    if status != "ok":
        action = "Upload more recently coded invoices so the estimate reflects current coding practice."
    return _cat("recency", "Recency", w, assessed=True, status=status, score=score,
                metrics={"last_12_months_share": round(share, 4), "newest_line": newest.date().isoformat(),
                         "newest_age_days": age_days},
                finding=finding, action=action)


# --------------------------------------------------------------------------- isolated replay


@contextmanager
def _isolated_backends(overrides: dict | None = None):
    """Throwaway store + hashing embedder; restore every global we touch. Never posts to an ERP."""
    saved_client = retrieval._client_override
    saved_embedder = embeddings._embedder_override
    saved_vendors = dict(mdm.VENDOR_MAP)
    keys = set(_ISOLATE_KEYS)
    if overrides:
        keys.update(overrides)
    saved = {k: getattr(settings, k) for k in keys}
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    try:
        root = Path(tmp.name)
        settings.audit_db_path = str(root / "audit.db")
        settings.erp_db_path = str(root / "erp.db")
        settings.predictions_db_path = str(root / "predictions.db")
        settings.erp_backend = "simulated"
        settings.erp_url = ""
        settings.xero_posting_enabled = False
        settings.sap_posting_enabled = False
        settings.telemetry_enabled = False
        for key, val in (overrides or {}).items():
            setattr(settings, key, val)
        retrieval.set_client(_inproc_store())
        embeddings.set_embedder(offline.HashingEmbedder())
        retrieval.ensure_collection()
        yield
    finally:
        retrieval.set_client(saved_client)
        embeddings.set_embedder(saved_embedder)
        mdm.VENDOR_MAP.clear()
        mdm.VENDOR_MAP.update(saved_vendors)
        for key, val in saved.items():
            setattr(settings, key, val)
        tmp.cleanup()


def _replay(rows: list[dict], *, profile: str = "standard", overrides: dict | None = None) -> dict:
    """Seed ~60%, replay ~40% through predict() + isolated write-back of review lines only."""
    w = WEIGHTS["replay"]
    insufficient = (
        f"insufficient data for a replay estimate (need at least {MIN_LINES_FOR_REPLAY} usable lines "
        f"and {MIN_REPLAYABLE} replayable lines)"
    )
    if len(rows) < MIN_LINES_FOR_REPLAY:
        return _cat("replay", "Replay estimate", w, assessed=False, status="not_provided", score=None,
                    metrics={"reason": insufficient, "lines": len(rows)},
                    finding=insufficient, action="Import more coded history and run this check again.")

    ordered = sorted(rows, key=lambda r: (r["when"], r["key"]))
    seed_n = int(len(ordered) * SEED_FRACTION)
    replay_rows = ordered[seed_n:]
    capped = False
    if len(replay_rows) > MAX_REPLAY_LINES:
        replay_rows = replay_rows[:MAX_REPLAY_LINES]
        capped = True
    if len(replay_rows) < MIN_REPLAYABLE:
        return _cat("replay", "Replay estimate", w, assessed=False, status="not_provided", score=None,
                    metrics={"reason": insufficient, "lines": len(rows), "replayable": len(replay_rows)},
                    finding=insufficient, action="Import more coded history and run this check again.")

    seed_rows = ordered[:seed_n]
    auto = correct_auto = correct_all = 0
    with _isolated_backends(overrides):
        for r in seed_rows:
            mdm.register_vendor(r["vendor"])  # in-memory only; persist_path is omitted
        for r in replay_rows:
            mdm.register_vendor(r["vendor"])
        from ap_automation.core.seeding import seed_corpus

        seed_corpus(
            [
                InvoiceLine(
                    invoice_id=uuid4(), raw_vendor_name=r["vendor"], description=r["description"],
                    amount=r["amount"], currency=r["currency"], entity_id=r["entity_id"],
                    cost_centre=r["cost_centre"] or None, ground_truth_gl=r["gl"],
                )
                for r in seed_rows
            ],
            timestamp=(seed_rows[-1]["when"] if seed_rows else datetime.now(timezone.utc)).isoformat(),
        )
        for r in replay_rows:
            inv = InvoiceLine(
                invoice_id=uuid4(), raw_vendor_name=r["vendor"], description=r["description"],
                amount=r["amount"], currency=r["currency"], entity_id=r["entity_id"],
                cost_centre=r["cost_centre"] or None, ground_truth_gl=r["gl"],
            )
            pred = predict(inv, batch_mode=True)
            is_auto = pred.final_routing == RoutingDecision.AUTO_POST
            is_correct = pred.predicted_gl == r["gl"]
            if is_auto:
                auto += 1
                correct_auto += int(is_correct)
            correct_all += int(is_correct)
            if not is_auto:
                source = (ConfirmationSource.HUMAN_CONFIRM if is_correct
                          else ConfirmationSource.HUMAN_CORRECT)
                write_back(ConfirmedOutcome(
                    prediction_id=pred.prediction_id, invoice_id=inv.invoice_id,
                    canonical_vendor_id=pred.resolved_line.canonical_vendor_id,
                    entity_id=inv.entity_id, amount_band=pred.resolved_line.amount_band,
                    currency=inv.currency, gl_account=r["gl"],
                    cost_centre=inv.cost_centre or pred.predicted_cost_centre or "",
                    tax_code="UNASSESSED", confirming_user_id="readiness_replay",
                    confirmation_timestamp=r["when"], prediction_confidence=pred.weighted_confidence,
                    source=source, description=inv.description, vendor_name=inv.raw_vendor_name,
                    amount=inv.amount,
                ))

    n = len(replay_rows)
    automation = auto / n if n else 0.0
    accuracy = (correct_auto / auto) if auto else 0.0
    overall = correct_all / n if n else 0.0
    wrong = auto - correct_auto
    score = 100.0 * accuracy * (0.4 + 0.6 * min(automation / 0.40, 1.0)) if auto else 20.0 * min(overall / 0.5, 1.0)
    if auto and accuracy >= 0.97 and automation >= 0.25:
        status = "ok"
    elif auto and accuracy >= 0.90:
        status = "warn"
    else:
        status = "fail"
    finding = (
        f"On {n} held-out lines (seeded {len(seed_rows)} earlier lines into a throwaway store), "
        f"{automation:.1%} would auto-post, of which {accuracy:.1%} matched the client's GL "
        f"({wrong} wrong auto-post{'' if wrong == 1 else 's'}). Overall GL accuracy {overall:.1%}. "
        f"This is an estimate on this file, using the offline hashing embedder and the "
        f"“{profile}” gate profile — not a live ERP post."
    )
    action = None
    if status != "ok":
        action = (
            "Improve description quality and vendor-name consistency, then add more recent coded lines "
            "and re-run this check before turning on auto-post."
        )
    return _cat(
        "replay", "Replay estimate", w, assessed=True, status=status, score=score,
        metrics={
            "lines_seeded": len(seed_rows), "lines_replayed": n, "capped": capped,
            "automation_rate": round(automation, 4), "auto_posts": auto,
            "auto_post_accuracy": round(accuracy, 4) if auto else None,
            "wrong_auto_posts": wrong, "overall_gl_accuracy": round(overall, 4),
            "embedder": "offline-hashing", "profile": profile,
        },
        finding=finding, action=action,
    )


# --------------------------------------------------------------------------- scoring / verdict / narrative


def _weighted_score(categories: list[dict]) -> tuple[float, dict[str, float]]:
    used = {c["id"]: c["weight"] for c in categories if c["assessed"] and c["score"] is not None}
    total = sum(used.values()) or 1
    weights_used = {k: round(v / total, 4) for k, v in used.items()}
    score = 0.0
    for c in categories:
        if c["id"] in used:
            score += c["score"] * used[c["id"]] / total
    return round(score), weights_used


def _verdict(score: float, categories: list[dict], rows_stat: dict) -> tuple[str, str]:
    by_id = {c["id"]: c for c in categories}
    replay = by_id.get("replay") or {}
    volume = by_id.get("volume_sufficiency") or {}
    chart = by_id.get("chart_coverage") or {}
    metrics = replay.get("metrics") or {}
    hard = []
    if volume.get("status") == "fail":
        hard.append("too few lines or vendors that meet the live gate")
    if rows_stat["invalid"] > 0 and rows_stat["invalid"] / max(rows_stat["total"], 1) > NOT_READY_INVALID:
        hard.append("most rows are invalid")
    off = (chart.get("metrics") or {}).get("off_chart_rate") or 0.0
    if off > NOT_READY_OFF_CHART:
        hard.append("most rows use a GL that is not in the chart")
    # "Poor" replay (not ready) is distinct from "below the auto-post bar" (suggest-only).
    acc = metrics.get("auto_post_accuracy")
    if replay.get("assessed") and (
        (acc is not None and acc < 0.80) or (acc is None and (metrics.get("automation_rate") or 0) < 0.10)
    ):
        hard.append("the replay estimate is poor")
    auto_ok = (
        replay.get("assessed")
        and (metrics.get("automation_rate") or 0) >= AUTO_POST_MIN_AUTOMATION
        and (metrics.get("auto_post_accuracy") or 0) >= AUTO_POST_MIN_ACCURACY
        and not hard
    )
    if auto_ok:
        return (
            "auto-post ready",
            f"Replay estimated {metrics.get('automation_rate', 0):.0%} automation at "
            f"{metrics.get('auto_post_accuracy', 0):.1%} auto-post accuracy, and no hard checks failed. "
            "Thresholds are provisional.",
        )
    if hard or score < NOT_READY_MAX_SCORE:
        return "not ready", ("Not ready: " + "; ".join(hard) + "." if hard else
                             f"Overall score {score} is below the provisional not-ready bar ({NOT_READY_MAX_SCORE}).")
    return (
        "suggest-only",
        "History is usable for suggestions, but automation rate, auto-post accuracy or data-quality "
        "warnings are below the provisional auto-post bar.",
    )


def _actions(categories: list[dict]) -> list[dict]:
    rank = {"fail": 0, "warn": 1, "ok": 2, "not_provided": 3}
    out = []
    for c in sorted(categories, key=lambda x: (rank.get(x["status"], 9), -x["weight"])):
        if c["action"]:
            out.append({"priority": len(out) + 1, "category": c["id"], "status": c["status"],
                        "action": c["action"]})
    return out


def _summary(score: int, mode: str, categories: list[dict], replay: dict, rows_stat: dict) -> str:
    assessed = [c for c in categories if c["assessed"]]
    warns = [c["name"] for c in assessed if c["status"] == "warn"]
    fails = [c["name"] for c in assessed if c["status"] == "fail"]
    bits = [
        f"This file scores {score}/100 with recommended mode “{mode}”.",
        f"{rows_stat['valid']} of {rows_stat['total']} rows were usable "
        f"({rows_stat['invalid']} invalid, {rows_stat['off_chart']} off-chart).",
    ]
    if replay.get("assessed"):
        m = replay["metrics"]
        bits.append(
            f"Replay of {m['lines_replayed']} lines estimated {m['automation_rate']:.0%} auto-post "
            f"at {m['auto_post_accuracy']:.0%} accuracy among auto-posts."
            if m.get("auto_post_accuracy") is not None else
            f"Replay of {m['lines_replayed']} lines estimated {m['automation_rate']:.0%} auto-post."
        )
    else:
        bits.append((replay.get("metrics") or {}).get("reason") or "No replay estimate.")
    if fails:
        bits.append("Failed checks: " + ", ".join(fails) + ".")
    if warns:
        bits.append("Warnings: " + ", ".join(warns) + ".")
    bits.append("Score and mode thresholds are provisional and should be recalibrated against real client data.")
    return " ".join(bits)


def _run_replays(valid: list[dict], used: str) -> tuple[dict, dict | None]:
    """Always run standard; also run `used` when different, otherwise tuned. At most two replays."""
    if not valid:
        empty = _insufficient_replay(0, "insufficient data for a replay estimate")
        return empty, None
    standard = _replay(valid, profile="standard")
    second_name = used if used != "standard" else "tuned"
    second = _replay(valid, profile=second_name, overrides=PROFILE_OVERRIDES[second_name])
    return standard, second


def assess(text: str, now: datetime | None = None, gate_profile: str = "standard") -> dict:
    """
    Score a history CSV. JSON-serialisable. No side effects on live stores.

    Raises HistoryImportError when the file as a whole cannot be read (same rules as import),
    or when gate_profile is not one of standard / tuned / tuned_context.
    """
    requested = _validate_profile(gate_profile)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    headers = _headers(text)
    has_cc = _has_col(headers, history_import._CC)
    has_date = _has_col(headers, history_import._DATE)
    has_vat = _has_col(headers, history_import._TAX)
    has_context = _has_context_columns(headers)
    used = requested
    context_unavailable = False
    if requested == "tuned_context" and not has_context:
        used = "tuned"
        context_unavailable = True

    with _ASSESS_LOCK:
        snap = _settings_snapshot()
        try:
            valid, rejected, rows_read, duplicates = parse_rows(text, now, allow_unknown_gl=True)
            standard_replay, extra_replay = _run_replays(valid, used)
            selected_replay = extra_replay if used != "standard" and extra_replay is not None else standard_replay
            categories = [
                _vendor_consistency(valid) if valid else _cat(
                    "vendor_consistency", "Vendor consistency", WEIGHTS["vendor_consistency"],
                    assessed=True, status="fail", score=0, metrics={},
                    finding="No usable rows, so vendor consistency cannot be measured.",
                    action="Fix the rejected rows and upload again."),
                _gl_consistency(valid) if valid else _cat(
                    "gl_coding_consistency", "GL coding consistency", WEIGHTS["gl_coding_consistency"],
                    assessed=True, status="fail", score=0, metrics={},
                    finding="No usable rows, so GL consistency cannot be measured.",
                    action="Fix the rejected rows and upload again."),
                _chart_coverage(valid) if valid else _cat(
                    "chart_coverage", "Chart coverage", WEIGHTS["chart_coverage"],
                    assessed=True, status="fail", score=0, metrics={"off_chart_rate": 1.0},
                    finding="No usable rows.", action="Fix the rejected rows and upload again."),
                _description_quality(valid) if valid else _cat(
                    "description_quality", "Description quality", WEIGHTS["description_quality"],
                    assessed=True, status="fail", score=0, metrics={},
                    finding="No usable rows.", action="Fix the rejected rows and upload again."),
                _volume(valid),
                _cost_centre(valid, has_cc),
                _vat(valid, has_vat),
                _recency(valid, has_date, now),
                selected_replay if valid else _insufficient_replay(
                    0, "insufficient data for a replay estimate"),
            ]
            off_chart = sum(1 for r in valid if not r.get("in_chart"))
            reason_counts: Counter[str] = Counter()
            for r in rejected:
                reason = r["reason"]
                reason_counts["off-chart GL" if reason.startswith("GL ") and "not in the active chart" in reason
                              else reason] += 1
            rows_stat = {
                "total": rows_read,
                "valid": len(valid),
                "invalid": len(rejected),
                "invalid_reasons": dict(reason_counts),
                "duplicates_in_file": duplicates,
                "off_chart": off_chart,
                "capped": False,
            }
            score, weights_used = _weighted_score(categories)
            mode, mode_reason = _verdict(score, categories, rows_stat)
            replay = next(c for c in categories if c["id"] == "replay")
            replay_out = _replay_payload(replay, used)
            replays = [_replay_payload(standard_replay, "standard")]
            if extra_replay is not None:
                extra_name = used if used != "standard" else "tuned"
                replays.append(_replay_payload(extra_replay, extra_name))
            if context_unavailable:
                replays.append({
                    "profile": "tuned_context",
                    "status": "not_available",
                    "reason": "not available: file has no context columns",
                    "lines": None,
                })
        finally:
            _restore_settings(snap)

    return {
        "score": score,
        "mode": mode,
        "mode_reason": mode_reason,
        "gate_profile": requested,
        "gate_profile_used": used,
        "context_available": has_context,
        "categories": categories,
        "replay": replay_out,
        "replays": replays,
        "replay_note": REPLAY_NOTE,
        "actions": _actions(categories),
        "summary": _summary(score, mode, categories, replay, rows_stat),
        "rows": rows_stat,
        "weights": WEIGHTS,
        "weights_used": weights_used,
        "thresholds_provisional": True,
        "thresholds": {
            "auto_post_min_automation": AUTO_POST_MIN_AUTOMATION,
            "auto_post_min_accuracy": AUTO_POST_MIN_ACCURACY,
            "not_ready_max_score": NOT_READY_MAX_SCORE,
            "note": "PROVISIONAL — recalibrate against real client data.",
        },
    }
