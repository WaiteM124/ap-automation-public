"""
Confidence scoring and routing gate.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from ap_automation.core.config import settings
from ap_automation.core.models import RetrievedEvidence, RoutingDecision

# Per-vendor confidence thresholds (override the default for specific vendors).
VENDOR_THRESHOLDS: dict[str, float] = {}

# Weighting: agreement is the primary signal, similarity is a secondary modifier
AGREEMENT_WEIGHT = 0.7
SIMILARITY_WEIGHT = 0.3


def get_threshold(vendor_id: str) -> float:
    return VENDOR_THRESHOLDS.get(vendor_id, settings.default_confidence_threshold)


def score(evidence: list[RetrievedEvidence]) -> tuple[str, str, float, float]:
    """
    Compute confidence from retrieved evidence.

    Returns:
        predicted_gl: the GL account most common in the evidence
        predicted_cost_centre: the cost centre most common among lines agreeing on that GL
        agreement_ratio: proportion of the evidence sharing predicted_gl
        weighted_confidence: (agreement_ratio * 0.7) + (avg similarity of agreeing lines * 0.3)

    Agreement is the primary signal. Similarity rewards closer semantic matches.
    """
    if not evidence:
        return "", "", 0.0, 0.0

    gl_counts = Counter(e.gl_account for e in evidence)
    predicted_gl, count = gl_counts.most_common(1)[0]
    agreement_ratio = count / len(evidence)

    agreeing = [e for e in evidence if e.gl_account == predicted_gl]
    avg_similarity = sum(e.similarity_score for e in agreeing) / len(agreeing)

    weighted_confidence = (agreement_ratio * AGREEMENT_WEIGHT) + (avg_similarity * SIMILARITY_WEIGHT)

    cc_counts = Counter(e.cost_centre for e in agreeing)
    predicted_cc = cc_counts.most_common(1)[0][0] if cc_counts else ""

    return predicted_gl, predicted_cc, agreement_ratio, weighted_confidence


def route(
    evidence: list[RetrievedEvidence],
    vendor_id: str,
    weighted_confidence: float,
    mdm_confidence: float = 1.0,
    agreement_ratio: float | None = None,
) -> tuple[RoutingDecision, str]:
    """
    Rule-gate routing decision. Auto-post requires ALL of:

    1. at least `min_evidence_lines` retrieved lines for this vendor (cold start -> review)
    2. vendor identity resolved with confidence >= `min_mdm_confidence`
    3. agreement_ratio >= `min_agreement_ratio` (a slim plurality is not enough)
    4. weighted confidence >= the vendor threshold

    Otherwise: human review. The first failing rule supplies the reason.
    """
    if len(evidence) < settings.min_evidence_lines:
        return (
            RoutingDecision.HUMAN_REVIEW,
            f"Sparse retrieval: only {len(evidence)} lines retrieved "
            f"(minimum {settings.min_evidence_lines} required)",
        )

    if mdm_confidence < settings.min_mdm_confidence:
        return (
            RoutingDecision.HUMAN_REVIEW,
            f"Vendor identity uncertain: MDM confidence {mdm_confidence:.2f} below "
            f"{settings.min_mdm_confidence:.2f}",
        )

    if agreement_ratio is None:
        counts = Counter(e.gl_account for e in evidence)
        agreement_ratio = counts.most_common(1)[0][1] / len(evidence)
    if agreement_ratio < settings.min_agreement_ratio:
        return (
            RoutingDecision.HUMAN_REVIEW,
            f"Evidence disagrees: agreement {agreement_ratio:.0%} below "
            f"{settings.min_agreement_ratio:.0%}",
        )

    threshold = get_threshold(vendor_id)
    if weighted_confidence < threshold:
        return (
            RoutingDecision.HUMAN_REVIEW,
            f"Confidence {weighted_confidence:.3f} below threshold {threshold:.3f} "
            f"for vendor {vendor_id}",
        )

    return (
        RoutingDecision.AUTO_POST,
        f"Confidence {weighted_confidence:.3f} meets threshold {threshold:.3f}",
    )


# --------------------------------------------------------------------------- gate variants (opt-in)
#
# Everything below is OFF unless a gate_* setting is changed: the shipped path is score() + route() above.
# The same pure functions are used by the replay sweeps (core/gate_analysis.py has the vectorised twins).

@dataclass(frozen=True)
class GateParams:
    """Knobs of the gate variants. The all-default instance means 'shipped behaviour'."""
    min_margin: float = 0.0
    min_agreement_lb: float = 0.0
    wilson_z: float = 1.0
    diversity_per_code: float = 0.0
    diversity_scope: str = "vendor"
    diversity_cap: int = 10
    recency_halflife_days: float = 0.0
    recency_window_days: float = 0.0
    required_context: tuple[str, ...] = ()

    @classmethod
    def from_settings(cls) -> GateParams:
        return cls(
            min_margin=settings.gate_min_margin,
            min_agreement_lb=settings.gate_min_agreement_lb,
            wilson_z=settings.gate_wilson_z,
            diversity_per_code=settings.gate_diversity_per_code,
            diversity_scope=settings.gate_diversity_scope,
            diversity_cap=settings.gate_diversity_cap,
            recency_halflife_days=settings.gate_recency_halflife_days,
            recency_window_days=settings.gate_recency_window_days,
            required_context=tuple(s.strip() for s in settings.gate_required_context.split(",") if s.strip()),
        )

    def uses_variants(self) -> bool:
        return (self.min_margin > 0 or self.min_agreement_lb > 0 or self.diversity_per_code > 0
                or self.recency_halflife_days > 0 or self.recency_window_days > 0 or bool(self.required_context))


@dataclass(frozen=True)
class VoteFeatures:
    """What the gate needs to know about an evidence set, after any recency weighting."""
    predicted_gl: str
    predicted_cc: str
    n: int                  # evidence lines that vote
    agree_count: int        # of those, lines agreeing with predicted_gl
    agreement: float        # (weighted) share of the vote won by predicted_gl
    margin: float           # winner's share minus runner-up's share
    avg_similarity: float   # (weighted) mean similarity of the agreeing lines
    weighted_confidence: float
    n_eff: float            # Kish effective sample size of the weights (== n when unweighted)


def recency_weight(age_days: float, halflife_days: float) -> float:
    """0.5 ** (age / halflife); 1.0 when halflife <= 0 (off). Future-dated evidence counts as age 0."""
    if halflife_days <= 0:
        return 1.0
    return 0.5 ** (max(age_days, 0.0) / halflife_days)


def wilson_lower(p: float, n: float, z: float = 1.0) -> float:
    """Lower end of the Wilson score interval for a proportion p observed over n (effective) trials."""
    if n <= 0:
        return 0.0
    z2 = z * z
    centre = p + z2 / (2 * n)
    spread = z * math.sqrt(max(p * (1 - p), 0.0) / n + z2 / (4 * n * n))
    return max((centre - spread) / (1 + z2 / n), 0.0)


def required_agreeing_lines(distinct_codes: int, per_code: float, cap: int) -> int:
    """Agreeing lines demanded of a vendor that uses `distinct_codes` account codes (0 when the rule is off)."""
    if per_code <= 0:
        return 0
    return min(cap, math.ceil(per_code * max(distinct_codes, 0)))


def vote_features(evidence: list[RetrievedEvidence], as_of: datetime, params: GateParams,
                  window_min_lines: int = 3) -> VoteFeatures:
    """
    score() generalised with recency: each line votes with weight 0.5 ** (age / halflife) and, if a window is set,
    only lines newer than the window vote (all lines vote when fewer than `window_min_lines` are that recent).
    With the default params this is exactly score(): same winner (first-seen on ties), same numbers.
    """
    if not evidence:
        return VoteFeatures("", "", 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    def age(e: RetrievedEvidence) -> float:
        ts = e.confirmation_timestamp if e.confirmation_timestamp.tzinfo else e.confirmation_timestamp.replace(tzinfo=timezone.utc)
        return (as_of - ts).total_seconds() / 86400.0

    used = list(evidence)
    if params.recency_window_days > 0:
        recent = [e for e in evidence if age(e) <= params.recency_window_days]
        if len(recent) >= window_min_lines:
            used = recent
    weights = [recency_weight(age(e), params.recency_halflife_days) for e in used]
    votes: dict[str, float] = {}
    for e, w in zip(used, weights, strict=True):
        votes[e.gl_account] = votes.get(e.gl_account, 0.0) + w
    total = sum(weights)
    predicted_gl = max(votes, key=lambda g: votes[g])  # first max in insertion order, like Counter.most_common
    top = votes[predicted_gl]
    runner_up = max((v for g, v in votes.items() if g != predicted_gl), default=0.0)
    agreeing = [(e, w) for e, w in zip(used, weights, strict=True) if e.gl_account == predicted_gl]
    agree_w = sum(w for _, w in agreeing)
    avg_sim = sum(e.similarity_score * w for e, w in agreeing) / agree_w if agree_w > 0 else 0.0
    agreement = top / total if total > 0 else 0.0
    cc = Counter(e.cost_centre for e, _ in agreeing)
    n_eff = (total * total) / sum(w * w for w in weights) if total > 0 else 0.0
    return VoteFeatures(
        predicted_gl=predicted_gl, predicted_cc=cc.most_common(1)[0][0] if cc else "", n=len(used),
        agree_count=len(agreeing), agreement=agreement, margin=(top - runner_up) / total if total > 0 else 0.0,
        avg_similarity=avg_sim, weighted_confidence=agreement * AGREEMENT_WEIGHT + avg_sim * SIMILARITY_WEIGHT,
        n_eff=n_eff,
    )


def route_variant(
    f: VoteFeatures,
    vendor_id: str,
    params: GateParams,
    mdm_confidence: float = 1.0,
    vendor_distinct_codes: int = 0,
    subset_distinct_codes: int = 0,
    context_level: str = "vendor",
) -> tuple[RoutingDecision, str]:
    """
    route() plus the opt-in extras. Order of checks: sparse evidence, vendor identity, agreement, margin, Wilson
    bound, diversity-scaled agreeing-line requirement, weighted confidence, required context level. With default
    params (and f from vote_features) the decision equals route()'s.
    """
    if f.n < settings.min_evidence_lines:
        return RoutingDecision.HUMAN_REVIEW, (
            f"Sparse retrieval: only {f.n} lines retrieved (minimum {settings.min_evidence_lines} required)")
    if mdm_confidence < settings.min_mdm_confidence:
        return RoutingDecision.HUMAN_REVIEW, (
            f"Vendor identity uncertain: MDM confidence {mdm_confidence:.2f} below {settings.min_mdm_confidence:.2f}")
    if f.agreement < settings.min_agreement_ratio:
        return RoutingDecision.HUMAN_REVIEW, (
            f"Evidence disagrees: agreement {f.agreement:.0%} below {settings.min_agreement_ratio:.0%}")
    if params.min_margin > 0 and f.margin < params.min_margin:
        return RoutingDecision.HUMAN_REVIEW, (
            f"Evidence disagrees: margin over the runner-up {f.margin:.0%} below {params.min_margin:.0%}")
    if params.min_agreement_lb > 0:
        lb = wilson_lower(f.agreement, f.n_eff, params.wilson_z)
        if lb < params.min_agreement_lb:
            return RoutingDecision.HUMAN_REVIEW, (
                f"Evidence disagrees: agreement lower bound {lb:.2f} below {params.min_agreement_lb:.2f}")
    if params.diversity_per_code > 0:
        d = subset_distinct_codes if params.diversity_scope == "subset" else vendor_distinct_codes
        need = required_agreeing_lines(d, params.diversity_per_code, params.diversity_cap)
        if f.agree_count < need:
            return RoutingDecision.HUMAN_REVIEW, (
                f"Sparse retrieval: only {f.agree_count} agreeing lines; a vendor with {d} account codes needs {need}")
    threshold = get_threshold(vendor_id)
    if f.weighted_confidence < threshold:
        return RoutingDecision.HUMAN_REVIEW, (
            f"Confidence {f.weighted_confidence:.3f} below threshold {threshold:.3f} for vendor {vendor_id}")
    if params.required_context and context_level not in params.required_context:
        return RoutingDecision.HUMAN_REVIEW, (
            f"Context evidence thin: evidence matched at '{context_level}', need one of {list(params.required_context)}")
    return RoutingDecision.AUTO_POST, f"Confidence {f.weighted_confidence:.3f} meets threshold {threshold:.3f}"
