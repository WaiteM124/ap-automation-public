"""
Vectorised, side-effect-free helpers for studying the gate offline on recorded evidence.

A replay records, once per line, the evidence the retriever returned (account, similarity, age). Everything here
then re-applies gate variants to those recordings without touching the vector store, which is what makes a
parameter sweep over thousands of gates cheap. The scalar twins live in core/confidence.py
(vote_features / route_variant) and tests/test_gate_analysis.py checks the two agree line for line.

Nothing here reads a label except to score an outcome (`correct`); it never sees any column derived from the
account being predicted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

AGREEMENT_WEIGHT = 0.7
SIMILARITY_WEIGHT = 0.3
CONTEXT_LEVELS: tuple[str, ...] = ("dept+fund", "fund", "dept", "vendor")


# --------------------------------------------------------------------------- statistics

def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%). (0, 1) when n == 0."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    denom = 1 + z2 / n
    return max((centre - spread) / denom, 0.0), min((centre + spread) / denom, 1.0)


def recency_weights(age_days: np.ndarray, halflife_days: float) -> np.ndarray:
    """0.5 ** (age / halflife) elementwise; all ones when halflife <= 0. Negative ages count as 0."""
    if halflife_days <= 0:
        return np.ones_like(age_days, dtype=np.float64)
    return np.power(0.5, np.maximum(age_days, 0.0) / halflife_days)


def calibration_table(confidence: np.ndarray, correct: np.ndarray, auto: np.ndarray,
                      edges: Sequence[tuple[float, float, str]]) -> dict:
    """Accuracy per confidence band (all predictions and auto-posts) plus ECE over the predictions."""
    rows, ece_num, total = [], 0.0, len(confidence)
    for lo, hi, label in edges:
        m = (confidence >= lo) & (confidence < hi)
        n = int(m.sum())
        a = m & auto
        acc = float(correct[m].mean()) if n else None
        mc = float(confidence[m].mean()) if n else None
        if n:
            ece_num += n * abs(acc - mc)
        rows.append({"band": label, "n": n, "accuracy": None if acc is None else round(acc, 4),
                     "mean_confidence": None if mc is None else round(mc, 4),
                     "n_auto_posted": int(a.sum()),
                     "auto_post_accuracy": round(float(correct[a].mean()), 4) if a.any() else None})
    return {"bins": rows, "ece_all_predictions": round(ece_num / total, 4) if total else None}


# --------------------------------------------------------------------------- vote features (vector twin of vote_features)

@dataclass
class Votes:
    """Per-line gate features, each an array of length N."""
    predicted: np.ndarray        # index into the recorded evidence of the winning line (its code is the prediction)
    n: np.ndarray
    agree_count: np.ndarray
    agreement: np.ndarray
    margin: np.ndarray
    avg_similarity: np.ndarray
    weighted_confidence: np.ndarray
    n_eff: np.ndarray
    predicted_code: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def vote_arrays(codes: np.ndarray, sims: np.ndarray, ages: np.ndarray, valid: np.ndarray, k: int,
                halflife_days: float = 0.0, window_days: float = 0.0, window_min_lines: int = 3) -> Votes:
    """
    Vote features for the first `k` evidence lines of every row.

    codes (N,K) int account ids, sims (N,K), ages (N,K) days, valid (N,K) bool. Rows are in retrieval order, so the
    winner on a tie is the code seen first, exactly as in confidence.vote_features / Counter.most_common.
    """
    c, s, a, v = codes[:, :k], sims[:, :k], ages[:, :k], valid[:, :k]
    use = v.copy()
    if window_days > 0:
        recent = v & (a <= window_days)
        enough = recent.sum(axis=1) >= window_min_lines
        use = np.where(enough[:, None], recent, v)
    w = np.where(use, recency_weights(a, halflife_days), 0.0)
    same = (c[:, :, None] == c[:, None, :]) & use[:, :, None] & use[:, None, :]       # (N,K,K)
    vote = (same * w[:, :, None]).sum(axis=1)                                           # vote of the code of line j
    vote = np.where(use, vote, -1.0)
    top_j = vote.argmax(axis=1)                                                         # first max wins ties
    rows = np.arange(len(c))
    top_code = c[rows, top_j]
    total = w.sum(axis=1)
    top_vote = vote[rows, top_j]
    other = np.where(use & (c != top_code[:, None]), vote, 0.0).max(axis=1)
    agreeing = use & (c == top_code[:, None])
    agree_w = np.where(agreeing, w, 0.0).sum(axis=1)
    avg_sim = np.divide((np.where(agreeing, s, 0.0) * w).sum(axis=1), agree_w, out=np.zeros(len(c)), where=agree_w > 0)
    safe_total = np.where(total > 0, total, 1.0)
    agreement = np.where(total > 0, top_vote / safe_total, 0.0)
    sum_w2 = (w * w).sum(axis=1)
    n_eff = np.divide(total * total, sum_w2, out=np.zeros(len(c)), where=sum_w2 > 0)
    return Votes(
        predicted=top_j, n=use.sum(axis=1), agree_count=agreeing.sum(axis=1), agreement=agreement,
        margin=np.where(total > 0, (top_vote - other) / safe_total, 0.0), avg_similarity=avg_sim,
        weighted_confidence=agreement * AGREEMENT_WEIGHT + avg_sim * SIMILARITY_WEIGHT, n_eff=n_eff,
        predicted_code=np.where(total > 0, top_code, -1),
    )


# --------------------------------------------------------------------------- gate (vector twin of route_variant)

@dataclass(frozen=True)
class SweepGate:
    min_n: int = 3
    min_agreement: float = 0.80
    min_wc: float = 0.50
    min_margin: float = 0.0
    min_lb: float = 0.0
    z: float = 1.0
    div_per_code: float = 0.0
    div_scope: str = "vendor"
    div_cap: int = 10
    required_levels: tuple[str, ...] = ()
    min_exact_lines: int = 0          # lines at dept+fund the vendor must already hold (0 = off)


def wilson_lower_array(p: np.ndarray, n: np.ndarray, z: float) -> np.ndarray:
    z2 = z * z
    safe_n = np.where(n > 0, n, 1.0)
    centre = p + z2 / (2 * safe_n)
    spread = z * np.sqrt(np.maximum(p * (1 - p), 0.0) / safe_n + z2 / (4 * safe_n * safe_n))
    return np.where(n > 0, np.maximum((centre - spread) / (1 + z2 / safe_n), 0.0), 0.0)


def gate_mask(v: Votes, g: SweepGate, vendor_distinct: np.ndarray, subset_distinct: np.ndarray,
              level_idx: np.ndarray, n_exact: np.ndarray | None = None, cache: dict | None = None) -> np.ndarray:
    """
    Boolean auto-post decision per line. Same conditions, same comparisons as confidence.route_variant.

    `cache` (optional, one per Votes object) memoises each single-condition mask, which makes sweeping thousands of
    gates over the same recording cheap: a gate is then just an AND of cached masks.
    """
    def cond(key: tuple, make):
        if cache is None:
            return make()
        if key not in cache:
            cache[key] = make()
        return cache[key]

    ok = cond(("n", g.min_n), lambda: v.n >= g.min_n) & cond(("agree", g.min_agreement),
                                                           lambda: v.agreement >= g.min_agreement)
    if g.min_margin > 0:
        ok = ok & cond(("margin", g.min_margin), lambda: v.margin >= g.min_margin)
    if g.min_lb > 0:
        ok = ok & cond(("lb", g.min_lb, g.z),
                       lambda: wilson_lower_array(v.agreement, v.n_eff, g.z) >= g.min_lb)
    if g.div_per_code > 0:
        def _div():
            d = subset_distinct if g.div_scope == "subset" else vendor_distinct
            need = np.minimum(g.div_cap, np.ceil(g.div_per_code * np.maximum(d, 0))).astype(int)
            return v.agree_count >= need
        ok = ok & cond(("div", g.div_per_code, g.div_scope, g.div_cap), _div)
    ok = ok & cond(("wc", g.min_wc), lambda: v.weighted_confidence >= g.min_wc)
    if g.required_levels:
        allowed = np.array([CONTEXT_LEVELS.index(lv) for lv in g.required_levels])
        ok = ok & cond(("lvl", g.required_levels), lambda: np.isin(level_idx, allowed))
    if g.min_exact_lines > 0 and n_exact is not None:
        ok = ok & (n_exact >= g.min_exact_lines)
    return ok


# --------------------------------------------------------------------------- coverage vs accuracy

def coverage_accuracy(auto: np.ndarray, correct: np.ndarray) -> tuple[float, float | None, int, int]:
    """(coverage, auto-post accuracy or None, n auto-posted, n auto-posted wrong)."""
    n_auto = int(auto.sum())
    wrong = int((auto & ~correct).sum())
    return (n_auto / len(auto) if len(auto) else 0.0,
            (n_auto - wrong) / n_auto if n_auto else None, n_auto, wrong)


def risk_coverage_curve(score: np.ndarray, correct: np.ndarray, min_posts: int = 1) -> list[dict]:
    """
    Selective-prediction curve for ONE ranking score: auto-post the top-m lines by score for every m. Ties in the
    score are kept together (a threshold cannot split them). Returns rows of coverage / accuracy / counts.
    """
    order = np.argsort(-score, kind="stable")
    s, ok = score[order], correct[order]
    cum_ok = np.cumsum(ok)
    n = len(s)
    ends = np.append(np.nonzero(s[1:] != s[:-1])[0], n - 1)  # last index of each tie group
    rows = []
    for e in ends:
        m = int(e) + 1
        if m < min_posts:
            continue
        rows.append({"threshold": float(s[e]), "n_auto": m, "coverage": m / n,
                     "accuracy": float(cum_ok[e]) / m, "n_wrong": m - int(cum_ok[e])})
    return rows


def coverage_at_accuracy(curve: list[dict], target: float, min_posts: int = 30) -> dict | None:
    """Largest-coverage point of a curve whose accuracy >= target with at least `min_posts` auto-posts."""
    best = None
    for r in curve:
        if r["n_auto"] >= min_posts and r["accuracy"] >= target and (best is None or r["coverage"] > best["coverage"]):
            best = r
    return best


def pareto_front(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of the points not dominated in (coverage, accuracy), both to be maximised; sorted by coverage."""
    order = sorted(range(len(points)), key=lambda i: (-points[i][0], -points[i][1]))
    front, best_acc = [], -1.0
    for i in order:
        if points[i][1] > best_acc:
            front.append(i)
            best_acc = points[i][1]
    return sorted(front, key=lambda i: points[i][0])
