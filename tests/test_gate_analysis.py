"""Gate variants, recency weights, coverage curves, context retrieval and the leak guard."""

import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from conftest import make_invoice

from ap_automation.core import clock, confidence, retrieval
from ap_automation.core.confidence import (
    GateParams,
    recency_weight,
    required_agreeing_lines,
    route,
    route_variant,
    score,
    vote_features,
    wilson_lower,
)
from ap_automation.core.config import settings
from ap_automation.core.gate_analysis import (
    CONTEXT_LEVELS,
    SweepGate,
    calibration_table,
    coverage_accuracy,
    coverage_at_accuracy,
    gate_mask,
    pareto_front,
    recency_weights,
    risk_coverage_curve,
    vote_arrays,
    wilson_interval,
    wilson_lower_array,
)
from ap_automation.core.models import ConfirmationSource, RetrievedEvidence, RoutingDecision
from ap_automation.core.pipeline import predict
from ap_automation.core.posting import post_outcome
from ap_automation.core.retrieval import context_matches, context_subset
from ap_automation.core.seeding import seed_corpus

AUTO, REVIEW = RoutingDecision.AUTO_POST, RoutingDecision.HUMAN_REVIEW
NOW = datetime(2026, 3, 1, 12, tzinfo=timezone.utc)
CODES = ["A", "B", "C", "D"]


def _ev(gls, sims=None, ages=None):
    sims = sims or [0.9] * len(gls)
    ages = ages or [0.0] * len(gls)
    return [
        RetrievedEvidence(
            invoice_id=str(i), description="x", gl_account=g, cost_centre="F1", vendor_id="V",
            similarity_score=s, dense_rank=i, sparse_rank=i, rrf_score=0.03,
            confirmation_timestamp=NOW - timedelta(days=a), source=ConfirmationSource.HUMAN_CONFIRM,
        )
        for i, (g, s, a) in enumerate(zip(gls, sims, ages, strict=True))
    ]


def _random_evidence(rng, n):
    gls = [rng.choice(CODES[: rng.randint(1, 4)]) for _ in range(n)]
    return _ev(gls, [round(rng.uniform(0.2, 1.0), 3) for _ in range(n)], [rng.uniform(0, 400) for _ in range(n)])


# ------------------------------------------------------------------ statistics

def test_wilson_interval_known_values():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = wilson_interval(0, 0)
    assert (lo, hi) == (0.0, 1.0)
    lo, hi = wilson_interval(100, 100)
    assert hi == pytest.approx(1.0) and lo < 0.97  # 100/100 still leaves room for error


def test_wilson_lower_scalar_matches_array():
    p, n = np.array([0.5, 0.9, 1.0, 0.0]), np.array([10.0, 20.0, 3.0, 0.0])
    for z in (1.0, 1.645):
        expect = [wilson_lower(float(a), float(b), z) for a, b in zip(p, n, strict=True)]
        assert wilson_lower_array(p, n, z).tolist() == pytest.approx(expect)


def test_recency_weight_halves_each_halflife_and_is_off_at_zero():
    assert recency_weight(90, 90) == pytest.approx(0.5)
    assert recency_weight(180, 90) == pytest.approx(0.25)
    assert recency_weight(500, 0) == 1.0
    assert recency_weight(-5, 30) == 1.0
    assert recency_weights(np.array([0.0, 30.0, 60.0]), 30).tolist() == pytest.approx([1, 0.5, 0.25])
    assert recency_weights(np.array([0.0, 30.0]), 0).tolist() == [1.0, 1.0]


def test_required_agreeing_lines_scales_with_code_diversity_and_caps():
    assert required_agreeing_lines(6, 0, 10) == 0
    assert required_agreeing_lines(1, 1.0, 10) == 1
    assert required_agreeing_lines(6, 1.0, 10) == 6
    assert required_agreeing_lines(40, 1.0, 10) == 10
    assert required_agreeing_lines(5, 0.5, 10) == 3


# ------------------------------------------------------------------ vote features / gate equivalences

def test_vote_features_with_default_params_equal_the_shipped_score():
    rng = random.Random(1)
    for _ in range(200):
        ev = _random_evidence(rng, rng.randint(1, 10))
        gl, cc, agreement, wc = score(ev)
        f = vote_features(ev, NOW, GateParams())
        assert (f.predicted_gl, f.predicted_cc) == (gl, cc)
        assert f.agreement == pytest.approx(agreement) and f.weighted_confidence == pytest.approx(wc)
        assert f.n == len(ev) and f.n_eff == pytest.approx(len(ev))


def test_route_variant_with_default_params_equals_the_shipped_route():
    rng = random.Random(2)
    for _ in range(300):
        ev = _random_evidence(rng, rng.randint(0, 10))
        mdm = rng.choice([0.5, 1.0])
        gl, _, agreement, wc = score(ev)
        old = route(ev, "V", wc, mdm, agreement)
        if not ev:
            continue
        new = route_variant(vote_features(ev, NOW, GateParams()), "V", GateParams(), mdm)
        assert new[0] == old[0]


def test_recency_decay_lets_recent_lines_outvote_old_majority():
    ev = _ev(["A"] * 6 + ["B"] * 4, ages=[300] * 6 + [1] * 4)          # old majority A, fresh minority B
    assert vote_features(ev, NOW, GateParams()).predicted_gl == "A"
    f = vote_features(ev, NOW, GateParams(recency_halflife_days=30))
    assert f.predicted_gl == "B" and f.agreement > 0.99


def test_recency_window_votes_recent_lines_only_when_enough_remain():
    ev = _ev(["A"] * 6 + ["B"] * 4, ages=[300] * 6 + [1] * 4)
    f = vote_features(ev, NOW, GateParams(recency_window_days=30))
    assert f.predicted_gl == "B" and f.n == 4
    thin = vote_features(_ev(["A"] * 6 + ["B"] * 2, ages=[300] * 6 + [1] * 2), NOW, GateParams(recency_window_days=30))
    assert thin.n == 8 and thin.predicted_gl == "A"      # only 2 recent lines: falls back to all of them


def test_margin_gate_needs_a_clear_winner():
    ev = _ev(["A"] * 8 + ["B"] * 2)
    f = vote_features(ev, NOW, GateParams())
    assert f.margin == pytest.approx(0.6)
    assert route_variant(f, "V", GateParams(min_margin=0.5))[0] == AUTO
    assert route_variant(f, "V", GateParams(min_margin=0.7))[0] == REVIEW


def test_diversity_scaled_evidence_requirement():
    f = vote_features(_ev(["A"] * 4), NOW, GateParams())
    p = GateParams(diversity_per_code=1.0)
    assert route_variant(f, "V", p, vendor_distinct_codes=1)[0] == AUTO
    assert route_variant(f, "V", p, vendor_distinct_codes=6)[0] == REVIEW
    assert route_variant(f, "V", GateParams(diversity_per_code=1.0, diversity_scope="subset"),
                         vendor_distinct_codes=6, subset_distinct_codes=2)[0] == AUTO


def test_required_context_level():
    f = vote_features(_ev(["A"] * 5), NOW, GateParams())
    p = GateParams(required_context=("dept+fund",))
    assert route_variant(f, "V", p, context_level="dept+fund")[0] == AUTO
    assert route_variant(f, "V", p, context_level="vendor")[0] == REVIEW


def test_vector_votes_equal_scalar_votes():
    rng = random.Random(3)
    K = 12
    for halflife, window in [(0, 0), (60, 0), (0, 90), (45, 120)]:
        evs = [_random_evidence(rng, rng.randint(1, K)) for _ in range(150)]
        codes = np.full((len(evs), K), -1)
        sims, ages, valid = np.zeros((len(evs), K)), np.zeros((len(evs), K)), np.zeros((len(evs), K), bool)
        for i, ev in enumerate(evs):
            for j, e in enumerate(ev):
                codes[i, j] = CODES.index(e.gl_account)
                sims[i, j], ages[i, j] = e.similarity_score, (NOW - e.confirmation_timestamp).total_seconds() / 86400
                valid[i, j] = True
        v = vote_arrays(codes, sims, ages, valid, K, halflife, window, 3)
        p = GateParams(recency_halflife_days=halflife, recency_window_days=window)
        for i, ev in enumerate(evs):
            f = vote_features(ev, NOW, p, 3)
            assert CODES[v.predicted_code[i]] == f.predicted_gl
            assert v.n[i] == f.n and v.agree_count[i] == f.agree_count
            assert v.agreement[i] == pytest.approx(f.agreement)
            assert v.margin[i] == pytest.approx(f.margin, abs=1e-9)
            assert v.weighted_confidence[i] == pytest.approx(f.weighted_confidence)
            assert v.n_eff[i] == pytest.approx(f.n_eff)


def test_vector_votes_truncate_to_k():
    ev = _ev(["A"] * 3 + ["B"] * 7)
    codes = np.array([[CODES.index(e.gl_account) for e in ev]])
    ones = np.ones_like(codes, dtype=float)
    v = vote_arrays(codes, ones, np.zeros_like(ones), np.ones_like(codes, dtype=bool), 3)
    assert v.n[0] == 3 and v.agreement[0] == 1.0 and CODES[v.predicted_code[0]] == "A"


def test_gate_mask_equals_route_variant_over_a_parameter_grid():
    rng = random.Random(4)
    K = 10
    evs = [_random_evidence(rng, rng.randint(1, K)) for _ in range(120)]
    codes = np.full((len(evs), K), -1)
    sims, ages, valid = np.zeros((len(evs), K)), np.zeros((len(evs), K)), np.zeros((len(evs), K), bool)
    for i, ev in enumerate(evs):
        for j, e in enumerate(ev):
            codes[i, j] = CODES.index(e.gl_account)
            sims[i, j], ages[i, j] = e.similarity_score, (NOW - e.confirmation_timestamp).total_seconds() / 86400
            valid[i, j] = True
    v = vote_arrays(codes, sims, ages, valid, K)
    vendor_d = np.array([rng.randint(1, 8) for _ in evs])
    subset_d = np.array([rng.randint(1, 4) for _ in evs])
    level = np.array([rng.randrange(len(CONTEXT_LEVELS)) for _ in evs])
    grid = [
        SweepGate(min_n=n, min_agreement=a, min_wc=w, min_margin=m, min_lb=lb, div_per_code=d, div_scope=sc,
                  required_levels=rl)
        for n in (3, 6) for a in (0.8, 0.9) for w in (0.5, 0.8) for m in (0.0, 0.6) for lb in (0.0, 0.5)
        for d in (0.0, 1.0) for sc in ("vendor", "subset") for rl in ((), ("dept+fund", "fund"))
    ]
    saved = settings.min_evidence_lines, settings.min_agreement_ratio
    try:
        for g in grid:
            settings.min_evidence_lines, settings.min_agreement_ratio = g.min_n, g.min_agreement
            confidence.VENDOR_THRESHOLDS["V"] = g.min_wc
            gp = GateParams(min_margin=g.min_margin, min_agreement_lb=g.min_lb, diversity_per_code=g.div_per_code,
                            diversity_scope=g.div_scope, required_context=g.required_levels)
            mask = gate_mask(v, g, vendor_d, subset_d, level)
            for i, ev in enumerate(evs):
                dec, _ = route_variant(vote_features(ev, NOW, gp), "V", gp, 1.0, int(vendor_d[i]), int(subset_d[i]),
                                       CONTEXT_LEVELS[level[i]])
                assert (dec == AUTO) == bool(mask[i]), (g, i)
    finally:
        settings.min_evidence_lines, settings.min_agreement_ratio = saved
        confidence.VENDOR_THRESHOLDS.pop("V", None)


# ------------------------------------------------------------------ coverage curve

def test_risk_coverage_curve_and_coverage_at_accuracy():
    score_ = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    correct = np.array([True, True, True, False, True, False])
    curve = risk_coverage_curve(score_, correct)
    assert [r["n_auto"] for r in curve] == [1, 2, 3, 4, 5, 6]
    assert curve[2]["accuracy"] == 1.0 and curve[3]["accuracy"] == pytest.approx(0.75)
    best = coverage_at_accuracy(curve, 0.99, min_posts=1)
    assert best and best["n_auto"] == 3 and best["coverage"] == 0.5
    assert coverage_at_accuracy(curve, 0.99, min_posts=4) is None          # cannot reach it with enough posts
    assert coverage_at_accuracy(curve, 0.7, min_posts=1)["n_auto"] == 5


def test_risk_coverage_curve_keeps_tied_scores_together():
    curve = risk_coverage_curve(np.array([0.9, 0.9, 0.5]), np.array([True, False, True]))
    assert [r["n_auto"] for r in curve] == [2, 3]


def test_coverage_accuracy_counts():
    cov, acc, n, wrong = coverage_accuracy(np.array([True, True, False, True]), np.array([True, False, True, True]))
    assert (cov, acc, n, wrong) == (0.75, pytest.approx(2 / 3), 3, 1)
    assert coverage_accuracy(np.zeros(3, bool), np.ones(3, bool))[1] is None


def test_pareto_front_keeps_non_dominated_points():
    pts = [(0.5, 0.90), (0.6, 0.85), (0.4, 0.95), (0.3, 0.80), (0.6, 0.80)]
    assert pareto_front(pts) == [2, 0, 1]


def test_calibration_table_ece():
    conf = np.array([0.1, 0.1, 0.9, 0.9])
    cal = calibration_table(conf, np.array([False, False, True, True]), np.array([False, False, True, True]),
                            [(0.0, 0.5, "lo"), (0.5, 1.01, "hi")])
    assert cal["ece_all_predictions"] == pytest.approx(0.1)
    assert cal["bins"][1]["n_auto_posted"] == 2 and cal["bins"][1]["auto_post_accuracy"] == 1.0


# ------------------------------------------------------------------ context retrieval (default OFF)

def _payload(entity, fund):
    return {"entity_id": entity, "cost_centre": fund, "gl_account": "A"}


def test_context_subset_ladder_and_fallback():
    pays = [_payload("D1", "F1")] * 3 + [_payload("D1", "F2")] * 2 + [_payload("D2", "F1")] * 4
    ladder = "dept+fund,fund,dept,vendor"
    assert context_subset(pays, "D1", "F1", ladder, 3) == ("dept+fund", [0, 1, 2])
    lvl, idx = context_subset(pays, "D1", "F2", ladder, 3)       # dept+fund has 2 (<3); fund F2 has 2; dept D1 has 5
    assert lvl == "dept" and len(idx) == 5
    assert context_subset(pays, "D9", "F9", ladder, 3)[0] == "vendor"
    assert context_subset(pays, "D1", None, "dept+fund,vendor", 3)[0] == "vendor"
    with pytest.raises(ValueError):
        context_subset(pays, "D1", "F1", "nonsense", 3)


def test_context_matches_requires_a_value():
    p = {"entity_id": "D1", "cost_centre": "F1", "amount_band": "small"}
    assert context_matches(p, "D1", "F1", "small") == (True, True, True)
    assert context_matches(p, "", None, "") == (False, False, False)


def test_context_signals_are_off_and_gate_variants_inactive_by_default():
    assert settings.context_signals == "off"
    assert not GateParams.from_settings().uses_variants()


def test_seed_and_write_back_payloads_carry_the_context_fields():
    inv = make_invoice(entity="UK001", cost_centre="CC300", amount=1234.5)
    seed_corpus([inv])
    pts = retrieval.get_qdrant().scroll(settings.qdrant_collection, limit=5, with_payload=True)[0]
    p = pts[0].payload
    assert p["entity_id"] == "UK001" and p["cost_centre"] == "CC300" and p["amount"] == 1234.5
    assert p["amount_band"] == "medium"
    # a human confirmation writes the same fields back
    pred = predict(make_invoice(entity="UK002", cost_centre="CC310", amount=77.0))
    res = post_outcome(pred, "6300", "CC310", "clerk", ConfirmationSource.HUMAN_CORRECT, mirror_to_sheet=False)
    assert res.learned
    stored = [q.payload for q in retrieval.get_qdrant().scroll(settings.qdrant_collection, limit=50,
                                                               with_payload=True)[0]
              if q.payload.get("confirming_user_id") == "clerk"][0]
    assert stored["entity_id"] == "UK002" and stored["cost_centre"] == "CC310"
    assert stored["amount"] == 77.0 and stored["amount_band"] == "micro"


def _history(dept, fund, gl, n, amount=500.0, desc="monthly service charge"):
    return [make_invoice(vendor="Meridian Facilities Ltd", description=f"{desc} {i}", amount=amount, entity=dept,
                         cost_centre=fund, gl=gl) for i in range(n)]


@pytest.fixture
def ctx_settings():
    keys = ("context_signals", "context_min_lines", "context_ladder", "context_boost", "context_use_amount")
    saved = {k: getattr(settings, k) for k in keys}
    yield
    for k, v in saved.items():
        setattr(settings, k, v)


def test_subset_mode_pulls_evidence_from_the_matching_department_and_fund(ctx_settings):
    seed_corpus(_history("UK001", "CC300", "6300", 6) + _history("UK002", "CC310", "6200", 6))
    inv = make_invoice(vendor="Meridian Facilities Ltd", description="monthly service charge 9", entity="UK002",
                       cost_centre="CC310")
    info: dict = {}
    off = retrieval.retrieve(inv.description, "MERIDIAN_FACILITIES", "UK002", retrieval.AmountBand.SMALL, "GBP",
                             fund="CC310", info=info)
    assert info.get("context_level") == "vendor"
    assert {e.gl_account for e in off} == {"6300", "6200"}                 # mixed: vendor-wide evidence disagrees
    settings.context_signals = "subset"
    info = {}
    sub = retrieval.retrieve(inv.description, "MERIDIAN_FACILITIES", "UK002", retrieval.AmountBand.SMALL, "GBP",
                             fund="CC310", info=info)
    assert info["context_level"] == "dept+fund" and info["subset_lines"] == 6 and info["vendor_lines"] == 12
    assert {e.gl_account for e in sub} == {"6200"} and all(e.entity_id == "UK002" for e in sub)


def test_subset_mode_falls_back_to_vendor_evidence_when_the_subset_is_thin(ctx_settings):
    seed_corpus(_history("UK001", "CC300", "6300", 6) + _history("UK002", "CC310", "6200", 1))
    settings.context_signals = "subset"
    info: dict = {}
    ev = retrieval.retrieve("monthly service charge 9", "MERIDIAN_FACILITIES", "UK002",
                            retrieval.AmountBand.SMALL, "GBP", fund="CC310", info=info)
    assert info["context_level"] == "vendor" and len(ev) == 7


def test_boost_mode_ranks_matching_context_first(ctx_settings):
    seed_corpus(_history("UK001", "CC300", "6300", 8) + _history("UK002", "CC310", "6200", 8))
    settings.context_signals, settings.context_boost = "boost", 0.5
    ev = retrieval.retrieve("monthly service charge 3", "MERIDIAN_FACILITIES", "UK002",
                            retrieval.AmountBand.SMALL, "GBP", top_k=8, fund="CC310")
    assert {e.gl_account for e in ev} == {"6200"}


def test_pipeline_with_context_and_gate_variants_runs_end_to_end(ctx_settings):
    seed_corpus(_history("UK001", "CC300", "6300", 8) + _history("UK002", "CC310", "6200", 8))
    saved = settings.gate_min_margin, settings.gate_recency_halflife_days
    settings.context_signals = "subset"
    settings.gate_min_margin, settings.gate_recency_halflife_days = 0.5, 60.0
    try:
        t = datetime(2026, 5, 1, tzinfo=timezone.utc)
        clock.set_clock(lambda: t)
        pred = predict(make_invoice(vendor="Meridian Facilities Ltd", description="monthly service charge 2",
                                    entity="UK002", cost_centre="CC310", amount=500.0), batch_mode=True)
    finally:
        clock.set_clock(None)
        settings.gate_min_margin, settings.gate_recency_halflife_days = saved
    assert pred.predicted_gl == "6200" and pred.provenance["retrieval_context"]["context_level"] == "dept+fund"


# ------------------------------------------------------------------ leak guard

def test_gate_code_never_references_leak_columns():
    root = Path(__file__).resolve().parents[1]
    files = [root / "ap_automation/core" / f for f in ("retrieval.py", "confidence.py", "gate_analysis.py")]
    files += [root / "scripts" / f for f in ("gate_record.py", "tuned_replay.py", "gate_sweep.py")]
    pattern = re.compile(r"account_name|expenditure_type|item_code")
    for f in files:
        if f.exists():
            assert not pattern.search(f.read_text(encoding="utf-8")), f"{f.name} references a leak column"


def test_clock_defaults_to_real_time_and_can_be_replaced():
    assert abs((clock.now() - datetime.now(timezone.utc)).total_seconds()) < 5
    fixed = datetime(2020, 1, 1, tzinfo=timezone.utc)
    clock.set_clock(lambda: fixed)
    try:
        assert clock.now() == fixed and clock.is_simulated()
    finally:
        clock.set_clock(None)
    assert not clock.is_simulated()
