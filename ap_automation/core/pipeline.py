"""
Main prediction pipeline:
  MDM -> retrieval -> confidence -> rule gate -> specialist agents -> governance gate -> LLM gate

Two decisions are produced and both are recorded:
  * routing_decision - the RULE gate (evidence sufficiency + confidence)
  * final_routing    - the GOVERNED decision (rule gate + agent flags + LLM escalation)

Only `final_routing` may authorise an auto-post, and `confirm()` enforces that itself
so that no caller can post something the gate did not approve.

batch_mode=True skips the LLM gate for throughput; the rule + agent gates still apply.
Telemetry (Langfuse / Supabase) is best-effort and never affects the decision.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import UUID, uuid4

from ap_automation.agents.orchestrator import AgentOrchestrator
from ap_automation.core import clock, governance
from ap_automation.core import confidence as conf_module
from ap_automation.core.config import settings
from ap_automation.core.mdm import resolve_vendor
from ap_automation.core.models import (
    ConfirmationSource,
    ConfirmedOutcome,
    ERPPostingPayload,
    InvoiceLine,
    Prediction,
    ResolvedInvoiceLine,
    RetrievedEvidence,
    RoutingDecision,
    amount_to_band,
)
from ap_automation.core.retrieval import _collection_size, index_outcome, point_id_for, retrieve
from ap_automation.graph.nodes import llm_gate_node as _llm_gate_fn

_orchestrator = AgentOrchestrator()
_lf_client = None
_lf_init_done = False

_BATCH_LLM_SKIP = {
    "llm_reasoning": "",
    "llm_reasoning_confidence": 0.0,
    "llm_escalated": False,
    "llm_conflicts": [],
    "llm_conflict_count": 0,
    "llm_latency_ms": 0,
}


class AutoPostNotPermitted(Exception):
    """Raised when something tries to auto-post a prediction the governed gate did not approve."""


# --------------------------------------------------------------------------- telemetry

def _get_langfuse():
    """Return a Langfuse client, or None when telemetry is off or unconfigured."""
    global _lf_client, _lf_init_done
    if not settings.telemetry_enabled:
        return None
    if _lf_init_done:
        return _lf_client
    _lf_init_done = True
    try:
        if "streamlit" in sys.modules:  # Streamlit Cloud keeps keys in st.secrets
            try:
                import streamlit as st
                for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
                    val = str(st.secrets.get(key, "") or "").strip()
                    if val:
                        os.environ[key] = val
            except Exception:
                pass
        for key, val in (
            ("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key),
            ("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key),
            ("LANGFUSE_HOST", settings.langfuse_host),
        ):
            if val and not os.environ.get(key):
                os.environ[key] = val
        if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
            return None
        from langfuse import get_client
        _lf_client = get_client()
    except Exception as e:
        print(f"[Langfuse] init error: {e}")
        _lf_client = None
    return _lf_client


class _NullSpan:
    def set_output(self, **_: object) -> None: ...
    def score(self, name: str, value: float, comment: str = "") -> None: ...


class _Span(_NullSpan):
    def __init__(self, lf):
        self._lf = lf

    def set_output(self, **output: object) -> None:
        try:
            self._lf.update_current_span(output=output)
        except Exception:
            pass

    def score(self, name: str, value: float, comment: str = "") -> None:
        try:
            self._lf.score_current_trace(name=name, value=value, comment=comment)
        except Exception:
            pass


@contextmanager
def _span(name: str, input: object = None, metadata: object = None, trace_id: str | None = None):
    """A Langfuse span that degrades to a no-op. Real exceptions in the body still propagate."""
    lf = _get_langfuse()
    cm = None
    if lf is not None:
        try:
            kwargs = {"trace_context": {"trace_id": trace_id}} if trace_id else {}
            cm = lf.start_as_current_observation(
                as_type="span", name=name, input=input, metadata=metadata, **kwargs
            )
            cm.__enter__()
        except Exception as e:
            print(f"[Langfuse] span '{name}' unavailable: {e}")
            cm = None
    try:
        yield _Span(lf) if cm is not None else _NullSpan()
    except BaseException as exc:
        if cm is not None:
            try:
                cm.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                pass
        raise
    else:
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass


def _flush() -> None:
    lf = _get_langfuse()
    if lf is not None:
        try:
            lf.flush()
        except Exception:
            pass


# --------------------------------------------------------------------------- predict

def predict(invoice: InvoiceLine, anthropic_api_key: str = "", batch_mode: bool = False) -> Prediction:
    """
    Full prediction pipeline.
    batch_mode=True skips the LLM gate (fast, cheap bulk runs); rule + agent gates still apply.

    When LANGGRAPH_API_URL is set, the run is a Cloud LangGraph thread. Otherwise the
    in-process pipeline is used (tests, readiness replay, local demo).
    """
    if settings.langgraph_api_url.strip():
        from ap_automation.graph.cloud import predict_remote
        return predict_remote(invoice, batch_mode=batch_mode)
    prediction_id = uuid4()
    with _span(
        "invoice_prediction",
        input={
            "invoice_id": str(invoice.invoice_id),
            "vendor_raw": invoice.raw_vendor_name,
            "description": invoice.description,
            "amount": invoice.amount,
            "currency": invoice.currency,
            "entity_id": invoice.entity_id,
            "ground_truth_gl": invoice.ground_truth_gl or "",
            "batch_mode": batch_mode,
        },
        metadata={"prediction_id": str(prediction_id)},
        trace_id=prediction_id.hex,
    ) as root:
        prediction = _predict(invoice, prediction_id, anthropic_api_key, batch_mode)
        root.set_output(
            predicted_gl=prediction.predicted_gl,
            routing_decision=prediction.routing_decision.value,
            final_routing=prediction.final_routing.value,
            agent_flags=prediction.agent_flags,
            weighted_confidence=round(prediction.weighted_confidence, 4),
            evidence_count=len(prediction.evidence),
            vat_treatment=prediction.provenance.get("vat_treatment"),
            batch_mode=batch_mode,
        )
    _flush()
    return prediction


def _predict(invoice: InvoiceLine, prediction_id: UUID, anthropic_api_key: str, batch_mode: bool) -> Prediction:
    pid = str(prediction_id)

    # -- MDM -----------------------------------------------------------------
    t0 = time.time()
    canonical_vendor_id, mdm_confidence = resolve_vendor(invoice.raw_vendor_name)
    amount_band = amount_to_band(invoice.amount)
    mdm_ms = int((time.time() - t0) * 1000)
    with _span("mdm", input={"raw_vendor_name": invoice.raw_vendor_name}, metadata={"latency_ms": mdm_ms}) as s:
        s.set_output(canonical_vendor_id=canonical_vendor_id, mdm_confidence=mdm_confidence,
                     amount_band=amount_band.value)
    resolved = ResolvedInvoiceLine(
        invoice_line=invoice, canonical_vendor_id=canonical_vendor_id,
        mdm_confidence=mdm_confidence, amount_band=amount_band,
    )

    # -- Retrieval (fail closed: an outage is routed to review, never guessed through) --
    t0 = time.time()
    retrieval_error: str | None = None
    retrieval_info: dict = {}
    try:
        evidence = retrieve(
            description=invoice.description, vendor_id=canonical_vendor_id,
            entity_id=invoice.entity_id, amount_band=amount_band, currency=invoice.currency,
            fund=invoice.cost_centre, amount=invoice.amount, info=retrieval_info,
        )
    except Exception as e:
        evidence = []
        retrieval_error = f"{type(e).__name__}: {e}"[:160]
        print(f"[Retrieval] unavailable - routing to human review: {retrieval_error}")
    retrieval_ms = int((time.time() - t0) * 1000)
    with _span("retrieval", input={"description": invoice.description, "vendor_id": canonical_vendor_id},
               metadata={"latency_ms": retrieval_ms}) as s:
        s.set_output(
            evidence_count=len(evidence),
            top_similarity=round(evidence[0].similarity_score, 4) if evidence else 0,
            top_gl_codes=[e.gl_account for e in evidence[:3]],
            error=retrieval_error,
        )

    # -- Confidence + rule gate ---------------------------------------------------
    gate_params = conf_module.GateParams.from_settings()
    if gate_params.uses_variants():  # opt-in gate variants (recency / margin / diversity / ...); off by default
        feats = conf_module.vote_features(evidence, clock.now(), gate_params, settings.min_evidence_lines)
        predicted_gl, evidence_cc = feats.predicted_gl, feats.predicted_cc
        agreement_ratio, weighted_confidence = feats.agreement, feats.weighted_confidence
        rule_decision, rule_reason = conf_module.route_variant(
            feats, canonical_vendor_id, gate_params, mdm_confidence,
            vendor_distinct_codes=retrieval_info.get("vendor_distinct_codes", 0),
            subset_distinct_codes=retrieval_info.get("subset_distinct_codes", 0),
            context_level=retrieval_info.get("context_level", "vendor"),
        )
    else:
        predicted_gl, evidence_cc, agreement_ratio, weighted_confidence = conf_module.score(evidence)
        rule_decision, rule_reason = conf_module.route(
            evidence=evidence, vendor_id=canonical_vendor_id, weighted_confidence=weighted_confidence,
            mdm_confidence=mdm_confidence, agreement_ratio=agreement_ratio,
        )
    if retrieval_error:
        rule_reason = f"Retrieval unavailable ({retrieval_error}); {rule_reason}"
    with _span("confidence", input={"evidence_count": len(evidence)}) as s:
        s.set_output(predicted_gl=predicted_gl, agreement_ratio=round(agreement_ratio, 4),
                     weighted_confidence=round(weighted_confidence, 4),
                     routing_decision=rule_decision.value, routing_reason=rule_reason)

    # -- Specialist agents + governance gate -----------------------------------------
    t0 = time.time()
    agent_result = _orchestrator.run(
        {
            "predicted_gl": predicted_gl,
            "predicted_cost_centre": evidence_cc,
            "submitted_cost_centre": invoice.cost_centre or "",
            "description": invoice.description,
            "amount": invoice.amount,
            "currency": invoice.currency,
            "vendor_id": canonical_vendor_id,
            "entity_id": invoice.entity_id,
            "routing_decision": rule_decision.value,
            "weighted_confidence": weighted_confidence,
        },
        langfuse_client=_get_langfuse(),
    )
    agent_ms = int((time.time() - t0) * 1000)
    signals = governance.extract_signals(agent_result)
    flags = governance.agent_flags(**signals)
    gate_decision, gate_reason = governance.apply_gate(rule_decision, rule_reason, flags)
    predicted_cc = signals["inferred_cost_centre"] or evidence_cc
    with _span("governance_gate", input={"rule_decision": rule_decision.value, "flags": flags}) as s:
        s.set_output(final_routing=gate_decision.value, reason=gate_reason, vat_treatment=signals["vat_treatment"])

    # -- LLM gate (escalation only; skipped in batch mode) ------------------------------
    api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "") or settings.anthropic_api_key
    llm_result: dict
    if batch_mode:
        llm_result = {**_BATCH_LLM_SKIP, "final_routing": gate_decision.value}
    else:
        llm_state = _build_llm_state(
            invoice, canonical_vendor_id, evidence, predicted_gl, predicted_cc, weighted_confidence,
            rule_decision, gate_decision, gate_reason, flags, agent_result, api_key,
        )
        with _span("llm_gate", input={"routing": gate_decision.value, "agent_flags": flags}) as s:
            llm_result = _llm_gate_fn(llm_state)
            s.set_output(routing=llm_result.get("final_routing"), escalated=llm_result.get("llm_escalated"),
                         conflict_count=llm_result.get("llm_conflict_count", 0))

    # The LLM may only make the decision stricter - this is enforced here, not trusted.
    final_decision = governance.never_loosen(gate_decision, llm_result.get("final_routing"))
    escalated = final_decision == RoutingDecision.HUMAN_REVIEW and gate_decision == RoutingDecision.AUTO_POST
    final_reason = (
        f"LLM gate escalated to human review: {llm_result.get('llm_reasoning', '')[:200]}"
        if escalated else gate_reason
    )

    provenance = {
        "prediction_id": pid,
        "invoice_id": str(invoice.invoice_id),
        "vendor_raw": invoice.raw_vendor_name,
        "vendor_canonical": canonical_vendor_id,
        "mdm_confidence": mdm_confidence,
        "amount_band": amount_band.value,
        "evidence_count": len(evidence),
        "predicted_gl": predicted_gl,
        "predicted_cost_centre": predicted_cc,
        "submitted_cost_centre": invoice.cost_centre,
        "agreement_ratio": round(agreement_ratio, 4),
        "weighted_confidence": round(weighted_confidence, 4),
        "routing_decision": rule_decision.value,
        "routing_reason": rule_reason,
        "agent_flags": flags,
        "vat_treatment": signals["vat_treatment"],
        "final_routing": final_decision.value,
        "final_routing_reason": final_reason,
        "llm_reasoning": llm_result.get("llm_reasoning", ""),
        "llm_reasoning_confidence": llm_result.get("llm_reasoning_confidence", 0.0),
        "llm_escalated": escalated,
        "llm_conflicts": llm_result.get("llm_conflicts", []),
        "llm_conflict_count": llm_result.get("llm_conflict_count", 0),
        "llm_latency_ms": llm_result.get("llm_latency_ms", 0),
        "llm_gate_reason": llm_result.get("llm_gate_reason", ""),
        "batch_mode": batch_mode,
        "retrieval_error": retrieval_error,
        "retrieval_context": dict(retrieval_info),
        "latency_ms": {"mdm": mdm_ms, "retrieval": retrieval_ms, "agents": agent_ms},
        "a2a_agents": agent_result.to_dict(),
        "retrieved_evidence": [
            {
                "description": e.description,
                "gl_account": e.gl_account,
                "similarity_score": round(e.similarity_score, 4),
                "rrf_score": round(e.rrf_score, 6),
                "dense_rank": e.dense_rank,
                "sparse_rank": e.sparse_rank,
                "source": e.source.value,
                "confirmation_timestamp": e.confirmation_timestamp.isoformat(),
            }
            for e in evidence
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _write_analytics_event(
        invoice, pid, canonical_vendor_id, predicted_gl, rule_decision, final_decision,
        weighted_confidence, agreement_ratio, signals, retrieval_ms, batch_mode, amount_band.value,
    )

    return Prediction(
        prediction_id=prediction_id,
        resolved_line=resolved,
        evidence=evidence,
        predicted_gl=predicted_gl,
        predicted_cost_centre=predicted_cc,
        agreement_ratio=agreement_ratio,
        weighted_confidence=weighted_confidence,
        routing_decision=rule_decision,
        routing_reason=rule_reason,
        final_routing=final_decision,
        final_routing_reason=final_reason,
        agent_flags=flags,
        provenance=provenance,
    )


def _write_analytics_event(invoice, pid, vendor, predicted_gl, rule_decision, final_decision,
                           weighted_confidence, agreement_ratio, signals, retrieval_ms, batch_mode, band):
    try:
        from ap_automation.core.supabase import write_prediction_event
        write_prediction_event({
            "prediction_id": pid,
            "vendor_raw": invoice.raw_vendor_name,
            "vendor_canonical": vendor,
            "description": invoice.description,
            "amount": invoice.amount,
            "currency": invoice.currency,
            "entity_id": invoice.entity_id,
            "predicted_gl": predicted_gl,
            "routing_decision": rule_decision.value,
            "final_routing": final_decision.value,
            "weighted_confidence": round(weighted_confidence, 4),
            "agreement_ratio": round(agreement_ratio, 4),
            "vat_treatment": signals["vat_treatment"],
            "cost_centre_mismatch": signals["cost_centre_mismatch"],
            "entity_valid": signals["entity_valid"],
            "retrieval_latency_ms": retrieval_ms,
            "llm_skipped": batch_mode,
            "batch_mode": batch_mode,
            "amount_band": band,
        })
    except Exception as e:
        print(f"[Supabase] prediction event write error: {e}")


def _build_llm_state(invoice, vendor_id, evidence: list[RetrievedEvidence], predicted_gl, predicted_cc,
                     weighted_confidence, rule_decision, gate_decision, gate_reason, flags,
                     agent_result, api_key) -> dict:
    """State for the LLM gate. Previously the retrieved evidence was omitted, so the LLM was told '0 lines'."""
    return {
        "description": invoice.description,
        "raw_vendor_name": invoice.raw_vendor_name,
        "canonical_vendor_id": vendor_id,
        "amount": invoice.amount,
        "currency": invoice.currency,
        "entity_id": invoice.entity_id,
        "predicted_gl": predicted_gl,
        "predicted_cost_centre": predicted_cc,
        "weighted_confidence": weighted_confidence,
        "routing_decision": rule_decision.value,
        "final_routing": gate_decision.value,
        "gate_reason": gate_reason,
        "agent_flags": flags,
        "agent_results": [r.to_dict() for r in agent_result.results],
        "evidence": [
            {"gl_account": e.gl_account, "similarity_score": e.similarity_score} for e in evidence
        ],
        "_api_key": api_key,
    }


# --------------------------------------------------------------------------- confirm / write-back

def confirm(
    prediction: Prediction,
    confirmed_gl: str,
    confirmed_cc: str,
    confirming_user_id: str,
    source: ConfirmationSource,
) -> tuple[ConfirmedOutcome, ERPPostingPayload]:
    """
    Record a confirmation and build the ERP payload.

    An AUTO_POST confirmation is only valid for a prediction whose GOVERNED decision
    (final_routing) is auto_post and which is posted exactly as predicted.
    """
    if source == ConfirmationSource.AUTO_POST:
        if prediction.final_routing != RoutingDecision.AUTO_POST:
            raise AutoPostNotPermitted(
                f"Prediction {prediction.prediction_id} is routed to {prediction.final_routing.value}: "
                f"{prediction.final_routing_reason}"
            )
        if confirmed_gl != prediction.predicted_gl:
            raise AutoPostNotPermitted("An auto-post must post the predicted GL unchanged")

    invoice = prediction.resolved_line.invoice_line
    now = clock.now()
    tax_code = prediction.provenance.get("vat_treatment") or "UNASSESSED"

    is_correct = (prediction.predicted_gl == confirmed_gl) if prediction.predicted_gl else None
    is_false_positive = (
        prediction.final_routing == RoutingDecision.AUTO_POST
        and source == ConfirmationSource.HUMAN_CORRECT
        and prediction.predicted_gl != confirmed_gl
    )
    with _span(
        "invoice_confirmation",
        input={
            "invoice_id": str(invoice.invoice_id), "predicted_gl": prediction.predicted_gl,
            "confirmed_gl": confirmed_gl, "source": source.value, "confirming_user": confirming_user_id,
            "ground_truth_gl": invoice.ground_truth_gl or "",
        },
        metadata={
            "confidence": round(prediction.weighted_confidence, 4),
            "final_routing": prediction.final_routing.value,
            "correct": is_correct, "false_positive": is_false_positive,
        },
        trace_id=prediction.prediction_id.hex,
    ) as s:
        s.score("gate.correct", 1.0 if is_correct else 0.0,
                f"Predicted {prediction.predicted_gl}, confirmed {confirmed_gl}")
        if is_false_positive:
            s.score("gate.false_positive", 1.0,
                    f"Auto-post corrected: {prediction.predicted_gl} -> {confirmed_gl}")
        if invoice.ground_truth_gl:
            s.score("retrieval.recall_at_10",
                    1.0 if prediction.predicted_gl == invoice.ground_truth_gl else 0.0,
                    f"Predicted {prediction.predicted_gl}, ground truth {invoice.ground_truth_gl}")
    _flush()

    outcome = ConfirmedOutcome(
        prediction_id=prediction.prediction_id,
        invoice_id=invoice.invoice_id,
        canonical_vendor_id=prediction.resolved_line.canonical_vendor_id,
        entity_id=invoice.entity_id,
        amount_band=prediction.resolved_line.amount_band,
        currency=invoice.currency,
        gl_account=confirmed_gl,
        cost_centre=confirmed_cc,
        tax_code=tax_code,
        confirming_user_id=confirming_user_id,
        confirmation_timestamp=now,
        prediction_confidence=prediction.weighted_confidence,
        source=source,
        description=invoice.description,
        vendor_name=invoice.raw_vendor_name,
        amount=invoice.amount,
    )
    erp_payload = ERPPostingPayload(
        invoice_id=str(invoice.invoice_id),
        vendor_id=prediction.resolved_line.canonical_vendor_id,
        vendor_name=invoice.raw_vendor_name,
        entity_id=invoice.entity_id,
        gl_account=confirmed_gl,
        cost_centre=confirmed_cc,
        tax_code=tax_code,
        amount=invoice.amount,
        currency=invoice.currency,
        description=invoice.description,
        posted_by=confirming_user_id,
        posted_at=now,
        trace_id=str(prediction.prediction_id),
    )
    return outcome, erp_payload


def write_back(outcome: ConfirmedOutcome) -> bool:
    """
    Add a confirmed line to the learning corpus. Returns True if it was indexed.

    Only HUMAN-confirmed (or human-corrected) lines are learned from. An auto-post is
    the system's own guess; indexing it as confirmed evidence would let the corpus
    agree with itself and inflate the automation rate without any new information.
    """
    if outcome.source == ConfirmationSource.AUTO_POST:
        return False

    corpus_before = _collection_size()
    index_outcome({
        "point_id": point_id_for(str(outcome.invoice_id)),
        "invoice_id": str(outcome.invoice_id),
        "vendor_id": outcome.canonical_vendor_id,
        "vendor_name": outcome.vendor_name,
        "entity_id": outcome.entity_id,
        "amount_band": outcome.amount_band.value,
        "amount": outcome.amount,
        "currency": outcome.currency,
        "gl_account": outcome.gl_account,
        "cost_centre": outcome.cost_centre,
        "tax_code": outcome.tax_code,
        "description": outcome.description,
        "confirming_user_id": outcome.confirming_user_id,
        "confirmation_timestamp": outcome.confirmation_timestamp.isoformat(),
        "prediction_confidence": outcome.prediction_confidence,
        "source": outcome.source.value,
    })
    corpus_after = _collection_size()
    with _span(
        "write_back",
        input={"vendor_id": outcome.canonical_vendor_id, "gl_account": outcome.gl_account,
               "source": outcome.source.value},
        metadata={"corpus_before": corpus_before, "corpus_after": corpus_after,
                  "delta": corpus_after - corpus_before},
        trace_id=outcome.prediction_id.hex,
    ) as s:
        s.set_output(corpus_size=corpus_after, delta=corpus_after - corpus_before)
    _flush()
    return True
