"""
LangGraph nodes - one function per pipeline step.
Each node receives the full InvoiceState, does one thing, and returns a partial dict
that LangGraph merges back into the state.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from uuid import uuid4

from ap_automation.agents.orchestrator import AgentOrchestrator
from ap_automation.core import confidence as conf_module
from ap_automation.core import governance
from ap_automation.core.mdm import resolve_vendor
from ap_automation.core.models import (
    ConfirmationSource,
    RoutingDecision,
    amount_to_band,
)
from ap_automation.core.retrieval import index_outcome, retrieve
from ap_automation.graph.state import InvoiceState

try:
    from langgraph.types import interrupt as _lg_interrupt
except ImportError:  # older langgraph: interrupt is a no-op so local tests still run
    def _lg_interrupt(value):  # type: ignore[no-redef]
        return value

_orchestrator = AgentOrchestrator()

LANGFUSE_PROMPT_NAME = "llm-gate-conflict-analysis"
_cached_prompt_template: str | None = None


def _get_prompt_template() -> str | None:
    """Fetch the LLM gate prompt from Langfuse. Caches in memory. Returns None on failure."""
    global _cached_prompt_template
    if _cached_prompt_template is not None:
        return _cached_prompt_template
    try:
        from langfuse import Langfuse
        client = Langfuse(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        p = client.get_prompt(LANGFUSE_PROMPT_NAME, label="production")
        _cached_prompt_template = p.prompt
        print(f"[LLM Gate] Loaded prompt '{LANGFUSE_PROMPT_NAME}' v{p.version} from Langfuse.")
        return _cached_prompt_template
    except Exception as e:
        print(f"[LLM Gate] Could not load prompt from Langfuse: {e} - using hardcoded fallback.")
        return None


def _compile_prompt(template: str | None, state: dict, agent_summary: list, evidence_summary: dict) -> str:
    """Compile prompt from Langfuse template or hardcoded fallback."""
    if template:
        return (
            template
            .replace("{{description}}", str(state.get("description", "")))
            .replace("{{vendor}}", str(state.get("raw_vendor_name", state.get("canonical_vendor_id", ""))))
            .replace("{{currency}}", str(state.get("currency", "GBP")))
            .replace("{{amount}}", str(state.get("amount", "")))
            .replace("{{entity_id}}", str(state.get("entity_id", "")))
            .replace("{{evidence_total_lines}}", str(evidence_summary["total_lines"]))
            .replace("{{evidence_gl_agreement_pct}}", str(evidence_summary["gl_agreement_pct"]))
            .replace("{{predicted_gl}}", str(evidence_summary["predicted_gl"]))
            .replace("{{evidence_top_similarity}}", str(evidence_summary["top_similarity"]))
            .replace("{{weighted_confidence}}", str(state.get("weighted_confidence", "")))
            .replace("{{final_routing}}", str(state.get("final_routing", "")))
            .replace("{{gate_reason}}", str(state.get("gate_reason", "")))
            .replace("{{agent_flags}}", str(state.get("agent_flags", [])))
            .replace("{{agent_results_json}}", json.dumps(agent_summary, indent=2))
        )
    # Hardcoded fallback
    return f"""You are a senior AP controller and AI governance officer reviewing an automated invoice coding decision.

Your job is to identify CONFLICTS in the decision - places where agents disagree with each other or where an agent flag contradicts the RAG evidence - and provide structured rationale for each conflict.

INVOICE:
- Description: {state.get("description")}
- Vendor: {state.get("raw_vendor_name", state.get("canonical_vendor_id"))}
- Amount: {state.get("currency")} {state.get("amount")}
- Entity: {state.get("entity_id")}

RAG EVIDENCE SUMMARY:
- Lines retrieved: {evidence_summary["total_lines"]}
- GL agreement: {evidence_summary["gl_agreement_pct"]}% of evidence lines coded to {evidence_summary["predicted_gl"]}
- Top similarity score: {evidence_summary["top_similarity"]}
- Weighted confidence: {state.get("weighted_confidence")}

RULE GATE DECISION:
- Routing: {state.get("final_routing")}
- Reason: {state.get("gate_reason")}
- Flags raised: {state.get("agent_flags", [])}

AGENT RESULTS (sequential chain: entity -> vat -> cost_centre):
{json.dumps(agent_summary, indent=2)}

INSTRUCTIONS:
1. Identify every conflict present. A conflict is one of:
   a) An agent flag that contradicts strong RAG evidence (e.g. cost centre mismatch flagged but GL evidence is 100% consistent)
   b) Two agents whose results are logically inconsistent with each other
   c) An agent result that appears incorrect given the invoice context
   d) A case where the rule gate decision seems disproportionate given the evidence

2. For each conflict, explain what the rule gate decided and whether you agree.

3. You MAY escalate auto_post to human_review if you find a genuine risk the rules missed.
   You MUST NOT reverse human_review to auto_post - governance rules are absolute.

4. If there are no conflicts, say so clearly in the summary.

5. Return ONLY valid JSON in this exact format - no preamble, no markdown fences:

{{
  "routing": "auto_post" or "human_review",
  "escalated": true or false,
  "reasoning_confidence": 0.0 to 1.0,
  "conflicts": [
    {{
      "conflict_id": "c1",
      "agents_involved": ["agent_name"] or ["agent_name", "rag_evidence"],
      "description": "What the conflict is - one sentence, specific to this invoice",
      "challenge": "What a reasonable reviewer would question about this flag or result",
      "resolution": "What should happen and why - reference governance rules if applicable",
      "severity": "high", "medium" or "low"
    }}
  ],
  "summary": "2-3 sentence plain English overall rationale. Be specific: name the GL, amounts, agents, flags."
}}"""


# -- Node: MDM resolution ------------------------------------------------------

def mdm_node(state: InvoiceState) -> dict:
    t0 = time.time()
    canonical_vendor_id, mdm_confidence = resolve_vendor(state["raw_vendor_name"])
    amount_band = amount_to_band(state["amount"])
    return {
        "canonical_vendor_id": canonical_vendor_id,
        "mdm_confidence": mdm_confidence,
        "amount_band": amount_band.value,
        "mdm_latency_ms": int((time.time() - t0) * 1000),
    }


# -- Node: Retrieval -----------------------------------------------------------

def retrieval_node(state: InvoiceState) -> dict:
    from ap_automation.core.models import AmountBand
    t0 = time.time()
    evidence_records = retrieve(
        description=state["description"],
        vendor_id=state["canonical_vendor_id"],
        entity_id=state["entity_id"],
        amount_band=AmountBand(state["amount_band"]),
        currency=state["currency"],
    )
    evidence_dicts = [
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
        for e in evidence_records
    ]
    return {
        "evidence": evidence_dicts,
        "evidence_count": len(evidence_dicts),
        "retrieval_latency_ms": int((time.time() - t0) * 1000),
        "_evidence_records": evidence_records,
    }


# -- Node: Confidence scoring --------------------------------------------------

def confidence_node(state: InvoiceState) -> dict:
    evidence_records = state.get("_evidence_records")
    if evidence_records is None:
        from ap_automation.core.models import AmountBand
        evidence_records = retrieve(
            description=state["description"],
            vendor_id=state["canonical_vendor_id"],
            entity_id=state["entity_id"],
            amount_band=AmountBand(state["amount_band"]),
            currency=state["currency"],
        )

    t0 = time.time()
    predicted_gl, predicted_cc, agreement_ratio, weighted_confidence = conf_module.score(evidence_records)
    return {
        "predicted_gl": predicted_gl,
        "predicted_cost_centre": predicted_cc,
        "agreement_ratio": round(agreement_ratio, 4),
        "weighted_confidence": round(weighted_confidence, 4),
        "confidence_latency_ms": int((time.time() - t0) * 1000),
        "_evidence_records": evidence_records,
    }


# -- Node: Routing -------------------------------------------------------------

def routing_node(state: InvoiceState) -> dict:
    evidence_records = state.get("_evidence_records", [])
    routing_decision, routing_reason = conf_module.route(
        evidence=evidence_records,
        vendor_id=state["canonical_vendor_id"],
        weighted_confidence=state["weighted_confidence"],
        mdm_confidence=state.get("mdm_confidence", 1.0),
        agreement_ratio=state.get("agreement_ratio"),
    )
    return {
        "routing_decision": routing_decision.value,
        "routing_reason": routing_reason,
    }


# -- Node: A2A agents ----------------------------------------------------------

def agents_node(state: InvoiceState) -> dict:
    context = {
        "predicted_gl": state.get("predicted_gl"),
        "predicted_cost_centre": state.get("predicted_cost_centre"),
        "submitted_cost_centre": state.get("submitted_cost_centre", ""),
        "description": state["description"],
        "amount": state["amount"],
        "currency": state["currency"],
        "vendor_id": state["canonical_vendor_id"],
        "entity_id": state["entity_id"],
        "routing_decision": state.get("routing_decision"),
        "weighted_confidence": state.get("weighted_confidence"),
    }

    t0 = time.time()
    result = _orchestrator.run(context)
    agent_ms = int((time.time() - t0) * 1000)
    signals = governance.extract_signals(result)

    return {
        "agent_results": [r.to_dict() for r in result.results],
        "agents_called": len(result.results),
        "agent_latency_ms": agent_ms,
        "vat_treatment": signals["vat_treatment"],
        "vat_review_required": signals["vat_review_required"],
        "cost_centre_mismatch": signals["cost_centre_mismatch"],
        "entity_valid": signals["entity_valid"],
        "agent_errors": signals["agent_errors"],
        "predicted_cost_centre": signals["inferred_cost_centre"] or state.get("predicted_cost_centre"),
        "_agent_orchestration_result": result,
    }


# -- Node: Rule gate -----------------------------------------------------------

def gate_node(state: InvoiceState) -> dict:
    # Shared with core/pipeline.py via core/governance.py - one flag vocabulary, one gate.
    flags = governance.agent_flags(
        cost_centre_mismatch=bool(state.get("cost_centre_mismatch")),
        entity_valid=state.get("entity_valid"),
        vat_treatment=state.get("vat_treatment"),
        vat_review_required=bool(state.get("vat_review_required")),
        agent_errors=state.get("agent_errors") or [],
    )
    base_routing = state.get("routing_decision", RoutingDecision.HUMAN_REVIEW.value)
    decision, gate_reason = governance.apply_gate(
        base_routing, state.get("routing_reason", "Rule gate"), flags
    )
    return {
        "final_routing": decision.value,
        "gate_reason": gate_reason,
        "agent_flags": flags,
    }


# -- Node: LLM reasoning gate --------------------------------------------------

def llm_gate_node(state: InvoiceState) -> dict:
    """
    LLM reasoning layer on top of the rule gate.
    Fetches prompt from Langfuse Prompt Management (llm-gate-conflict-analysis, production label).
    Falls back to hardcoded prompt if Langfuse is unavailable.
    Can escalate auto_post to human_review but CANNOT reverse human_review to auto_post.
    """
    t0 = time.time()

    agent_summary = []
    for r in state.get("agent_results", []):
        agent_summary.append({
            "agent": r.get("agent", r.get("agent_name", "unknown")),
            "success": r.get("success"),
            "agent_confidence": r.get("confidence"),
            "reasoning": r.get("reasoning", ""),
            "data": {
                k: v for k, v in r.items()
                if k not in ("agent", "agent_name", "success", "confidence", "reasoning", "latency_ms", "error")
            },
        })

    evidence = state.get("evidence", [])
    evidence_summary = {
        "total_lines": len(evidence),
        "gl_agreement_pct": round(
            sum(1 for e in evidence if e.get("gl_account") == state.get("predicted_gl")) / len(evidence) * 100
            if evidence else 0,
            1,
        ),
        "top_similarity": round(max((e.get("similarity_score", 0) for e in evidence), default=0), 3),
        "predicted_gl": state.get("predicted_gl"),
    }

    try:
        import httpx
        api_key = state.get("_api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("No ANTHROPIC_API_KEY found")

        template = _get_prompt_template()
        prompt = _compile_prompt(template, state, agent_summary, evidence_summary)

        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=15.0,
        )

        response.raise_for_status()
        raw = response.json()["content"][0]["text"].strip()
        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        # Extract JSON object - find first { and last }
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in LLM response")
        raw = raw[start:end]
        result = json.loads(raw)

        reasoning_confidence = float(result.get("reasoning_confidence", 0.0))
        conflicts = result.get("conflicts", [])
        summary = result.get("summary", "")

        # The LLM may only make the decision stricter. Anything it returns is validated
        # against the incoming decision here, and again by the caller.
        base = state.get("final_routing", RoutingDecision.HUMAN_REVIEW.value)
        routing = governance.never_loosen(base, result.get("routing")).value
        escalated = routing == RoutingDecision.HUMAN_REVIEW.value and base == RoutingDecision.AUTO_POST.value

        return {
            "final_routing": routing,
            "llm_reasoning": summary,
            "llm_reasoning_confidence": reasoning_confidence,
            "llm_escalated": escalated,
            "llm_conflicts": conflicts,
            "llm_conflict_count": len(conflicts),
            "llm_gate_reason": f"LLM: {summary}",
            "llm_latency_ms": int((time.time() - t0) * 1000),
        }

    except Exception as e:
        print(f"[LLM Gate] Error: {e} - falling back to rule gate decision")
        return {
            "llm_reasoning": f"LLM gate unavailable: {str(e)[:100]}. Rule gate decision stands.",
            "llm_reasoning_confidence": 0.0,
            "llm_escalated": False,
            "llm_conflicts": [],
            "llm_conflict_count": 0,
            "llm_gate_reason": "Rule gate only (LLM unavailable)",
            "llm_latency_ms": int((time.time() - t0) * 1000),
            "final_routing": state.get("final_routing", RoutingDecision.HUMAN_REVIEW.value),
        }


# -- Node: Auto post -----------------------------------------------------------

def auto_post_node(state: InvoiceState) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "confirmed_gl": state.get("predicted_gl"),
        "confirmed_cc": state.get("predicted_cost_centre"),
        "confirmation_source": ConfirmationSource.AUTO_POST.value,
        "confirmed_at": now,
        "erp_posted": True,
        "erp_posted_at": now,
        "is_correct": None,
        "is_false_positive": False,
    }


# -- Node: Human review --------------------------------------------------------

def human_review_node(state: InvoiceState) -> dict:
    payload = {
        "invoice": {
            "invoice_id": state.get("invoice_id"),
            "raw_vendor_name": state.get("raw_vendor_name"),
            "description": state.get("description"),
            "amount": state.get("amount"),
            "currency": state.get("currency"),
            "entity_id": state.get("entity_id"),
            "submitted_cost_centre": state.get("submitted_cost_centre"),
        },
        "prediction": {
            "predicted_gl": state.get("predicted_gl"),
            "predicted_cost_centre": state.get("predicted_cost_centre"),
            "agreement_ratio": state.get("agreement_ratio"),
            "weighted_confidence": state.get("weighted_confidence"),
            "routing_decision": state.get("routing_decision"),
            "final_routing": state.get("final_routing"),
        },
        "evidence": state.get("evidence") or [],
        "agent_results": state.get("agent_results") or [],
        "agent_flags": state.get("agent_flags") or [],
        "confidence": state.get("weighted_confidence"),
        "routing_reason": state.get("gate_reason") or state.get("routing_reason") or "",
        "final_routing_reason": state.get("gate_reason") or state.get("routing_reason") or "",
        "llm_reasoning": state.get("llm_reasoning"),
        "llm_conflicts": state.get("llm_conflicts") or [],
    }
    updates: dict = {
        "review_queued_at": datetime.now(timezone.utc).isoformat(),
        "erp_posted": False,
    }
    # A resume that already carries the reviewer's decision must not pause again.
    if not state.get("confirmed_gl") and not state.get("review_notes") == "rejected":
        decision = _lg_interrupt(payload)
        if isinstance(decision, dict):
            if decision.get("confirmed_gl"):
                updates["confirmed_gl"] = decision["confirmed_gl"]
            if decision.get("confirmed_cc") is not None:
                updates["confirmed_cc"] = decision["confirmed_cc"]
            if decision.get("confirmed_by") or decision.get("reviewer_id"):
                updates["reviewer_id"] = decision.get("confirmed_by") or decision.get("reviewer_id")
            if decision.get("confirmation_source"):
                updates["confirmation_source"] = decision["confirmation_source"]
            if decision.get("action") == "reject":
                updates["review_notes"] = "rejected"
    return updates


# -- Node: Confirmation --------------------------------------------------------

def confirmation_node(state: InvoiceState) -> dict:
    predicted = state.get("predicted_gl")
    confirmed = state.get("confirmed_gl")
    is_correct = predicted == confirmed if predicted and confirmed else None
    is_false_positive = (
        state.get("final_routing") == RoutingDecision.AUTO_POST.value
        and state.get("confirmation_source") == ConfirmationSource.HUMAN_CORRECT.value
        and not is_correct
    )
    return {
        "is_correct": is_correct,
        "is_false_positive": is_false_positive,
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
    }


# -- Node: Write-back ----------------------------------------------------------

def write_back_node(state: InvoiceState) -> dict:
    from ap_automation.core.retrieval import _collection_size

    # Same rule as core.pipeline.write_back: only human decisions are learned from.
    if state.get("confirmation_source", ConfirmationSource.AUTO_POST.value) == ConfirmationSource.AUTO_POST.value:
        return {
            "write_back_complete": False,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    corpus_before = _collection_size()

    payload = {
        "point_id": str(uuid4()),
        "invoice_id": state["invoice_id"],
        "vendor_id": state["canonical_vendor_id"],
        "entity_id": state["entity_id"],
        "amount_band": state["amount_band"],
        "currency": state["currency"],
        "gl_account": state.get("confirmed_gl") or state.get("predicted_gl"),
        "cost_centre": state.get("confirmed_cc") or state.get("predicted_cost_centre"),
        "tax_code": state.get("vat_treatment") or "UNASSESSED",
        "description": state["description"],
        "confirming_user_id": state.get("reviewer_id", "system"),
        "confirmation_timestamp": state.get("confirmed_at", datetime.now(timezone.utc).isoformat()),
        "prediction_confidence": state.get("weighted_confidence", 0.0),
        "source": state.get("confirmation_source", ConfirmationSource.AUTO_POST.value),
    }
    index_outcome(payload)

    corpus_after = _collection_size()
    completed_at = datetime.now(timezone.utc).isoformat()

    started = state.get("started_at")
    total_ms = None
    if started:
        try:
            delta = datetime.now(timezone.utc) - datetime.fromisoformat(started)
            total_ms = int(delta.total_seconds() * 1000)
        except Exception:
            pass

    return {
        "write_back_complete": True,
        "corpus_before": corpus_before,
        "corpus_after": corpus_after,
        "completed_at": completed_at,
        "total_latency_ms": total_ms,
    }


# -- Edge router ---------------------------------------------------------------

def route_after_gate(state: InvoiceState) -> str:
    final = state.get("final_routing", RoutingDecision.HUMAN_REVIEW.value)
    if final == RoutingDecision.AUTO_POST.value:
        return "auto_post"
    return "human_review"
