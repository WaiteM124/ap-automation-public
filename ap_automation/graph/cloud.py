"""
Cloud LangGraph (LangSmith Deployment) client.

When LANGGRAPH_API_URL is set, predict / review-queue / thread confirm go through
the SDK. When it is empty the in-process pipeline is used (tests, readiness, local demo).
URL and key are read from settings only — never hardcoded.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from ap_automation.core.config import settings
from ap_automation.core.models import (
    AmountBand,
    ConfirmationSource,
    InvoiceLine,
    Prediction,
    ResolvedInvoiceLine,
    RetrievedEvidence,
    RoutingDecision,
    amount_to_band,
)

# Must match the key in langgraph.json. Not a secret.
GRAPH_ID = "invoice_pipeline"


def configured() -> bool:
    return bool((settings.langgraph_api_url or "").strip())


def get_lg_client():
    if not configured():
        raise RuntimeError("LANGGRAPH_API_URL is not set")
    from langgraph_sdk import get_sync_client

    return get_sync_client(
        url=settings.langgraph_api_url.strip(),
        api_key=(settings.langgraph_api_key or None) or None,
    )


def invoice_input(invoice: InvoiceLine, prediction_id: str | None = None) -> dict[str, Any]:
    return {
        "invoice_id": str(invoice.invoice_id),
        "raw_vendor_name": invoice.raw_vendor_name,
        "description": invoice.description,
        "amount": invoice.amount,
        "currency": invoice.currency,
        "entity_id": invoice.entity_id,
        "submitted_cost_centre": invoice.cost_centre,
        "ground_truth_gl": invoice.ground_truth_gl,
        "prediction_id": prediction_id or str(uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def _id(obj: Any, *keys: str) -> str:
    if isinstance(obj, dict):
        for k in keys:
            if obj.get(k):
                return str(obj[k])
    for k in keys:
        if hasattr(obj, k) and getattr(obj, k):
            return str(getattr(obj, k))
    return ""


def _values(state: Any) -> dict[str, Any]:
    if isinstance(state, dict):
        return state.get("values") or state
    vals = getattr(state, "values", None)
    return vals if isinstance(vals, dict) else {}


def _interrupts(state: Any) -> list[Any]:
    if isinstance(state, dict):
        tasks = state.get("tasks") or []
        found = []
        for t in tasks:
            ints = t.get("interrupts") if isinstance(t, dict) else getattr(t, "interrupts", None)
            if ints:
                found.extend(ints)
        if state.get("interrupts"):
            found.extend(state["interrupts"])
        return found
    return list(getattr(state, "interrupts", None) or [])


def _run_status(run: Any) -> str:
    if isinstance(run, dict):
        return str(run.get("status") or "")
    return str(getattr(run, "status", "") or "")


def _evidence(raw: list[dict] | None) -> list[RetrievedEvidence]:
    out: list[RetrievedEvidence] = []
    for i, e in enumerate(raw or []):
        ts_raw = e.get("confirmation_timestamp")
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")) if ts_raw else datetime.now(timezone.utc)
        except ValueError:
            ts = datetime.now(timezone.utc)
        src = e.get("source") or ConfirmationSource.HUMAN_CONFIRM.value
        try:
            source = ConfirmationSource(src)
        except ValueError:
            source = ConfirmationSource.HUMAN_CONFIRM
        out.append(RetrievedEvidence(
            invoice_id=str(e.get("invoice_id") or i),
            description=str(e.get("description") or ""),
            gl_account=str(e.get("gl_account") or ""),
            cost_centre=str(e.get("cost_centre") or ""),
            vendor_id=str(e.get("vendor_id") or ""),
            similarity_score=float(e.get("similarity_score") or 0),
            dense_rank=int(e.get("dense_rank") or i + 1),
            sparse_rank=int(e.get("sparse_rank") or i + 1),
            rrf_score=float(e.get("rrf_score") or 0),
            confirmation_timestamp=ts,
            source=source,
        ))
    return out


def state_to_prediction(state: Any, invoice: InvoiceLine, thread_id: str, prediction_id: UUID) -> Prediction:
    values = _values(state)
    routing = values.get("routing_decision") or RoutingDecision.HUMAN_REVIEW.value
    final = values.get("final_routing") or RoutingDecision.HUMAN_REVIEW.value
    try:
        rule = RoutingDecision(routing)
    except ValueError:
        rule = RoutingDecision.HUMAN_REVIEW
    try:
        governed = RoutingDecision(final)
    except ValueError:
        governed = RoutingDecision.HUMAN_REVIEW
    band = values.get("amount_band") or amount_to_band(invoice.amount).value
    try:
        amount_band = AmountBand(band)
    except ValueError:
        amount_band = amount_to_band(invoice.amount)
    evidence = _evidence(values.get("evidence"))
    flags = list(values.get("agent_flags") or [])
    reason = values.get("gate_reason") or values.get("routing_reason") or ""
    provenance = {
        "prediction_id": str(prediction_id),
        "thread_id": thread_id,
        "invoice_id": str(invoice.invoice_id),
        "vendor_raw": invoice.raw_vendor_name,
        "vendor_canonical": values.get("canonical_vendor_id") or "",
        "mdm_confidence": values.get("mdm_confidence", 0.0),
        "amount_band": amount_band.value,
        "evidence_count": len(evidence),
        "predicted_gl": values.get("predicted_gl") or "",
        "predicted_cost_centre": values.get("predicted_cost_centre") or "",
        "submitted_cost_centre": invoice.cost_centre,
        "agreement_ratio": values.get("agreement_ratio") or 0.0,
        "weighted_confidence": values.get("weighted_confidence") or 0.0,
        "routing_decision": rule.value,
        "routing_reason": values.get("routing_reason") or reason,
        "agent_flags": flags,
        "vat_treatment": values.get("vat_treatment"),
        "final_routing": governed.value,
        "final_routing_reason": reason,
        "llm_reasoning": values.get("llm_reasoning") or "",
        "llm_reasoning_confidence": values.get("llm_reasoning_confidence") or 0.0,
        "llm_escalated": values.get("llm_escalated") or False,
        "llm_conflicts": values.get("llm_conflicts") or [],
        "llm_conflict_count": values.get("llm_conflict_count") or 0,
        "a2a_agents": {"results": values.get("agent_results") or []},
        "retrieved_evidence": values.get("evidence") or [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return Prediction(
        prediction_id=prediction_id,
        resolved_line=ResolvedInvoiceLine(
            invoice_line=invoice,
            canonical_vendor_id=str(values.get("canonical_vendor_id") or ""),
            mdm_confidence=float(values.get("mdm_confidence") or 0.0),
            amount_band=amount_band,
        ),
        evidence=evidence,
        predicted_gl=values.get("predicted_gl") or "",
        predicted_cost_centre=values.get("predicted_cost_centre") or "",
        agreement_ratio=float(values.get("agreement_ratio") or 0.0),
        weighted_confidence=float(values.get("weighted_confidence") or 0.0),
        routing_decision=rule,
        routing_reason=values.get("routing_reason") or reason,
        final_routing=governed,
        final_routing_reason=reason,
        agent_flags=flags,
        provenance=provenance,
    )


def predict_remote(invoice: InvoiceLine, *, batch_mode: bool = False) -> Prediction:
    """Create a thread + run. Interrupted runs come back as HUMAN_REVIEW with a thread_id."""
    del batch_mode  # the deployed graph always runs the full node chain
    client = get_lg_client()
    prediction_id = uuid4()
    thread = client.threads.create()
    thread_id = _id(thread, "thread_id", "threadId")
    run = client.runs.create(thread_id, GRAPH_ID, input=invoice_input(invoice, str(prediction_id)))
    run_id = _id(run, "run_id", "runId")
    if hasattr(client.runs, "join"):
        client.runs.join(thread_id, run_id)
    run = client.runs.get(thread_id, run_id)
    state = client.threads.get_state(thread_id)
    pred = state_to_prediction(state, invoice, thread_id, prediction_id)
    if _run_status(run) in {"interrupted", "interrupt"} or (getattr(state, "next", None) or _values(state).get("next")):
        pred.final_routing = RoutingDecision.HUMAN_REVIEW
        if not pred.final_routing_reason:
            pred.final_routing_reason = "Paused for human review"
        pred.provenance["thread_id"] = thread_id
        pred.provenance["run_status"] = _run_status(run)
    return pred


def resume_remote(
    thread_id: str,
    *,
    gl_account: str,
    cost_centre: str,
    confirmed_by: str,
    action: str,
    predicted_gl: str = "",
) -> dict[str, Any]:
    """Resume a paused thread with the reviewer's decision. Does not call write_back."""
    client = get_lg_client()
    if action == "reject":
        decision = {"action": "reject", "confirmed_by": confirmed_by, "review_notes": "rejected"}
        source = None
        confirmed_gl = ""
    else:
        confirmed_gl = gl_account
        source = (
            ConfirmationSource.HUMAN_CONFIRM
            if not predicted_gl or gl_account == predicted_gl
            else ConfirmationSource.HUMAN_CORRECT
        )
        decision = {
            "action": action,
            "confirmed_gl": confirmed_gl,
            "confirmed_cc": cost_centre,
            "confirmed_by": confirmed_by,
            "confirmation_source": source.value,
        }
    client.threads.update_state(thread_id, {
        "confirmed_gl": confirmed_gl,
        "confirmed_cc": cost_centre,
        "reviewer_id": confirmed_by,
        "confirmation_source": source.value if source else "",
        "review_notes": "rejected" if action == "reject" else "",
    })
    run = client.runs.create(thread_id, GRAPH_ID, command={"resume": decision})
    run_id = _id(run, "run_id", "runId")
    if run_id and hasattr(client.runs, "join"):
        client.runs.join(thread_id, run_id)
    state = client.threads.get_state(thread_id)
    return {"thread_id": thread_id, "state": _values(state), "action": action, "source": source.value if source else ""}


def list_paused() -> list[dict[str, Any]]:
    """Paused (interrupted) threads and their interrupt / state payload."""
    if not configured():
        return []
    client = get_lg_client()
    search = getattr(client.threads, "search", None)
    threads = search(status="interrupted", limit=100) if search else []
    items: list[dict[str, Any]] = []
    for thread in threads or []:
        thread_id = _id(thread, "thread_id", "threadId")
        if not thread_id:
            continue
        try:
            state = client.threads.get_state(thread_id)
        except Exception:
            continue
        values = _values(state)
        payload = None
        for it in _interrupts(state):
            if isinstance(it, dict) and ("invoice" in it or "value" in it):
                payload = it.get("value") if "invoice" not in it else it
                break
            val = getattr(it, "value", None)
            if isinstance(val, dict):
                payload = val
                break
        if payload is None:
            payload = {
                "invoice": {
                    "invoice_id": values.get("invoice_id"),
                    "raw_vendor_name": values.get("raw_vendor_name"),
                    "description": values.get("description"),
                    "amount": values.get("amount"),
                    "currency": values.get("currency"),
                    "entity_id": values.get("entity_id"),
                    "submitted_cost_centre": values.get("submitted_cost_centre"),
                },
                "prediction": {
                    "predicted_gl": values.get("predicted_gl"),
                    "predicted_cost_centre": values.get("predicted_cost_centre"),
                    "weighted_confidence": values.get("weighted_confidence"),
                    "final_routing": values.get("final_routing") or RoutingDecision.HUMAN_REVIEW.value,
                },
                "evidence": values.get("evidence") or [],
                "agent_results": values.get("agent_results") or [],
                "agent_flags": values.get("agent_flags") or [],
                "confidence": values.get("weighted_confidence"),
                "routing_reason": values.get("gate_reason") or values.get("routing_reason") or "",
            }
        items.append({"thread_id": thread_id, "payload": payload})
    return items
