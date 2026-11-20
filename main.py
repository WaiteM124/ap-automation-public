# AP Automation API - the single entry point (local and Render).
# Run locally: python -m uvicorn main:app --reload --port 8000

from __future__ import annotations

import asyncio
import csv
import hmac
import io
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ap_automation.core import (
    chart_of_accounts,
    database,
    history_import,
    post_ready,
    readiness,
    retrieval,
    sap_poster,
    xero_auth,
    xero_poster,
)
from ap_automation.core import demo as demo_support
from ap_automation.core.chart_of_accounts import CHART
from ap_automation.core.config import settings
from ap_automation.core.mdm import load_client_vendors, load_vendor_map
from ap_automation.core.models import (
    ConfirmationSource,
    InvoiceCategory,
    InvoiceLine,
    Prediction,
    RoutingDecision,
)
from ap_automation.core.pipeline import AutoPostNotPermitted, predict
from ap_automation.core.posting import post_outcome
from ap_automation.core.prediction_store import PredictionStore
from ap_automation.core.retrieval import _collection_size

load_vendor_map()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not settings.api_key:
        print("[API] WARNING: API_KEY is not set - predict/confirm/batch are open to anyone who can reach this service.")
    if sap_poster.backend_selected():
        s = sap_poster.status()
        print(f"[API] ERP backend: SAP ({s['host']}); posting enabled={s['posting_enabled']}; "
              f"ready={s['ready_to_post']}{'' if s['ready_to_post'] else ' - ' + s['not_ready_reason']}")
    if xero_poster.backend_selected():
        s = xero_poster.status()
        print(f"[API] ERP backend: Xero; posting enabled={s['posting_enabled']}; connected={s['connected']}; "
              f"ready={s['ready_to_post']}{'' if s['ready_to_post'] else ' - ' + s['not_ready_reason']}")
    if chart_of_accounts.load_saved_chart(settings.chart_path):
        print(f"[API] Loaded the client chart of accounts from {settings.chart_path} "
              f"({len(chart_of_accounts.CHART)} accounts)")
    n_vendors = load_client_vendors(settings.client_vendors_path)
    if n_vendors:
        print(f"[API] Loaded {n_vendors} imported vendors from {settings.client_vendors_path}")
    yield


app = FastAPI(title="AP Automation API", version="2.0.0", lifespan=lifespan)

_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """If API_KEY is configured, every endpoint except /health requires `X-API-Key`."""
    if settings.api_key and not hmac.compare_digest(x_api_key or "", settings.api_key):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")


# --------------------------------------------------------------------------- state

_store = PredictionStore()
_jobs: dict[str, dict] = {}            # in-memory job state (authoritative for this process)
_cancelled: set[str] = set()
_jobs_lock = threading.Lock()


# --------------------------------------------------------------------------- models

class PredictRequest(BaseModel):
    raw_vendor_name: str
    description: str
    amount: float = Field(default=0.0, ge=0)
    currency: str = "GBP"
    entity_id: str = "UK001"
    cost_centre: str = "CC300"


class ConfirmRequest(BaseModel):
    confirmed_gl: str
    confirmed_cc: str = ""
    source: Literal["human_confirm", "human_correct"] = "human_confirm"
    confirming_user_id: str = "api_user"


class ThreadConfirmRequest(BaseModel):
    gl_account: str = ""
    cost_centre: str = ""
    confirmed_by: str = "api_user"
    action: Literal["confirm", "override", "reject"] = "confirm"


def _evidence_json(pred: Prediction) -> list[dict]:
    return [
        {
            "description": e.description,
            "gl_account": e.gl_account,
            "similarity_score": e.similarity_score,
            "rrf_score": e.rrf_score,
            "source": e.source.value,
            "confirmation_timestamp": e.confirmation_timestamp.isoformat(),
        }
        for e in pred.evidence
    ]


# --------------------------------------------------------------------------- endpoints

@app.get("/health")
def health():
    return {"status": "ok", "corpus_size": _collection_size()}


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict_invoice(req: PredictRequest):
    invoice = InvoiceLine(
        invoice_id=uuid.uuid4(),
        raw_vendor_name=req.raw_vendor_name,
        description=req.description,
        amount=req.amount,
        currency=req.currency,
        entity_id=req.entity_id,
        cost_centre=req.cost_centre,
        category=InvoiceCategory.ROUTINE,
    )
    pred = predict(invoice)
    _store.put(pred)
    prov = pred.provenance
    return {
        "prediction_id": str(pred.prediction_id),
        "predicted_gl": pred.predicted_gl,
        "predicted_cost_centre": pred.predicted_cost_centre,
        "weighted_confidence": pred.weighted_confidence,
        # `routing_decision` is the rule gate only; `final_routing` is the governed decision
        # and the only one that may authorise an auto-post.
        "routing_decision": pred.routing_decision.value,
        "routing_reason": pred.routing_reason,
        "final_routing": pred.final_routing.value,
        "final_routing_reason": pred.final_routing_reason,
        "agent_flags": pred.agent_flags,
        "vat_treatment": prov.get("vat_treatment"),
        "evidence": _evidence_json(pred),
        "a2a_agents": prov.get("a2a_agents", {}),
        "llm_conflicts": prov.get("llm_conflicts", []),
        "llm_reasoning": prov.get("llm_reasoning", ""),
        "llm_reasoning_confidence": prov.get("llm_reasoning_confidence", 0.0),
        "llm_escalated": prov.get("llm_escalated", False),
        "explanation": _build_explanation(pred),
        "thread_id": prov.get("thread_id"),
    }


@app.get("/review-queue", dependencies=[Depends(require_api_key)])
def review_queue():
    """Paused Cloud LangGraph threads. Empty when LANGGRAPH_API_URL is not set."""
    from ap_automation.graph.cloud import list_paused
    return {"items": list_paused()}


@app.post("/predict/{thread_id}/confirm", dependencies=[Depends(require_api_key)])
def confirm_thread(thread_id: str, req: ThreadConfirmRequest):
    """Resume a paused Cloud LangGraph thread with the reviewer's decision."""
    from ap_automation.graph.cloud import configured, resume_remote, state_to_prediction

    if not configured():
        raise HTTPException(status_code=503, detail="Cloud LangGraph is not configured")
    if req.action != "reject":
        if not req.gl_account:
            raise HTTPException(status_code=422, detail="gl_account is required")
        if req.gl_account not in CHART:
            raise HTTPException(status_code=422, detail=f"Unknown GL account {req.gl_account!r}")
    try:
        resumed = resume_remote(
            thread_id,
            gl_account=req.gl_account,
            cost_centre=req.cost_centre,
            confirmed_by=req.confirmed_by,
            action=req.action,
            predicted_gl="",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not resume the review thread: {e}") from e

    values = resumed.get("state") or {}
    if req.action == "reject":
        return {"status": "rejected", "thread_id": thread_id, "learned": False, "erp_posted": False}

    invoice = InvoiceLine(
        invoice_id=uuid.UUID(values["invoice_id"]) if values.get("invoice_id") else uuid.uuid4(),
        raw_vendor_name=values.get("raw_vendor_name") or "",
        description=values.get("description") or "",
        amount=float(values.get("amount") or 0),
        currency=values.get("currency") or "GBP",
        entity_id=values.get("entity_id") or "UK001",
        cost_centre=values.get("submitted_cost_centre") or req.cost_centre,
        category=InvoiceCategory.ROUTINE,
    )
    pred = state_to_prediction(values, invoice, thread_id, invoice.invoice_id)
    source = (ConfirmationSource.HUMAN_CONFIRM if not pred.predicted_gl or req.gl_account == pred.predicted_gl
              else ConfirmationSource.HUMAN_CORRECT)
    _store.put(pred)
    claimed = _store.claim(str(pred.prediction_id))
    if claimed is None:
        claimed = pred
    try:
        result = post_outcome(
            claimed, req.gl_account, req.cost_centre or pred.predicted_cost_centre,
            req.confirmed_by, source,
        )
    except Exception:
        _store.release(str(pred.prediction_id))
        raise
    _raise_if_erp_failed(str(pred.prediction_id), pred, result)
    return {
        "status": "confirmed",
        "thread_id": thread_id,
        "prediction_id": str(pred.prediction_id),
        "predicted_gl": pred.predicted_gl,
        "source": source.value,
        "learned": result.learned,
        "erp_posted": result.erp_posted,
        "erp_backend": result.erp_backend,
        "erp_ref": result.erp_ref,
        "warnings": result.error,
    }


@app.post("/confirm/{prediction_id}", dependencies=[Depends(require_api_key)])
def confirm_invoice(prediction_id: str, req: ConfirmRequest):
    if req.confirmed_gl not in CHART:
        raise HTTPException(status_code=422, detail=f"Unknown GL account {req.confirmed_gl!r}")
    item = _store.get(prediction_id)
    if item is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found (it may have been pruned, or the demo was reset)",
        )
    pred = _store.claim(prediction_id)
    if pred is None:
        raise HTTPException(status_code=409, detail="Prediction has already been posted")

    # A confirmation that changes the GL is a correction, whatever the caller labelled it. When the system made
    # no prediction (cold start) there is nothing to correct: the human is coding the invoice.
    source = (ConfirmationSource.HUMAN_CONFIRM if not pred.predicted_gl or req.confirmed_gl == pred.predicted_gl
              else ConfirmationSource.HUMAN_CORRECT)
    try:
        result = post_outcome(
            pred, req.confirmed_gl, req.confirmed_cc or pred.predicted_cost_centre,
            req.confirming_user_id, source,
        )
    except Exception:
        _store.release(prediction_id)
        raise
    _raise_if_erp_failed(prediction_id, pred, result)
    return {
        "status": "confirmed",
        "source": source.value,
        "learned": result.learned,
        "erp_posted": result.erp_posted,
        "erp_backend": result.erp_backend,
        "erp_ref": result.erp_ref,
        "warnings": result.error,
        "erp_payload": result.erp_payload.model_dump(mode="json"),
    }


_ERP_LABEL = {"sap": "SAP", "xero": "Xero"}


def _erp_failure_reason(result) -> str:
    ref = result.erp_ref or {}
    return f"{_ERP_LABEL.get(result.erp_backend, 'ERP')} post not completed ({ref.get('status')}): {ref.get('error')}"


def _raise_if_erp_failed(prediction_id: str, pred: Prediction, result) -> None:
    """
    A post that SAP / Xero did not accept must not look like a success: the invoice goes back to the review queue
    (unposted, no longer cleared for auto-post) carrying the sanitised error, and the caller gets a 502 with it.
    """
    if not result.erp_failed:
        return
    ref = result.erp_ref or {}
    label = _ERP_LABEL.get(result.erp_backend, "ERP")
    reason = _erp_failure_reason(result)
    if pred.final_routing_reason and not pred.final_routing_reason.startswith(
            tuple(f"{name} post not completed" for name in _ERP_LABEL.values())):
        reason += f" | earlier routing reason: {pred.final_routing_reason}"
    _store.return_to_review(prediction_id, reason[:900])
    raise HTTPException(status_code=502, detail={
        "status": f"{result.erp_backend}_post_failed",
        "message": f"{label} did not accept the posting. The invoice stays in human review and nothing was posted.",
        "prediction_id": prediction_id,
        result.erp_backend: {k: ref.get(k) for k in ("status", "reference", "http_status", "error")},
    })


@app.post("/auto-post/{prediction_id}", dependencies=[Depends(require_api_key)])
def auto_post_invoice(prediction_id: str):
    """Post a prediction the governed gate approved, with no human in the loop.

    Refused (403) unless `final_routing` is auto_post, whatever the caller claims. Never written back to the
    corpus: the system does not learn from its own decisions.
    """
    item = _store.get(prediction_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Prediction not found (it may have been pruned, or the demo was reset)")
    pred = item["pred"]
    if pred.final_routing != RoutingDecision.AUTO_POST:
        raise HTTPException(status_code=403, detail=f"Not approved for auto-post: {pred.final_routing_reason}")
    if _store.claim(prediction_id) is None:
        raise HTTPException(status_code=409, detail="Prediction has already been posted")
    try:
        result = post_outcome(pred, pred.predicted_gl, pred.predicted_cost_centre, "auto_post_service",
                              ConfirmationSource.AUTO_POST)
    except AutoPostNotPermitted as e:
        _store.release(prediction_id)
        raise HTTPException(status_code=403, detail=str(e)) from e
    except Exception:
        _store.release(prediction_id)
        raise
    _raise_if_erp_failed(prediction_id, pred, result)
    return {
        "status": "auto_posted",
        "source": ConfirmationSource.AUTO_POST.value,
        "learned": result.learned,
        "erp_posted": result.erp_posted,
        "erp_backend": result.erp_backend,
        "erp_ref": result.erp_ref,
        "warnings": result.error,
        "erp_payload": result.erp_payload.model_dump(mode="json"),
    }


@app.post("/batch", dependencies=[Depends(require_api_key)])
async def batch_predict(
    file: UploadFile = File(...),
    governance_mode: str = "false",
    simulate_reviewer: str = "false",
):
    """
    Async batch. Returns a job_id immediately; poll GET /batch/{job_id}.

    governance_mode=true runs the LLM gate on every row (slower, costs tokens).
    simulate_reviewer=true resolves review-routed rows using the CSV's `ground_truth_gl`
    column, standing in for a human reviewer - this is what lets the corpus learn during a
    demo run. Auto-posted rows are never learned from. Without it, review rows stay pending
    and can be resolved through POST /confirm/{prediction_id}.
    """
    content = await file.read()
    try:
        rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded")
    if not rows:
        raise HTTPException(status_code=400, detail="CSV contains no rows")

    gov = governance_mode.lower() == "true"
    sim = simulate_reviewer.lower() == "true"
    job_id = str(uuid.uuid4())
    total = len(rows)

    from ap_automation.core.supabase import create_batch_job

    state = {
        "id": job_id, "status": "running", "total": total, "completed": 0,
        "auto_confirmed": 0, "human_review": 0, "errors": 0, "results": [], "error": None,
        "governance_mode": gov, "simulate_reviewer": sim,
    }
    with _jobs_lock:
        _jobs[job_id] = state
    create_batch_job(job_id, total, gov)

    threading.Thread(target=_run_batch, args=(job_id, rows, gov, sim), daemon=True).start()
    return {"job_id": job_id, "total": total, "status": "running",
            "governance_mode": gov, "simulate_reviewer": sim}


def _process_row(job_id: str, row: dict, gov: bool, sim: bool) -> dict:
    if job_id in _cancelled:
        return {"skipped": True}
    try:
        invoice = InvoiceLine(
            invoice_id=uuid.uuid4(),
            raw_vendor_name=row.get("vendor_name") or row.get("raw_vendor_name") or "",
            description=row.get("description", ""),
            amount=float(row.get("amount") or 0),
            currency=row.get("currency") or "GBP",
            entity_id=row.get("entity_id") or "UK001",
            cost_centre=row.get("cost_centre") or None,
            category=InvoiceCategory.ROUTINE,
            ground_truth_gl=row.get("ground_truth_gl") or None,
        )
        pred = predict(invoice, batch_mode=(not gov))
        _store.put(pred)

        posted, learned, warning = False, False, None
        routing, reason = pred.final_routing.value, pred.final_routing_reason
        if pred.final_routing == RoutingDecision.AUTO_POST:
            res = post_outcome(pred, pred.predicted_gl, pred.predicted_cost_centre,
                               settings.auto_post_user_id, ConfirmationSource.AUTO_POST)
            posted, warning = res.erp_posted, res.error
            if res.erp_failed:  # SAP / Xero did not accept it: it is a review item now, not an auto-post
                routing = RoutingDecision.HUMAN_REVIEW.value
                reason = _erp_failure_reason(res)
                _store.return_to_review(str(pred.prediction_id), reason[:900])
        elif sim and invoice.ground_truth_gl and _store.claim(str(pred.prediction_id)):
            source = (ConfirmationSource.HUMAN_CONFIRM if invoice.ground_truth_gl == pred.predicted_gl
                      else ConfirmationSource.HUMAN_CORRECT)
            res = post_outcome(pred, invoice.ground_truth_gl, pred.predicted_cost_centre,
                               "simulated_reviewer", source)
            posted, learned, warning = res.erp_posted, res.learned, res.error
            if res.erp_failed:
                _store.return_to_review(str(pred.prediction_id), _erp_failure_reason(res)[:900])

        return {
            "vendor": invoice.raw_vendor_name,
            "description": invoice.description,
            "predicted_gl": pred.predicted_gl,
            "confidence": pred.weighted_confidence,
            "routing": routing,
            "rule_routing": pred.routing_decision.value,
            "reason": reason,
            "agent_flags": pred.agent_flags,
            "prediction_id": str(pred.prediction_id),
            "erp_posted": posted,
            "learned": learned,
            "warning": warning,
        }
    except AutoPostNotPermitted as e:  # defence in depth - should be unreachable
        return {"error": str(e), "row": dict(row)}
    except Exception as e:
        return {"error": str(e), "row": dict(row)}


def _run_batch(job_id: str, rows: list[dict], gov: bool, sim: bool) -> None:
    from ap_automation.core.supabase import fail_batch_job, update_batch_job

    state = _jobs[job_id]
    total = len(rows)

    def sync(status: str) -> None:
        with _jobs_lock:
            snapshot = dict(state, results=list(state["results"]))
            state["status"] = status
        update_batch_job(job_id, snapshot["completed"], snapshot["auto_confirmed"],
                         snapshot["human_review"], snapshot["results"], status)

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(_process_row, job_id, row, gov, sim) for row in rows]
            for fut in as_completed(futures):
                result = fut.result()
                if result.get("skipped"):
                    continue
                with _jobs_lock:
                    state["results"].append(result)
                    state["completed"] += 1
                    if "error" in result:
                        state["errors"] += 1
                        state["human_review"] += 1
                    elif result["routing"] == RoutingDecision.AUTO_POST.value:
                        state["auto_confirmed"] += 1
                    else:
                        state["human_review"] += 1
                    completed = state["completed"]
                if completed % 5 == 0 or completed == total:
                    sync("running")

        if job_id in _cancelled:
            sync("cancelled")
        else:
            sync("complete")
    except Exception as e:
        with _jobs_lock:
            state["status"], state["error"] = "failed", str(e)
        fail_batch_job(job_id, str(e))
        print(f"[Batch] job {job_id[:8]} failed: {e}")
    finally:
        _cancelled.discard(job_id)


@app.get("/batch/{job_id}", dependencies=[Depends(require_api_key)])
async def batch_status(job_id: str):
    """Poll for batch job status. Returns progress and results so far."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        job = dict(job, results=list(job["results"])) if job else None
    if job is None:  # job from another process / before a restart
        from ap_automation.core.supabase import get_batch_job
        job = get_batch_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    total, completed = job.get("total", 0), job.get("completed", 0)
    auto = job.get("auto_confirmed", 0)
    return {
        "job_id": job_id,
        "status": job.get("status", "running"),
        "total": total,
        "completed": completed,
        "auto_confirmed": auto,
        "human_review": job.get("human_review", 0),
        "errors": job.get("errors", 0),
        "automation_rate": round(auto / completed * 100, 1) if completed else 0,
        "progress_pct": round(completed / total * 100) if total else 0,
        "results": job.get("results", []),
        "error": job.get("error"),
    }


@app.get("/batch/{job_id}/stream", dependencies=[Depends(require_api_key)])
async def batch_stream(job_id: str):
    """SSE stream of batch progress (replaces polling GET /batch/{job_id})."""

    async def events():
        last = (-1, "")
        while True:
            with _jobs_lock:
                job = _jobs.get(job_id)
                snapshot = dict(job, results=list(job["results"])) if job else None
            if snapshot is None:
                from ap_automation.core.supabase import get_batch_job
                snapshot = get_batch_job(job_id)
            if not snapshot:
                yield f"event: error\ndata: {json.dumps({'detail': 'Job not found'})}\n\n"
                return
            total, completed = snapshot.get("total", 0), snapshot.get("completed", 0)
            auto = snapshot.get("auto_confirmed", 0)
            payload = {
                "job_id": job_id,
                "status": snapshot.get("status", "running"),
                "total": total,
                "completed": completed,
                "auto_confirmed": auto,
                "human_review": snapshot.get("human_review", 0),
                "errors": snapshot.get("errors", 0),
                "automation_rate": round(auto / completed * 100, 1) if completed else 0,
                "progress_pct": round(completed / total * 100) if total else 0,
                "results": snapshot.get("results", []),
                "error": snapshot.get("error"),
            }
            key = (completed, payload["status"])
            if key != last:
                yield f"data: {json.dumps(payload)}\n\n"
                last = key
            if payload["status"] in ("complete", "failed", "cancelled"):
                yield f"event: done\ndata: {json.dumps(payload)}\n\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


@app.delete("/batch/{job_id}", dependencies=[Depends(require_api_key)])
async def cancel_batch(job_id: str):
    """Cancel a running job. Rows not yet started are skipped immediately."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] != "running":
            return {"status": job["status"], "message": "Job is not running"}
        _cancelled.add(job_id)
    return {"status": "cancelling", "job_id": job_id}


@app.get("/performance", dependencies=[Depends(require_api_key)])
def performance(days: int = 30):
    """Measured from recorded prediction events. Nothing here is hard-coded."""
    from ap_automation.core.supabase import query_analytics

    data = query_analytics(days=days)
    return {
        "corpus_size": _collection_size(),
        "data_available": bool(data),
        "period_days": days,
        "invoices_processed": data["total_predictions"] if data else 0,
        "automation_rate": data["automation_rate"] if data else None,
        # Accuracy needs ground truth, which production events do not carry.
        # Run `python scripts/evaluate_poc.py` for measured accuracy / false positives.
        "false_positive_rate": None,
        "recall_at_10": None,
    }


# --------------------------------------------------------------------------- ledger, audit, corpus

@app.get("/ledger", dependencies=[Depends(require_api_key)])
def ledger(limit: int = 200):
    """ERP postings (newest first) with how each was decided, plus totals computed from the rows."""
    rows = database.get_ledger(min(max(limit, 1), 1000))
    totals: dict[str, float] = {}
    by_source: dict[str, int] = {}
    for r in rows:
        totals[r["currency"]] = round(totals.get(r["currency"], 0.0) + float(r["amount"]), 2)
        by_source[r.get("source") or "unknown"] = by_source.get(r.get("source") or "unknown", 0) + 1
    return {"postings": rows, "summary": {"count": len(rows), "total_by_currency": totals, "by_source": by_source}}


@app.get("/sap/status", dependencies=[Depends(require_api_key)])
def sap_status():
    """Is SAP posting configured and enabled? Host only; no key, password or URL path is ever returned."""
    return sap_poster.status()


@app.get("/sap/postings", dependencies=[Depends(require_api_key)])
def sap_postings(status: str | None = None, limit: int = 100):
    """SAP posting attempts (newest first) with their reference key, SAP document and sanitised error.
    `status` = posted | failed | unknown | in_progress. Failed and unknown ones are the invoices still in review."""
    return {"postings": database.get_sap_postings(min(max(limit, 1), 1000), status)}


@app.get("/xero/status", dependencies=[Depends(require_api_key)])
def xero_status():
    """Is Xero posting configured, connected and enabled? Org name if connected. No id, secret or token is returned."""
    return xero_poster.status()


@app.get("/xero/postings", dependencies=[Depends(require_api_key)])
def xero_postings(status: str | None = None, limit: int = 100):
    """Xero posting attempts (newest first) with reference key, Xero invoice id, deep link and sanitised error.
    `status` = posted | failed | unknown | in_progress. Failed and unknown ones are the invoices still in review."""
    return {"postings": database.get_xero_postings(min(max(limit, 1), 1000), status)}


@app.post("/xero/sync-chart", dependencies=[Depends(require_api_key)])
async def xero_sync_chart(dry_run: str = "false"):
    """
    Replace the chart of accounts with the connected Xero organisation's expense accounts (EXPENSE, DIRECTCOSTS,
    OVERHEADS by default), so that every GL code the system predicts is a valid Xero AccountCode. Same effect as POST
    /chart. Use dry_run=true to see what would change. Needs the app to be connected (scripts/xero_authorize.py).
    """
    try:
        return await run_in_threadpool(xero_poster.sync_chart, dry_run.lower() != "true")
    except xero_auth.XeroError as e:
        raise HTTPException(status_code=409 if e.kind in ("config", "auth") else 502,
                            detail={"message": xero_auth.sanitize(e), "kind": e.kind}) from None


@app.get("/post-ready", dependencies=[Depends(require_api_key)])
def post_ready_list(status: str | None = None, date_from: str | None = None, date_to: str | None = None,
                    limit: int = 200, format: Literal["json", "csv"] = "json"):
    """
    ERP-neutral records of every invoice cleared to post (governed auto_post, or human-confirmed): reference key,
    vendor, description, date, amount, GL (+name), cost centre, VAT treatment, entity, routing, confidence and who/what
    approved it. `status` = ready_to_post | posted | all. `posted` means an external system (SAP / Xero) returned a
    document id; anything else is ready_to_post. Dates are YYYY-MM-DD, inclusive. format=csv downloads a spreadsheet.
    """
    try:
        records = post_ready.build_records(_store, status=status, date_from=date_from, date_to=date_to, limit=limit)
    except post_ready.PostReadyError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    rows = [post_ready.public(r) for r in records]
    if format == "csv":
        return Response(post_ready.to_csv(rows), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="post-ready.csv"'})
    return {"records": rows, "summary": post_ready.summarise(rows),
            "filters": {"status": status or "all", "date_from": date_from, "date_to": date_to,
                        "limit": min(max(limit, 1), post_ready.MAX_LIMIT)}}


@app.get("/post-ready/{reference_key}/payload", dependencies=[Depends(require_api_key)])
def post_ready_payload(reference_key: str, target: str = "neutral"):
    """
    Preview what would be sent to `target` (neutral | xero | sap) for one record. Nothing is sent anywhere. A mapping
    problem (for example no SAP supplier configured) comes back as `error` with a null payload.
    """
    record = post_ready.find_record(reference_key, _store)
    if record is None:
        raise HTTPException(status_code=404, detail="No post-ready record with that reference key")
    try:
        return post_ready.payload_for(record, target)
    except post_ready.PostReadyError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


@app.get("/audit", dependencies=[Depends(require_api_key)])
def audit_trail(limit: int = 200):
    """Audit records (newest first). `provenance` is the full decision record: evidence, agents, gate reasons."""
    database.init_db()
    events = {str(e.get("prediction_id")): e for e in database.get_events(limit=max(limit * 3, 2000))}
    records = []
    for a in database.get_audit_records(min(max(limit, 1), 1000)):
        a["provenance"] = json.loads(a.pop("provenance_json"))
        ev = events.get(a["prediction_id"], {})
        a["vendor_name"] = ev.get("vendor_raw") or a["provenance"].get("vendor_raw") or a["vendor_id"]
        a["description"] = ev.get("description") or ""
        records.append(a)
    return {"records": records}


@app.get("/corpus/vendors", dependencies=[Depends(require_api_key)])
def corpus_vendor_list():
    """Every vendor in the learning corpus with its size, dominant GL and who taught it."""
    vendors = retrieval.corpus_vendors()
    return {"corpus_size": sum(v["points"] for v in vendors), "vendors": vendors}


@app.get("/corpus/vendors/{vendor_id}", dependencies=[Depends(require_api_key)])
def corpus_vendor_points(vendor_id: str):
    points = retrieval.corpus_points(vendor_id)
    return {"vendor_id": vendor_id, "points": points}


@app.delete("/corpus/points/{point_id}", dependencies=[Depends(require_api_key)])
def corpus_delete_point(point_id: str):
    """Remove one bad evidence line (data-steward correction). Retrieval stops using it immediately."""
    if not retrieval.delete_corpus_point(point_id):
        raise HTTPException(status_code=404, detail="Corpus point not found")
    return {"status": "deleted", "corpus_size": _collection_size()}


MAX_UPLOAD_BYTES = 5 * 1024 * 1024


async def _read_csv_upload(file: UploadFile) -> str:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 5 MB - split it and upload in parts")
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded") from None


@app.get("/chart", dependencies=[Depends(require_api_key)])
def get_chart():
    """The active chart of accounts (the built-in one, or the client's uploaded chart)."""
    rows = chart_of_accounts.chart_rows()
    return {"custom": chart_of_accounts.is_custom(), "accounts": len(rows), "chart": rows}


@app.post("/chart", dependencies=[Depends(require_api_key)])
async def upload_chart(file: UploadFile = File(...)):
    """
    Replace the chart of accounts with the client's (CSV: code, name[, default_cost_centre, vat_treatment]).
    Confirmation, the agents and history import all validate against it. Single-tenant: it applies to everyone using
    this service. Corpus lines already coded with accounts outside the new chart are NOT removed.
    """
    text = await _read_csv_upload(file)
    try:
        accounts = chart_of_accounts.parse_chart_csv(text)
    except chart_of_accounts.ChartError as e:
        raise HTTPException(status_code=422, detail={"message": "Chart rejected", "problems": e.problems}) from None
    chart_of_accounts.set_chart(accounts)
    persisted = chart_of_accounts.save_chart(settings.chart_path)
    return {"status": "chart_replaced", "accounts": len(accounts), "persisted": persisted}


@app.delete("/chart", dependencies=[Depends(require_api_key)])
def reset_chart():
    """Go back to the built-in chart of accounts."""
    chart_of_accounts.reset_chart(settings.chart_path)
    return {"status": "chart_reset", "accounts": len(chart_of_accounts.CHART)}


@app.post("/readiness", dependencies=[Depends(require_api_key)])
async def readiness_report(
    file: UploadFile = File(...),
    gate_profile: Literal["standard", "tuned", "tuned_context"] = "standard",
):
    """
    Score a coded-history CSV (same format as /corpus/import) without writing anything.

    Off-chart GLs are reported, not fatal. The optional replay estimate runs the live
    decision path on a throwaway in-process store (offline hashing embedder; no ERP, no
    OpenAI). At most two 2,000-line replays (standard plus the selected or tuned profile)
    typically finish in well under two minutes and run in a threadpool so the event loop
    is not blocked. Live gate settings are not changed.
    """
    text = await _read_csv_upload(file)
    try:
        return await run_in_threadpool(readiness.assess, text, None, gate_profile)
    except history_import.HistoryImportError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


@app.post("/corpus/import", dependencies=[Depends(require_api_key)])
async def import_corpus_history(file: UploadFile = File(...), dry_run: str = "false"):
    """
    Bulk-load already-coded invoices (CSV) as human-confirmed corpus evidence. Validated against the active chart,
    idempotent (re-uploading is safe), at most 5,000 rows per file. Use dry_run=true to validate without writing.
    Required columns: vendor_name, description, gl (or account_code / gl_account). Optional: amount, currency,
    entity_id, cost_centre, date.
    """
    text = await _read_csv_upload(file)
    try:
        return await run_in_threadpool(history_import.import_history, text, dry_run=dry_run.lower() == "true")
    except history_import.HistoryImportError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


@app.get("/evaluation")
def evaluation():
    """The latest offline evaluation (`python scripts/evaluate_poc.py --json docs/evaluation.json`)."""
    path = Path(__file__).parent / "docs" / "evaluation.json"
    if not path.exists():
        return {"available": False}
    return {"available": True, "generated_at": path.stat().st_mtime, **json.loads(path.read_text(encoding="utf-8"))}


# --------------------------------------------------------------------------- demo

@app.get("/demo/status")
def demo_status():
    """Pre-flight for the demo: is everything it depends on ready, and what is it running on?"""
    return demo_support.status(_store)


@app.get("/demo/scenarios")
def demo_scenarios():
    """The scripted scenarios. Each is a real invoice; `expect` is what the system should do."""
    return demo_support.SCENARIOS


@app.post("/demo/reset", dependencies=[Depends(require_api_key)])
def demo_reset():
    """Restore the known starting state (seeded corpus, empty ledger). Demo mode only."""
    if not settings.demo_mode:
        raise HTTPException(status_code=403, detail="Demo reset is disabled (set DEMO_MODE=true to enable it)")
    with _jobs_lock:
        # Stop any batch still running, or it would keep posting into the freshly reset ledger.
        _cancelled.update(j for j, st in _jobs.items() if st["status"] == "running")
        _jobs.clear()
    return {"status": "reset", **demo_support.reset(_store)}



@app.get("/analytics", dependencies=[Depends(require_api_key)])
async def analytics(days: int = 7):
    """Governance analytics computed from recorded prediction events (Supabase)."""
    from ap_automation.core.supabase import query_analytics

    result = query_analytics(days=days)
    if result:
        return {**result, "data_available": True}
    return _empty_analytics(days)


def _empty_analytics(days: int) -> dict:
    """Same shape as real analytics, all zeros. Never invents figures."""
    return {
        "data_available": False,
        "period_days": days,
        "total_predictions": 0,
        "automation_rate": 0.0,
        "auto_count": 0,
        "human_review_count": 0,
        "governance_overrides": 0,
        "governance_override_rate": 0.0,
        "agent_flags": {},
        "vat_treatments": {},
        "cost_centre_mismatches": 0,
        "cost_centre_correction_rate": 0.0,
        "cold_starts": 0,
        "cold_start_rate": 0.0,
        "confidence_bands": {},
        "avg_retrieval_latency_by_day": {},
    }


def _build_explanation(pred: Prediction) -> str:
    n = len(pred.evidence)
    top_gl = pred.predicted_gl
    conf = round(pred.weighted_confidence * 100, 1)
    agreement = round(sum(1 for e in pred.evidence if e.gl_account == top_gl) / n * 100) if n else 0
    top_sim = round(max(e.similarity_score for e in pred.evidence) * 100, 1) if pred.evidence else 0

    if pred.final_routing == RoutingDecision.AUTO_POST:
        return (
            f"Retrieved {n} historical lines for this vendor; {agreement}% were coded to GL {top_gl} "
            f"and the closest scored {top_sim}% similarity. Confidence {conf}% passes every gate "
            f"and no specialist agent raised a flag, so it is cleared to auto-post."
        )
    return f"Routed to human review. {pred.final_routing_reason}"
