"""
Streamlit UI  AP Automation POC demo.
Polished demo version with improved layout, plain-English explanations, and narrative captions.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from uuid import uuid4

import httpx
import pandas as pd
import streamlit as st

from ap_automation.core.chart_of_accounts import CHART
from ap_automation.core.config import settings
from ap_automation.core.database import get_postings, init_db
from ap_automation.core.mdm import load_vendor_map
from ap_automation.core.models import (
    ConfirmationSource,
    InvoiceCategory,
    InvoiceLine,
    RoutingDecision,
)
from ap_automation.core.pipeline import predict
from ap_automation.core.posting import post_outcome
from ap_automation.core.retrieval import ensure_collection, get_qdrant
from ap_automation.core.seeding import seed_corpus
from ap_automation.core.sheets import ensure_sheet_headers
from ap_automation.ui.graph_tab import render_graph_tab

st.set_page_config(
    page_title="AP Automation POC",
    page_icon="AP",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .poc-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem; color: white;
    }
    .poc-header h1 { color: white; margin: 0; font-size: 2rem; font-weight: 700; }
    .poc-header p { color: #a8b2d8; margin: 0.3rem 0 0; font-size: 0.95rem; }
    .badge-auto { background: #d1fae5; color: #065f46; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; display: inline-block; }
    .badge-review { background: #fef3c7; color: #92400e; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; display: inline-block; }
    .badge-coldstart { background: #fee2e2; color: #991b1b; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; display: inline-block; }
    .evidence-row { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; }
    .evidence-desc { font-weight: 500; color: #1e293b; }
    .evidence-meta { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }
    .conf-bar-bg { background: #e2e8f0; border-radius: 4px; height: 8px; margin-top: 0.3rem; }
    .conf-bar-fill { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #f59e0b, #10b981); }
    .section-header { font-size: 0.75rem; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.8rem; padding-bottom: 0.4rem; border-bottom: 1px solid #f1f5f9; }
    .why-box { background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 1rem 1.2rem; margin: 0.8rem 0; }
    .why-box p { margin: 0; color: #0c4a6e; font-size: 0.9rem; line-height: 1.5; }
    .narrative { background: #f8fafc; border-left: 3px solid #6366f1; padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem; color: #374151; font-size: 0.9rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: #f8fafc; padding: 4px; border-radius: 10px; border: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { border-radius: 7px; padding: 0.5rem 1.2rem; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: white !important; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    .stButton > button { border-radius: 8px; font-weight: 500; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

if "metrics" not in st.session_state:
    st.session_state.metrics = {"total": 0, "auto_posted": 0, "batch_results": []}
if "initialised" not in st.session_state:
    st.session_state.initialised = False
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "graph_state" not in st.session_state:
    st.session_state.graph_state = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def _get_langfuse_keys():
    try:
        all_keys = dict(st.secrets)
        pub = str(all_keys.get("LANGFUSE_PUBLIC_KEY", "") or "").replace("\n", "").replace("\r", "").strip()
        sec = str(all_keys.get("LANGFUSE_SECRET_KEY", "") or "").replace("\n", "").replace("\r", "").strip()
        host = str(all_keys.get("LANGFUSE_HOST", "https://cloud.langfuse.com") or "https://cloud.langfuse.com").replace("\n", "").replace("\r", "").strip()
        return pub, sec, host
    except Exception:
        return "", "", "https://cloud.langfuse.com"


def _expose_secrets_to_env() -> None:
    """Copy Streamlit secrets into os.environ so non-UI modules can read them."""
    import os
    keys = ["ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
    for k in keys:
        try:
            val = str(st.secrets.get(k, "") or "").strip()
            if val:
                os.environ[k] = val
        except Exception:
            pass

def initialise() -> None:
    if not st.session_state.initialised:
        init_db()
        try:
            ensure_collection()
        except Exception:
            pass
        load_vendor_map("data/vendor_map.csv")
        try:
            ensure_sheet_headers()
        except Exception:
            pass
        st.session_state.initialised = True


def describe_post(res) -> str:
    """One-line, honest summary of what a posting actually did."""
    parts = [
        "Posted to ERP ledger" if res.erp_posted else "ERP post FAILED",
        "audit record written" if res.audited else "audit write FAILED",
        "added to learning corpus" if res.learned else "not added to corpus",
    ]
    return " | ".join(parts) + (f"  ({res.error})" if res.error else "")


def generate_explanation(prediction) -> str:
    evidence = prediction.evidence
    predicted_gl = prediction.predicted_gl
    confidence = prediction.weighted_confidence

    if not evidence:
        return (
            "No usable history for this vendor, so there is nothing to retrieve and the system "
            "will not guess. The invoice is routed to human review. Once a reviewer codes it, "
            "that decision is written back and future invoices from this vendor benefit."
        )

    gl_counts = Counter(e.gl_account for e in evidence)
    top_gl, top_count = gl_counts.most_common(1)[0]
    agreement_pct = int(top_count / len(evidence) * 100)
    top_sim = max(e.similarity_score for e in evidence)
    best_match = next(e for e in evidence if e.similarity_score == top_sim)

    if prediction.final_routing == RoutingDecision.AUTO_POST:
        return (
            f"Retrieved {len(evidence)} historical lines for this vendor; {agreement_pct}% were coded "
            f"to GL {predicted_gl} and the closest (\"{best_match.description}\") scored {top_sim:.0%} "
            f"similarity. Confidence {confidence:.0%} passes every gate (evidence, vendor identity, "
            f"agreement, threshold) and no specialist agent raised a flag, so it is cleared to auto-post."
        )

    text = f"Routed to human review. {prediction.final_routing_reason}."
    if prediction.agent_flags and prediction.routing_decision == RoutingDecision.AUTO_POST:
        text += (
            f" The confidence gate alone would have auto-posted this ({confidence:.0%}), "
            f"but a specialist agent objected - the governed decision wins."
        )
    return text


def process_invoice(inv: InvoiceLine, auto_confirm: bool = False,
                    simulate_reviewer: bool = False, batch_mode: bool = False) -> dict:
    prediction = predict(inv, batch_mode=batch_mode)
    result = {
        "invoice_id": str(inv.invoice_id),
        "vendor": inv.raw_vendor_name,
        "description": inv.description,
        "amount": inv.amount,
        "currency": inv.currency,
        "predicted_gl": prediction.predicted_gl,
        "confidence": prediction.weighted_confidence,
        # `routing` is the GOVERNED decision (rule gate + agent flags + LLM escalation)
        "routing": prediction.final_routing.value,
        "rule_routing": prediction.routing_decision.value,
        "routing_reason": prediction.final_routing_reason,
        "agent_flags": prediction.agent_flags,
        "ground_truth_gl": inv.ground_truth_gl,
        "category": inv.category.value,
        "provenance": prediction.provenance,
        "prediction": prediction,
        "correct": prediction.predicted_gl == inv.ground_truth_gl if inv.ground_truth_gl else None,
        "posted": False,
        "learned": False,
    }
    if auto_confirm and prediction.final_routing == RoutingDecision.AUTO_POST:
        res = post_outcome(
            prediction, prediction.predicted_gl, prediction.predicted_cost_centre,
            settings.auto_post_user_id, ConfirmationSource.AUTO_POST,
        )
        result["posted"] = res.erp_posted
    elif simulate_reviewer and prediction.final_routing != RoutingDecision.AUTO_POST and inv.ground_truth_gl:
        # Stand-in for a human reviewer: applies the ground-truth GL. This is the only
        # source of learning - auto-posts are never written back.
        source = (ConfirmationSource.HUMAN_CONFIRM if prediction.predicted_gl == inv.ground_truth_gl
                  else ConfirmationSource.HUMAN_CORRECT)
        res = post_outcome(prediction, inv.ground_truth_gl, prediction.predicted_cost_centre,
                           "simulated_reviewer", source)
        result["posted"], result["learned"] = res.erp_posted, res.learned
    return result


def vendor_history(question_lower: str):
    """Find a known vendor named in the question and return its GL history from the corpus."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from ap_automation.core.mdm import VENDOR_MAP

    for key, vendor_id in VENDOR_MAP.items():
        if len(key) >= 4 and key in question_lower:
            points, _ = get_qdrant().scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=Filter(must=[FieldCondition(key="vendor_id", match=MatchValue(value=vendor_id))]),
                limit=500, with_payload=True,
            )
            counts = Counter(p.payload.get("gl_account") for p in points
                             if p.payload.get("source") != "auto_post")
            return vendor_id, counts
    return None, Counter()


def answer_question(question: str) -> str:
    """
    Rule-based assistant. Every number it states is read from the live corpus, the ERP ledger,
    or the most recent batch run in this session - nothing is hard-coded.
    """
    q = question.lower().strip()
    last = st.session_state.metrics.get("last_batch")

    vendor_id, history = vendor_history(q)
    if vendor_id and ("gl" in q or "code" in q or "account" in q or "use" in q):
        if not history:
            return f"**{vendor_id}** has no human-confirmed history in the corpus yet, so its invoices go to human review."
        total = sum(history.values())
        dist = ", ".join(f"GL {gl} ({n / total:.0%})" for gl, n in history.most_common())
        return f"**{vendor_id}** - {total} human-confirmed lines in the corpus: {dist}."

    if any(w in q for w in ["corpus", "points", "size"]):
        try:
            count = get_qdrant().get_collection(settings.qdrant_collection).points_count
            return f"The corpus holds **{count:,} human-confirmed lines**. Only reviewer decisions are added; auto-posts are not."
        except Exception:
            return "I couldn't reach the corpus right now."

    if any(w in q for w in ["false positive", "wrong auto", "incorrect post"]):
        if not last or last["auto_posted"] == 0:
            return "No batch with ground truth has been run in this session, so there is no measured false-positive rate. Run one in the Batch Processing tab, or run `python scripts/evaluate_poc.py`."
        return (f"In the last batch, **{last['false_positives']} of {last['auto_posted']}** auto-posts "
                f"({last['false_positives'] / last['auto_posted']:.1%}) did not match the ground-truth GL.")

    if any(w in q for w in ["accurate", "accuracy", "correct", "wrong", "error"]):
        if not last or last["auto_posted"] == 0:
            return "No batch with ground truth has been run in this session, so accuracy has not been measured."
        return (f"In the last batch, auto-posted invoices were **{1 - last['false_positives'] / last['auto_posted']:.1%}** "
                f"correct; GL accuracy across all {last['n']:,} predictions was **{last['accuracy']:.1%}**.")

    if any(w in q for w in ["automation", "auto-post", "rate", "percentage"]):
        if not last:
            return "No batch has been run yet. Upload a CSV in the Batch Processing tab."
        return (f"The last batch auto-posted **{last['auto_posted']:,} of {last['n']:,}** invoices "
                f"(**{last['auto_posted'] / last['n']:.1%}**).")

    if any(w in q for w in ["cold start", "new vendor", "unknown vendor", "cold-start"]):
        return ("A vendor with fewer than "
                f"{settings.min_evidence_lines} human-confirmed lines, or a name the vendor master cannot resolve confidently, "
                "is routed to human review - the system does not borrow another vendor's history. "
                "Once reviewers code a few invoices, that vendor starts to auto-post.")

    if any(w in q for w in ["how does", "how it works", "explain", "what is"]):
        return (
            "An invoice auto-posts only if **all** of these hold:\n\n"
            f"1. at least {settings.min_evidence_lines} history lines exist for the resolved vendor\n"
            f"2. the vendor name resolves with confidence >= {settings.min_mdm_confidence:.0%}\n"
            f"3. at least {settings.min_agreement_ratio:.0%} of the retrieved lines agree on the GL, "
            f"and confidence >= {settings.default_confidence_threshold:.0%}\n"
            "4. no specialist agent flags VAT, entity or cost-centre problems\n\n"
            "The LLM gate can only make that decision stricter. Only **human-confirmed** decisions are written back to the corpus."
        )

    if any(w in q for w in ["fast", "speed", "latency", "slow", "quick", "cost", "price", "expensive"]):
        if last and last.get("avg_ms"):
            return f"The last batch averaged **{last['avg_ms']:.0f} ms per invoice** end-to-end. I have no cost data - it depends on your embedding and LLM usage."
        return "Run a batch to measure latency. I have no cost data - it depends on your embedding and LLM usage."

    if any(w in q for w in ["posted", "erp", "ledger", "how many invoices"]):
        return f"**{len(get_postings(limit=10000)):,} invoices** are recorded in the local ERP ledger."

    return (
        "I can answer questions about this system using live data. Try asking:\n\n"
        "- *What GL code does Meridian Facilities use?*\n"
        "- *What is the last batch's automation rate?*\n"
        "- *What is the false positive rate?*\n"
        "- *How does the gate work?*\n"
        "- *How many invoices have been posted?*"
    )

_expose_secrets_to_env()
initialise()

st.markdown("""
<div class="poc-header">
    <h1> AP Automation</h1>
    <p>Self-learning non-PO invoice coding  RAG + confidence gate + write-back loop</p>
</div>
""", unsafe_allow_html=True)

try:
    qc = get_qdrant()
    corpus_size = qc.get_collection(settings.qdrant_collection).points_count
except Exception:
    corpus_size = 0

postings_count = len(get_postings(limit=10000))
batch_results = st.session_state.metrics.get("batch_results", [])
current_rate = f"{batch_results[-1]['auto_post_rate']:.0%}" if batch_results else ""

col1, col2, col3, col4 = st.columns(4)
col1.metric(" Corpus size", f"{corpus_size:,}")
col2.metric(" Invoices posted", f"{postings_count:,}")
col3.metric(" Automation rate", current_rate)
col4.metric(" Batches run", len(batch_results))

st.divider()

tabs = st.tabs([
    "Review Queue",
    "Batch Processing",
    "Performance",
    "ERP Ledger",
    "Audit Trail",
    "Ask AP",
    "Graph",
    "Setup",
])

#  TAB 1: Review Queue 
with tabs[0]:
    st.markdown('<div class="narrative">Submit an invoice to see the system predict a GL code from historical evidence. Confirm the prediction to post it to the ERP and write it back into the learning corpus.</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")
    with col_form:
        st.markdown('<div class="section-header">Submit Invoice</div>', unsafe_allow_html=True)
        vendor = st.text_input("Vendor name", value=st.session_state.get("vendor_input", "Meridian Facilities Ltd"), key="vendor_input")
        description = st.text_input("Description", value=st.session_state.get("desc_input", "Office cleaning services monthly"), key="desc_input")
        col_a, col_b = st.columns(2)
        with col_a:
            amount = st.number_input("Amount (-)", value=450.0, min_value=0.01, format="%.2f")
            entity = st.selectbox("Entity", ["UK001", "UK002", "IE001"])
        with col_b:
            currency = st.selectbox("Currency", ["GBP", "EUR", "USD", "CHF"])
            cost_centre = st.selectbox("Cost centre", ["CC100", "CC200", "CC300", "CC400", "CC500"], index=2)

        st.markdown("---")
        st.markdown("**Try a hard-edge invoice (unknown vendor):**")
        if st.button("Load hard-edge example", use_container_width=True):
            st.session_state["hard_edge_vendor"] = "Zephyr Global Holdings"
            st.session_state["hard_edge_desc"] = "Professional advisory services Q3"
            st.rerun()

        if "hard_edge_vendor" in st.session_state:
            vendor = st.session_state.pop("hard_edge_vendor")
            description = st.session_state.pop("hard_edge_desc")

        if st.button("Predict GL Code", type="primary", use_container_width=True):
            inv = InvoiceLine(
                invoice_id=uuid4(), raw_vendor_name=vendor, description=description,
                amount=amount, currency=currency, entity_id=entity,
                cost_centre=cost_centre, category=InvoiceCategory.ROUTINE,
            )
            with st.spinner("Running pipeline..."):
                # Secrets are copied into the environment at start-up; reading st.secrets
                # directly raises when no secrets.toml exists (e.g. local runs).
                _anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
                prediction = predict(inv, anthropic_api_key=_anthropic_key)
            st.session_state.last_prediction = prediction
            prov = prediction.provenance or {}
            a2a = prov.get("a2a_agents", {})
            st.session_state.graph_state = {
                "invoice_id": str(inv.invoice_id),
                "raw_vendor_name": inv.raw_vendor_name,
                "description": inv.description,
                "amount": inv.amount,
                "currency": inv.currency,
                "entity_id": inv.entity_id,
                "canonical_vendor_id": prediction.resolved_line.canonical_vendor_id,
                "mdm_confidence": prediction.resolved_line.mdm_confidence,
                "amount_band": prediction.resolved_line.amount_band.value,
                "evidence_count": len(prediction.evidence),
                "predicted_gl": prediction.predicted_gl,
                "predicted_cost_centre": prediction.predicted_cost_centre,
                "agreement_ratio": prediction.agreement_ratio,
                "weighted_confidence": prediction.weighted_confidence,
                "routing_decision": prediction.routing_decision.value,
                "routing_reason": prediction.routing_reason,
                "agents_called": a2a.get("agents_called", 0),
                "agent_results": a2a.get("results", []),
                "vat_treatment": next((r.get("vat_treatment") for r in a2a.get("results", []) if "vat_treatment" in r), None),
                "cost_centre_mismatch": any(r.get("cost_centre_mismatch") for r in a2a.get("results", [])),
                "entity_valid": next((r.get("entity_valid") for r in a2a.get("results", []) if "entity_valid" in r), None),
                "agent_flags": prediction.agent_flags,
                "final_routing": prov.get("final_routing", prediction.routing_decision.value),
                "gate_reason": prov.get("final_routing_reason", prediction.routing_reason),
                "llm_reasoning": prov.get("llm_reasoning"),
                "llm_reasoning_confidence": prov.get("llm_reasoning_confidence"),
                "llm_escalated": prov.get("llm_escalated", False),
                "llm_latency_ms": prov.get("llm_latency_ms"),
                "started_at": prov.get("timestamp", ""),
                "total_latency_ms": int((datetime.now(timezone.utc) - datetime.fromisoformat(prov["timestamp"])).total_seconds() * 1000) if prov.get("timestamp") else None,
            }
            st.session_state.last_invoice = inv

    with col_result:
        pred = st.session_state.get("last_prediction")
        if pred:
            st.markdown('<div class="section-header">Prediction</div>', unsafe_allow_html=True)

            is_cold_start = len(pred.evidence) == 0
            if is_cold_start:
                st.markdown('<span class="badge-coldstart"> Cold start  no vendor history</span>', unsafe_allow_html=True)
            elif pred.final_routing == RoutingDecision.AUTO_POST:
                st.markdown('<span class="badge-auto"> Auto-post approved</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-review"> Routed to human review</span>', unsafe_allow_html=True)

            st.caption(pred.final_routing_reason or pred.routing_reason)
            if pred.agent_flags:
                st.caption("Agent flags: " + ", ".join(pred.agent_flags))
            st.write("")

            m1, m2, m3 = st.columns(3)
            m1.metric("GL Account", pred.predicted_gl or "")
            m2.metric("Confidence", f"{pred.weighted_confidence:.1%}")
            m3.metric("Evidence lines", len(pred.evidence))

            pct = int(pred.weighted_confidence * 100)
            st.markdown(f"""
            <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{pct}%"></div></div>
            <div style="font-size:0.75rem;color:#6b7280;margin-top:0.2rem">
                Threshold: {settings.default_confidence_threshold:.0%} &nbsp;|&nbsp; Score: {pred.weighted_confidence:.1%}
            </div>""", unsafe_allow_html=True)

            st.write("")
            st.markdown('<div class="section-header">Why this prediction?</div>', unsafe_allow_html=True)
            explanation = generate_explanation(pred)
            st.markdown(f'<div class="why-box"><p>{explanation}</p></div>', unsafe_allow_html=True)

            if pred.evidence:
                st.write("")
                st.markdown('<div class="section-header">Retrieved Evidence</div>', unsafe_allow_html=True)
                for e in pred.evidence[:3]:
                    st.markdown(f"""
                    <div class="evidence-row">
                        <div class="evidence-desc">{e.description}</div>
                        <div class="evidence-meta">GL: <strong>{e.gl_account}</strong> &nbsp;&nbsp; Similarity: <strong>{e.similarity_score:.3f}</strong> &nbsp;&nbsp; RRF: {e.rrf_score:.5f} &nbsp;&nbsp; Source: {e.source.value}</div>
                    </div>""", unsafe_allow_html=True)

            st.write("")
            st.markdown('<div class="section-header">Action</div>', unsafe_allow_html=True)
            act1, act2 = st.columns(2)
            with act1:
                if st.button(" Confirm & Post to ERP", use_container_width=True, type="primary"):
                    res = post_outcome(pred, pred.predicted_gl, pred.predicted_cost_centre, "demo_user", ConfirmationSource.HUMAN_CONFIRM)
                    st.success(describe_post(res))
            with act2:
                corrected_gl = st.text_input("Override GL code", value=pred.predicted_gl or "", placeholder="e.g. 6300")
                if st.button(" Submit Correction", use_container_width=True):
                    if corrected_gl not in CHART:
                        st.error(f"Unknown GL account {corrected_gl}. Valid: {', '.join(CHART)}")
                    else:
                        res = post_outcome(pred, corrected_gl, pred.predicted_cost_centre, "demo_user",
                                           ConfirmationSource.HUMAN_CORRECT if corrected_gl != pred.predicted_gl else ConfirmationSource.HUMAN_CONFIRM)
                        st.success(describe_post(res))

            # A2A Agent Results
            a2a = pred.provenance.get("a2a_agents", {})
            agent_results = a2a.get("results", [])
            if agent_results:
                st.write("")
                st.markdown('<div class="section-header">A2A Agent Insights</div>', unsafe_allow_html=True)
                for agent in agent_results:
                    agent_name = agent.get("agent", "")
                    if not agent.get("success", False):
                        continue
                    if agent_name == "vat_agent":
                        vat = agent.get("vat_treatment", "UNKNOWN")
                        rate = agent.get("vat_rate", "")
                        flag = agent.get("vat_flag", False)
                        review = agent.get("review_required", False)
                        note = agent.get("vat_note", "")
                        colour = "#fee2e2" if flag else "#f0fdf4"
                        border = "#fca5a5" if flag else "#86efac"
                        label = "VAT [REVIEW]" if flag else "VAT [OK]"
                        extra = "<br><small style='color:#b91c1c;font-weight:600'>Manual VAT review recommended</small>" if review else ""
                        st.markdown(f'<div style="background:{colour};border:1px solid {border};border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.5rem"><strong>{label}</strong> &nbsp; Treatment: <strong>{vat}</strong> &nbsp; Rate: <strong>{rate}</strong><br><span style="font-size:0.82rem;color:#374151">{note}</span>{extra}</div>', unsafe_allow_html=True)
                    elif agent_name == "cost_centre_agent":
                        inferred = agent.get("inferred_cost_centre", "")
                        mismatch = agent.get("cost_centre_mismatch", False)
                        note = agent.get("cost_centre_note", "")
                        colour = "#fef9c3" if mismatch else "#f0fdf4"
                        border = "#fde047" if mismatch else "#86efac"
                        label = "Cost Centre [MISMATCH]" if mismatch else "Cost Centre [OK]"
                        st.markdown(f'<div style="background:{colour};border:1px solid {border};border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.5rem"><strong>{label}</strong> &nbsp; Inferred: <strong>{inferred}</strong><br><span style="font-size:0.82rem;color:#374151">{note}</span></div>', unsafe_allow_html=True)
                    elif agent_name == "entity_agent":
                        valid = agent.get("entity_valid", True)
                        expected = agent.get("expected_entity", "")
                        sensitive = agent.get("entity_sensitive", False)
                        note = agent.get("entity_note", "")
                        colour = "#fee2e2" if not valid else "#f0fdf4"
                        border = "#fca5a5" if not valid else "#86efac"
                        label = "Entity [MISMATCH]" if not valid else "Entity [OK]"
                        extra = "<br><small style='color:#b91c1c;font-weight:600'>Entity-sensitive GL - confirm before posting</small>" if sensitive else ""
                        st.markdown(f'<div style="background:{colour};border:1px solid {border};border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.5rem"><strong>{label}</strong> &nbsp; Expected: <strong>{expected}</strong> &nbsp; Valid: <strong>{"Yes" if valid else "No"}</strong><br><span style="font-size:0.82rem;color:#374151">{note}</span>{extra}</div>', unsafe_allow_html=True)

            with st.expander(" Full provenance JSON"):
                st.json(pred.provenance)
        else:
            st.info("Submit an invoice on the left to see the prediction. Try the hard-edge example to see cold-start handling.")

#  TAB 2: Batch Processing 
with tabs[1]:
    st.markdown('<div class="narrative">Upload a CSV of invoices and run them through the full pipeline. Only invoices the governed gate approves are auto-posted. Review-routed invoices stay pending, or - with the simulated reviewer on - are resolved from the ground truth in the CSV, and ONLY those decisions are written back to the corpus.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload invoices CSV", type=["csv"], label_visibility="collapsed")
    batch_size = st.select_slider("Reporting batch size", options=[50, 100, 150, 200], value=100)
    if uploaded:
        df = pd.read_csv(StringIO(uploaded.getvalue().decode()))
        st.info(f"**{len(df):,}** invoices loaded")
        sim_reviewer = st.checkbox(
            "Simulate a reviewer using the CSV's ground_truth_gl",
            value="ground_truth_gl" in df.columns,
            help="Review-routed invoices are resolved with the ground-truth GL and written back. "
                 "This is the ONLY source of learning - auto-posts are never written back.",
        )
        if st.button("Run Batch", type="primary"):
            progress = st.progress(0, text="Starting...")
            _batch_t0 = time.time()
            results = []
            batch_metrics = []
            for i, row in df.iterrows():
                inv = InvoiceLine(
                    invoice_id=uuid4(), raw_vendor_name=str(row["raw_vendor_name"]),
                    description=str(row["description"]), amount=float(row["amount"]),
                    currency=str(row.get("currency", "GBP")), entity_id=str(row.get("entity_id", "UK001")),
                    cost_centre=str(row.get("cost_centre", "CC100")),
                    category=InvoiceCategory(row.get("category", "routine")),
                    ground_truth_gl=(str(row["ground_truth_gl"]) if "ground_truth_gl" in row and pd.notna(row["ground_truth_gl"]) else None),
                )
                result = process_invoice(inv, auto_confirm=True, simulate_reviewer=sim_reviewer, batch_mode=True)
                results.append(result)
                n = i + 1
                progress.progress(n / len(df), text=f"Processing {n:,}/{len(df):,}  {result['vendor'][:35]}")
                if n % batch_size == 0 or n == len(df):
                    batch = results[-batch_size:]
                    auto = sum(1 for r in batch if r["routing"] == "auto_post")
                    correct = sum(1 for r in batch if r.get("correct") is True)
                    batch_metrics.append({"batch": n, "auto_post_rate": auto / len(batch), "accuracy": correct / len(batch) if batch else 0})
            st.session_state.metrics["batch_results"] = batch_metrics
            _auto = [r for r in results if r["routing"] == "auto_post"]
            st.session_state.metrics["last_batch"] = {
                "n": len(results), "auto_posted": len(_auto),
                "false_positives": sum(1 for r in _auto if r.get("correct") is False),
                "accuracy": sum(1 for r in results if r.get("correct") is True) / max(len(results), 1),
                "avg_ms": (time.time() - _batch_t0) * 1000 / max(len(results), 1),
            }
            try:
                pub, sec, host = _get_langfuse_keys()
                if pub and sec:
                    os.environ["LANGFUSE_PUBLIC_KEY"] = pub
                    os.environ["LANGFUSE_SECRET_KEY"] = sec
                    os.environ["LANGFUSE_HOST"] = host
                    from langfuse import get_client as _lf_gc
                    _lf_gc().flush()
            except Exception:
                pass
            progress.empty()
            auto_count = sum(1 for r in results if r["routing"] == "auto_post")
            correct_count = sum(1 for r in results if r.get("correct") is True)
            st.success(f" Batch complete  {len(results):,} invoices processed")
            c1, c2, c3 = st.columns(3)
            c1.metric("Auto-posted", f"{auto_count:,}", f"{auto_count/len(results):.0%}")
            c2.metric("Human review", f"{len(results)-auto_count:,}")
            c3.metric("Correct predictions", f"{correct_count:,}", f"{correct_count/len(results):.0%}")
            _lb = st.session_state.metrics["last_batch"]
            if _lb["auto_posted"]:
                st.caption(f"Auto-post accuracy: {1 - _lb['false_positives'] / _lb['auto_posted']:.1%} "
                           f"({_lb['false_positives']} wrong of {_lb['auto_posted']} auto-posts)")
            st.dataframe(pd.DataFrame([{
                "Vendor": r["vendor"][:40], "Description": r["description"][:50],
                "Amount": f"-{r['amount']:,.2f}", "GL": r["predicted_gl"] or "",
                "Confidence": f"{r['confidence']:.0%}",
                "Routing": "Auto" if r["routing"] == "auto_post" else "Review",
                "Correct": "Yes" if r.get("correct") else ("No" if r.get("correct") is False else ""),
            } for r in results]), use_container_width=True, height=400)

#  TAB 3: Performance 
with tabs[2]:
    st.markdown('<div class="narrative">Automation rate by reporting batch. The corpus grows only from reviewer-confirmed decisions, so a rising line means the system is resolving vendors it previously had no history for - not agreeing with itself. For a with-vs-without-learning comparison, run <code>python scripts/evaluate_poc.py</code>.</div>', unsafe_allow_html=True)

    batch_results = st.session_state.metrics.get("batch_results", [])
    if batch_results:
        first = batch_results[0]["auto_post_rate"]
        last = batch_results[-1]["auto_post_rate"]
        delta = last - first
        c1, c2, c3 = st.columns(3)
        c1.metric("Start automation rate", f"{first:.1%}")
        c2.metric("End automation rate", f"{last:.1%}")
        c3.metric("Write-back improvement", f"{delta:+.1%}", delta=f"{delta:+.1%}")
        st.write("")
        chart_df = pd.DataFrame(batch_results).set_index("batch")
        st.line_chart(chart_df[["auto_post_rate", "accuracy"]])
        st.caption("auto_post_rate = % straight-through posted without human review | accuracy = % correct vs ground truth GL code")
    else:
        st.info("Run a batch in the Batch Processing tab to see performance metrics.")

#  TAB 4: ERP Ledger 
with tabs[3]:
    st.markdown('<div class="narrative">Every auto-posted and human-confirmed invoice made through this app is recorded in the local ERP ledger database (and mirrored to Google Sheets if configured).</div>', unsafe_allow_html=True)

    postings = get_postings(limit=500)
    if postings:
        df_postings = pd.DataFrame(postings)
        st.dataframe(df_postings[["invoice_id", "vendor_id", "entity_id", "gl_account", "cost_centre", "amount", "currency", "posted_by", "posted_at"]], use_container_width=True, height=500)
        st.caption(f"{len(postings):,} postings in ledger")
    else:
        st.info("No invoices posted yet.")

#  TAB 5: Audit Trail 
with tabs[4]:
    st.markdown('<div class="narrative">Every posting writes an audit record: the prediction, the evidence that drove it, agent flags, the rule and governed routing decisions, and who confirmed it.</div>', unsafe_allow_html=True)

    from ap_automation.core.database import get_audit_records
    records = get_audit_records(limit=100)
    if records:
        for rec in records[:20]:
            routing_label = "Auto-posted" if rec["routing_decision"] == "auto_post" else "Human review"
            source_label = {"auto_post": "Auto-posted", "human_confirm": "Human confirmed", "human_correct": "Human corrected"}.get(rec["source"], rec["source"])
            with st.expander(f"[{routing_label}] Invoice {rec['invoice_id'][:8]}...  GL {rec['confirmed_gl']}  {source_label}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted GL", rec["predicted_gl"])
                c2.metric("Confirmed GL", rec["confirmed_gl"])
                c3.metric("Confidence", f"{rec['confidence']:.1%}")
                st.json(json.loads(rec["provenance_json"]))
    else:
        st.info("No audit records yet.")

#  TAB 6: Ask AP 
with tabs[5]:
    st.markdown('<div class="narrative">Ask about the corpus, the GL history of a vendor, the results of the last batch, or how the gate works. Answers use live data; nothing is hard-coded.</div>', unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested questions
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        suggestions = [
            "What GL code does Meridian Facilities use?",
            "What is the current automation rate?",
            "How does the system work?",
            "What is the false positive rate?",
            "How fast is the retrieval?",
            "How many invoices have been posted?",
        ]
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, use_container_width=True, key=f"sugg_{i}"):
                    st.session_state.chat_history.append({"role": "user", "content": suggestion})
                    response = answer_question(suggestion)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask anything about the AP Automation system..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = answer_question(prompt)
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    if st.session_state.chat_history:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

#  TAB 7: Setup 
with tabs[6]:
    render_graph_tab(st.session_state.graph_state)

with tabs[7]:
    col_setup, col_status = st.columns([1, 1], gap="large")
    with col_setup:
        st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
        if st.button("Generate 1,000-invoice dataset", use_container_width=True):
            with st.spinner("Generating..."):
                import random
                random.seed(42)
                from ap_automation.core.dataset import generate_dataset
                invoices = generate_dataset()
            st.success(f" Generated {len(invoices):,} invoices  data/invoices.csv")
        st.write("")
        st.markdown('<div class="section-header">Seed Corpus</div>', unsafe_allow_html=True)
        per_vendor = st.slider("History lines per known vendor", 5, 40, 15)
        if st.button("Seed Qdrant corpus", use_container_width=True):
            from ap_automation.core.dataset import generate_history
            load_vendor_map("data/vendor_map.csv")
            counts = seed_corpus(generate_history(per_vendor))
            st.success(f"Seeded {sum(counts.values()):,} human-coded lines across {len(counts)} vendors")

    with col_status:
        st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
        try:
            qc2 = get_qdrant()
            info = qc2.get_collection(settings.qdrant_collection)
            st.success(f"Qdrant  {info.points_count:,} points in `{settings.qdrant_collection}`")
        except Exception as e:
            st.error(f"Qdrant  {e}")
        if settings.erp_url:
            try:
                r = httpx.get(f"{settings.erp_url.rstrip('/')}/health", timeout=2)
                st.success("Mock ERP service reachable") if r.status_code == 200 else st.warning("Mock ERP service: unexpected response")
            except Exception:
                st.error(f"ERP_URL is set ({settings.erp_url}) but the service is not reachable - postings will fail")
        else:
            st.info("ERP: posting to the local ledger database (set ERP_URL to use the mock ERP service)")
        try:
            from ap_automation.core.sheets import _get_service
            _get_service()
            st.success("Google Sheets  connected")
        except Exception as e:
            st.warning(f"Google Sheets  {e}")
        dataset_path = Path("data/invoices.csv")
        if dataset_path.exists():
            st.success(f"Dataset  invoices.csv ({dataset_path.stat().st_size/1024:.0f} KB)")
        else:
            st.warning("Dataset  not generated yet")

        st.write("")
        st.markdown('<div class="section-header">Langfuse Observability</div>', unsafe_allow_html=True)
        try:
            pub, sec, host = _get_langfuse_keys()
            if pub and sec:
                os.environ["LANGFUSE_PUBLIC_KEY"] = pub
                os.environ["LANGFUSE_SECRET_KEY"] = sec
                os.environ["LANGFUSE_HOST"] = host
                from langfuse import get_client as _lf_get
                _lf = _lf_get()
                _auth = _lf.auth_check()
                if _auth:
                    st.success(f"Langfuse connected ({host})")
                    if st.button("Send test trace to Langfuse", use_container_width=True):
                        with _lf.start_as_current_observation(as_type="span", name="test_trace", input={"source": "setup_tab"}):
                            _lf.update_current_span(output={"result": "ok"})
                        _lf.flush()
                        st.success("Test trace sent - check cloud.langfuse.com")
                else:
                    st.error("Langfuse auth check failed")
            else:
                st.warning(f"Langfuse keys missing (pub_len={len(pub)}, sec_len={len(sec)})")
        except Exception as _e:
            st.error(f"Langfuse error: {_e}")

        st.write("")
        st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
        st.code(f"""Confidence threshold: {settings.default_confidence_threshold}
Min evidence lines:   {settings.min_evidence_lines}
Min vendor match:     {settings.min_mdm_confidence}
Min GL agreement:     {settings.min_agreement_ratio}
Top-k retrieval:      {settings.top_k}
Embedding model:      {settings.embedding_model}
Collection:           {settings.qdrant_collection}""")

