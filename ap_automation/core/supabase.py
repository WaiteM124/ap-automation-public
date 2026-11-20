"""
Supabase client for analytics event storage and async batch job tracking.
"""
from __future__ import annotations

import os

import httpx

from ap_automation.core.config import settings


def _supabase_url() -> str:
    return (os.environ.get("SUPABASE_URL", "") or settings.supabase_url).rstrip("/")


def _key() -> str:
    """Service key, or "" when unset, when no SUPABASE_URL is configured, or telemetry is disabled."""
    if not settings.telemetry_enabled:
        return ""
    if not _supabase_url():
        return ""
    return os.environ.get("SUPABASE_SECRET_KEY", "") or settings.supabase_secret_key


def _headers(extra: dict | None = None) -> dict:
    key = _key()
    h = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    if extra:
        h.update(extra)
    return h


# -- Prediction events ---------------------------------------------------------

def write_prediction_event(data: dict) -> None:
    """Record a prediction event: always locally (so analytics work offline), and to Supabase when configured.

    Fire-and-forget - never raises.
    """
    try:
        from ap_automation.core import database
        database.record_event(data)
    except Exception as e:
        print(f"[Events] local write error: {e}")
    if not _key():
        return
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                f"{_supabase_url()}/rest/v1/prediction_events",
                headers=_headers(),
                json=data,
            )
            if resp.status_code not in (200, 201):
                print(f"[Supabase] Write failed: {resp.status_code} {resp.text[:100]}")
    except Exception as e:
        print(f"[Supabase] Write error: {e}")


# -- Batch jobs ----------------------------------------------------------------

def create_batch_job(job_id: str, total: int, governance_mode: bool) -> None:
    """Create a new batch job record."""
    if not _key():
        return
    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(
                f"{_supabase_url()}/rest/v1/batch_jobs",
                headers=_headers(),
                json={
                    "id": job_id,
                    "status": "running",
                    "total": total,
                    "completed": 0,
                    "auto_confirmed": 0,
                    "human_review": 0,
                    "governance_mode": governance_mode,
                    "results": [],
                },
            )
    except Exception as e:
        print(f"[Supabase] create_batch_job error: {e}")


def update_batch_job(job_id: str, completed: int, auto_confirmed: int,
                     human_review: int, results: list, status: str = "running") -> None:
    """Update batch job progress."""
    if not _key():
        return
    try:
        with httpx.Client(timeout=5.0) as client:
            client.patch(
                f"{_supabase_url()}/rest/v1/batch_jobs",
                headers=_headers({"Prefer": "return=minimal"}),
                params={"id": f"eq.{job_id}"},
                json={
                    "completed": completed,
                    "auto_confirmed": auto_confirmed,
                    "human_review": human_review,
                    "results": results,
                    "status": status,
                },
            )
    except Exception as e:
        print(f"[Supabase] update_batch_job error: {e}")


def fail_batch_job(job_id: str, error: str) -> None:
    """Mark a batch job as failed."""
    if not _key():
        return
    try:
        with httpx.Client(timeout=5.0) as client:
            client.patch(
                f"{_supabase_url()}/rest/v1/batch_jobs",
                headers=_headers({"Prefer": "return=minimal"}),
                params={"id": f"eq.{job_id}"},
                json={"status": "failed", "error": error},
            )
    except Exception as e:
        print(f"[Supabase] fail_batch_job error: {e}")


def get_batch_job(job_id: str) -> dict | None:
    """Fetch a batch job by ID."""
    if not _key():
        return None
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(
                f"{_supabase_url()}/rest/v1/batch_jobs",
                headers=_headers(),
                params={"id": f"eq.{job_id}", "select": "*"},
            )
            rows = resp.json()
            return rows[0] if rows else None
    except Exception as e:
        print(f"[Supabase] get_batch_job error: {e}")
        return None


# -- Analytics -----------------------------------------------------------------

def query_analytics(days: int = 7) -> dict | None:
    """Aggregated analytics: from Supabase when configured, otherwise from the local event log."""
    if not _key():
        try:
            from ap_automation.core import database
            return _aggregate(database.get_events(days), days)
        except Exception as e:
            print(f"[Events] local query error: {e}")
            return None
    try:
        from datetime import datetime, timedelta, timezone
        from_ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{_supabase_url()}/rest/v1/prediction_events",
                headers=_headers(),
                params={
                    "created_at": f"gte.{from_ts}",
                    "select": "final_routing,routing_decision,weighted_confidence,vat_treatment,cost_centre_mismatch,entity_valid,retrieval_latency_ms,created_at,amount_band,vendor_canonical",
                    "limit": 2000,
                },
            )
            resp.raise_for_status()
            return _aggregate(resp.json(), days)
    except Exception as e:
        print(f"[Supabase] Query error: {e}")
        return None


def _aggregate(rows: list[dict], days: int) -> dict | None:
    """Aggregate prediction events into the analytics payload. None when there are no events."""
    from collections import defaultdict

    if not rows:
        return None

    total = len(rows)
    auto_count = sum(1 for r in rows if r.get("final_routing") == "auto_post")
    human_count = total - auto_count

    high_conf_overridden = sum(
        1 for r in rows
        if r.get("routing_decision") == "auto_post"
        and r.get("final_routing") == "human_review"
        and float(r.get("weighted_confidence") or 0) >= 0.75
    )

    vat_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        vat_counts[r.get("vat_treatment") or "UNASSESSED"] += 1

    cc_mismatches = sum(1 for r in rows if r.get("cost_centre_mismatch"))
    cold_starts = sum(1 for r in rows if str(r.get("vendor_canonical") or "").startswith("NEW_") or not r.get("vendor_canonical"))

    bands: dict[str, dict] = defaultdict(lambda: {"total": 0, "auto": 0})
    for r in rows:
        c = float(r.get("weighted_confidence") or 0)
        b_low = int(c * 10) * 10
        band = f"{b_low}-{b_low + 10}%"
        bands[band]["total"] += 1
        if r.get("final_routing") == "auto_post":
            bands[band]["auto"] += 1

    latencies: dict[str, list] = defaultdict(list)
    for r in rows:
        ts = (r.get("created_at") or "")[:10]
        ms = r.get("retrieval_latency_ms")
        if ts and ms:
            latencies[ts].append(float(ms))

    return {
        "period_days": days,
        "total_predictions": total,
        "automation_rate": round(auto_count / total * 100, 1),
        "auto_count": auto_count,
        "human_review_count": human_count,
        "governance_overrides": high_conf_overridden,
        "governance_override_rate": round(high_conf_overridden / total * 100, 1),
        "agent_flags": {
            "cost_centre_mismatch": cc_mismatches,
            "vat_blocked": vat_counts.get("BLOCKED", 0),
            "entity_validation_failed": sum(1 for r in rows if r.get("entity_valid") is False),
            "vat_reverse_charge": vat_counts.get("REVERSE_CHARGE", 0),
            "vat_not_assessed": vat_counts.get("NOT_ASSESSED", 0),
        },
        "vat_treatments": dict(sorted(vat_counts.items(), key=lambda x: -x[1])),
        "cost_centre_mismatches": cc_mismatches,
        "cost_centre_correction_rate": round(cc_mismatches / total * 100, 1),
        "cold_starts": cold_starts,
        "cold_start_rate": round(cold_starts / total * 100, 1),
        "confidence_bands": dict(sorted(bands.items(), key=lambda x: -int(x[0].split("-")[0]))),
        "avg_retrieval_latency_by_day": {
            day: round(sum(v) / len(v), 0)
            for day, v in sorted(latencies.items())
        },
    }
