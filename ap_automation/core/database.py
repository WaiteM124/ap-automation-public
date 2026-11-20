"""
SQLite persistence - ERP ledger and audit trail.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ap_automation.core.config import settings

# `source` recorded on the audit row of a post that SAP / Xero rejected / that could not be completed.
SAP_FAILED_SOURCE = "sap_post_failed"
XERO_FAILED_SOURCE = "xero_post_failed"
ERP_FAILED_SOURCES = frozenset({SAP_FAILED_SOURCE, XERO_FAILED_SOURCE})


@contextmanager
def _get_conn(path: str) -> Iterator[sqlite3.Connection]:
    """Open a connection, commit on success, roll back on error, ALWAYS close.

    (`with sqlite3.connect(...)` only manages the transaction - it never closes the
    connection, which leaked handles and locked the database file on Windows.)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initialise both databases with schema."""
    # ERP ledger
    with _get_conn(settings.erp_db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS erp_postings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_id TEXT NOT NULL,
                vendor_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                gl_account TEXT NOT NULL,
                cost_centre TEXT NOT NULL,
                tax_code TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT NOT NULL,
                description TEXT NOT NULL,
                posted_by TEXT NOT NULL,
                posted_at TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        # SAP posting state, keyed by the idempotency reference. One row per invoice: it is what stops a retry
        # from posting twice, and it keeps the (sanitised) SAP error for any post that did not succeed.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sap_postings (
                reference TEXT PRIMARY KEY,
                invoice_id TEXT NOT NULL,
                prediction_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                sap_document TEXT NOT NULL DEFAULT '',
                fiscal_year TEXT NOT NULL DEFAULT '',
                company_code TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                attempts INTEGER NOT NULL DEFAULT 0,
                summary_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        # Xero posting state, same idea as sap_postings: one row per reference key. No token or secret is ever stored.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS xero_postings (
                reference TEXT PRIMARY KEY,
                invoice_id TEXT NOT NULL,
                prediction_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                xero_invoice_id TEXT NOT NULL DEFAULT '',
                xero_url TEXT NOT NULL DEFAULT '',
                tenant_id TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                attempts INTEGER NOT NULL DEFAULT 0,
                summary_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.commit()

    # Audit trail
    with _get_conn(settings.audit_db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_trail (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT NOT NULL,
                invoice_id TEXT NOT NULL,
                vendor_id TEXT NOT NULL,
                predicted_gl TEXT NOT NULL,
                confirmed_gl TEXT NOT NULL,
                confidence REAL NOT NULL,
                routing_decision TEXT NOT NULL,
                routing_reason TEXT NOT NULL,
                source TEXT NOT NULL,
                confirming_user TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        # Every prediction, local and always-on, so analytics work without Supabase.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT (datetime('now')),
                body TEXT NOT NULL
            )
        """)
        conn.commit()


_ready: set[tuple[str, str]] = set()


def _ensure() -> None:
    """init_db once per database location (cheap to call on hot paths)."""
    key = (settings.erp_db_path, settings.audit_db_path)
    if key not in _ready:
        init_db()
        _ready.add(key)


def clear_all() -> None:
    """Empty the ERP ledger, audit trail and event log (demo reset only)."""
    init_db()
    with _get_conn(settings.erp_db_path) as conn:
        conn.execute("DELETE FROM erp_postings")
        conn.execute("DELETE FROM sap_postings")
        conn.execute("DELETE FROM xero_postings")
    with _get_conn(settings.audit_db_path) as conn:
        conn.execute("DELETE FROM audit_trail")
        conn.execute("DELETE FROM prediction_events")


def record_event(data: dict) -> None:
    """Append a prediction event to the local log."""
    _ensure()
    with _get_conn(settings.audit_db_path) as conn:
        conn.execute("INSERT INTO prediction_events (body) VALUES (?)", (json.dumps(data, default=str),))


def get_events(days: int | None = None, limit: int = 20000) -> list[dict]:
    """Prediction events newest first, each with an ISO `created_at`. `days=None` means all time."""
    _ensure()
    sql = "SELECT created_at, body FROM prediction_events"
    args: tuple = ()
    if days is not None:
        sql += " WHERE created_at >= datetime('now', ?)"
        args = (f"-{int(days)} days",)
    sql += " ORDER BY id DESC LIMIT ?"
    with _get_conn(settings.audit_db_path) as conn:
        rows = conn.execute(sql, (*args, limit)).fetchall()
    events = []
    for r in rows:
        ev = json.loads(r["body"])
        ev["created_at"] = str(r["created_at"]).replace(" ", "T") + "+00:00"
        events.append(ev)
    return events


def get_ledger(limit: int = 200) -> list[dict]:
    """ERP postings newest first, enriched with how each was decided (source, confidence) and the raw vendor name."""
    _ensure()
    postings = get_postings(limit)
    # A failed SAP / Xero attempt is audited too (source "sap_post_failed" / "xero_post_failed"); it must not
    # describe a later successful post.
    audit = {a["prediction_id"]: a for a in get_audit_records(max(limit * 2, 500))
             if a["source"] not in ERP_FAILED_SOURCES}
    names: dict[str, str] = {}
    for ev in get_events(limit=max(limit * 3, 2000)):
        names.setdefault(str(ev.get("prediction_id")), ev.get("vendor_raw") or "")
    sap = {r["invoice_id"]: r for r in get_sap_postings(limit=max(limit * 2, 500), status="posted")}
    xero = {r["invoice_id"]: r for r in get_xero_postings(limit=max(limit * 2, 500), status="posted")}
    for p in postings:
        a = audit.get(p["trace_id"], {})
        p["source"] = a.get("source")
        p["confidence"] = a.get("confidence")
        p["vendor_name"] = names.get(p["trace_id"]) or p["vendor_id"]
        s = sap.get(p["invoice_id"])
        x = xero.get(p["invoice_id"])
        p["erp_backend"] = "sap" if s else "xero" if x else "simulated"
        p["erp_document"] = ""        # the id the external system gave it; empty for the simulated ledger
        if s:  # the SAP document this ledger row was posted as
            p["sap_document"], p["sap_fiscal_year"], p["sap_company_code"] = (
                s["sap_document"], s["fiscal_year"], s["company_code"])
            p["erp_document"] = s["sap_document"]
        if x:  # the Xero bill this ledger row was posted as
            p["xero_invoice_id"], p["xero_url"] = x["xero_invoice_id"], x["xero_url"]
            p["erp_document"] = x["xero_invoice_id"]
    return postings


def post_to_erp_ledger(payload: dict) -> None:
    """Write a confirmed invoice posting to the ERP ledger."""
    with _get_conn(settings.erp_db_path) as conn:
        conn.execute("""
            INSERT INTO erp_postings
            (invoice_id, vendor_id, entity_id, gl_account, cost_centre,
             tax_code, amount, currency, description, posted_by, posted_at, trace_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            payload["invoice_id"],
            payload["vendor_id"],
            payload["entity_id"],
            payload["gl_account"],
            payload["cost_centre"],
            payload["tax_code"],
            payload["amount"],
            payload["currency"],
            payload["description"],
            payload["posted_by"],
            payload["posted_at"],
            payload["trace_id"],
        ))
        conn.commit()


# --------------------------------------------------------------------------- SAP posting state (idempotency)

def sap_claim(reference: str, invoice_id: str, prediction_id: str, stale_after: float = 600.0) -> tuple[bool, dict]:
    """
    Atomically take the right to post `reference` to SAP. Returns (claimed, row).

    A new reference is inserted as in_progress. An existing one can be re-claimed only if its last attempt failed,
    its outcome is unknown, or it has been in_progress longer than `stale_after` seconds (the process died mid-post).
    A posted reference, or one another worker is posting right now, is NOT claimed.
    """
    import time

    _ensure()
    now = time.time()
    with _get_conn(settings.erp_db_path) as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO sap_postings (reference, invoice_id, prediction_id, status, attempts, created_at, "
            "updated_at) VALUES (?, ?, ?, 'in_progress', 1, ?, ?)",
            (reference, invoice_id, prediction_id, now, now),
        )
        claimed = cur.rowcount == 1
        if not claimed:
            cur = conn.execute(
                "UPDATE sap_postings SET status = 'in_progress', attempts = attempts + 1, updated_at = ? "
                "WHERE reference = ? AND (status IN ('failed', 'unknown') OR (status = 'in_progress' AND updated_at < ?))",
                (now, reference, now - stale_after),
            )
            claimed = cur.rowcount == 1
        row = conn.execute("SELECT * FROM sap_postings WHERE reference = ?", (reference,)).fetchone()
    return claimed, dict(row)


def sap_finish(reference: str, status: str, *, sap_document: str = "", fiscal_year: str = "",
               company_code: str = "", error: str = "", summary: dict | None = None) -> None:
    """Record the outcome of a post: status is 'posted', 'failed' (SAP refused) or 'unknown' (outcome unconfirmed)."""
    import time

    _ensure()
    with _get_conn(settings.erp_db_path) as conn:
        conn.execute(
            "UPDATE sap_postings SET status = ?, sap_document = ?, fiscal_year = ?, company_code = ?, error = ?, "
            "summary_json = ?, updated_at = ? WHERE reference = ?",
            (status, sap_document, fiscal_year, company_code, error, json.dumps(summary or {}, default=str),
             time.time(), reference),
        )


def get_sap_posting(reference: str) -> dict | None:
    _ensure()
    with _get_conn(settings.erp_db_path) as conn:
        row = conn.execute("SELECT * FROM sap_postings WHERE reference = ?", (reference,)).fetchone()
    return dict(row) if row else None


def get_sap_postings(limit: int = 100, status: str | None = None) -> list[dict]:
    """SAP posting attempts, newest first. `status` filters (posted / failed / unknown / in_progress)."""
    _ensure()
    sql, args = "SELECT * FROM sap_postings", ()
    if status:
        sql, args = sql + " WHERE status = ?", (status,)
    with _get_conn(settings.erp_db_path) as conn:
        rows = conn.execute(sql + " ORDER BY updated_at DESC LIMIT ?", (*args, limit)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["summary"] = json.loads(d.pop("summary_json") or "{}")
        out.append(d)
    return out


# --------------------------------------------------------------------------- Xero posting state (idempotency)

def xero_claim(reference: str, invoice_id: str, prediction_id: str, stale_after: float = 600.0) -> tuple[bool, dict]:
    """
    Atomically take the right to post `reference` to Xero. Same rules as sap_claim: a new reference is inserted as
    in_progress; an existing one can be re-claimed only if its last attempt failed, its outcome is unknown, or it has
    been in_progress longer than `stale_after` seconds. A posted reference is NOT claimed.
    """
    import time

    _ensure()
    now = time.time()
    with _get_conn(settings.erp_db_path) as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO xero_postings (reference, invoice_id, prediction_id, status, attempts, created_at, "
            "updated_at) VALUES (?, ?, ?, 'in_progress', 1, ?, ?)",
            (reference, invoice_id, prediction_id, now, now),
        )
        claimed = cur.rowcount == 1
        if not claimed:
            cur = conn.execute(
                "UPDATE xero_postings SET status = 'in_progress', attempts = attempts + 1, updated_at = ? "
                "WHERE reference = ? AND (status IN ('failed', 'unknown') OR (status = 'in_progress' AND updated_at < ?))",
                (now, reference, now - stale_after),
            )
            claimed = cur.rowcount == 1
        row = conn.execute("SELECT * FROM xero_postings WHERE reference = ?", (reference,)).fetchone()
    return claimed, dict(row)


def xero_finish(reference: str, status: str, *, xero_invoice_id: str = "", xero_url: str = "", tenant_id: str = "",
                error: str = "", summary: dict | None = None) -> None:
    """Record the outcome of a post: status is 'posted', 'failed' (Xero refused / nothing was sent) or 'unknown'."""
    import time

    _ensure()
    with _get_conn(settings.erp_db_path) as conn:
        conn.execute(
            "UPDATE xero_postings SET status = ?, xero_invoice_id = ?, xero_url = ?, tenant_id = ?, error = ?, "
            "summary_json = ?, updated_at = ? WHERE reference = ?",
            (status, xero_invoice_id, xero_url, tenant_id, error, json.dumps(summary or {}, default=str),
             time.time(), reference),
        )


def get_xero_posting(reference: str) -> dict | None:
    _ensure()
    with _get_conn(settings.erp_db_path) as conn:
        row = conn.execute("SELECT * FROM xero_postings WHERE reference = ?", (reference,)).fetchone()
    return dict(row) if row else None


def get_xero_postings(limit: int = 100, status: str | None = None) -> list[dict]:
    """Xero posting attempts, newest first. `status` filters (posted / failed / unknown / in_progress)."""
    _ensure()
    sql, args = "SELECT * FROM xero_postings", ()
    if status:
        sql, args = sql + " WHERE status = ?", (status,)
    with _get_conn(settings.erp_db_path) as conn:
        rows = conn.execute(sql + " ORDER BY updated_at DESC LIMIT ?", (*args, limit)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["summary"] = json.loads(d.pop("summary_json") or "{}")
        out.append(d)
    return out


def write_audit(record: dict) -> None:
    """Write a prediction audit record."""
    with _get_conn(settings.audit_db_path) as conn:
        conn.execute("""
            INSERT INTO audit_trail
            (prediction_id, invoice_id, vendor_id, predicted_gl, confirmed_gl,
             confidence, routing_decision, routing_reason, source, confirming_user, provenance_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["prediction_id"],
            record["invoice_id"],
            record["vendor_id"],
            record["predicted_gl"],
            record["confirmed_gl"],
            record["confidence"],
            record["routing_decision"],
            record["routing_reason"],
            record["source"],
            record["confirming_user"],
            json.dumps(record["provenance"]),
        ))
        conn.commit()


def has_posting(invoice_id: str) -> bool:
    """True if the ERP ledger already has a row for this invoice."""
    _ensure()
    with _get_conn(settings.erp_db_path) as conn:
        return conn.execute("SELECT 1 FROM erp_postings WHERE invoice_id = ? LIMIT 1", (invoice_id,)).fetchone() is not None


def get_postings(limit: int = 100) -> list[dict]:
    """Fetch recent ERP postings."""
    with _get_conn(settings.erp_db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM erp_postings ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_audit_records(limit: int = 100) -> list[dict]:
    """Fetch recent audit trail records."""
    with _get_conn(settings.audit_db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM audit_trail ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
