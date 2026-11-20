"""
Bulk-load already-coded historical invoices into the learning corpus.

This is the "Import history" step of onboarding: the lines are what a human already coded, so they enter the corpus
as human-confirmed evidence (the system still never learns from its own decisions). The import is:

  * validated  - every line must have a vendor, a description and a GL that exists in the ACTIVE chart of accounts;
  * idempotent - a line's corpus id is derived from its content, so re-uploading a file does not duplicate anything
                 and lines already present are not re-embedded;
  * reported   - the caller gets counts and the first rejected rows with the reason, never a silent drop;
  * bounded    - at most MAX_ROWS lines per file, so one request stays within a web request's time budget.

`dry_run=True` validates and reports without writing anything.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from collections import Counter
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from ap_automation.core import chart_of_accounts
from ap_automation.core.config import settings
from ap_automation.core.embeddings import embed_batch
from ap_automation.core.mdm import register_vendor
from ap_automation.core.models import ConfirmationSource, amount_to_band
from ap_automation.core.retrieval import _collection_size, get_qdrant, point_id_for

MAX_ROWS = 5_000
MAX_REJECTS_REPORTED = 50
MAX_DESCRIPTION_CHARS = 2_000
EMBED_CHUNK = 100
IMPORT_USER = "history_import"

_VENDOR = ("vendor_name", "raw_vendor_name", "vendor")
_DESC = ("description", "detailed_item_description", "line_description")
_GL = ("gl", "gl_account", "account_code", "ground_truth_gl", "confirmed_gl")
_AMOUNT = ("amount", "dollar_amount")
_CURRENCY = ("currency",)
_ENTITY = ("entity_id", "entity", "department_name")
_CC = ("cost_centre", "cost_center", "fund")
_DATE = ("date", "confirmed_date", "transaction_date", "invoice_date")
_TAX = ("vat", "vat_code", "tax_code", "tax")


class HistoryImportError(ValueError):
    """The file as a whole cannot be imported (not a CSV, no rows, missing columns, too many rows)."""


def _first(row: dict, names: tuple[str, ...]) -> str:
    for n in names:
        v = row.get(n)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _parse_amount(text: str) -> float:
    cleaned = text.replace(",", "").replace("$", "").replace("£", "").replace("€", "").strip()
    if not cleaned:
        return 0.0
    try:
        value = float(cleaned)
    except ValueError:
        raise ValueError(f"amount {text!r} is not a number") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"amount {text!r} must be a non-negative number")
    return value


def _parse_date(text: str, now: datetime) -> datetime:
    if not text:
        return now
    t = text.strip()
    try:
        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"date {t!r} is not ISO (YYYY-MM-DD)") from None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if dt > now:
        raise ValueError(f"date {t} is in the future")
    return dt


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def parse_rows(text: str, now: datetime | None = None, *, allow_unknown_gl: bool = False
               ) -> tuple[list[dict], list[dict], int, int]:
    """
    Validate a CSV against the ACTIVE chart. Returns (valid_rows, rejected, rows_read, duplicates_in_file).
    A valid row carries: vendor, description, gl, amount, currency, entity_id, cost_centre, when, key,
    in_chart, tax_code.

    `allow_unknown_gl=True` keeps rows whose GL is not in the chart (used by the readiness report:
    off-chart codes are reported, not fatal). Import still rejects them.
    """
    now = now or datetime.now(timezone.utc)
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HistoryImportError("the file is empty")
    reader.fieldnames = [(h or "").strip().lower() for h in reader.fieldnames]
    for label, cols in (("vendor", _VENDOR), ("description", _DESC), ("GL account", _GL)):
        if not any(c in reader.fieldnames for c in cols):
            raise HistoryImportError(f"no {label} column (expected one of: {', '.join(cols)})")

    chart = chart_of_accounts.CHART
    valid: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    duplicates = 0
    rows_read = 0
    for line_no, raw in enumerate(reader, start=2):
        row = {k: v for k, v in raw.items() if k}
        if not any((v or "").strip() for v in row.values()):
            continue
        rows_read += 1
        if rows_read > MAX_ROWS:
            raise HistoryImportError(f"more than {MAX_ROWS:,} rows - split the file and import it in parts")
        vendor, desc, gl = _first(row, _VENDOR), _first(row, _DESC), _first(row, _GL)

        def reject(reason: str, _line=line_no, _vendor=vendor, _gl=gl) -> None:
            rejected.append({"line": _line, "vendor": _vendor[:40], "gl": _gl, "reason": reason})

        if not vendor:
            reject("missing vendor")
            continue
        if not re.search(r"[A-Za-z0-9]", vendor):
            reject("vendor name has no letters or digits")
            continue
        if not re.search(r"[A-Za-z0-9]", vendor):
            reject("vendor name has no letters or digits")
            continue
        if not desc:
            reject("missing description")
            continue
        if len(desc) > MAX_DESCRIPTION_CHARS:
            reject(f"description longer than {MAX_DESCRIPTION_CHARS} characters")
            continue
        if not gl:
            reject("missing GL account")
            continue
        in_chart = gl in chart
        if not in_chart and not allow_unknown_gl:
            reject(f"GL {gl} is not in the active chart of accounts")
            continue
        try:
            amount = _parse_amount(_first(row, _AMOUNT))
            when = _parse_date(_first(row, _DATE), now)
        except ValueError as e:
            reject(str(e))
            continue
        key = hashlib.sha1(f"{_norm(vendor)}|{_norm(desc)}|{amount:.2f}|{gl}|{when.date().isoformat()}".encode()).hexdigest()
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        valid.append({
            "vendor": vendor, "description": desc, "gl": gl, "amount": amount,
            "currency": _first(row, _CURRENCY) or "GBP", "entity_id": _first(row, _ENTITY) or "UK001",
            "cost_centre": _first(row, _CC), "when": when, "key": key,
            "in_chart": in_chart, "tax_code": _first(row, _TAX),
        })
    if rows_read == 0:
        raise HistoryImportError("the file has no data rows")
    return valid, rejected, rows_read, duplicates


def import_history(text: str, *, dry_run: bool = False, now: datetime | None = None) -> dict:
    """Validate and (unless dry_run) load the CSV into the corpus. See the module docstring."""
    now = now or datetime.now(timezone.utc)
    valid, rejected, rows_read, duplicates = parse_rows(text, now)

    for r in valid:
        r["invoice_id"] = str(uuid5(NAMESPACE_URL, f"ap-automation/history-import/{r['key']}"))
        r["point_id"] = point_id_for(r["invoice_id"])

    already: set[str] = set()
    to_write = valid
    if valid and not dry_run:
        # Register every vendor (including those whose lines are already present), so a restart or redeploy that
        # lost the registrations is repaired by simply re-uploading the same file.
        for r in valid:
            r["vendor_id"] = register_vendor(r["vendor"], settings.client_vendors_path)
        client = get_qdrant()
        ids = [r["point_id"] for r in valid]
        for i in range(0, len(ids), 200):
            found = client.retrieve(settings.qdrant_collection, ids=ids[i:i + 200], with_payload=False, with_vectors=False)
            already.update(str(p.id) for p in found)
        to_write = [r for r in valid if r["point_id"] not in already]

    imported = 0
    if to_write and not dry_run:
        from qdrant_client.models import PointStruct

        client = get_qdrant()
        for i in range(0, len(to_write), EMBED_CHUNK):
            chunk = to_write[i:i + EMBED_CHUNK]
            vectors = embed_batch([r["description"] for r in chunk])
            points = []
            for r, vec in zip(chunk, vectors, strict=True):
                points.append(PointStruct(id=r["point_id"], vector=vec, payload={
                    "point_id": r["point_id"], "invoice_id": r["invoice_id"], "vendor_id": r["vendor_id"],
                    "vendor_name": r["vendor"], "entity_id": r["entity_id"],
                    "amount_band": amount_to_band(r["amount"]).value, "amount": r["amount"],
                    "currency": r["currency"], "gl_account": r["gl"], "cost_centre": r["cost_centre"],
                    "tax_code": "UNASSESSED", "description": r["description"],
                    "confirming_user_id": IMPORT_USER, "confirmation_timestamp": r["when"].isoformat(),
                    "prediction_confidence": 1.0, "source": ConfirmationSource.HUMAN_CONFIRM.value,
                }))
            client.upsert(collection_name=settings.qdrant_collection, points=points)
            imported += len(points)

    top = Counter(r["gl"] for r in valid).most_common(5)
    return {
        "dry_run": dry_run,
        "rows_read": rows_read,
        "valid": len(valid),
        "imported": imported if not dry_run else 0,
        "would_import": len(valid) if dry_run else None,
        "already_present": len(already),
        "duplicates_in_file": duplicates,
        "rejected_count": len(rejected),
        "rejected": rejected[:MAX_REJECTS_REPORTED],
        "vendors": len({_norm(r["vendor"]) for r in valid}),
        "top_gl": [{"gl": g, "name": chart_of_accounts.GL_NAMES.get(g, ""), "lines": n} for g, n in top],
        "corpus_size": None if dry_run else _collection_size(),
    }
