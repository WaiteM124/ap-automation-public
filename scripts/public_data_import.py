"""
Import a replay sample from the City of Los Angeles "Checkbook L.A." open dataset.

    Dataset : Checkbook L.A. (Socrata id pggv-e4fn), City of Los Angeles Controller
    URL     : https://controllerdata.lacity.org/resource/pggv-e4fn.csv
    Licence : CC BY 4.0 (per the dataset's API metadata) - attribute "City of Los Angeles"

    python scripts/public_data_import.py --fiscal-year 2026 --max-rows 30000
    (re-running re-uses the cached raw pages under data/public/)

What it does
  1. Pulls one fiscal year, filtered server-side to expenditure_type = 'SUPPLIES & OTHER SERVICES',
     dollar_amount > 0 and a non-empty detailed_item_description. Pages with $limit/$offset and a stable
     $order, retries with backoff, and caches every raw page under data/public/ (gitignored; never commit it).
  2. Normalises (trim, drop exact duplicate lines, drop non-positive amounts / empty descriptions).
  3. Builds a sample that keeps history structure: the busiest vendors (by vendor_id) so vendors repeat over
     time, capped at --max-rows, written in transaction_date order.

Leak guard: only the whitelisted columns in RAW_COLUMNS are ever downloaded, and `row_to_invoice` reads only
the whitelist below. account_name, expenditure_type, item_code and item_code_name are derived from the account
being predicted, so they are never pulled, never mapped and never used as a feature (see LEAK_COLUMNS and
tests/test_public_data.py). expenditure_type is used ONLY in the server-side WHERE clause to select the population.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ap_automation.core.models import InvoiceLine  # noqa: E402
from ap_automation.core.preprocessing import preprocess_description  # noqa: E402

DATASET_NAME = "Checkbook L.A. (City of Los Angeles Controller)"
DATASET_ID = "pggv-e4fn"
DATASET_URL = "https://controllerdata.lacity.org/resource/pggv-e4fn.csv"
LICENCE = "CC BY 4.0"
ATTRIBUTION = "Source: Checkbook L.A., City of Los Angeles (controllerdata.lacity.org, dataset pggv-e4fn), CC BY 4.0"
EXPENDITURE_TYPE = "SUPPLIES & OTHER SERVICES"  # used only as a server-side population filter
PRIVACY_PREFIX = "PRIVACY-"  # Checkbook L.A. masks individuals' payees as "PRIVACY-<DEPARTMENT>"

# Columns derived from the account being predicted. They must never reach an invoice field.
LEAK_COLUMNS: tuple[str, ...] = ("account_name", "expenditure_type", "item_code", "item_code_name")

# The ONLY columns downloaded. `:id` is Socrata's stable row id (used for ordering and invoice ids).
RAW_COLUMNS: tuple[str, ...] = (
    ":id", "fiscal_year", "department_name", "vendor_name", "vendor_id", "transaction_date", "dollar_amount",
    "fund_name", "fund", "account_code", "detailed_item_description", "inv_num", "inv_date", "po_num", "currency",
)
assert not set(LEAK_COLUMNS) & set(RAW_COLUMNS)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "public"
PAGE_SIZE = 50_000


# --------------------------------------------------------------------------- fetch

def build_where(fiscal_year: int) -> str:
    return (
        f"fiscal_year={fiscal_year} AND expenditure_type='{EXPENDITURE_TYPE}' AND dollar_amount > 0 "
        "AND detailed_item_description IS NOT NULL AND detailed_item_description != '' AND vendor_id IS NOT NULL"
    )


def _get_page(client, params: dict, retries: int = 6) -> bytes:
    delay = 3.0
    for attempt in range(1, retries + 1):
        try:
            resp = client.get(DATASET_URL, params=params)
            resp.raise_for_status()
            return resp.content
        except Exception as e:  # network error / 5xx / timeout - back off and retry
            if attempt == retries:
                raise
            print(f"  page fetch failed ({type(e).__name__}); retry {attempt}/{retries - 1} in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("unreachable")


def pull_fiscal_year(fiscal_year: int, cache_dir: Path = DEFAULT_CACHE_DIR, page_size: int = PAGE_SIZE) -> list[Path]:
    """Pull (or reuse cached) raw pages for the fiscal year. Returns the page files in order."""
    import httpx

    raw_dir = cache_dir / f"la_checkbook_fy{fiscal_year}_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    where = build_where(fiscal_year)
    manifest = raw_dir / "manifest.json"
    pages: list[Path] = []
    offset = 0
    t0 = time.time()
    with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0), headers={"Accept": "text/csv"}) as client:
        while True:
            page = raw_dir / f"page_{offset:08d}.csv"
            if not page.exists():
                params = {"$select": ",".join(RAW_COLUMNS), "$where": where, "$order": ":id",
                          "$limit": str(page_size), "$offset": str(offset)}
                body = _get_page(client, params)
                page.with_suffix(".tmp").write_bytes(body)
                page.with_suffix(".tmp").replace(page)
                print(f"  fetched offset {offset:>8}  ({len(body) / 1e6:.1f} MB, {time.time() - t0:.0f}s)", flush=True)
            else:
                print(f"  cached  offset {offset:>8}", flush=True)
            pages.append(page)
            with page.open(newline="", encoding="utf-8") as f:
                n = sum(1 for _ in f) - 1  # header
            if n < page_size:
                break
            offset += page_size
    manifest.write_text(json.dumps({
        "dataset": DATASET_NAME, "url": DATASET_URL, "where": where, "order": ":id", "select": list(RAW_COLUMNS),
        "page_size": page_size, "pages": [p.name for p in pages],
        "pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    return pages


def read_raw(pages: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in pages:
        with p.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


# --------------------------------------------------------------------------- normalise + map

def _clean(s: str | None) -> str:
    return " ".join((s or "").split())


def _parse_date(s: str | None) -> str:
    """'2025-06-13T00:00:00.000' -> '2025-06-13' ('' if unparseable)."""
    s = (s or "").strip()[:10]
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return ""
    return s


def normalise_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """Trim, drop non-positive amounts, empty/unusable descriptions and exact duplicate lines."""
    dropped: Counter[str] = Counter()
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in rows:
        try:
            amount = float(r.get("dollar_amount") or "")
        except ValueError:
            dropped["bad_amount"] += 1
            continue
        if amount <= 0:
            dropped["non_positive_amount"] += 1
            continue
        desc = _clean(r.get("detailed_item_description"))
        vendor_id = _clean(r.get("vendor_id"))
        vendor_name = _clean(r.get("vendor_name"))
        code = _clean(r.get("account_code"))
        date = _parse_date(r.get("transaction_date"))
        if not desc or not preprocess_description(desc):
            dropped["empty_description"] += 1
            continue
        if not vendor_id or not vendor_name:
            dropped["no_vendor"] += 1
            continue
        if vendor_name.upper().startswith(PRIVACY_PREFIX) or vendor_id.upper().startswith(PRIVACY_PREFIX):
            dropped["privacy_masked_payee"] += 1  # the city's masked bucket for individuals' payouts, not a vendor
            continue
        if not code:
            dropped["no_account_code"] += 1
            continue
        if not date:
            dropped["no_date"] += 1
            continue
        clean = {
            ":id": _clean(r.get(":id")), "vendor_id": vendor_id, "vendor_name": vendor_name,
            "department_name": _clean(r.get("department_name")), "fund": _clean(r.get("fund")),
            "fund_name": _clean(r.get("fund_name")), "account_code": code, "dollar_amount": f"{amount:.2f}",
            "transaction_date": date, "detailed_item_description": desc,
            "inv_num": _clean(r.get("inv_num")), "inv_date": _parse_date(r.get("inv_date")),
            "po_num": _clean(r.get("po_num")), "currency": _clean(r.get("currency")) or "USD",
            "fiscal_year": _clean(r.get("fiscal_year")),
        }
        key = (vendor_id, clean["inv_num"], date, clean["dollar_amount"], desc.lower(), code, clean["fund"],
               clean["department_name"], clean["po_num"])
        if key in seen:
            dropped["duplicate_line"] += 1
            continue
        seen.add(key)
        out.append(clean)
    return out, {"input_rows": len(rows), "kept": len(out), "dropped": dict(dropped)}


def row_to_invoice(row: dict) -> InvoiceLine:
    """
    Map one normalised row to the pipeline's invoice model. Reads ONLY the whitelisted keys below.

        vendor = vendor_name           description = detailed_item_description
        amount = dollar_amount (USD)   entity      = department_name
        cost_centre = fund             label       = account_code  (ground_truth_gl; evaluation-only)

    account_name / expenditure_type / item_code / item_code_name are never read (leak guard).
    """
    return InvoiceLine(
        invoice_id=uuid5(NAMESPACE_URL, f"la-checkbook/pggv-e4fn/{row[':id']}"),
        raw_vendor_name=row["vendor_name"],
        description=row["detailed_item_description"],
        amount=float(row["dollar_amount"]),
        currency="USD",
        entity_id=row["department_name"],
        cost_centre=row["fund"] or None,
        ground_truth_gl=row["account_code"],
    )


# --------------------------------------------------------------------------- sample

def select_sample(rows: list[dict], max_rows: int = 30_000, max_per_vendor: int = 250,
                  min_vendor_rows: int = 50) -> tuple[list[dict], dict]:
    """
    Keep the busiest vendors (by vendor_id) until `max_rows` is reached, so vendors repeat over time.

    * A vendor needs >= `min_vendor_rows` lines in the year to be eligible.
    * A vendor with more than `max_per_vendor` lines is thinned evenly (every k-th line in date order), which
      keeps its full date span but stops one utility-type vendor eating the whole budget.
    * Vendors are taken in descending order of (capped) size; a vendor that does not fit is skipped, so
      the total never exceeds `max_rows`. Output is in (transaction_date, :id) order.
    * Vendor names whose normalised form collides with another vendor's (MDM keys on name) are dropped
      from the smaller vendor so vendor identity stays consistent.
    """
    from ap_automation.core.mdm import _normalise

    by_vendor: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_vendor[r["vendor_id"]].append(r)
    eligible = {v: rs for v, rs in by_vendor.items() if len(rs) >= min_vendor_rows}

    def capped(rs: list[dict]) -> list[dict]:
        rs = sorted(rs, key=lambda r: (r["transaction_date"], r[":id"]))
        if len(rs) <= max_per_vendor:
            return rs
        step = len(rs) / max_per_vendor
        return [rs[int(i * step)] for i in range(max_per_vendor)]

    ranked = sorted(eligible, key=lambda v: (-min(len(eligible[v]), max_per_vendor), -len(eligible[v]), v))
    chosen: list[str] = []
    claimed: dict[str, str] = {}  # normalised name -> vendor_id
    total = 0
    name_collisions = 0
    for v in ranked:
        n = min(len(eligible[v]), max_per_vendor)
        if total + n > max_rows:
            continue
        keys = {_normalise(r["vendor_name"]) for r in eligible[v]}
        if any(claimed.get(k, v) != v for k in keys):
            name_collisions += 1
            continue
        for k in keys:
            claimed[k] = v
        chosen.append(v)
        total += n
        if total >= max_rows:
            break

    sample = [r for v in chosen for r in capped(eligible[v])]
    sample.sort(key=lambda r: (r["transaction_date"], r[":id"]))
    params = {
        "max_rows": max_rows, "max_per_vendor": max_per_vendor, "min_vendor_rows": min_vendor_rows,
        "method": "busiest vendors by vendor_id; vendors > max_per_vendor thinned evenly in date order; "
                  "rows kept in (transaction_date, :id) order",
        "eligible_vendors": len(eligible), "vendors_in_population": len(by_vendor),
        "population_rows": len(rows), "vendors_skipped_for_name_collision": name_collisions,
    }
    return sample, params


def sample_stats(rows: list[dict]) -> dict:
    """Descriptive stats for a sample, including label-noise indicators (codes per vendor, concentration)."""
    by_vendor: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_vendor[r["vendor_id"]].append(r)
    sizes = sorted(len(v) for v in by_vendor.values())

    def q(p: float) -> int:
        return sizes[min(int(p * len(sizes)), len(sizes) - 1)] if sizes else 0

    multi, tops, majority = 0, [], 0
    for rs in by_vendor.values():
        counts = Counter(r["account_code"] for r in rs)
        multi += len(counts) > 1
        top = counts.most_common(1)[0][1] / len(rs)
        tops.append((len(rs), top))
        majority += top >= 0.8
    n = len(rows)
    weighted_top = sum(sz * t for sz, t in tops) / n if n else 0.0
    code_counts = Counter(r["account_code"] for r in rows)
    dates = sorted(r["transaction_date"] for r in rows)
    distinct_codes_per_vendor = sorted(len({r["account_code"] for r in rs}) for rs in by_vendor.values())
    return {
        "rows": n, "vendors": len(by_vendor), "account_codes": len(code_counts),
        "non_numeric_account_codes": sum(1 for c in code_counts if not c.isdigit()),
        "departments": len({r["department_name"] for r in rows}), "funds": len({r["fund"] for r in rows}),
        "date_min": dates[0] if dates else None, "date_max": dates[-1] if dates else None,
        "rows_per_vendor": {"min": sizes[0] if sizes else 0, "median": q(0.5), "p90": q(0.9),
                            "max": sizes[-1] if sizes else 0, "mean": round(n / len(sizes), 1) if sizes else 0},
        "vendors_with_gt1_account_code": multi,
        "share_vendors_with_gt1_account_code": round(multi / len(by_vendor), 4) if by_vendor else None,
        "distinct_codes_per_vendor_median": distinct_codes_per_vendor[len(distinct_codes_per_vendor) // 2]
        if distinct_codes_per_vendor else 0,
        "vendors_with_top_code_share_ge_80pct": majority,
        "share_vendors_with_top_code_share_ge_80pct": round(majority / len(by_vendor), 4) if by_vendor else None,
        "row_weighted_top_code_share": round(weighted_top, 4),
        "top_account_codes": code_counts.most_common(10),
        "rows_with_po_num": sum(1 for r in rows if r["po_num"]),
    }


def write_sample(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [":id", "vendor_id", "vendor_name", "department_name", "fund", "fund_name", "account_code",
              "dollar_amount", "transaction_date", "detailed_item_description", "inv_num", "inv_date", "po_num",
              "currency", "fiscal_year"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)


def read_sample(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample_path(fiscal_year: int, max_rows: int, max_per_vendor: int = 250, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return cache_dir / f"la_checkbook_fy{fiscal_year}_sample_{max_rows}_pv{max_per_vendor}.csv"


def ensure_sample(fiscal_year: int = 2026, max_rows: int = 30_000, max_per_vendor: int = 250,
                  min_vendor_rows: int = 50, cache_dir: Path = DEFAULT_CACHE_DIR,
                  page_size: int = PAGE_SIZE) -> tuple[list[dict], dict]:
    """Return (sample rows, import report), pulling / normalising / sampling only if not already cached."""
    out = sample_path(fiscal_year, max_rows, max_per_vendor, cache_dir)
    report_file = out.with_suffix(".report.json")
    if out.exists() and report_file.exists():
        return read_sample(out), json.loads(report_file.read_text())
    print(f"Pulling {DATASET_NAME} FY{fiscal_year} (cached pages are reused)", flush=True)
    pages = pull_fiscal_year(fiscal_year, cache_dir, page_size)
    raw = read_raw(pages)
    clean, norm = normalise_rows(raw)
    sample, params = select_sample(clean, max_rows, max_per_vendor, min_vendor_rows)
    write_sample(sample, out)
    report = {
        "dataset": DATASET_NAME, "dataset_id": DATASET_ID, "url": DATASET_URL, "licence": LICENCE,
        "attribution": ATTRIBUTION, "fiscal_year": fiscal_year, "where": build_where(fiscal_year),
        "normalisation": norm, "sample_params": params, "sample_stats": sample_stats(sample),
        "population_stats": sample_stats(clean), "leak_guard_columns": list(LEAK_COLUMNS),
        "sample_file": out.name,
    }
    report_file.write_text(json.dumps(report, indent=2))
    return sample, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiscal-year", type=int, default=2026, help="latest full fiscal year (Jul-Jun); default 2026")
    ap.add_argument("--max-rows", type=int, default=30_000)
    ap.add_argument("--max-per-vendor", type=int, default=250)
    ap.add_argument("--min-vendor-rows", type=int, default=50)
    ap.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--page-size", type=int, default=PAGE_SIZE)
    ap.add_argument("--stats-out", type=str, default="", help="write sample + normalisation stats JSON here")
    args = ap.parse_args()
    sample, report = ensure_sample(args.fiscal_year, args.max_rows, args.max_per_vendor, args.min_vendor_rows,
                                   Path(args.cache_dir), args.page_size)
    print(json.dumps(report, indent=2))
    if args.stats_out:
        Path(args.stats_out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
