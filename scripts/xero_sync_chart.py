"""
Pull the connected Xero organisation's chart of accounts into this system's per-client chart.

    python scripts/xero_sync_chart.py              # replace the saved chart (data/client_chart.json) with Xero's
    python scripts/xero_sync_chart.py --dry-run    # show what would be loaded, change nothing
    python scripts/xero_sync_chart.py --csv out.csv  # also write a CSV you can upload with POST /chart (e.g. to Render)

Only ACTIVE accounts of the types in XERO_CHART_ACCOUNT_TYPES (default EXPENSE, DIRECTCOSTS, OVERHEADS: the types a
supplier bill is coded to) that have a code are loaded, so every GL code the system predicts is a valid Xero
AccountCode. Each account's default Xero tax type becomes our VAT treatment; one we cannot map becomes NOT_ASSESSED,
which always sends the invoice to human review. Corpus lines coded to accounts outside the new chart are NOT removed.

A running API keeps its chart in memory: restart it, or call POST /xero/sync-chart instead (same effect, live).
Needs the app to be connected first: python scripts/xero_authorize.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ap_automation.core import xero_auth, xero_poster  # noqa: E402
from ap_automation.core.config import settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="read Xero and report, but do not change the chart")
    ap.add_argument("--csv", metavar="PATH", help="write the chart as a CSV (code,name,vat_treatment) for POST /chart")
    args = ap.parse_args()

    try:
        result = xero_poster.sync_chart(apply=not args.dry_run)
    except xero_auth.XeroError as e:
        print(f"Chart sync failed: {xero_auth.sanitize(e)}")
        return 1

    print(f"Read {result['xero_accounts_read']} active Xero accounts; kept {result['accounts']} of type "
          f"{', '.join(result['types'])}.")
    skipped = ", ".join(f"{k}={v}" for k, v in result["skipped"].items() if v)
    print(f"Skipped: {skipped or 'none'}")
    if result["unmapped_tax_types"]:
        print("Accounts whose Xero tax type has no VAT-treatment mapping (set to NOT_ASSESSED, always reviewed): "
              + ", ".join(f"{k} x{v}" for k, v in result["unmapped_tax_types"].items())
              + ". Add them to XERO_TAX_TYPE_MAP to change this.")
    for row in result["chart"][:10]:
        print(f"  {row['code']:<10} {row['name']}  [{row['vat_treatment']}]")
    if len(result["chart"]) > 10:
        print(f"  ... and {len(result['chart']) - 10} more")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["code", "name", "vat_treatment"])
            w.writerows([r["code"], r["name"], r["vat_treatment"]] for r in result["chart"])
        print(f"Wrote {args.csv}")
    if args.dry_run:
        print("Dry run: the chart was not changed.")
    elif result["persisted"]:
        print(f"Chart saved to {settings.chart_path}. Restart the API (or call POST /xero/sync-chart) to use it.")
    else:
        print("The chart was applied in this process but could not be saved to disk (read-only?).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
