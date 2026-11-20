"""
Run the demo backend with one command - deterministic, resettable, and nothing to install or host.

    python scripts/run_demo.py            # offline: in-memory corpus, no Docker / OpenAI / network
    python scripts/run_demo.py --live     # your configured Qdrant + OpenAI embeddings (real semantic model)

It starts the API on http://localhost:8000 in demo mode, seeds the corpus, and points the ERP ledger,
audit trail and pending predictions at a throw-away folder, so recording a take never touches real data.
Use "Reset demo" in the UI (or POST /demo/reset) to return to the starting state between takes.

Then, in the UI repo:   AP_API_URL=http://localhost:8000 npm run dev
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Must be set before the application settings are imported.
os.environ["DEMO_MODE"] = "true"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="use the configured Qdrant and OpenAI instead of the offline backend")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--telemetry", action="store_true", help="send demo traffic to Langfuse / Supabase (off by default)")
    ap.add_argument("--api-key", default="", help="require this X-API-Key (default: none, for local use)")
    ap.add_argument("--llm", action="store_true",
                    help="enable the Claude gate (needs ANTHROPIC_API_KEY). Off by default: the LLM can only make "
                         "decisions stricter but it is not deterministic, so two takes of the script could differ")
    args = ap.parse_args()

    if not args.llm:
        os.environ.pop("ANTHROPIC_API_KEY", None)

    if not args.telemetry:
        os.environ["TELEMETRY_ENABLED"] = "false"

    from ap_automation.core import dataset, mdm, offline
    from ap_automation.core.config import settings

    settings.demo_mode = True
    settings.api_key = args.api_key
    settings.erp_url = ""
    settings.erp_backend = "simulated"  # the scripted demo never posts to SAP or Xero
    if not args.llm:
        settings.anthropic_api_key = ""
    scratch = Path(tempfile.mkdtemp(prefix="ap_demo_"))
    settings.erp_db_path = str(scratch / "erp.db")
    settings.audit_db_path = str(scratch / "audit.db")
    settings.predictions_db_path = str(scratch / "predictions.db")
    if not args.telemetry:
        settings.telemetry_enabled = False

    if not args.live:
        offline.use_offline_backend()

    import main as api  # imports the app and loads the vendor master from data/vendor_map.csv

    _, vendor_rows = dataset.generate_invoices(10, 0, 0, seed=1)
    mdm.set_vendor_map(vendor_rows)  # the vendor master the scenarios are written against

    from ap_automation.core import demo

    info = demo.reset(api._store)
    status = demo.status(api._store)
    print(f"\nDemo backend ready on http://localhost:{args.port}")
    print(f"  embeddings : {status['embeddings']}")
    print(f"  corpus     : {info['corpus_size']} human-coded lines across {info['vendors_seeded']} vendors")
    print(f"  llm gate   : {[c['detail'] for c in status['checks'] if c['name'] == 'LLM gate'][0]}")
    print(f"  scratch    : {scratch}")
    print("  UI         : AP_API_URL=http://localhost:%d npm run dev   (in ap-automation-ui)\n" % args.port)

    import uvicorn

    uvicorn.run(api.app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
