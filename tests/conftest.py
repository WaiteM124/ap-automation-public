"""
Test fixtures. Everything runs offline: in-memory Qdrant, hashing embedder, temp SQLite
databases, no Langfuse / Supabase / Anthropic / OpenAI traffic.
"""

import os

# Must be set before ap_automation.core.config is imported anywhere.
for _k, _v in {
    "TELEMETRY_ENABLED": "false",
    "API_KEY": "",
    "ERP_URL": "",
    "ANTHROPIC_API_KEY": "",
    "SUPABASE_SECRET_KEY": "",
    "LANGFUSE_PUBLIC_KEY": "",
    "LANGFUSE_SECRET_KEY": "",
    "GOOGLE_SHEET_ID": "",
    # A developer's .env may hold a real SAP sandbox key. Tests must never see it or reach SAP.
    "SAP_API_KEY": "",
    "SAP_USERNAME": "",
    "SAP_PASSWORD": "",
    "ERP_BACKEND": "simulated",
    "SAP_POSTING_ENABLED": "false",
    # Same for Xero: a developer's .env may hold real Xero credentials. Tests must never see them or reach Xero.
    "XERO_POSTING_ENABLED": "false",
    "XERO_CLIENT_ID": "",
    "XERO_CLIENT_SECRET": "",
    "XERO_TENANT_ID": "",
    "LANGGRAPH_API_URL": "",
    "LANGGRAPH_API_KEY": "",
}.items():
    os.environ[_k] = _v

import pytest  # noqa: E402

from ap_automation.core import (  # noqa: E402
    chart_of_accounts,
    dataset,
    mdm,
    offline,
    posting,
    sap_poster,
    xero_auth,
)
from ap_automation.core.config import settings  # noqa: E402
from ap_automation.core.models import InvoiceLine  # noqa: E402
from ap_automation.core.seeding import seed_corpus  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Fresh corpus, fresh databases and the synthetic vendor master for every test."""
    saved = (settings.erp_db_path, settings.audit_db_path, settings.erp_url, settings.telemetry_enabled,
             settings.predictions_db_path, settings.chart_path, settings.client_vendors_path,
             settings.langgraph_api_url, settings.langgraph_api_key)
    settings.predictions_db_path = str(tmp_path / "predictions.db")
    settings.chart_path = str(tmp_path / "client_chart.json")
    settings.client_vendors_path = str(tmp_path / "client_vendors.csv")
    saved_xero = settings.xero_token_path
    settings.xero_token_path = str(tmp_path / "xero_tokens.json")      # never the real data/xero_tokens.json
    xero_auth.HARDEN_WINDOWS_ACL = False                                # icacls is slow; one test turns it on
    xero_auth.reset_state()
    chart_of_accounts.reset_chart()
    settings.erp_db_path = str(tmp_path / "erp.db")
    settings.audit_db_path = str(tmp_path / "audit.db")
    settings.erp_url = ""
    settings.telemetry_enabled = False
    settings.langgraph_api_url = ""
    settings.langgraph_api_key = ""
    posting._db_ready = False

    _, vendor_rows = dataset.generate_invoices(10, 0, 0, seed=1)
    mdm.set_vendor_map(vendor_rows)
    offline.use_offline_backend()
    yield
    offline.reset_backend()
    sap_poster.set_transport(None)
    xero_auth.set_transport(None)
    xero_auth.reset_state()
    settings.xero_token_path = saved_xero
    chart_of_accounts.reset_chart()
    (settings.erp_db_path, settings.audit_db_path, settings.erp_url, settings.telemetry_enabled,
     settings.predictions_db_path, settings.chart_path, settings.client_vendors_path,
     settings.langgraph_api_url, settings.langgraph_api_key) = saved
    posting._db_ready = False


@pytest.fixture
def seeded():
    """Corpus seeded with human-coded history for the nine known vendors."""
    return seed_corpus(dataset.generate_history(15))


def make_invoice(vendor="Meridian Facilities Ltd", description="Office cleaning services monthly",
                 amount=1200.0, entity="UK001", cost_centre=None, gl="6300", currency="GBP") -> InvoiceLine:
    return InvoiceLine(
        raw_vendor_name=vendor, description=description, amount=amount, currency=currency,
        entity_id=entity, cost_centre=cost_centre, ground_truth_gl=gl,
    )
