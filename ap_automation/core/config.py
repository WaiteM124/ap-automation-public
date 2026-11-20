"""
Configuration - loaded from the environment and .env via pydantic-settings.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": a shared .env often carries keys this app does not model
    # (deployment tokens etc.). They must not stop the application starting.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    # Embeddings / vector store
    openai_api_key: str = ""
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_collection: str = "invoice_corpus"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Persistence
    erp_db_path: str = "data/erp_ledger.db"
    # Predictions awaiting confirmation. Persisted so a restart does not break /confirm.
    predictions_db_path: str = "data/predictions.db"
    # Where an uploaded per-client chart of accounts is persisted (see core/chart_of_accounts.py).
    chart_path: str = "data/client_chart.json"
    # Vendors registered by history import (raw_name,canonical_id) so they resolve at full MDM confidence.
    client_vendors_path: str = "data/client_vendors.csv"
    audit_db_path: str = "data/audit_trail.db"
    # Optional: URL of the standalone mock ERP service (e.g. http://localhost:8001).
    # Empty = post to the local ledger database in-process.
    erp_url: str = ""

    # ---- SAP posting (core/sap_poster.py). OFF by default: nothing reaches SAP unless ERP_BACKEND=sap AND
    # SAP_POSTING_ENABLED=true. Sandbox only for now: a non-sandbox host is refused unless SAP_ALLOW_NON_SANDBOX=true.
    erp_backend: str = "simulated"            # "simulated" (local ledger / ERP_URL) | "sap" | "xero"
    sap_posting_enabled: bool = False         # the explicit write switch
    sap_allow_non_sandbox: bool = False       # second switch needed before any host other than sandbox.api.sap.com
    sap_base_url: str = "https://sandbox.api.sap.com/s4hanacloud"
    sap_api_key: str = ""                     # API Business Hub sandbox key (header APIKey). Never logged.
    sap_username: str = ""                    # real tenant: communication user (basic auth)
    sap_password: str = ""
    sap_company_code: str = "1110"            # default company code (1110 = UK company in the sandbox demo data)
    sap_default_currency: str = "GBP"         # used only when the invoice carries no currency
    sap_default_tax_code: str = "V0"          # used when a VAT treatment has no entry in SAP_TAX_CODE_MAP
    sap_default_supplier: str = ""            # fallback InvoicingParty when a vendor has no SAP_SUPPLIER_MAP entry
    # JSON objects, e.g. {"UK001": "1110"}. Unmapped keys fall back to the default (company code) or pass through (GL/CC).
    sap_company_code_map: str = ""            # our entity_id            -> SAP company code
    sap_supplier_map: str = ""                # our canonical vendor id  -> SAP supplier (InvoicingParty)
    sap_gl_map: str = ""                      # our GL account           -> SAP G/L account
    sap_cost_centre_map: str = ""             # our cost centre          -> SAP cost centre
    sap_tax_code_map: str = ""                # our VAT treatment        -> SAP tax code
    sap_tax_calculated_automatically: bool = True
    sap_timeout_seconds: float = 20.0
    sap_max_retries: int = 2                  # extra attempts for SAFE requests only (GET / CSRF fetch); never a bare POST

    # ---- Xero posting (core/xero_poster.py). OFF by default: nothing reaches Xero unless ERP_BACKEND=xero AND
    # XERO_POSTING_ENABLED=true AND the app has been connected once with scripts/xero_authorize.py.
    xero_posting_enabled: bool = False
    xero_client_id: str = ""
    xero_client_secret: str = ""              # never logged, never stored in the database or in tracked files
    xero_tenant_id: str = ""                  # optional: discovered from GET /connections when empty
    xero_redirect_uri: str = "http://localhost:8765/callback"   # must be registered on the Xero app, character for character
    # Scopes requested at consent. Xero apps created on/after 2 March 2026 use granular scopes (no accounting.transactions).
    xero_scopes: str = "offline_access accounting.invoices accounting.contacts accounting.settings.read"
    xero_bill_status: str = "DRAFT"           # DRAFT | SUBMITTED | AUTHORISED
    xero_line_amount_types: str = "Inclusive" # our amount is the invoice TOTAL: Inclusive | Exclusive | NoTax
    xero_due_days: int = 30                   # DueDate = bill date + N days; 0 = omit and let Xero apply the org default
    xero_account_map: str = ""                # JSON: our GL account -> Xero AccountCode (unmapped = passed through)
    xero_tax_type_map: str = ""               # JSON: our VAT treatment -> Xero TaxType (see docs/XERO_POSTING.md)
    xero_default_tax_type: str = "NONE"       # demo company only: used for a treatment with no mapping
    xero_tracking_category: str = ""          # optional: Xero tracking category NAME that cost centres map to
    xero_cost_centre_map: str = ""            # JSON: our cost centre -> Xero tracking OPTION name (default: pass through)
    xero_token_path: str = "data/xero_tokens.json"
    xero_chart_account_types: str = "EXPENSE,DIRECTCOSTS,OVERHEADS"   # Xero account types pulled into our chart
    xero_timeout_seconds: float = 20.0
    xero_max_retries: int = 2                 # extra attempts for SAFE requests only; never a bare bill creation

    # Google Sheets mirror (optional)
    google_sheets_credentials: str = "credentials/google_service_account.json"
    google_sheet_id: str = ""

    # LLM gate (escalation-only)
    anthropic_api_key: str = ""

    # Cloud LangGraph (LangSmith Deployment). Empty = use the in-process pipeline
    # (tests, readiness replay, local demo). Values come from the environment only.
    langgraph_api_url: str = Field(default="", alias="LANGGRAPH_API_URL")
    langgraph_api_key: str = Field(default="", alias="LANGGRAPH_API_KEY")

    # Observability (all optional)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    supabase_url: str = ""  # set SUPABASE_URL; leave empty to disable Supabase
    supabase_secret_key: str = ""
    # Set TELEMETRY_ENABLED=false for tests / offline evaluation so that nothing is
    # sent to Langfuse or written to the production analytics table.
    telemetry_enabled: bool = True

    # Retrieval
    top_k: int = 10
    rrf_k: int = 60
    bm25_max_candidates: int = 1000

    # Routing gate. An invoice auto-posts only if ALL of these hold.
    min_evidence_lines: int = 3
    min_mdm_confidence: float = 0.85
    min_agreement_ratio: float = 0.80
    default_confidence_threshold: float = 0.50
    auto_post_user_id: str = "system"

    # ---- Optional gate / retrieval variants (experiments). EVERY default below reproduces the shipped behaviour
    # exactly; docs/evaluation.json is produced with these defaults. Recommendations live in docs/, not here.
    # Context signals: use department (entity_id), fund (cost_centre) and amount as extra evidence signals.
    #   "off"    - vendor-scoped retrieval only (shipped)
    #   "subset" - evidence is drawn from the vendor's lines in the matching department/fund (ladder, with fallback)
    #   "boost"  - vendor-wide evidence, re-ranked with a bonus for matching department / fund / amount
    context_signals: str = "off"
    context_ladder: str = "dept+fund,fund,dept,vendor"  # subset mode: most specific level with enough lines wins
    context_min_lines: int = 3                          # lines a subset needs before it is used
    context_boost: float = 0.02                         # boost mode: RRF bonus per matching signal
    context_use_amount: bool = False                    # amount-band match counts as a signal (boost) / tie-break
    # Gate variants (applied on top of min_evidence_lines / min_agreement_ratio / default_confidence_threshold).
    gate_min_margin: float = 0.0            # top account's share minus the runner-up's share
    gate_min_agreement_lb: float = 0.0      # Wilson lower bound of the (weighted) agreement
    gate_wilson_z: float = 1.0
    gate_diversity_per_code: float = 0.0    # extra agreeing lines required per additional distinct code of the vendor
    gate_diversity_scope: str = "vendor"    # "vendor" or "subset": which line set the distinct-code count comes from
    gate_diversity_cap: int = 10            # ceiling on the agreeing-lines requirement
    gate_recency_halflife_days: float = 0.0  # 0 = off. Evidence votes are weighted 0.5 ** (age_days / halflife)
    gate_recency_window_days: float = 0.0   # 0 = off. Only evidence newer than this votes (falls back if too thin)
    gate_required_context: str = ""         # e.g. "dept+fund,fund": auto-post only if the evidence came from these levels

    # API hardening. If API_KEY is set, mutating endpoints require X-API-Key.
    api_key: str = ""
    cors_origins: str = "*"  # comma separated list, or "*"
    max_tracked_predictions: int = 5000

    # Demo: enables the destructive POST /demo/reset. Never enable on a real deployment.
    demo_mode: bool = False


settings = Settings()
