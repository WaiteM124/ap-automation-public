# Posting coded invoices to SAP (sandbox first)

Non-PO invoices that the system has coded can be posted to SAP S/4HANA Cloud as supplier invoices with a G/L line
(OData v2 `API_SUPPLIERINVOICE_PROCESS_SRV`, `POST A_SupplierInvoice` with the line deep-inserted under
`to_SupplierInvoiceItemGLAcct`). Code: `ap_automation/core/sap_poster.py`; it plugs in behind `posting.post_outcome`,
so `/auto-post`, `/confirm` and `/batch` all use it. **The simulated ERP stays the default.**

## Status: what is and is not verified

| | |
|---|---|
| Request/response shape, field names and lengths | Checked against the live sandbox `$metadata` (`tests/fixtures/sap_supplierinvoice_metadata.json`). |
| CSRF fetch, reading an invoice by reference | **Verified live** against the SAP API Business Hub sandbox (HTTP 200). |
| `POST A_SupplierInvoice` | **Not verified against SAP.** The API Business Hub sandbox is GET-only: it answers every POST with `405 OPERATION_NOT_SUPPORTED`. Creating a document needs your own S/4HANA Cloud tenant (a communication arrangement for the supplier-invoice scenario, believed to be `SAP_COM_0057`; confirm in your tenant). Everything on the write path is covered by mocked-HTTP tests only. |

## Safety rules

- **Off by default.** Nothing is sent unless `ERP_BACKEND=sap` **and** `SAP_POSTING_ENABLED=true`. With the backend set
  to `sap` but posting disabled, an invoice is *not* silently diverted to the simulated ledger: it stays in review with
  the message "SAP posting is disabled".
- **Sandbox only.** A host other than `sandbox.api.sap.com` is refused unless `SAP_ALLOW_NON_SANDBOX=true` is also set,
  and on such a host an unmapped VAT treatment is an error rather than a default tax code. HTTPS only.
- **Auto-post rule unchanged.** Only `final_routing == auto_post` may auto-post; `/auto-post` still returns 403
  otherwise. SAP failures can only make a decision stricter.
- **Idempotent.** Each invoice carries one reference key, `SupplierInvoiceIDByInvcgParty = "AP" + first 14 hex chars of
  SHA-256(invoice_id)` (16 chars). The `sap_postings` table remembers the outcome per key, and before every POST SAP is
  asked whether an invoice with that key already exists (company code + supplier + reference). A retry, a lost response
  or a wiped local database therefore cannot create a second document.
- **No blind POST retries.** A POST is repeated only when it provably never left the machine (connection could not be
  established, max `SAP_MAX_RETRIES`) or after a CSRF-token rejection (once). A timeout / 5xx after sending is
  *reconciled* by reading the reference back; if inconclusive the post is recorded as `unknown` and is re-checked
  before any further attempt.
- **Failures are visible.** If SAP does not accept a post: nothing is written to the ledger, the audit trail gets a row
  with `source = sap_post_failed` and SAP's sanitised error, the invoice is put back in the review queue (unposted,
  `final_routing` forced to `human_review`, reason = the SAP error), nothing is learned from it, `/auto-post` and
  `/confirm` answer **502** with the error, and `GET /sap/postings?status=failed` lists it. A human can then fix the
  master data and `/confirm` it; the reference key guarantees one document.
- **No secrets in output.** The key/password and CSRF tokens are stripped from every stored or returned message; error
  text is parsed (never echoed raw) and capped at 600 characters. `GET /sap/status` returns host only.

## Field mapping

| Our field | SAP field | Value / default |
|---|---|---|
| `invoice_id` | `SupplierInvoiceIDByInvcgParty` | `reference_for(invoice_id)` (idempotency key) |
| `entity_id` | `CompanyCode` (header + line) | `SAP_COMPANY_CODE_MAP[entity_id]`, else `SAP_COMPANY_CODE` (default `1110`) |
| `vendor_id` (canonical) | `InvoicingParty` | `SAP_SUPPLIER_MAP[vendor_id]`, else `SAP_DEFAULT_SUPPLIER`; **error if neither** |
| `currency` | `DocumentCurrency` (header + line) | invoice currency, else `SAP_DEFAULT_CURRENCY` (`GBP`) |
| `amount` | `InvoiceGrossAmount`, line `SupplierInvoiceItemAmount` | 2 dp. Treated as the **gross** invoice amount; must be > 0 |
| `description` | `DocumentHeaderText` (25), line `SupplierInvoiceItemText` (50) | truncated |
| `posted_at` | `PostingDate`, `DocumentDate` | the post date (UTC). We hold no invoice date |
| `gl_account` | line `GLAccount` | `SAP_GL_MAP[gl]`, else passed through unchanged |
| `cost_centre` | line `CostCenter` | `SAP_COST_CENTRE_MAP[cc]`, else passed through; omitted if empty |
| `tax_code` (our VAT treatment) | line `TaxCode` | `SAP_TAX_CODE_MAP[treatment]`, else `SAP_DEFAULT_TAX_CODE` (`V0`, sandbox only) |
| - | `TaxIsCalculatedAutomatically` | `SAP_TAX_CALCULATED_AUTOMATICALLY` (`true`) |
| - | line `SupplierInvoiceItem`, `DebitCreditCode` | `1`, `S` (debit: expense line of an invoice) |

SAP mandatory data we **cannot** supply from our own data, and what we do instead:

- **Supplier (`InvoicingParty`)** - our vendor ids are not SAP supplier numbers. Must be mapped (`SAP_SUPPLIER_MAP`).
- **Company code** - we only have `entity_id`. Mapped, with a single default.
- **Tax code** - our VAT agent produces a *treatment* (STANDARD, ZERO, ...), not an SAP tax code. Mapped; the default
  `V0` (no tax) is wrong for a taxed invoice and is therefore only allowed on the sandbox.
- **Document / invoice date and the supplier's own invoice number** - not in our data. Posting date = document date =
  the day of posting; the reference is our generated key. Consequence: the same real-world invoice submitted twice gets
  two different keys, so SAP's own duplicate check does not catch it.
- **GL account / cost centre** - our codes are passed through unchanged unless mapped; they must exist in the client's chart.
- Not supported: credit memos, multiple lines, withholding tax, reverse charge / non-standard tax, payment terms.

## Enabling it

Locally: copy the `SAP_*` entries from `.env.example` into `.env`, set `ERP_BACKEND=sap`, `SAP_POSTING_ENABLED=true`
and the credentials, then check `GET /sap/status` (`ready_to_post: true`). On Render (variable names only; set the
secrets in the dashboard, never in `render.yaml`): `ERP_BACKEND`, `SAP_POSTING_ENABLED`, `SAP_BASE_URL`, `SAP_API_KEY`
(or `SAP_USERNAME` + `SAP_PASSWORD`), `SAP_COMPANY_CODE`, `SAP_SUPPLIER_MAP`, `SAP_GL_MAP`, `SAP_COST_CENTRE_MAP`,
`SAP_TAX_CODE_MAP`, and only for a non-sandbox tenant `SAP_ALLOW_NON_SANDBOX`. Render's free tier has an ephemeral disk:
the local `sap_postings` table is lost on redeploy (the reference check against SAP still prevents duplicates), so use
a persistent disk before relying on it.

Sandbox demo data that exists (company code `1110`, GBP): supplier `11300001`, G/L `68000000`, cost centre `11101101`,
tax code `V0`; company code `1710` (USD) uses supplier `USSU-VSF01`. Reading works against the sandbox; writes do not.

## Endpoints

- `GET /sap/status` - configured / posting enabled / ready, host only, mapping counts.
- `GET /sap/postings?status=failed|unknown|posted|in_progress` - posting attempts with reference, SAP document, sanitised error.
- `GET /ledger` rows now carry `erp_backend` and, for SAP posts, `sap_document`, `sap_fiscal_year`, `sap_company_code`.
- `/auto-post`, `/confirm` return `erp_backend` and `erp_ref`; a SAP failure is a 502 with `detail.sap.error`.
