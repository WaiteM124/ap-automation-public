# Posting to Xero

An opt-in ERP backend: a cleared invoice becomes a **bill** (`ACCPAY`) in Xero, **DRAFT** by default. It is off unless
you switch it on, and it shares the safety model of the SAP adapter (`docs/SAP_POSTING.md`).

> **Verification status.** The adapter was built from Xero's public documentation and OpenAPI spec and tested against
> a **mocked** Xero (`tests/test_xero_poster.py`). **It has not been run against a real Xero organisation.** Step 7
> below is the first live test. Items still unverified live are listed at the end.

## Safety rules

- Off by default. Nothing is sent unless **both** `ERP_BACKEND=xero` **and** `XERO_POSTING_ENABLED=true`.
- Only invoices the governed gate cleared (`final_routing == auto_post`) can be auto-posted. `POST /auto-post/{id}`
  still answers 403 for anything else. A human can still `POST /confirm/{id}`.
- A bill is created with `PUT /Invoices` (create-only, so an existing bill can never be updated or overwritten).
  Before every create the adapter looks the bill up by `InvoiceNumber` (our reference key). A create is never blindly
  retried; if the outcome is unclear the invoice is marked `unknown` and reconciled by reading Xero, not by posting again.
- If Xero refuses or cannot be reached the invoice goes back to review with a sanitised error, the audit trail records
  `xero_post_failed`, and `/auto-post` and `/confirm` answer **502**. The UI only says "Posted to Xero" when Xero
  returned an invoice id.
- Evaluation, replay and demo scripts force the simulated backend and never post.
- Secrets: `XERO_CLIENT_SECRET`, access and refresh tokens are never logged, returned by the API, put in audit rows or
  the database. Error text is scrubbed of them. The token file is created owner-only.

## What goes to Xero

| Xero field | Source | Default / note |
|---|---|---|
| `Type` | fixed | `ACCPAY` (supplier bill) |
| `Contact.Name` | vendor name as received on the invoice | matched or created by name; falls back to the vendor id |
| `InvoiceNumber` | our reference key (`AP` + 14 hex of SHA-256(invoice id)) | the idempotency key; same key in every ERP |
| `Date` | processing / posting date | |
| `DueDate` | `Date` + `XERO_DUE_DAYS` | 30; `0` omits it |
| `CurrencyCode` | invoice currency | required |
| `LineAmountTypes` | `XERO_LINE_AMOUNT_TYPES` | `Inclusive` (our amounts are gross); `Exclusive` / `NoTax` |
| `Status` | `XERO_BILL_STATUS` | `DRAFT`; `AUTHORISED` is allowed |
| `LineItems[0].Description` | invoice description | one line per invoice |
| `LineItems[0].Quantity` / `UnitAmount` | `1` / invoice amount | amount must be > 0 (no credit notes) |
| `LineItems[0].AccountCode` | GL account, through `XERO_ACCOUNT_MAP` (JSON `{"6300":"400"}`) | the GL itself if unmapped |
| `LineItems[0].TaxType` | VAT treatment, through `XERO_TAX_TYPE_MAP` | see below |
| `LineItems[0].Tracking` | cost centre, only if `XERO_TRACKING_CATEGORY` is set | option from `XERO_COST_CENTRE_MAP`, else the cost centre code |
| Header `xero-tenant-id` | `XERO_TENANT_ID`, else the tenant saved by `xero_authorize.py` | |
| Header `Idempotency-Key` | `<reference>-<attempt>` | sent; the server's behaviour is not documented, so the `InvoiceNumber` lookup is what protects you |

**Tax.** For a *real* organisation there is no default: a VAT treatment with no entry in `XERO_TAX_TYPE_MAP` stops the
post with a mapping error (zero requests are sent). Only the **Xero Demo Company (UK)** uses a built-in map
(`STANDARD→INPUT2`, `REDUCED→RRINPUT`, `ZERO→ZERORATEDINPUT`, `EXEMPT→EXEMPTINPUT`, `OUTSIDE_SCOPE→NONE`), with
`XERO_DEFAULT_TAX_TYPE` (`NONE`) for anything else. Xero requires tax types that exist on your chart, so for a real
organisation fill in the map from your own tax rates.

## Set-up, step by step

1. **Xero account and demo company.** Sign up for a free Xero account, then use *Try the demo company* in the
   organisation menu. Use the Demo Company, never a real ledger, for this first test.
2. **Create the app** at <https://developer.xero.com/app/manage>: *New app* → integration type **Web app**, any
   name that does **not** contain the word "Xero", company URL anything, **redirect URI**
   `http://localhost:8765/callback` (it must match `XERO_REDIRECT_URI` exactly). Note the client id and generate a
   client secret.
3. **Put the credentials in `.env`** (never commit it):
   ```
   XERO_CLIENT_ID=...
   XERO_CLIENT_SECRET=...
   ```
   The scopes the app asks for default to `offline_access accounting.invoices accounting.contacts
   accounting.settings.read` (`XERO_SCOPES`). Xero is moving apps to more granular scopes; if
   Xero rejects the consent with `invalid_scope`, set `XERO_SCOPES` to the equivalents your app's configuration page
   offers (this was not verified live).
4. **Connect once:** `python scripts/xero_authorize.py` (add `--no-browser` to print the URL instead). Approve the
   Demo Company in the browser. The script stores the tokens in `data/xero_tokens.json` (git-ignored, owner-only) and
   prints only the organisation name. Xero refresh tokens are single-use and rotate on every refresh; the adapter
   persists each new one atomically before using it.
5. **Sync the chart of accounts:** `python scripts/xero_sync_chart.py --dry-run`, then without `--dry-run` (or call
   `POST /xero/sync-chart`). This replaces the chart with your Xero expense accounts (types `EXPENSE`, `DIRECTCOSTS`,
   `OVERHEADS`, configurable with `XERO_CHART_ACCOUNT_TYPES`), so predicted GL codes are real Xero account codes.
   Review the printed VAT treatments; unrecognised Xero tax types become `NOT_ASSESSED`.
6. **Enable it:** in `.env` set `ERP_BACKEND=xero` and `XERO_POSTING_ENABLED=true`, restart the API.
   `GET /xero/status` should show `ready_to_post: true`.
7. **Post a test invoice.** Predict an invoice for a known vendor, then auto-post it (or confirm it):
   ```
   POST /predict          {"raw_vendor_name": "...", "description": "...", "amount": 120, "entity_id": "UK001"}
   POST /auto-post/{prediction_id}      # only if the gate cleared it; otherwise POST /confirm/{prediction_id}
   ```
   Send `X-API-Key` if `API_KEY` is set. The response contains `erp_ref.xero_invoice_id` and a deep link.
8. **Find the bill in Xero:** *Business → Bills to pay*; a DRAFT bill appears under the **Draft** tab, its reference
   is the 16-character reference key (the "Post-ready" table shows it). `GET /xero/postings` lists every attempt.

The ERP page "Post-ready" tab (UI) and `GET /post-ready?format=csv|json` list every cleared invoice with the status
"Ready to post" or "Posted to Xero". `GET /post-ready/{reference}/payload?target=xero` previews exactly what would be sent.
Invoices already in the *simulated* ledger can be exported but are not re-posted to Xero.

## Demo company caveats and limits

- The Demo Company is **reset about every 28 days**; after a reset the organisation (tenant id) changes and you must
  repeat steps 4-5.
- Limits (Xero documentation): 60 calls/minute and 5,000/day per organisation on most plans; the Starter plan is lower
  (1,000/day per organisation, 5 connections). A bill costs about 2-3 calls (lookup, create, one organisation read for
  the deep link). A rate-limit answer longer than 30 s fails the post (invoice back to review); nothing is retried blindly.
- Tokens: an access token lasts 30 minutes; a refresh token is valid 60 days and is replaced on every refresh, so use
  the app at least every 60 days or run `xero_authorize.py` again.

## Render (or any host)

Set these as environment variables (names only; values are yours):
`ERP_BACKEND`, `XERO_POSTING_ENABLED`, `XERO_CLIENT_ID`, `XERO_CLIENT_SECRET`, `XERO_TENANT_ID`, `XERO_REDIRECT_URI`,
`XERO_SCOPES`, `XERO_BILL_STATUS`, `XERO_LINE_AMOUNT_TYPES`, `XERO_DUE_DAYS`, `XERO_ACCOUNT_MAP`, `XERO_TAX_TYPE_MAP`,
`XERO_DEFAULT_TAX_TYPE`, `XERO_TRACKING_CATEGORY`, `XERO_COST_CENTRE_MAP`, `XERO_TOKEN_PATH`, `XERO_CHART_ACCOUNT_TYPES`.

**Tokens live in a file (`XERO_TOKEN_PATH`).** On Render's free tier the disk is ephemeral: every redeploy or restart
loses it, and posting stops with "Xero is not connected" until the one-time consent is repeated. The consent script
needs a browser and a local redirect, so repeating it on the host is awkward: use a persistent disk and point
`XERO_TOKEN_PATH` at that mount (authorise once there), or keep Xero posting for local runs on the free tier.

## Not verified live

Nothing above has been run against Xero. Specifically: the exact tax type codes of the UK Demo Company, whether the
default `GET /Invoices` filter includes VOIDED/DELETED bills (a found voided bill is treated as an error needing
manual action), the deep-link format built from the organisation short code, the server's handling of
`Idempotency-Key`, the granular-scope names, and the plan-specific rate limits (taken from documentation).
