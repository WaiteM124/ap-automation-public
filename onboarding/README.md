# Onboarding templates

Two CSV templates for loading a new client. The example rows are fictional; replace them with the client's own data
and keep the header row exactly as it is.

1. `chart_of_accounts_template.csv` - the client's GL codes. Upload it first (Import & chart page, or `POST /chart`).
2. `invoice_history_template.csv` - previously coded invoices, one line per row. Upload it second
   (Import & chart page, or `POST /corpus/import`). Run it as a dry run first.

## Columns

Chart of accounts: `code`, `name`.

Invoice history:

| Column | Notes |
|---|---|
| `vendor_name` | The vendor as it appears on the invoice. |
| `description` | A meaningful line description. Generic text such as "Invoice" or "Services" lowers accuracy. |
| `amount` | A positive number. |
| `currency` | For example GBP. |
| `entity_id` | The client's entity code. |
| `cost_centre` | Optional. |
| `gl_account` | Must be a code in the uploaded chart of accounts. |

CSV has no comment syntax, so do not add notes inside the files: any extra row would be read as data.

Aim for at least 3 lines per vendor, a consistent GL code for the same kind of invoice, and recent history.
`POST /readiness` (or "Check data readiness" on the Import & chart page) scores a history file before you load it.
