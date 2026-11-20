# Jira setup for the AP Automation POC

`scripts/jira_setup.py` creates one Epic ("AP Automation POC") and nine child Tasks in a Jira Cloud project.
Ticket wording lives in `scripts/jira_tickets.json` (edit that, not the code). The script is a **dry run by
default**, idempotent (re-running skips anything that already exists), and never prints or stores your token.

## 1. Get an API token

1. Go to <https://id.atlassian.com/manage-profile/security/api-tokens>.
2. Either:
   - **Create API token** (classic, no scopes): simplest. Works against `https://yourco.atlassian.net`.
   - **Create API token with scopes**: more secure, but you must also set `JIRA_CLOUD_ID` (see "Scoped tokens").
3. Copy the token once (Atlassian will not show it again). Never paste it into chat, a file or a commit.

## 2. Set environment variables (current PowerShell session only)

```powershell
$env:JIRA_BASE_URL    = "https://yourco.atlassian.net"
$env:JIRA_EMAIL       = "you@company.com"
$env:JIRA_PROJECT_KEY = "APAUTO"          # uppercase letters/digits, starts with a letter, 2-10 chars

# Prompt for the token without echoing it or saving it anywhere:
$t = Read-Host "Jira API token" -AsSecureString
$env:JIRA_API_TOKEN = [System.Net.NetworkCredential]::new("", $t).Password
Remove-Variable t

# optional
$env:NOTION_PAGE_URL = "https://www.notion.so/..."   # adds a "Notion plan" line to each ticket
```

These variables disappear when you close the window. To clear the token sooner: `Remove-Item Env:JIRA_API_TOKEN`.
Do not put the token in `.env` (that file is for the app, and the script deliberately does not read it).

## 3. Run order

```powershell
python scripts/jira_setup.py --check     # 1. read-only: auth OK? project visible? which issue types?
python scripts/jira_setup.py             # 2. dry run: prints exactly what would be created, makes no writes
python scripts/jira_setup.py --apply     # 3. creates the Epic + tickets, saves keys/URLs to jira_created.json
```

Useful extras: `--show-adf` (dry run: print the exact description JSON), `--transition-done` (move the
embeddings ticket to Done after creating it), `--no-priority`, `--child-type Story`,
`--tickets-file other.json`. Without credentials set, the plain dry run still works as an offline preview.

`jira_created.json` holds issue keys and URLs only (no secrets) and is git-ignored.

## 4. Project must exist first (or use `--create-project`)

- Easiest: create an empty project in Jira yourself with the key you chose (Jira Software, Kanban or Scrum).
- `--create-project` calls `POST /rest/api/3/project` (team-managed Kanban template by default, you as lead).
  It needs the **Administer Jira** global permission. Many company instances restrict this.
- **If creation is blocked** (403, or you are not an admin): ask a Jira admin to create an *empty* project with
  your key and add you with permission to create issues, then run `--check` and continue without `--create-project`.
- Override the template with `--project-template-key` / `--project-type-key` if needed (for example
  `com.pyxis.greenhopper.jira:gh-simplified-kanban-classic` for company-managed).

## 5. Scoped tokens

A token created with scopes is rejected on your normal site URL (it can look like an anonymous response or a 401).
It must go through the Atlassian gateway:

```powershell
$env:JIRA_CLOUD_ID = "<your cloud id>"   # open https://yourco.atlassian.net/_edge/tenant_info
```

With `JIRA_CLOUD_ID` set, the script uses `https://api.atlassian.com/ex/jira/<cloudId>/...`. `JIRA_BASE_URL` is
then optional and only used to print browse links. Authentication is still Basic (`email:token`). Select at least
these scopes when creating the token: `read:jira-work`, `write:jira-work`, `read:jira-user` (for `/myself`), and
for `--create-project` also `manage:jira-configuration`. If a call returns 401/403 with a scoped token, a missing
scope is the most likely cause; scopes cannot be edited, create a new token.

## 6. Company-managed vs team-managed

- Children are linked to the Epic with the modern **`parent`** field, which works for both project styles.
  The old `Epic Link` custom field was removed from the REST API by Atlassian (announced for 13 Sep 2025).
- If a project rejects `parent` on create, the script creates the ticket anyway, then tries `PUT` with `parent`,
  then (if such a field still exists) the `Epic Link` custom field, and otherwise prints a clear message so you
  can link by hand.
- If the project does not have the Priority or Labels field on its create screen, the script drops that field
  and retries (it says so). Older company-managed projects that require **Epic Name** are handled by discovering
  the field via `GET /rest/api/3/field`.
- Descriptions are Atlassian Document Format (required by API v3). Acceptance criteria are rendered as a
  bullet checklist ("☐ ..."). Real checkbox task lists (`--task-list`) are not officially supported in Jira's
  ADF, so they are opt-in and fall back to bullets if rejected.

## 7. How idempotency works

Before creating anything the script runs a JQL search (`POST /rest/api/3/search/jql`, the current endpoint; the old
`/rest/api/3/search` has been removed) for `project = KEY`, pages through all results with `nextPageToken`, and matches
**exact summaries** (case and whitespace insensitive) client-side. A summary that already exists is skipped. Epics
only match Epics. Consequences:

- Editing a summary in `jira_tickets.json` and re-running will create a *new* issue (the old one is untouched).
- Jira's search index is eventually consistent: wait a minute before re-running if the first run was interrupted.
- Projects with more than about 5,000 issues are refused in `--apply` mode rather than risk duplicates.

## 8. Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| 401 | wrong email/token, or scoped token without `JIRA_CLOUD_ID` (or classic token with it set) |
| 403 | account lacks Create Issues / Administer Jira, or token scope missing |
| 404 on project | key wrong, or project not visible to you |
| 400 with field names | the project's screens differ; the script retries without priority/labels/parent automatically |
| 429 | rate limited; the script waits for `Retry-After` and retries |
| 410 on search | an old endpoint was used; this script uses the current one |
