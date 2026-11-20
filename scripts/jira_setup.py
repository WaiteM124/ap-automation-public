#!/usr/bin/env python
"""
Set up a Jira Cloud project for the AP Automation POC (one Epic + child Tasks) via the REST API.

SAFE BY DEFAULT: with no flags this is a DRY RUN and makes no write calls. Credentials are read from
environment variables only; the token is never printed, logged or written to disk (it is redacted from
all output, including error messages).

Environment variables
    JIRA_BASE_URL      https://yourco.atlassian.net      (site URL; used for the API in classic-token mode)
    JIRA_EMAIL         the Atlassian account email that owns the token
    JIRA_API_TOKEN     API token from https://id.atlassian.com/manage-profile/security/api-tokens
    JIRA_PROJECT_KEY   e.g. APAUTO (uppercase letters/digits, starts with a letter, 2-10 chars)
    JIRA_CLOUD_ID      optional; REQUIRED for tokens created with scopes ("Create API token with scopes").
                       Calls then go via https://api.atlassian.com/ex/jira/<cloudId>. Find it at
                       https://yourco.atlassian.net/_edge/tenant_info . JIRA_BASE_URL is then only
                       used to print browse links.
    JIRA_PROJECT_NAME  optional, project name when using --create-project (default "AP Automation POC")
    NOTION_PAGE_URL    optional; added as a "Notion plan" line in each ticket's Links section

Run order (PowerShell, current session only; the token is never echoed or stored):
    $env:JIRA_BASE_URL = "https://yourco.atlassian.net"
    $env:JIRA_EMAIL = "you@company.com"
    $env:JIRA_PROJECT_KEY = "APAUTO"
    $t = Read-Host "Jira API token" -AsSecureString
    $env:JIRA_API_TOKEN = [System.Net.NetworkCredential]::new("", $t).Password
    python scripts/jira_setup.py --check      # 1. auth + project visible + issue types (read-only)
    python scripts/jira_setup.py              # 2. dry run: prints exactly what would be created
    python scripts/jira_setup.py --apply      # 3. create the Epic and tickets (idempotent; safe to re-run)
    # optional: --create-project (needs Administer Jira), --transition-done, --show-adf, --no-priority, --task-list
    Remove-Item Env:JIRA_API_TOKEN            # when finished

See docs/JIRA_SETUP.md for details (company-managed vs team-managed, scoped tokens, blocked project creation).
Ticket wording lives in scripts/jira_tickets.json; edit that, not this file.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

DEFAULT_TICKETS_FILE = Path(__file__).with_name("jira_tickets.json")
DEFAULT_OUTPUT_FILE = Path("jira_created.json")
SCOPED_GATEWAY = "https://api.atlassian.com/ex/jira"
PROJECT_KEY_RE = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")
VALID_STATUSES = ("todo", "in_progress", "done")
VALID_PRIORITIES = ("Highest", "High", "Medium", "Low", "Lowest")
STATUS_LABELS = {"todo": "To do", "in_progress": "In progress", "done": "Done"}
# Fields Jira may reject if they are not on the project's create screen; we retry without them.
OPTIONAL_FIELDS = ("priority", "labels")
MAX_429_RETRIES = 5
DEFAULT_PROJECT_TEMPLATE = "com.pyxis.greenhopper.jira:gh-simplified-agility-kanban"  # team-managed Kanban
DEFAULT_PROJECT_TYPE = "software"


# --------------------------------------------------------------------------------------------------
# Output + secret redaction
# --------------------------------------------------------------------------------------------------

_SECRETS: list[str] = []
_AUTH_HEADER_RE = re.compile(r"\b(Basic|Bearer)\s+[A-Za-z0-9+/=._~-]+", re.IGNORECASE)


def register_secrets(*values: str | None) -> None:
    """Remember values that must never appear in output."""
    for v in values:
        if v and v not in _SECRETS:
            _SECRETS.append(v)


def secret_variants(email: str, token: str) -> list[str]:
    """The token plus its base64 'email:token' form (what a Basic auth header would carry)."""
    out = [token]
    if email and token:
        out.append(base64.b64encode(f"{email}:{token}".encode()).decode())
    return [v for v in out if v]


def redact(text: object, secrets: list[str] | None = None) -> str:
    """Replace any known secret (and any Authorization header value) with ***."""
    s = str(text)
    for secret in sorted(secrets if secrets is not None else _SECRETS, key=len, reverse=True):
        if secret:
            s = s.replace(secret, "***")
    return _AUTH_HEADER_RE.sub(lambda m: f"{m.group(1)} ***", s)


def say(message: object = "") -> None:
    print(redact(message), flush=True)


# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------


class ConfigError(Exception):
    pass


@dataclass
class Config:
    base_url: str
    email: str
    token: str
    project_key: str
    cloud_id: str = ""
    project_name: str = "AP Automation POC"
    notion_url: str = ""

    @property
    def api_root(self) -> str:
        """Scoped tokens must use the Atlassian API gateway; classic tokens use the site URL."""
        if self.cloud_id:
            return f"{SCOPED_GATEWAY}/{self.cloud_id}"
        return self.base_url.rstrip("/")

    def browse_url(self, issue_key: str) -> str:
        return f"{self.base_url.rstrip('/')}/browse/{issue_key}" if self.base_url else issue_key


def load_config(env: dict[str, str] | None = None, *, require_credentials: bool = True) -> Config:
    """Read config from environment variables. Error messages mention variable NAMES only, never values."""
    e = os.environ if env is None else env
    get = lambda k: (e.get(k) or "").strip()  # noqa: E731
    cloud_id = get("JIRA_CLOUD_ID")
    base_url = get("JIRA_BASE_URL").rstrip("/")
    required = ["JIRA_EMAIL", "JIRA_API_TOKEN", "JIRA_PROJECT_KEY"]
    if not cloud_id:
        required.append("JIRA_BASE_URL")
    missing = [k for k in required if not get(k)]
    if missing and require_credentials:
        raise ConfigError("Missing environment variable(s): " + ", ".join(missing))
    key = get("JIRA_PROJECT_KEY").upper()
    if key and not PROJECT_KEY_RE.match(key):
        raise ConfigError(
            "JIRA_PROJECT_KEY must start with a letter and contain only uppercase letters/digits (2-10 chars)."
        )
    if base_url and not base_url.lower().startswith("https://"):
        raise ConfigError("JIRA_BASE_URL must start with https:// (credentials are never sent over plain http).")
    cfg = Config(
        base_url=base_url,
        email=get("JIRA_EMAIL"),
        token=get("JIRA_API_TOKEN"),
        project_key=key or "APAUTO",
        cloud_id=cloud_id,
        project_name=get("JIRA_PROJECT_NAME") or "AP Automation POC",
        notion_url=get("NOTION_PAGE_URL"),
    )
    register_secrets(*secret_variants(cfg.email, cfg.token))
    return cfg


# --------------------------------------------------------------------------------------------------
# HTTP layer (the only place that talks to the network)
# --------------------------------------------------------------------------------------------------


class JiraError(Exception):
    def __init__(self, status: int, messages: list[str], errors: dict[str, str] | None = None, hint: str = ""):
        self.status = status
        self.messages = messages
        self.errors = errors or {}
        self.hint = hint
        super().__init__(self.describe())

    def describe(self) -> str:
        parts = list(self.messages) + [f"{k}: {v}" for k, v in self.errors.items()]
        text = f"HTTP {self.status}: " + ("; ".join(parts) if parts else "(no error details from Jira)")
        if self.hint:
            text += f"\n  hint: {self.hint}"
        return redact(text)


def _hint_for(status: int, path: str, scoped: bool) -> str:
    if status == 401:
        return (
            "authentication failed. Check JIRA_EMAIL and JIRA_API_TOKEN. A token created with scopes only works "
            "with JIRA_CLOUD_ID set (api.atlassian.com gateway); a classic token only works with JIRA_BASE_URL."
        )
    if status == 403:
        return "authenticated but not permitted. Your account (or the token's scopes) lacks the needed permission."
    if status == 404 and "/project" in path:
        return "project not found or not visible to this account (Jira returns 404 for both)."
    if status == 410:
        return "this endpoint has been removed from Jira Cloud."
    return ""


def parse_error_body(response: httpx.Response) -> tuple[list[str], dict[str, str]]:
    try:
        body = response.json()
    except ValueError:
        return ([response.text[:300]] if response.text else []), {}
    if not isinstance(body, dict):
        return [str(body)[:300]], {}
    messages = [str(m) for m in body.get("errorMessages", []) or []]
    errors = {str(k): str(v) for k, v in (body.get("errors") or {}).items()}
    if not messages and not errors and body.get("message"):
        messages.append(str(body["message"]))
    return messages, errors


class JiraClient:
    """Thin wrapper over httpx. Non-GET requests are refused unless allow_writes=True."""

    def __init__(
        self,
        cfg: Config,
        *,
        allow_writes: bool = False,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
    ):
        self.cfg = cfg
        self.allow_writes = allow_writes
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=cfg.api_root,
            auth=httpx.BasicAuth(cfg.email, cfg.token),
            headers={"Accept": "application/json"},
            timeout=30.0,
            transport=transport,
            follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: dict[str, Any] | None = None,
        write: bool | None = None,
    ) -> Any:
        """`write` defaults to (method != GET). Read-only POSTs (JQL search) pass write=False."""
        is_write = (method.upper() != "GET") if write is None else write
        if is_write and not self.allow_writes:
            raise RuntimeError(f"Refusing write call {method} {path}: writes are disabled (dry run).")
        attempt = 0
        while True:
            try:
                resp = self._client.request(method, path, json=json_body, params=params)
            except httpx.HTTPError as exc:
                raise JiraError(0, [f"network error: {type(exc).__name__}: {redact(exc)}"]) from None
            if resp.status_code == 429 and attempt < MAX_429_RETRIES:
                attempt += 1
                wait = _retry_after_seconds(resp.headers.get("Retry-After"))
                say(f"  rate limited (429); waiting {wait:.0f}s (retry {attempt}/{MAX_429_RETRIES})")
                self._sleep(wait)
                continue
            if resp.status_code >= 400:
                messages, errors = parse_error_body(resp)
                raise JiraError(
                    resp.status_code, messages, errors, _hint_for(resp.status_code, path, bool(self.cfg.cloud_id))
                )
            if resp.status_code == 204 or not resp.content:
                return None
            try:
                return resp.json()
            except ValueError:
                raise JiraError(resp.status_code, ["response was not valid JSON"]) from None

    # -- read helpers -------------------------------------------------------------------------
    def myself(self) -> dict:
        return self.request("GET", "/rest/api/3/myself")

    def get_project(self, key: str) -> dict | None:
        try:
            return self.request("GET", f"/rest/api/3/project/{key}")
        except JiraError as exc:
            if exc.status == 404:
                return None
            raise

    def search(self, jql: str, fields: list[str], *, max_pages: int = 50, page_size: int = 100) -> tuple[list[dict], bool]:
        """POST /rest/api/3/search/jql (enhanced search, nextPageToken pagination). Returns (issues, truncated)."""
        issues: list[dict] = []
        token: str | None = None
        for _ in range(max_pages):
            body: dict[str, Any] = {"jql": jql, "fields": fields, "maxResults": page_size}
            if token:
                body["nextPageToken"] = token
            data = self.request("POST", "/rest/api/3/search/jql", json_body=body, write=False) or {}
            issues.extend(data.get("issues", []))
            token = data.get("nextPageToken")
            if data.get("isLast", not token) or not token:
                return issues, False
        return issues, True


def _retry_after_seconds(value: str | None, default: float = 5.0, cap: float = 120.0) -> float:
    try:
        return min(max(float(value), 1.0), cap) if value is not None else default
    except ValueError:
        return default


# --------------------------------------------------------------------------------------------------
# Atlassian Document Format (ADF) builders (API v3 requires ADF for description fields)
# --------------------------------------------------------------------------------------------------


def adf_text(text: str, marks: list[dict] | None = None) -> dict:
    node: dict[str, Any] = {"type": "text", "text": text}
    if marks:
        node["marks"] = marks
    return node


def adf_inline(text: str) -> list[dict]:
    """Turn 'see `docs/x.json` now' into text nodes, with backticked spans as inline code."""
    nodes: list[dict] = []
    for i, chunk in enumerate(text.split("`")):
        if chunk:
            nodes.append(adf_text(chunk, [{"type": "code"}] if i % 2 == 1 else None))
    return nodes or [adf_text(" ")]


def adf_paragraph(text: str) -> dict:
    return {"type": "paragraph", "content": adf_inline(text)}


def adf_heading(text: str, level: int = 3) -> dict:
    return {"type": "heading", "attrs": {"level": level}, "content": [adf_text(text)]}


def adf_bullets(items: list[str]) -> dict:
    return {
        "type": "bulletList",
        "content": [{"type": "listItem", "content": [adf_paragraph(i)]} for i in items],
    }


def adf_checklist(items: list[str], *, prefix: str = "ac") -> dict:
    """Checkbox list (ADF taskList). Items are unchecked (TODO)."""
    return {
        "type": "taskList",
        "attrs": {"localId": f"{prefix}-list"},
        "content": [
            {
                "type": "taskItem",
                "attrs": {"localId": f"{prefix}-{n}", "state": "TODO"},
                "content": adf_inline(item),
            }
            for n, item in enumerate(items, start=1)
        ],
    }


def adf_plain_checklist(items: list[str]) -> dict:
    """Fallback if taskList is rejected: bullets prefixed with a ballot box."""
    return adf_bullets([f"\u2610 {i}" for i in items])


def adf_doc(content: list[dict]) -> dict:
    return {"type": "doc", "version": 1, "content": content}


def build_description(
    item: dict,
    *,
    common_links: list[str],
    notion_url: str = "",
    plain_checklist: bool = True,
) -> dict:
    """Build the ADF description for an epic or ticket definition.

    Acceptance criteria default to bullets prefixed with a ballot box. Atlassian's ADF docs list taskList/taskItem
    as Confluence-only (Jira: not supported), and the REST API has been reported to answer 400 INVALID_INPUT for
    them, so real checkboxes are opt-in via --task-list.
    """
    content: list[dict] = []
    status = item.get("status", "todo")
    if status != "todo":
        note = (item.get("status_note") or "").strip()
        line = f"Status at creation: {STATUS_LABELS[status]}."
        content.append(adf_paragraph(f"{line} {note}".strip()))
    for section in item.get("sections", []):
        content.append(adf_heading(section["heading"]))
        for p in section.get("paragraphs", []):
            content.append(adf_paragraph(p))
        if section.get("bullets"):
            content.append(adf_bullets(section["bullets"]))
    criteria = item.get("acceptance_criteria", [])
    if criteria:
        content.append(adf_heading("Acceptance criteria"))
        content.append(adf_plain_checklist(criteria) if plain_checklist else adf_checklist(criteria))
    links = list(common_links) + list(item.get("links", []))
    content.append(adf_heading("Links"))
    link_items = [f"`{p}`" for p in links]
    if link_items:
        content.append(adf_bullets(link_items))
    if notion_url:
        content.append(
            {
                "type": "paragraph",
                "content": [
                    adf_text("Notion plan: "),
                    adf_text(notion_url, [{"type": "link", "attrs": {"href": notion_url}}]),
                ],
            }
        )
    return adf_doc(content)


# --------------------------------------------------------------------------------------------------
# Ticket definitions
# --------------------------------------------------------------------------------------------------


class TicketFileError(Exception):
    pass


def validate_ticket_file(data: Any) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["top level must be a JSON object"]
    if not isinstance(data.get("epic"), dict):
        problems.append("missing 'epic' object")
    tickets = data.get("tickets")
    if not isinstance(tickets, list) or not tickets:
        problems.append("'tickets' must be a non-empty list")
        tickets = []
    common = data.get("common_links", [])
    if not isinstance(common, list) or not all(isinstance(x, str) for x in common):
        problems.append("'common_links' must be a list of strings")

    items: list[tuple[str, Any]] = []
    if isinstance(data.get("epic"), dict):
        items.append(("epic", data["epic"]))
    items.extend((f"tickets[{i}]", t) for i, t in enumerate(tickets))

    seen: dict[str, str] = {}
    for where, t in items:
        if not isinstance(t, dict):
            problems.append(f"{where}: must be an object")
            continue
        summary = t.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            problems.append(f"{where}: 'summary' is required")
        else:
            if len(summary) > 255:
                problems.append(f"{where}: summary longer than Jira's 255-character limit")
            if "\n" in summary:
                problems.append(f"{where}: summary must be a single line")
            norm = summary.strip().lower()
            if norm in seen:
                problems.append(f"{where}: duplicate summary (also {seen[norm]}); idempotency needs unique summaries")
            seen[norm] = where
        status = t.get("status", "todo")
        if status not in VALID_STATUSES:
            problems.append(f"{where}: status {status!r} not one of {list(VALID_STATUSES)}")
        priority = t.get("priority")
        if priority is not None and priority not in VALID_PRIORITIES:
            problems.append(f"{where}: priority {priority!r} not one of {list(VALID_PRIORITIES)}")
        labels = t.get("labels", [])
        if not isinstance(labels, list) or not all(isinstance(x, str) and x and not re.search(r"\s", x) for x in labels):
            problems.append(f"{where}: 'labels' must be a list of non-empty strings without spaces")
        ac = t.get("acceptance_criteria", [])
        if not isinstance(ac, list) or not all(isinstance(x, str) and x.strip() for x in ac):
            problems.append(f"{where}: 'acceptance_criteria' must be a list of non-empty strings")
        links = t.get("links", [])
        if not isinstance(links, list) or not all(isinstance(x, str) for x in links):
            problems.append(f"{where}: 'links' must be a list of strings")
        sections = t.get("sections", [])
        if not isinstance(sections, list):
            problems.append(f"{where}: 'sections' must be a list")
            sections = []
        for j, s in enumerate(sections):
            if not isinstance(s, dict) or not isinstance(s.get("heading"), str) or not s["heading"].strip():
                problems.append(f"{where}.sections[{j}]: needs a 'heading' string")
                continue
            for fld in ("paragraphs", "bullets"):
                v = s.get(fld, [])
                if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
                    problems.append(f"{where}.sections[{j}]: '{fld}' must be a list of non-empty strings")
    return problems


def load_ticket_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise TicketFileError(f"Ticket file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise TicketFileError(f"Ticket file {path} is not valid JSON: {exc}") from None
    problems = validate_ticket_file(data)
    if problems:
        raise TicketFileError(f"Ticket file {path} has problems:\n  - " + "\n  - ".join(problems))
    return data


# --------------------------------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------------------------------


def _norm(summary: str) -> str:
    return " ".join(summary.split()).casefold()


def build_existing_index(issues: list[dict]) -> dict[tuple[str, bool], str]:
    """Map (normalised summary, is_epic) -> issue key from search results."""
    index: dict[tuple[str, bool], str] = {}
    for issue in issues:
        fields = issue.get("fields") or {}
        summary = fields.get("summary")
        key = issue.get("key")
        if not summary or not key:
            continue
        itype = (fields.get("issuetype") or {}).get("name", "")
        index.setdefault((_norm(summary), itype.casefold() == "epic"), key)
    return index


def find_existing(index: dict[tuple[str, bool], str], summary: str, *, is_epic: bool) -> str | None:
    return index.get((_norm(summary), is_epic))


def jql_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


# --------------------------------------------------------------------------------------------------
# Issue-type selection
# --------------------------------------------------------------------------------------------------


@dataclass
class IssueTypes:
    epic: dict
    child: dict
    available: list[dict] = field(default_factory=list)


def pick_issue_types(project: dict | None, child_type_name: str | None = None) -> IssueTypes:
    """Choose the Epic type and a standard child type from the project's issue types."""
    available = list((project or {}).get("issueTypes") or [])
    if not available:  # offline dry run, or scoped token without the field: assume the defaults
        return IssueTypes({"name": "Epic"}, {"name": child_type_name or "Task"}, [])
    standard = [t for t in available if not t.get("subtask")]
    epic = next((t for t in standard if t.get("name", "").casefold() == "epic"), None) or next(
        (t for t in standard if t.get("hierarchyLevel") == 1), None
    )
    if epic is None:
        raise JiraError(0, [f"Project has no Epic issue type (available: {[t.get('name') for t in available]})."])
    wanted = (child_type_name or "Task").casefold()
    child = next((t for t in standard if t.get("name", "").casefold() == wanted), None)
    if child is None and child_type_name is None:
        child = next((t for t in standard if t.get("hierarchyLevel", 0) == 0), None)
    if child is None:
        raise JiraError(0, [f"Child issue type {child_type_name!r} not found (available: {[t.get('name') for t in standard]})."])
    return IssueTypes(epic, child, available)


def _type_ref(t: dict) -> dict:
    return {"id": str(t["id"])} if t.get("id") else {"name": t["name"]}


# --------------------------------------------------------------------------------------------------
# The workflow
# --------------------------------------------------------------------------------------------------


@dataclass
class Options:
    apply: bool = False
    check: bool = False
    create_project: bool = False
    transition_done: bool = False
    show_adf: bool = False
    no_priority: bool = False
    task_list: bool = False
    child_type: str | None = None
    project_type_key: str = DEFAULT_PROJECT_TYPE
    project_template_key: str = DEFAULT_PROJECT_TEMPLATE
    tickets_file: Path = DEFAULT_TICKETS_FILE
    output: Path = DEFAULT_OUTPUT_FILE


@dataclass
class Result:
    key: str
    url: str
    summary: str
    outcome: str  # created | exists | would-create | failed
    note: str = ""


class Creator:
    """Creates issues with graceful fallbacks. Holds state learned from earlier failures."""

    def __init__(self, client: JiraClient, cfg: Config, spec: dict, types: IssueTypes, opts: Options):
        self.client, self.cfg, self.spec, self.types, self.opts = client, cfg, spec, types, opts
        self.drop_fields: set[str] = {"priority"} if opts.no_priority else set()
        self.use_parent = True
        self._epic_link_field: str | None | bool = False  # False = not looked up yet
        self._epic_name_field: str | None | bool = False

    # -- field discovery ----------------------------------------------------------------------
    def _find_field(self, predicate) -> str | None:
        try:
            fields = self.client.request("GET", "/rest/api/3/field") or []
        except JiraError as exc:
            say(f"  (could not list fields: {exc})")
            return None
        for f in fields:
            if predicate(f):
                return f.get("id")
        return None

    def epic_link_field(self) -> str | None:
        if self._epic_link_field is False:
            self._epic_link_field = self._find_field(
                lambda f: str((f.get("schema") or {}).get("custom", "")).endswith("gh-epic-link")
                or f.get("name") == "Epic Link"
            )
        return self._epic_link_field or None

    def epic_name_field(self) -> str | None:
        if self._epic_name_field is False:
            self._epic_name_field = self._find_field(
                lambda f: str((f.get("schema") or {}).get("custom", "")).endswith("gh-epic-label")
                or f.get("name") == "Epic Name"
            )
        return self._epic_name_field or None

    # -- core create with fallbacks -------------------------------------------------------------
    def _fields_for(self, item: dict, issue_type: dict, *, plain_checklist: bool) -> dict:
        labels = ["ap-automation"] + [x for x in item.get("labels", []) if x != "ap-automation"]
        fields: dict[str, Any] = {
            "project": {"key": self.cfg.project_key},
            "summary": item["summary"].strip(),
            "issuetype": _type_ref(issue_type),
            "description": build_description(
                item,
                common_links=self.spec.get("common_links", []),
                notion_url=self.cfg.notion_url,
                plain_checklist=plain_checklist,
            ),
            "labels": labels,
        }
        if item.get("priority"):
            fields["priority"] = {"name": item["priority"]}
        for name in self.drop_fields:
            fields.pop(name, None)
        return fields

    def post_issue(self, item: dict, issue_type: dict, parent_key: str | None) -> str:
        plain = not self.opts.task_list
        for _ in range(6):
            fields = self._fields_for(item, issue_type, plain_checklist=plain)
            if parent_key and self.use_parent:
                fields["parent"] = {"key": parent_key}
            try:
                data = self.client.request("POST", "/rest/api/3/issue", json_body={"fields": fields})
                return data["key"]
            except JiraError as exc:
                if exc.status != 400:
                    raise
                bad = [k for k in exc.errors if k in OPTIONAL_FIELDS and k in fields]
                if bad:
                    for k in bad:
                        self.drop_fields.add(k)
                        say(f"  note: project rejected field '{k}' ({exc.errors[k]}); retrying without it")
                    continue
                if "parent" in exc.errors and "parent" in fields:
                    self.use_parent = False
                    say(f"  note: 'parent' not accepted on create ({exc.errors['parent']}); will link after creation")
                    continue
                if not plain and ("description" in exc.errors or "INVALID_INPUT" in exc.messages):
                    plain = True
                    say("  note: description (taskList checkboxes) rejected; retrying with a plain bullet checklist")
                    continue
                epic_name_err = any("epic name" in v.lower() for v in exc.errors.values())
                if epic_name_err and issue_type is self.types.epic:
                    cf = self.epic_name_field()
                    if cf and cf not in fields:
                        # older company-managed projects require the Epic Name custom field
                        fields[cf] = item["summary"].strip()
                        data = self.client.request("POST", "/rest/api/3/issue", json_body={"fields": fields})
                        return data["key"]
                raise
        raise JiraError(400, ["gave up after repeated field rejections"])

    def link_to_epic(self, child_key: str, epic_key: str) -> str:
        """Fallback linking after creation. Returns '' on success or a message describing the failure."""
        try:
            self.client.request("PUT", f"/rest/api/3/issue/{child_key}", json_body={"fields": {"parent": {"key": epic_key}}})
            return ""
        except JiraError as exc:
            parent_err = exc.describe()
        cf = self.epic_link_field()
        if cf:
            try:
                self.client.request("PUT", f"/rest/api/3/issue/{child_key}", json_body={"fields": {cf: epic_key}})
                return ""
            except JiraError as exc:
                return f"could not link {child_key} to {epic_key}: parent -> {parent_err}; Epic Link ({cf}) -> {exc.describe()}"
        return (
            f"could not link {child_key} to {epic_key} via 'parent' ({parent_err}) and no Epic Link field was found "
            "(Atlassian removed Epic Link from the REST API in Sept 2025). Link it manually: open the issue > Parent."
        )

    def transition_done(self, key: str) -> str:
        try:
            data = self.client.request("GET", f"/rest/api/3/issue/{key}/transitions") or {}
        except JiraError as exc:
            return f"could not read transitions: {exc}"
        options = data.get("transitions", [])
        match = next(
            (t for t in options if ((t.get("to") or {}).get("statusCategory") or {}).get("key") == "done"), None
        ) or next((t for t in options if t.get("name", "").casefold() == "done"), None)
        if not match:
            return "no transition to a Done status is available from the current status; left as-is (noted in description)"
        try:
            self.client.request("POST", f"/rest/api/3/issue/{key}/transitions", json_body={"transition": {"id": match["id"]}})
        except JiraError as exc:
            return f"transition to Done failed: {exc}"
        return ""


def create_project(client: JiraClient, cfg: Config, opts: Options, account_id: str) -> None:
    body = {
        "key": cfg.project_key,
        "name": cfg.project_name,
        "description": "Self-learning AP automation POC (non-PO invoice GL coding).",
        "projectTypeKey": opts.project_type_key,
        "projectTemplateKey": opts.project_template_key,
        "leadAccountId": account_id,
        "assigneeType": "PROJECT_LEAD",
    }
    client.request("POST", "/rest/api/3/project", json_body=body)


def run_check(client: JiraClient, cfg: Config, opts: Options) -> int:
    say(f"Check mode (read-only). API root: {cfg.api_root}")
    try:
        me = client.myself()
    except JiraError as exc:
        say(f"FAIL  auth: {exc}")
        return 1
    say(f"OK    authenticated as {me.get('displayName', '?')} (accountId {me.get('accountId', '?')})")
    try:
        project = client.get_project(cfg.project_key)
    except JiraError as exc:
        say(f"FAIL  project lookup: {exc}")
        return 1
    if project is None:
        say(f"MISSING  project {cfg.project_key} does not exist or is not visible to you.")
        say("         Ask a Jira admin to create an empty project with this key, or use --create-project (needs admin).")
        return 1
    style = "team-managed" if project.get("style") == "next-gen" or project.get("simplified") else "company-managed"
    say(f"OK    project {project.get('key')} - {project.get('name')} ({style})")
    types = project.get("issueTypes") or []
    if types:
        say("      issue types: " + ", ".join(
            f"{t.get('name')}{' (subtask)' if t.get('subtask') else ''}" for t in types
        ))
        try:
            chosen = pick_issue_types(project, opts.child_type)
            say(f"      will use: epic='{chosen.epic['name']}', child='{chosen.child['name']}'")
        except JiraError as exc:
            say(f"WARN  {exc}")
            return 1
    else:
        say("WARN  project response did not list issue types (scoped token?). Defaults Epic/Task will be assumed.")
    say("All checks passed. Next: run without flags for a dry run, then with --apply.")
    return 0


def run_plan(client: JiraClient | None, cfg: Config, spec: dict, opts: Options) -> int:
    apply = opts.apply
    say(f"{'APPLY' if apply else 'DRY RUN (no write calls will be made)'} - project {cfg.project_key}")
    say(f"API root: {cfg.api_root if client else '(credentials not set; offline preview)'}")

    project: dict | None = None
    index: dict[tuple[str, bool], str] = {}
    if client:
        try:
            me = client.myself()
            say(f"auth OK as {me.get('displayName', '?')}")
            project = client.get_project(cfg.project_key)
        except JiraError as exc:
            say(f"ERROR talking to Jira: {exc}")
            return 1
        if project is None:
            if opts.create_project:
                body_desc = f"{opts.project_type_key} / {opts.project_template_key}"
                if apply:
                    say(f"Creating project {cfg.project_key} '{cfg.project_name}' ({body_desc}) ...")
                    try:
                        create_project(client, cfg, opts, me["accountId"])
                        project = client.get_project(cfg.project_key)
                    except JiraError as exc:
                        say(f"ERROR creating project: {exc}")
                        say("Project creation needs the 'Administer Jira' permission and is often blocked in company "
                            "instances. Ask an admin to create an empty project with key "
                            f"{cfg.project_key}, then re-run without --create-project.")
                        return 1
                    say("  project created.")
                else:
                    say(f"WOULD CREATE project {cfg.project_key} '{cfg.project_name}' ({body_desc}), lead = you")
            else:
                say(f"project {cfg.project_key} not found or not visible.")
                say("  Create it in Jira (or ask an admin), or re-run with --create-project (needs admin rights).")
                if apply:
                    return 1
        else:
            say(f"project OK: {project.get('key')} - {project.get('name')}")
        if project is not None:
            try:
                issues, truncated = client.search(
                    f"project = {jql_quote(cfg.project_key)} ORDER BY created DESC", ["summary", "issuetype"]
                )
                index = build_existing_index(issues)
                say(f"found {len(issues)} existing issue(s) in project for the idempotency check"
                    + (" (truncated; very large project)" if truncated else ""))
                if truncated and apply:
                    say("ERROR: project too large to check for duplicates safely; aborting.")
                    return 1
            except JiraError as exc:
                say(f"{'ERROR' if apply else 'WARN'}: duplicate check (JQL search) failed: {exc}")
                if apply:
                    return 1
    else:
        say("WARN: no credentials in environment; skipping existence checks (everything shown as 'would create').")

    try:
        types = pick_issue_types(project, opts.child_type)
    except JiraError as exc:
        say(f"ERROR: {exc}")
        return 1
    say(f"issue types: epic='{types.epic['name']}', child='{types.child['name']}'")
    say("")

    creator = Creator(client, cfg, spec, types, opts) if (client and apply) else None
    results: list[Result] = []
    epic_item = spec["epic"]

    def preview(item: dict, label: str, itype: dict) -> None:
        bits = [f"type={itype['name']}", f"status={item.get('status', 'todo')}"]
        if item.get("priority") and not opts.no_priority:
            bits.append(f"priority={item['priority']}")
        bits.append("labels=" + ",".join(["ap-automation"] + [x for x in item.get("labels", []) if x != "ap-automation"]))
        say(f"  {label}: {item['summary']}")
        say("      " + "  ".join(bits))
        if opts.show_adf:
            doc = build_description(
                item,
                common_links=spec.get("common_links", []),
                notion_url=cfg.notion_url,
                plain_checklist=not opts.task_list,
            )
            say("      ADF description:\n" + json.dumps(doc, indent=2))

    # ---- epic
    epic_key = find_existing(index, epic_item["summary"], is_epic=True)
    if epic_key:
        say(f"EXISTS  epic {epic_key}: {epic_item['summary']} (skipping)")
        results.append(Result(epic_key, cfg.browse_url(epic_key), epic_item["summary"], "exists"))
    elif apply and creator:
        say(f"CREATE  epic: {epic_item['summary']}")
        try:
            epic_key = creator.post_issue(epic_item, types.epic, None)
        except JiraError as exc:
            say(f"ERROR creating epic: {exc}")
            return 1
        say(f"  -> {epic_key}")
        results.append(Result(epic_key, cfg.browse_url(epic_key), epic_item["summary"], "created"))
    else:
        say("WOULD CREATE epic")
        preview(epic_item, "epic", types.epic)
        epic_key = "<new epic>"
        results.append(Result("(new)", "", epic_item["summary"], "would-create"))

    # ---- children
    failures = 0
    for item in spec["tickets"]:
        existing = find_existing(index, item["summary"], is_epic=False)
        if existing:
            say(f"EXISTS  {existing}: {item['summary']} (skipping)")
            results.append(Result(existing, cfg.browse_url(existing), item["summary"], "exists"))
            continue
        if not (apply and creator):
            say(f"WOULD CREATE task, parent={epic_key}")
            preview(item, "task", types.child)
            if item.get("status") == "done":
                say(f"      then {'transition to Done' if opts.transition_done else 'noted as Done in description (use --transition-done to transition)'}")
            results.append(Result("(new)", "", item["summary"], "would-create"))
            continue
        say(f"CREATE  task: {item['summary']}")
        try:
            key = creator.post_issue(item, types.child, epic_key)
        except JiraError as exc:
            say(f"  FAILED: {exc}")
            failures += 1
            results.append(Result("", "", item["summary"], "failed", exc.describe()))
            continue
        note = ""
        if not creator.use_parent:
            msg = creator.link_to_epic(key, epic_key)
            if msg:
                say(f"  WARN: {msg}")
                note = msg
        say(f"  -> {key}")
        if item.get("status") == "done" and opts.transition_done:
            msg = creator.transition_done(key)
            if msg:
                say(f"  note: {msg}")
                note = (note + " " + msg).strip()
            else:
                say("  transitioned to Done")
        results.append(Result(key, cfg.browse_url(key), item["summary"], "created", note))

    # ---- summary
    say("")
    say("Summary")
    for r in results:
        say(f"  [{r.outcome:<12}] {r.key or '-':<10} {r.summary}")
    if apply:
        write_created_file(opts.output, cfg, results)
        say(f"\nWrote {opts.output} (keys and URLs only, no secrets).")
        if failures:
            say(f"{failures} ticket(s) failed; fix the cause and re-run --apply (it skips ones that exist).")
            return 1
    else:
        say("\nDry run only. Nothing was created. Re-run with --apply to create these.")
    return 0


def write_created_file(path: Path, cfg: Config, results: list[Result]) -> None:
    payload = {
        "project": cfg.project_key,
        "site": cfg.base_url,
        "issues": [
            {"key": r.key, "url": r.url, "summary": r.summary, "outcome": r.outcome}
            for r in results
            if r.key
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> Options:
    p = argparse.ArgumentParser(description="Set up the AP Automation POC Jira project (dry run by default).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="perform the writes (default is a dry run)")
    mode.add_argument("--check", action="store_true", help="only verify auth, project visibility and issue types")
    p.add_argument("--create-project", action="store_true", help="create the project if missing (needs Administer Jira)")
    p.add_argument("--transition-done", action="store_true", help="move tickets marked status=done to Done after creating")
    p.add_argument("--show-adf", action="store_true", help="in a dry run, print each ADF description")
    p.add_argument("--no-priority", action="store_true", help="do not set priority on issues")
    p.add_argument("--task-list", action="store_true",
                   help="use ADF taskList checkboxes for acceptance criteria (not officially supported in Jira; "
                        "falls back to bullets if rejected)")
    p.add_argument("--child-type", default=None, help="issue type name for child tickets (default Task)")
    p.add_argument("--project-type-key", default=DEFAULT_PROJECT_TYPE, help="for --create-project (software|business)")
    p.add_argument("--project-template-key", default=DEFAULT_PROJECT_TEMPLATE, help="for --create-project")
    p.add_argument("--tickets-file", type=Path, default=DEFAULT_TICKETS_FILE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE, help="where to save created keys/URLs")
    a = p.parse_args(argv)
    return Options(
        apply=a.apply,
        check=a.check,
        create_project=a.create_project,
        transition_done=a.transition_done,
        show_adf=a.show_adf,
        no_priority=a.no_priority,
        task_list=a.task_list,
        child_type=a.child_type,
        project_type_key=a.project_type_key,
        project_template_key=a.project_template_key,
        tickets_file=a.tickets_file,
        output=a.output,
    )


def main(argv: list[str] | None = None) -> int:
    opts = parse_args(argv)
    needs_creds = opts.apply or opts.check
    try:
        cfg = load_config(require_credentials=needs_creds)
        spec = load_ticket_file(opts.tickets_file)
    except (ConfigError, TicketFileError) as exc:
        say(f"Error: {exc}")
        return 2

    have_creds = bool(cfg.email and cfg.token and (cfg.base_url or cfg.cloud_id))
    client = JiraClient(cfg, allow_writes=opts.apply) if have_creds else None
    try:
        if opts.check:
            assert client is not None
            return run_check(client, cfg, opts)
        return run_plan(client, cfg, spec, opts)
    except KeyboardInterrupt:
        say("Interrupted.")
        return 130
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
