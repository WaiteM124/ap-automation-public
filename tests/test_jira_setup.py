"""Offline tests for scripts/jira_setup.py. The HTTP layer is mocked; nothing here touches the network."""

import copy
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import jira_setup as js  # noqa: E402

TOKEN = "ATATT-super-secret-token-123"
ENV = {
    "JIRA_BASE_URL": "https://example.atlassian.net",
    "JIRA_EMAIL": "me@example.com",
    "JIRA_API_TOKEN": TOKEN,
    "JIRA_PROJECT_KEY": "APAUTO",
}
TICKETS_FILE = Path(__file__).resolve().parents[1] / "scripts" / "jira_tickets.json"


@pytest.fixture(autouse=True)
def _clean_secrets():
    js._SECRETS.clear()
    yield
    js._SECRETS.clear()


@pytest.fixture
def cfg():
    return js.load_config(ENV)


@pytest.fixture
def spec():
    return js.load_ticket_file(TICKETS_FILE)


# ---- a tiny stateful fake of the Jira endpoints we use ---------------------------------------------


class FakeJira:
    def __init__(self, *, existing=None, project_exists=True, reject_fields=(), reject_parent=False, fail_ratelimit=0):
        self.requests: list[tuple[str, str]] = []
        self.issues: list[dict] = list(existing or [])
        self.project_exists = project_exists
        self.reject_fields = set(reject_fields)
        self.reject_parent = reject_parent
        self.ratelimit_left = fail_ratelimit
        self.bodies: list[dict] = []
        self.next_id = 1

    def handler(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        self.requests.append((method, path))
        body = json.loads(request.content) if request.content else None
        if self.ratelimit_left > 0:
            self.ratelimit_left -= 1
            return httpx.Response(429, headers={"Retry-After": "2"}, json={})
        if method == "GET" and path == "/rest/api/3/myself":
            return httpx.Response(200, json={"accountId": "acc-1", "displayName": "Miles"})
        if method == "GET" and path == "/rest/api/3/project/APAUTO":
            if not self.project_exists:
                return httpx.Response(404, json={"errorMessages": ["No project could be found with key 'APAUTO'."]})
            types = [
                {"id": "10000", "name": "Epic", "subtask": False, "hierarchyLevel": 1},
                {"id": "10001", "name": "Task", "subtask": False, "hierarchyLevel": 0},
                {"id": "10002", "name": "Sub-task", "subtask": True, "hierarchyLevel": -1},
            ]
            return httpx.Response(200, json={"key": "APAUTO", "name": "AP", "issueTypes": types})
        if method == "POST" and path == "/rest/api/3/search/jql":
            return httpx.Response(200, json={"issues": copy.deepcopy(self.issues), "isLast": True})
        if method == "POST" and path == "/rest/api/3/issue":
            fields = body["fields"]
            self.bodies.append(body)
            errors = {k: "not on screen" for k in self.reject_fields if k in fields}
            if self.reject_parent and "parent" in fields:
                errors["parent"] = "Parent not allowed"
            if errors:
                return httpx.Response(400, json={"errorMessages": [], "errors": errors})
            key = f"APAUTO-{self.next_id}"
            self.next_id += 1
            itype = "Epic" if fields["issuetype"]["id"] == "10000" else "Task"
            self.issues.append({"key": key, "fields": {"summary": fields["summary"], "issuetype": {"name": itype}}})
            return httpx.Response(201, json={"id": str(self.next_id), "key": key})
        if method == "PUT" and path.startswith("/rest/api/3/issue/"):
            return httpx.Response(204)
        return httpx.Response(500, json={"errorMessages": [f"unhandled {method} {path}"]})


def make_client(cfg, fake, *, apply):
    return js.JiraClient(cfg, allow_writes=apply, transport=httpx.MockTransport(fake.handler), sleep=lambda s: None)


# ---- ADF -------------------------------------------------------------------------------------------


def test_adf_inline_code_marks():
    nodes = js.adf_inline("see `docs/x.json` now")
    assert [n["text"] for n in nodes] == ["see ", "docs/x.json", " now"]
    assert nodes[1]["marks"] == [{"type": "code"}] and "marks" not in nodes[0]


def test_build_description_structure(spec):
    ticket = spec["tickets"][0]
    doc = js.build_description(
        ticket, common_links=spec["common_links"], notion_url="https://notion.so/page", plain_checklist=False
    )
    assert doc["type"] == "doc" and doc["version"] == 1
    types = [n["type"] for n in doc["content"]]
    headings = [n["content"][0]["text"] for n in doc["content"] if n["type"] == "heading"]
    assert "taskList" in types and "bulletList" in types
    assert headings[-2:] == ["Acceptance criteria", "Links"]
    task_items = next(n for n in doc["content"] if n["type"] == "taskList")["content"]
    assert len(task_items) == len(ticket["acceptance_criteria"])
    assert all(t["attrs"]["state"] == "TODO" for t in task_items)
    ids = [t["attrs"]["localId"] for t in task_items]
    assert len(set(ids)) == len(ids)
    assert any("notion.so/page" in json.dumps(n) for n in doc["content"][-1:])
    json.dumps(doc)  # serialisable


def test_notion_line_omitted_when_unset(spec):
    doc = js.build_description(spec["tickets"][0], common_links=spec["common_links"], notion_url="")
    assert "Notion" not in json.dumps(doc)


def test_status_line_for_done_ticket_and_plain_checklist(spec):
    done = next(t for t in spec["tickets"] if t.get("status") == "done")
    doc = js.build_description(done, common_links=[])  # default is the plain ballot-box checklist
    assert "Status at creation: Done." in json.dumps(doc)
    assert "taskList" not in json.dumps(doc)
    assert "\u2610" in json.dumps(doc, ensure_ascii=False)


def test_every_ticket_ends_with_links_section(spec):
    for t in [spec["epic"], *spec["tickets"]]:
        doc = js.build_description(t, common_links=spec["common_links"])
        last_heading = [n for n in doc["content"] if n["type"] == "heading"][-1]
        assert last_heading["content"][0]["text"] == "Links"
        assert "docs/evaluation.json" in json.dumps(doc["content"][-1])


# ---- ticket file validation ------------------------------------------------------------------------


def test_shipped_ticket_file_is_valid(spec):
    assert js.validate_ticket_file(spec) == []
    assert spec["tickets"][0]["summary"].startswith("Client pilot")
    assert len(spec["tickets"]) >= 9  # client pilot + plan steps 1-8, plus any added improvement tickets
    assert any(t.get("status") == "done" for t in spec["tickets"])
    assert any(t.get("status") == "in_progress" for t in spec["tickets"])


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda d: d["tickets"][0].update(summary=""), "'summary' is required"),
        (lambda d: d["tickets"][1].update(summary=d["tickets"][0]["summary"]), "duplicate summary"),
        (lambda d: d["tickets"][0].update(status="blocked"), "status"),
        (lambda d: d["tickets"][0].update(priority="Urgent"), "priority"),
        (lambda d: d["tickets"][0].update(labels=["has space"]), "labels"),
        (lambda d: d["tickets"][0].update(acceptance_criteria="nope"), "acceptance_criteria"),
        (lambda d: d["tickets"][0].update(sections=[{"paragraphs": ["x"]}]), "heading"),
        (lambda d: d["tickets"][0].update(summary="x" * 300), "255"),
        (lambda d: d.pop("epic"), "epic"),
        (lambda d: d.update(tickets=[]), "tickets"),
    ],
)
def test_validation_catches_problems(spec, mutate, fragment):
    bad = copy.deepcopy(spec)
    mutate(bad)
    problems = js.validate_ticket_file(bad)
    assert problems and any(fragment in p for p in problems), problems


def test_load_ticket_file_errors(tmp_path):
    with pytest.raises(js.TicketFileError):
        js.load_ticket_file(tmp_path / "missing.json")
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(js.TicketFileError):
        js.load_ticket_file(p)


# ---- idempotency logic -----------------------------------------------------------------------------


def test_find_existing_matches_exact_summary_normalised():
    issues = [
        {"key": "A-1", "fields": {"summary": "AP  Automation POC", "issuetype": {"name": "Epic"}}},
        {"key": "A-2", "fields": {"summary": "Housekeeping", "issuetype": {"name": "Task"}}},
        {"key": "A-3", "fields": {"summary": "Housekeeping and more", "issuetype": {"name": "Task"}}},
    ]
    idx = js.build_existing_index(issues)
    assert js.find_existing(idx, "ap automation poc", is_epic=True) == "A-1"
    assert js.find_existing(idx, "AP Automation POC", is_epic=False) is None  # a Task with that name is not the Epic
    assert js.find_existing(idx, "Housekeeping", is_epic=False) == "A-2"
    assert js.find_existing(idx, "Housekeeping and", is_epic=False) is None  # no fuzzy matching


def test_jql_quote_escapes():
    assert js.jql_quote('a"b\\c') == '"a\\"b\\\\c"'


def test_second_apply_creates_nothing(cfg, spec, tmp_path):
    fake = FakeJira()
    opts = js.Options(apply=True, output=tmp_path / "created.json")
    assert js.run_plan(make_client(cfg, fake, apply=True), cfg, spec, opts) == 0
    created = [r for r in fake.requests if r == ("POST", "/rest/api/3/issue")]
    assert len(created) == 1 + len(spec["tickets"])
    out = json.loads((tmp_path / "created.json").read_text())
    assert len(out["issues"]) == 1 + len(spec["tickets"])

    before = len(fake.requests)
    assert js.run_plan(make_client(cfg, fake, apply=True), cfg, spec, opts) == 0
    assert not [r for r in fake.requests[before:] if r == ("POST", "/rest/api/3/issue")]
    out2 = json.loads((tmp_path / "created.json").read_text())
    assert {i["outcome"] for i in out2["issues"]} == {"exists"}


def test_created_file_contains_no_secrets(cfg, spec, tmp_path):
    fake = FakeJira()
    opts = js.Options(apply=True, output=tmp_path / "created.json")
    js.run_plan(make_client(cfg, fake, apply=True), cfg, spec, opts)
    text = (tmp_path / "created.json").read_text()
    assert TOKEN not in text and cfg.email not in text
    assert "https://example.atlassian.net/browse/APAUTO-1" in text


def test_children_use_parent_field_and_labels(cfg, spec, tmp_path):
    fake = FakeJira()
    js.run_plan(make_client(cfg, fake, apply=True), cfg, spec, js.Options(apply=True, output=tmp_path / "c.json"))
    epic_body, child_body = fake.bodies[0]["fields"], fake.bodies[1]["fields"]
    assert "parent" not in epic_body
    assert child_body["parent"] == {"key": "APAUTO-1"}
    assert child_body["issuetype"] == {"id": "10001"} and epic_body["issuetype"] == {"id": "10000"}
    assert "ap-automation" in child_body["labels"] and "client" in child_body["labels"]
    assert child_body["description"]["type"] == "doc"


# ---- graceful fallbacks ----------------------------------------------------------------------------


def test_priority_and_labels_rejection_retries_without(cfg, spec, tmp_path):
    fake = FakeJira(reject_fields={"priority"})
    rc = js.run_plan(make_client(cfg, fake, apply=True), cfg, spec, js.Options(apply=True, output=tmp_path / "c.json"))
    assert rc == 0
    assert "priority" not in fake.bodies[-1]["fields"]
    assert len([i for i in fake.issues]) == 1 + len(spec["tickets"])


def test_default_description_has_no_tasklist_and_flag_falls_back_on_invalid_input(cfg, spec, tmp_path):
    fake = FakeJira()
    js.run_plan(make_client(cfg, fake, apply=True), cfg, spec, js.Options(apply=True, output=tmp_path / "c.json"))
    assert all("taskList" not in json.dumps(b) for b in fake.bodies)

    class RejectTaskList(FakeJira):
        def handler(self, request):
            if request.method == "POST" and request.url.path == "/rest/api/3/issue" and b"taskList" in request.content:
                self.requests.append(("POST", "/rest/api/3/issue"))
                return httpx.Response(400, json={"errorMessages": ["INVALID_INPUT"], "errors": {}})
            return super().handler(request)

    fake2 = RejectTaskList()
    opts = js.Options(apply=True, task_list=True, output=tmp_path / "c2.json")
    assert js.run_plan(make_client(cfg, fake2, apply=True), cfg, spec, opts) == 0
    assert len(fake2.issues) == 1 + len(spec["tickets"])
    assert all("taskList" not in json.dumps(b) for b in fake2.bodies)


def test_parent_rejected_falls_back_to_link_after_create(cfg, spec, tmp_path):
    fake = FakeJira(reject_parent=True)
    rc = js.run_plan(make_client(cfg, fake, apply=True), cfg, spec, js.Options(apply=True, output=tmp_path / "c.json"))
    assert rc == 0
    puts = [r for r in fake.requests if r[0] == "PUT"]
    assert len(puts) == len(spec["tickets"])


def test_rate_limit_honours_retry_after(cfg):
    fake = FakeJira(fail_ratelimit=2)
    sleeps: list[float] = []
    client = js.JiraClient(cfg, transport=httpx.MockTransport(fake.handler), sleep=sleeps.append)
    assert client.myself()["accountId"] == "acc-1"
    assert sleeps == [2.0, 2.0]


def test_http_error_shows_status_and_jira_messages_but_no_headers_or_token(cfg):
    def handler(request):
        return httpx.Response(400, json={"errorMessages": ["Bad thing"], "errors": {"summary": "too long"}})

    client = js.JiraClient(cfg, transport=httpx.MockTransport(handler))
    with pytest.raises(js.JiraError) as ei:
        client.myself()
    text = str(ei.value)
    assert "HTTP 400" in text and "Bad thing" in text and "summary: too long" in text
    assert TOKEN not in text and "Authorization" not in text and "Basic" not in text


def test_401_hint_mentions_scoped_tokens(cfg):
    client = js.JiraClient(cfg, transport=httpx.MockTransport(lambda r: httpx.Response(401, json={})))
    with pytest.raises(js.JiraError) as ei:
        client.myself()
    assert "JIRA_CLOUD_ID" in str(ei.value)


# ---- dry run makes zero write calls ----------------------------------------------------------------


def test_dry_run_makes_no_writes(cfg, spec, tmp_path, capsys):
    fake = FakeJira()
    opts = js.Options(apply=False, output=tmp_path / "c.json")
    assert js.run_plan(make_client(cfg, fake, apply=False), cfg, spec, opts) == 0
    writes = [r for r in fake.requests if r[0] != "GET" and r != ("POST", "/rest/api/3/search/jql")]
    assert writes == []
    assert not (tmp_path / "c.json").exists()
    out = capsys.readouterr().out
    assert "WOULD CREATE epic" in out and out.count("WOULD CREATE task") == len(spec["tickets"])
    assert TOKEN not in out


def test_dry_run_shows_existing_as_skipped(cfg, spec, capsys):
    existing = [{"key": "APAUTO-7", "fields": {"summary": spec["tickets"][0]["summary"], "issuetype": {"name": "Task"}}}]
    fake = FakeJira(existing=existing)
    js.run_plan(make_client(cfg, fake, apply=False), cfg, spec, js.Options())
    assert "EXISTS  APAUTO-7" in capsys.readouterr().out


def test_dry_run_with_create_project_does_not_post(cfg, spec, capsys):
    fake = FakeJira(project_exists=False)
    js.run_plan(make_client(cfg, fake, apply=False), cfg, spec, js.Options(create_project=True))
    assert ("POST", "/rest/api/3/project") not in fake.requests
    assert "WOULD CREATE project APAUTO" in capsys.readouterr().out


def test_client_refuses_writes_when_not_allowed(cfg):
    client = js.JiraClient(cfg, allow_writes=False, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    with pytest.raises(RuntimeError):
        client.request("POST", "/rest/api/3/issue", json_body={})
    with pytest.raises(RuntimeError):
        client.request("PUT", "/rest/api/3/issue/X-1", json_body={})


def test_offline_dry_run_without_credentials(spec, capsys):
    cfg = js.load_config({"JIRA_PROJECT_KEY": "APAUTO"}, require_credentials=False)
    assert js.run_plan(None, cfg, spec, js.Options()) == 0
    assert "offline preview" in capsys.readouterr().out


def test_apply_creates_project_when_requested(cfg, spec, tmp_path):
    fake = FakeJira(project_exists=False)
    original = fake.handler

    def handler(request):
        if request.method == "POST" and request.url.path == "/rest/api/3/project":
            fake.requests.append(("POST", "/rest/api/3/project"))
            fake.project_exists = True
            fake.project_body = json.loads(request.content)
            return httpx.Response(201, json={"key": "APAUTO"})
        return original(request)

    client = js.JiraClient(cfg, allow_writes=True, transport=httpx.MockTransport(handler), sleep=lambda s: None)
    assert js.run_plan(client, cfg, spec, js.Options(apply=True, create_project=True, output=tmp_path / "c.json")) == 0
    assert fake.project_body["leadAccountId"] == "acc-1" and fake.project_body["key"] == "APAUTO"


def test_apply_without_project_and_no_create_flag_fails(cfg, spec, tmp_path):
    fake = FakeJira(project_exists=False)
    rc = js.run_plan(make_client(cfg, fake, apply=True), cfg, spec, js.Options(apply=True, output=tmp_path / "c.json"))
    assert rc == 1 and ("POST", "/rest/api/3/issue") not in fake.requests


# ---- transition to done ----------------------------------------------------------------------------


def test_transition_done_picks_done_category(cfg, spec):
    posted = {}

    def handler(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"transitions": [
                    {"id": "11", "name": "Start", "to": {"statusCategory": {"key": "indeterminate"}}},
                    {"id": "31", "name": "Finish", "to": {"statusCategory": {"key": "done"}}},
                ]},
            )
        posted.update(json.loads(request.content))
        return httpx.Response(204)

    client = js.JiraClient(cfg, allow_writes=True, transport=httpx.MockTransport(handler))
    creator = js.Creator(client, cfg, spec, js.pick_issue_types(None), js.Options())
    assert creator.transition_done("APAUTO-3") == ""
    assert posted == {"transition": {"id": "31"}}


def test_transition_done_missing_returns_note(cfg, spec):
    client = js.JiraClient(
        cfg, allow_writes=True, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"transitions": []}))
    )
    creator = js.Creator(client, cfg, spec, js.pick_issue_types(None), js.Options())
    assert "no transition" in creator.transition_done("APAUTO-3")


# ---- secrets / config ------------------------------------------------------------------------------


def test_redact_removes_token_and_basic_header(cfg):
    import base64

    b64 = base64.b64encode(f"{cfg.email}:{TOKEN}".encode()).decode()
    text = f"oops token={TOKEN} header=Authorization: Basic {b64} and Bearer abc.def-123"
    out = js.redact(text)
    assert TOKEN not in out and b64 not in out and "abc.def-123" not in out
    assert out.count("***") >= 3


def test_load_config_registers_secret_and_never_echoes_values():
    js.load_config(ENV)
    assert TOKEN in js._SECRETS


def test_missing_env_reports_names_only():
    with pytest.raises(js.ConfigError) as ei:
        js.load_config({"JIRA_API_TOKEN": TOKEN})
    msg = str(ei.value)
    assert "JIRA_EMAIL" in msg and "JIRA_PROJECT_KEY" in msg and "JIRA_BASE_URL" in msg
    assert TOKEN not in msg


def test_config_validation():
    with pytest.raises(js.ConfigError):
        js.load_config({**ENV, "JIRA_PROJECT_KEY": "1bad"})
    with pytest.raises(js.ConfigError):
        js.load_config({**ENV, "JIRA_BASE_URL": "http://example.atlassian.net"})


def test_scoped_mode_uses_gateway_and_does_not_need_base_url():
    env = {k: v for k, v in ENV.items() if k != "JIRA_BASE_URL"} | {"JIRA_CLOUD_ID": "abc-123"}
    c = js.load_config(env)
    assert c.api_root == "https://api.atlassian.com/ex/jira/abc-123"
    assert js.load_config(ENV).api_root == "https://example.atlassian.net"


def test_main_error_output_never_contains_token(monkeypatch, capsys, tmp_path):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    bad = tmp_path / "t.json"
    bad.write_text("{}", encoding="utf-8")
    assert js.main(["--tickets-file", str(bad)]) == 2
    captured = capsys.readouterr()
    assert TOKEN not in captured.out + captured.err


def test_apply_and_check_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        js.parse_args(["--apply", "--check"])


def test_check_mode_reports_issue_types(cfg, capsys):
    fake = FakeJira()
    rc = js.run_check(make_client(cfg, fake, apply=False), cfg, js.Options(check=True))
    out = capsys.readouterr().out
    assert rc == 0 and "Epic" in out and "Task" in out and "Sub-task (subtask)" in out
    assert all(r[0] == "GET" for r in fake.requests)


def test_check_mode_missing_project(cfg, capsys):
    fake = FakeJira(project_exists=False)
    assert js.run_check(make_client(cfg, fake, apply=False), cfg, js.Options(check=True)) == 1
    assert "--create-project" in capsys.readouterr().out
