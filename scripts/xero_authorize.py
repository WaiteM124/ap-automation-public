"""
One-time Xero consent: connect this app to YOUR Xero organisation (for example the Demo Company).

    python scripts/xero_authorize.py              # opens your browser, listens for the redirect, stores the tokens
    python scripts/xero_authorize.py --no-browser # prints the consent URL instead of opening it
    python scripts/xero_authorize.py --tenant-id <id>   # pick an organisation when several were connected

Needs XERO_CLIENT_ID and XERO_CLIENT_SECRET in .env (see docs/XERO_POSTING.md) and the redirect URI
(XERO_REDIRECT_URI, default http://localhost:8765/callback) registered on your Xero app exactly as written.

What it does: builds the consent URL with a random `state`, waits on a local port for Xero's redirect, checks the state,
exchanges the one-time code for tokens (HTTP Basic client auth), stores them atomically in XERO_TOKEN_PATH
(data/xero_tokens.json, git-ignored, owner-only permissions where the OS allows), looks up the organisation with
GET /connections and remembers its id and name. Tokens and the client secret are never printed or logged.
"""

from __future__ import annotations

import argparse
import hmac
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ap_automation.core import xero_auth  # noqa: E402
from ap_automation.core.config import settings  # noqa: E402

PAGE = ("<html><body style='font-family:sans-serif;max-width:32em;margin:3em auto'><h2>{title}</h2>"
        "<p>{body}</p></body></html>")


class _Result:
    code: str | None = None
    error: str | None = None
    done = threading.Event()


def _make_handler(path: str, state: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server API)
            url = urlparse(self.path)
            if url.path != path:
                self.send_response(404)
                self.end_headers()
                return
            q = parse_qs(url.query)
            if "error" in q:
                _Result.error = f"Xero reported: {q['error'][0][:80]}"
            elif not hmac.compare_digest(q.get("state", [""])[0].encode("utf-8"), state.encode("utf-8")):
                _Result.error = "The state value did not match (possible forged redirect); nothing was stored"
            elif not q.get("code"):
                _Result.error = "The redirect carried no authorization code"
            else:
                _Result.code = q["code"][0]
            ok = _Result.error is None
            body = PAGE.format(title="Xero connected" if ok else "Xero connection failed",
                               body="You can close this tab and return to the terminal." if ok else _Result.error)
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            _Result.done.set()

        def log_message(self, *args):  # the request line contains the one-time code: never log it
            return

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-browser", action="store_true", help="print the consent URL instead of opening a browser")
    ap.add_argument("--timeout", type=int, default=300, help="seconds to wait for the redirect (default 300)")
    ap.add_argument("--tenant-id", default="", help="organisation to use when more than one was connected")
    args = ap.parse_args()

    if not xero_auth.has_client_credentials():
        print("XERO_CLIENT_ID and XERO_CLIENT_SECRET are not set. Put them in .env (see docs/XERO_POSTING.md).")
        return 2
    redirect = urlparse(settings.xero_redirect_uri)
    if redirect.scheme != "http" or redirect.hostname not in ("localhost", "127.0.0.1") or not redirect.port:
        print("This helper listens locally, so XERO_REDIRECT_URI must look like http://localhost:8765/callback "
              "and be registered on your Xero app exactly as written.")
        return 2
    path = redirect.path or "/"

    state = secrets.token_urlsafe(32)
    try:
        server = HTTPServer(("127.0.0.1", redirect.port), _make_handler(path, state))
    except OSError as e:
        print(f"Cannot listen on port {redirect.port}: {e}. Change XERO_REDIRECT_URI (and the app's redirect URI).")
        return 2
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = xero_auth.authorize_url(state)
    print("Sign in to Xero and choose the organisation to connect (for a demo: 'Demo Company').")
    if args.no_browser or not webbrowser.open(url):
        print("Open this URL in your browser:\n  " + url)
    print(f"Waiting up to {args.timeout}s for the redirect to {settings.xero_redirect_uri} ...")
    got = _Result.done.wait(args.timeout)
    server.shutdown()
    if not got:
        print("Timed out waiting for the redirect. Run the script again.")
        return 1
    if _Result.error or not _Result.code:
        print(_Result.error or "No authorization code received.")
        return 1

    try:
        tokens = xero_auth.exchange_code(_Result.code)
        orgs = xero_auth.list_connections(tokens["access_token"], xero_auth.auth_event_id(tokens["access_token"]))
    except xero_auth.XeroError as e:
        print(f"Could not complete the connection: {xero_auth.sanitize(e)}")
        return 1

    if args.tenant_id:
        orgs = [o for o in orgs if o["tenantId"] == args.tenant_id]
    if not orgs:
        print("The connection succeeded but no Xero organisation was connected. Run the script again and tick one.")
        return 1
    if len(orgs) > 1:
        print("More than one organisation was connected. Run again with --tenant-id <id> to choose one:")
        for o in orgs:
            print(f"  {o['tenantId']}  {o.get('tenantName') or '(unnamed)'}")
        return 1

    org = orgs[0]
    xero_auth.update_connection(org["tenantId"], org.get("tenantName") or "")
    name = org.get("tenantName") or "(unnamed)"
    print(f"Connected to Xero organisation: {name}")
    if name.lower().startswith("demo company"):
        print("This is the Xero Demo Company: it resets every 28 days and gets a NEW tenant id; "
              "re-run this script after each reset.")
    print(f"Tokens stored in {settings.xero_token_path} (git-ignored). Next: python scripts/xero_sync_chart.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
