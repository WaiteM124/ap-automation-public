"""
Xero OAuth 2.0: consent URL, code exchange, token storage and refresh.

What Xero requires (checked against developer.xero.com docs and the published OpenAPI spec, October 2026):

* Authorization-code flow. Authorize at login.xero.com/identity/connect/authorize, exchange the code at
  identity.xero.com/connect/token with HTTP Basic client authentication (client id : client secret).
* `offline_access` is what yields a refresh token. Access tokens last 30 minutes.
* REFRESH TOKENS ARE SINGLE USE. A refresh returns a NEW refresh token; the old one is accepted for a further
  30-minute grace period (so a lost response can be retried) and is then dead. An unused refresh token expires after
  60 days. Whoever fails to store the new refresh token loses the connection.

How this module keeps the connection safe:

* The new token set is written to disk IMMEDIATELY after the token endpoint answers, before anything else happens, by
  write-to-temp + flush + fsync + atomic rename. A crash can therefore only leave the OLD complete file or the NEW
  complete file - never a half-written one - and the old refresh token still works for 30 minutes after it was used.
* If the token endpoint's answer is lost (timeout), the refresh is retried with the SAME old token (allowed during the
  grace period) instead of starting a new rotation.
* If the disk write fails (read-only / full disk) the new tokens are kept in memory for this process, the write is
  retried on the next use, and the failure is reported (no token in the message).
* One refresh at a time: an in-process lock plus a lock file for other processes. After taking the lock the token file
  is re-read, because another worker may already have refreshed.
* The token file is created with owner-only permissions where the OS allows (0600 on POSIX; on Windows the inherited
  ACL is replaced by one that grants only the current user). It lives under data/ and is git-ignored.

Secrets (client secret, access / refresh tokens) are removed from everything this module returns, logs or raises.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

from ap_automation.core.config import settings

AUTHORIZE_URL = "https://login.xero.com/identity/connect/authorize"
TOKEN_URL = "https://identity.xero.com/connect/token"
CONNECTIONS_URL = "https://api.xero.com/connections"

REFRESH_MARGIN_SECONDS = 120        # refresh a little before the 30-minute access token really expires
LOCK_STALE_SECONDS = 60
MAX_ERROR_CHARS = 600

_transport: httpx.BaseTransport | None = None     # tests install httpx.MockTransport here
_sleep = time.sleep
HARDEN_WINDOWS_ACL = True                          # tests switch the (slow) icacls call off

_thread_lock = threading.RLock()
_unsaved: dict | None = None        # newest token set that could not be written to disk yet
_seen_secrets: set[str] = set()     # every token this process has handled, so sanitize() can always strip them


def set_transport(transport: httpx.BaseTransport | None) -> None:
    """Route every Xero request (identity, connections, accounting) through `transport` (tests)."""
    global _transport
    _transport = transport


def transport() -> httpx.BaseTransport | None:
    return _transport


def reset_state() -> None:
    """Forget in-memory token state (tests, and after a deliberate re-authorisation)."""
    global _unsaved
    _unsaved = None
    _seen_secrets.clear()


class XeroError(Exception):
    """
    A Xero operation could not be completed.
    kind: config | auth | mapping | rejected | unavailable | rate_limited | unknown_outcome
    """

    def __init__(self, message: str, kind: str, http_status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.http_status = http_status


# --------------------------------------------------------------------------- sanitising

def _remember(*values: object) -> None:
    for v in values:
        if isinstance(v, str) and len(v) >= 8:
            _seen_secrets.add(v)


def sanitize(text: object, extra_secrets: tuple[str, ...] = ()) -> str:
    """Make `text` safe to store, log and return: secrets and tokens removed, one line, bounded length."""
    s = str(text)
    known = {settings.xero_client_secret, *_seen_secrets, *extra_secrets}
    for secret in sorted((k for k in known if k and len(k) >= 4), key=len, reverse=True):
        s = s.replace(secret, "***")
    s = re.sub(r'(?i)"(access_token|refresh_token|id_token|client_secret)"\s*:\s*"[^"]*"',
               r'"\1":"***"', s)
    s = re.sub(r"(?i)\b(access_token|refresh_token|id_token|client_secret|authorization|cookie|password)"
               r"\s*[:=]\s*(Bearer\s+|Basic\s+)?[^\s,;&\"']+", r"\1=***", s)
    s = re.sub(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{8,}", r"\1 ***", s)
    s = re.sub(r"eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*", "***", s)    # JWTs (Xero access tokens)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= MAX_ERROR_CHARS else s[: MAX_ERROR_CHARS - 3] + "..."


# --------------------------------------------------------------------------- token file

def token_path() -> Path:
    return Path(settings.xero_token_path)


def _harden(path: Path) -> None:
    """Owner-only permissions, as tight as the OS allows. Never raises."""
    try:
        if os.name == "nt":
            user = os.environ.get("USERNAME")
            if HARDEN_WINDOWS_ACL and user:
                subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:(F)"],
                               capture_output=True, timeout=15, check=False)
        else:
            os.chmod(path, 0o600)
    except Exception:
        pass


def _atomic_write(path: Path, text: str) -> None:
    """Write `text` to `path` so that a crash at any point leaves either the old or the new complete file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".xero_tokens.", suffix=".tmp")    # created 0600 on POSIX
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        _harden(tmp_path)
        os.replace(tmp_path, path)
        if os.name != "nt":                      # make the rename itself durable
            try:
                dfd = os.open(str(path.parent), os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except OSError:
                pass
    except BaseException:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def _read_file() -> dict | None:
    path = token_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data.get("refresh_token"):
            raise ValueError("no refresh token")
    except (OSError, ValueError):
        raise XeroError("The Xero token file is unreadable; run scripts/xero_authorize.py again", "auth") from None
    _remember(data.get("access_token"), data.get("refresh_token"))
    return data


def load_tokens() -> dict | None:
    """The newest known token set (an unsaved one wins), or None when this app has never been connected."""
    with _thread_lock:
        if _unsaved is not None:
            _flush_unsaved()
        return _unsaved if _unsaved is not None else _read_file()


def save_tokens(data: dict) -> bool:
    """
    Persist `data` atomically. Returns False if the disk write failed (the set is then held in memory and retried on
    next use). Never includes a token in an exception message.
    """
    global _unsaved
    data = {**data, "saved_at": time.time()}
    _remember(data.get("access_token"), data.get("refresh_token"))
    with _thread_lock:
        try:
            _atomic_write(token_path(), json.dumps(data, indent=1))
        except OSError as e:
            _unsaved = data
            print(f"[Xero] WARNING: could not write the token file ({type(e).__name__}); keeping the new tokens in "
                  "memory and retrying on next use. If the process stops now the connection must be re-authorised.")
            return False
        _unsaved = None
        return True


def _flush_unsaved() -> None:
    global _unsaved
    if _unsaved is None:
        return
    try:
        _atomic_write(token_path(), json.dumps(_unsaved, indent=1))
        _unsaved = None
    except OSError:
        pass


def update_connection(tenant_id: str, tenant_name: str = "", short_code: str | None = None) -> None:
    """Remember which organisation is connected (and its deep-link short code) in the token file."""
    with _thread_lock:
        data = load_tokens()
        if not data:
            return
        changed = data.get("tenant_id") != tenant_id
        data["tenant_id"] = tenant_id
        data["tenant_name"] = tenant_name or ("" if changed else data.get("tenant_name", ""))
        if changed:
            data["short_code"] = ""
        if short_code is not None:
            data["short_code"] = short_code
        save_tokens(data)


# --------------------------------------------------------------------------- the lock around a refresh

class _RefreshLock:
    """In-process lock + lock file so two processes never rotate the single-use refresh token at the same time."""

    def __init__(self, timeout: float = 30.0):
        self.path = Path(str(token_path()) + ".lock")
        self.timeout = timeout
        self._held = False

    def __enter__(self):
        _thread_lock.acquire()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self.timeout
        while True:
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                self._held = True
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > LOCK_STALE_SECONDS:
                        self.path.unlink()      # the holder died
                        continue
                except OSError:
                    continue
                if time.time() > deadline:
                    _thread_lock.release()
                    raise XeroError("Another process is refreshing the Xero token; try again shortly",
                                    "unavailable") from None
                _sleep(0.2)
            except OSError:
                self._held = False              # cannot create a lock file (read-only disk): rely on the thread lock
                return self

    def __exit__(self, *exc):
        try:
            if self._held:
                self.path.unlink()
        except OSError:
            pass
        _thread_lock.release()


# --------------------------------------------------------------------------- OAuth calls

def has_client_credentials() -> bool:
    return bool(settings.xero_client_id and settings.xero_client_secret)


def _http() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(settings.xero_timeout_seconds), transport=_transport,
                        follow_redirects=False)


def authorize_url(state: str) -> str:
    return AUTHORIZE_URL + "?" + urlencode({
        "response_type": "code", "client_id": settings.xero_client_id, "redirect_uri": settings.xero_redirect_uri,
        "scope": settings.xero_scopes, "state": state,
    })


def _token_error(resp: httpx.Response) -> XeroError:
    try:
        body = resp.json()
    except ValueError:
        body = {}
    code = str(body.get("error", "")) if isinstance(body, dict) else ""
    if code == "invalid_client":
        return XeroError("Xero rejected the client id / secret (invalid_client); check XERO_CLIENT_ID and "
                         "XERO_CLIENT_SECRET", "config", resp.status_code)
    if code == "invalid_grant":
        return XeroError("The Xero connection has expired or was revoked (invalid_grant); run "
                         "scripts/xero_authorize.py again", "auth", resp.status_code)
    if resp.status_code >= 500:
        return XeroError(f"Xero token endpoint unavailable (HTTP {resp.status_code})", "unavailable", resp.status_code)
    return XeroError(f"Xero token request refused (HTTP {resp.status_code}{': ' + code if code else ''})",
                     "auth", resp.status_code)


def _token_request(form: dict) -> dict:
    """POST the token endpoint once with Basic client auth. Raises XeroError; returns the parsed token response."""
    if not has_client_credentials():
        raise XeroError("Xero client credentials are not configured (XERO_CLIENT_ID, XERO_CLIENT_SECRET)", "config")
    with _http() as http:
        resp = http.post(TOKEN_URL, data=form, auth=(settings.xero_client_id, settings.xero_client_secret),
                         headers={"Accept": "application/json"})
    if resp.status_code != 200:
        raise _token_error(resp)
    try:
        body = resp.json()
        if not body.get("access_token") or not body.get("refresh_token"):
            raise ValueError("incomplete")
    except (ValueError, AttributeError):
        raise XeroError("Xero returned an unreadable token response", "unavailable", resp.status_code) from None
    _remember(body.get("access_token"), body.get("refresh_token"))
    return body


def _token_set(body: dict, previous: dict | None) -> dict:
    prev = previous or {}
    return {
        "access_token": body["access_token"],
        "refresh_token": body["refresh_token"],
        "expires_at": time.time() + float(body.get("expires_in", 1800)),
        "scope": body.get("scope", prev.get("scope", "")),
        "tenant_id": prev.get("tenant_id", ""),
        "tenant_name": prev.get("tenant_name", ""),
        "short_code": prev.get("short_code", ""),
    }


def exchange_code(code: str) -> dict:
    """
    Exchange the one-time authorization code for tokens and store them. Returns the stored set. The code is valid
    once and for 5 minutes. The caller must not print the result.
    """
    body = _token_request({"grant_type": "authorization_code", "code": code,
                           "redirect_uri": settings.xero_redirect_uri})
    data = _token_set(body, None)
    with _thread_lock:
        if not save_tokens(data):
            raise XeroError("Could not write the Xero token file (check permissions of the data/ folder)", "config")
    return data


def _refresh_locked(force: bool) -> dict:
    """Refresh under the lock. Returns the current valid token set."""
    with _RefreshLock():
        current = load_tokens()
        if current is None:
            raise XeroError("Xero is not connected: run scripts/xero_authorize.py", "auth")
        if not force and current.get("access_token") and current.get("expires_at", 0) - time.time() > REFRESH_MARGIN_SECONDS:
            return current                      # another worker refreshed while we waited for the lock
        old_refresh = current["refresh_token"]
        attempts = 1 + max(0, settings.xero_max_retries)
        for i in range(attempts):
            try:
                body = _token_request({"grant_type": "refresh_token", "refresh_token": old_refresh})
                break
            except httpx.TransportError as e:
                # The answer may have been lost AFTER Xero rotated the token. Xero accepts the old refresh token for
                # 30 minutes precisely for this: retry with the same one, never start a second rotation.
                if i == attempts - 1:
                    raise XeroError(f"Xero token endpoint unreachable: {type(e).__name__}", "unavailable") from None
                _sleep(min(0.5 * 2 ** i, 4.0))
        new = _token_set(body, current)
        # Persist before doing anything else with it. If this write fails the set stays in memory (see save_tokens).
        save_tokens(new)
        return new


def access_token(force_refresh: bool = False) -> str:
    """A valid access token, refreshing (and persisting the rotated refresh token) when needed."""
    with _thread_lock:
        cur = load_tokens()
    if cur is None:
        raise XeroError("Xero is not connected: run scripts/xero_authorize.py", "auth")
    if (not force_refresh and cur.get("access_token")
            and cur.get("expires_at", 0) - time.time() > REFRESH_MARGIN_SECONDS):
        return cur["access_token"]
    return _refresh_locked(force_refresh)["access_token"]


def auth_event_id(token: str) -> str:
    """The `authentication_event_id` claim of an access token (a JWT), used to list only what was just connected."""
    import base64

    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return str(claims.get("authentication_event_id", ""))
    except (IndexError, ValueError, TypeError):
        return ""


def list_connections(token: str, auth_event: str = "") -> list[dict]:
    """The organisations the user connected (GET /connections). Practices etc. are filtered out."""
    with _http() as http:
        resp = http.get(CONNECTIONS_URL, params={"authEventId": auth_event} if auth_event else None,
                        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    if resp.status_code != 200:
        raise XeroError(sanitize(f"Could not list Xero connections (HTTP {resp.status_code})"),
                        "auth" if resp.status_code in (401, 403) else "unavailable", resp.status_code)
    try:
        rows = resp.json()
    except ValueError:
        raise XeroError("Xero returned an unreadable connections response", "unavailable") from None
    return [r for r in rows if isinstance(r, dict) and r.get("tenantType", "ORGANISATION") == "ORGANISATION"
            and r.get("tenantId")]
