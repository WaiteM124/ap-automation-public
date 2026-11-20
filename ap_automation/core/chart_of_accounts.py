"""
Single source of truth for the chart of accounts.

The VAT agent, the cost-centre agent, the entity agent and the synthetic dataset
generator all read from here. Previously each kept its own GL table and they
disagreed (for example GL 6800 was "Training" in the data but "Business
entertainment - VAT blocked" in the VAT agent).

In production this would be loaded from the ERP / tenant configuration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Account:
    code: str
    name: str
    vat_treatment: str          # default UK VAT treatment for the account
    default_cost_centre: str    # cost centre this account is normally charged to
    vat_note: str = ""
    entity_sensitive: bool = False  # capex: must sit on the entity that owns the asset


COST_CENTRES: dict[str, str] = {
    "CC100": "Administration",
    "CC200": "IT & Operations",
    "CC300": "Facilities",
    "CC400": "Professional Services",
    "CC500": "Travel",
}

CHART: dict[str, Account] = {
    a.code: a
    for a in [
        Account("6100", "Office Supplies", "STANDARD", "CC100",
                "Office supplies - standard rated"),
        Account("6200", "IT & Software", "STANDARD", "CC200",
                "IT services and software - standard rated"),
        Account("6300", "Facilities & Maintenance", "STANDARD", "CC300",
                "Facilities services - standard rated"),
        Account("6400", "Professional Services", "STANDARD", "CC400",
                "Professional services - standard rated"),
        Account("6500", "Travel & Accommodation", "STANDARD", "CC500",
                "Travel and accommodation - standard rated"),
        Account("6600", "Utilities", "STANDARD", "CC200",
                "Business energy and water - standard rated"),
        Account("6700", "Marketing & Advertising", "STANDARD", "CC100",
                "Marketing and advertising - standard rated"),
        Account("6800", "Training & Development", "STANDARD", "CC400",
                "Training - standard rated unless supplied by an eligible body"),
        Account("7100", "Equipment (CapEx)", "STANDARD", "CC200",
                "Equipment purchases - standard rated", entity_sensitive=True),
        Account("7200", "Leasehold Improvements (CapEx)", "STANDARD", "CC300",
                "Leasehold improvements - standard rated", entity_sensitive=True),
    ]
}

GL_NAMES: dict[str, str] = {code: acct.name for code, acct in CHART.items()}

# --------------------------------------------------------------------------- per-client chart
#
# CHART and GL_NAMES are mutated IN PLACE (never rebound), so every module that did
# `from ... import CHART` sees the active chart. The built-in chart above is the default; a client's own chart
# can replace it at runtime (POST /chart) and is persisted to a small JSON file so it survives a restart.

DEFAULT_CHART: dict[str, Account] = dict(CHART)
MAX_CHART_ACCOUNTS = 5_000
VALID_VAT_TREATMENTS = frozenset({
    "STANDARD", "REDUCED", "ZERO", "EXEMPT", "OUTSIDE_SCOPE", "REVERSE_CHARGE", "BLOCKED", "NOT_ASSESSED",
    "NOT_APPLICABLE",
})
_CODE_COLS = ("code", "account_code", "gl", "gl_account", "account")
_NAME_COLS = ("name", "account_name", "description")
_CC_COLS = ("default_cost_centre", "default_cost_center", "cost_centre", "cost_center")
_VAT_COLS = ("vat_treatment", "vat")

_custom = False


class ChartError(ValueError):
    """The uploaded chart is invalid. `problems` lists what is wrong (first 20), one line each."""

    def __init__(self, problems: list[str]):
        self.problems = problems[:20]
        super().__init__("; ".join(self.problems))


def _col(row: dict, names: tuple[str, ...]) -> str:
    for n in names:
        if n in row and row[n] is not None:
            return str(row[n]).strip()
    return ""


def parse_chart_csv(text: str) -> dict[str, Account]:
    """
    Parse a chart-of-accounts CSV. Required columns: a code (`code`, `account_code`, `gl`...) and a `name`.
    Optional: `default_cost_centre` (blank = no expectation, so the cost-centre agent will not flag a mismatch) and
    `vat_treatment` (default STANDARD). Raises ChartError listing every problem found.
    """
    import csv
    import io

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ChartError(["the file is empty"])
    reader.fieldnames = [(h or "").strip().lower() for h in reader.fieldnames]
    if not any(c in reader.fieldnames for c in _CODE_COLS):
        raise ChartError([f"no account-code column (expected one of: {', '.join(_CODE_COLS)})"])
    if not any(c in reader.fieldnames for c in _NAME_COLS):
        raise ChartError([f"no name column (expected one of: {', '.join(_NAME_COLS)})"])

    accounts: dict[str, Account] = {}
    problems: list[str] = []
    for line_no, raw in enumerate(reader, start=2):
        row = {k: v for k, v in raw.items() if k}
        code, name = _col(row, _CODE_COLS), _col(row, _NAME_COLS)
        if not code and not name and not any((v or "").strip() for v in row.values()):
            continue  # blank line
        if not code:
            problems.append(f"line {line_no}: missing account code")
            continue
        if not name:
            problems.append(f"line {line_no}: account {code} has no name")
            continue
        vat = (_col(row, _VAT_COLS) or "STANDARD").upper()
        if vat not in VALID_VAT_TREATMENTS:
            problems.append(f"line {line_no}: unknown vat_treatment {vat!r} for account {code}")
            continue
        if code in accounts:
            problems.append(f"line {line_no}: account {code} appears more than once")
            continue
        accounts[code] = Account(code=code, name=name, vat_treatment=vat,
                                 default_cost_centre=_col(row, _CC_COLS), vat_note="")
        if len(accounts) > MAX_CHART_ACCOUNTS:
            raise ChartError([f"more than {MAX_CHART_ACCOUNTS:,} accounts"])
    if problems:
        raise ChartError(problems)
    if not accounts:
        raise ChartError(["the chart has no accounts"])
    return accounts


def _apply(accounts: dict[str, Account], custom: bool) -> None:
    global _custom
    CHART.clear()
    CHART.update(accounts)
    GL_NAMES.clear()
    GL_NAMES.update({code: acct.name for code, acct in accounts.items()})
    _custom = custom


def set_chart(accounts: dict[str, Account]) -> None:
    """Make `accounts` the active chart (in place)."""
    _apply(accounts, custom=True)


def reset_chart(path: str | None = None) -> None:
    """Restore the built-in chart; also delete the persisted custom chart at `path` if given."""
    _apply(DEFAULT_CHART, custom=False)
    if path:
        try:
            import os

            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def is_custom() -> bool:
    return _custom


def chart_rows() -> list[dict]:
    return [{"code": a.code, "name": a.name, "default_cost_centre": a.default_cost_centre,
             "vat_treatment": a.vat_treatment} for a in CHART.values()]


def save_chart(path: str) -> bool:
    """Persist the active custom chart. Returns False (never raises) if the disk is read-only."""
    import json
    from pathlib import Path

    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(chart_rows(), indent=1), encoding="utf-8")
        return True
    except OSError:
        return False


def load_saved_chart(path: str) -> bool:
    """Load a persisted custom chart at startup. Returns True if one was loaded."""
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return False
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
        accounts = {r["code"]: Account(code=r["code"], name=r["name"], vat_treatment=r.get("vat_treatment", "STANDARD"),
                                       default_cost_centre=r.get("default_cost_centre", "")) for r in rows}
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if not accounts:
        return False
    _apply(accounts, custom=True)
    return True


def get_account(gl_code: str) -> Account | None:
    return CHART.get(gl_code)
