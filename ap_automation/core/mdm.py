"""
MDM resolution - vendor name -> canonical ID.
POC stub: CSV lookup with conservative fuzzy matching.
Production: MDM service API call.

Matching is deliberately conservative. A wrong vendor match is worse than no
match: it would borrow another vendor's coding history. Previously ANY name that
merely started with a known short key (e.g. "apex") resolved to that vendor at
0.9 confidence, and the MDM confidence was never used for routing. Now:

  exact normalised match        -> confidence 1.0
  near match (token Jaccard)    -> confidence = Jaccard, accepted from 0.5
  otherwise                     -> NEW_<name>, confidence 0.3 (cold start)

The router sends anything below `min_mdm_confidence` (0.85) to human review.
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

VENDOR_MAP: dict[str, str] = {}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STOP_WORDS = r"\b(ltd|limited|plc|inc|llc|gmbh|co|corp|group|agency|services|solutions|partners)\b\.?"
FUZZY_ACCEPT = 0.5


def load_vendor_map(path: str = "data/vendor_map.csv") -> None:
    """Load vendor name -> canonical ID mapping from CSV (relative paths resolve from the repo root)."""
    p = Path(path)
    if not p.exists() and not p.is_absolute():
        p = _REPO_ROOT / path
    if not p.exists():
        print(f"[MDM] vendor map not found at {path} - every vendor will resolve as NEW")
        return
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            VENDOR_MAP[_normalise(row["raw_name"])] = row["canonical_id"].strip()


def set_vendor_map(rows: list[dict]) -> None:
    """Replace the vendor master with the given {raw_name, canonical_id} rows (tests / evaluation)."""
    VENDOR_MAP.clear()
    for row in rows:
        VENDOR_MAP[_normalise(row["raw_name"])] = row["canonical_id"].strip()


def load_client_vendors(path: str) -> int:
    """Load vendors registered by history import (raw_name,canonical_id rows). Returns how many were loaded."""
    p = Path(path)
    if not p.exists():
        return 0
    n = 0
    try:
        with p.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = _normalise(row.get("raw_name", ""))
                if key and row.get("canonical_id", "").strip():
                    VENDOR_MAP[key] = row["canonical_id"].strip()
                    n += 1
    except (OSError, csv.Error):
        return n
    return n


def register_vendor(raw_name: str, persist_path: str | None = None) -> str:
    """
    Make `raw_name` an exact-match vendor and return its canonical id. Used by history import, where the client's
    own vendor names are authoritative: without this a freshly imported vendor resolves at 0.3 confidence and could
    never auto-post. An already-known name keeps its id. A new name gets a deterministic id derived from the
    normalised name (no fuzzy joining: the import must not silently merge two vendors), so re-registering after a
    restart yields the same id the corpus already carries.
    """
    if not VENDOR_MAP:
        load_vendor_map()
    key = _normalise(raw_name)
    if key in VENDOR_MAP:
        return VENDOR_MAP[key]
    vendor_id = "IMP_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10].upper()
    VENDOR_MAP[key] = vendor_id
    if persist_path:
        try:
            p = Path(persist_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            new_file = not p.exists()
            with p.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["raw_name", "canonical_id"])
                w.writerow([raw_name.strip(), vendor_id])
        except OSError:
            pass  # in-memory registration still applies; re-importing the file restores it after a restart
    return vendor_id


def _normalise(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(_STOP_WORDS, "", name)
    name = re.sub(r"[^a-z0-9\s]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def _new_vendor_id(raw_name: str) -> str:
    return f"NEW_{re.sub(r'[^A-Z0-9]', '_', raw_name.upper())[:20]}"


def resolve_vendor(raw_name: str) -> tuple[str, float]:
    """Resolve a raw vendor name to (canonical_vendor_id, confidence)."""
    if not VENDOR_MAP:
        load_vendor_map()

    normalised = _normalise(raw_name)
    if normalised in VENDOR_MAP:
        return VENDOR_MAP[normalised], 1.0

    tokens = set(normalised.split())
    best_score, best_id = 0.0, ""
    if tokens:
        for key, canonical_id in VENDOR_MAP.items():
            key_tokens = set(key.split())
            if not key_tokens:
                continue
            jaccard = len(tokens & key_tokens) / len(tokens | key_tokens)
            if jaccard > best_score:
                best_score, best_id = jaccard, canonical_id

    if best_score >= FUZZY_ACCEPT:
        return best_id, round(best_score, 3)
    return _new_vendor_id(raw_name), 0.3
