"""
Retrieval module - hybrid dense + sparse (BM25) with RRF score fusion and re-ranking.

Design rules (see README "How the gate works"):
  * Evidence is ALWAYS restricted to the resolved vendor. If the vendor has no history
    the result is empty (cold start -> human review). There is no cross-vendor fallback:
    borrowing another vendor's coding history is exactly the failure a cold-start gate
    exists to prevent.
  * Lines whose source is `auto_post` are never evidence. Only human-confirmed lines
    (or seeded history) can teach the system, so it cannot learn from its own guesses.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

from ap_automation.core.config import settings
from ap_automation.core.embeddings import embed
from ap_automation.core.models import (
    AmountBand,
    ConfirmationSource,
    RetrievedEvidence,
)

_client: QdrantClient | None = None
_client_override: QdrantClient | None = None


def set_client(client: QdrantClient | None) -> None:
    """Override the Qdrant client (used for in-memory runs). None restores the configured one."""
    global _client_override
    _client_override = client


def get_qdrant() -> QdrantClient:
    global _client
    if _client_override is not None:
        return _client_override
    if _client is None:
        if settings.qdrant_api_key:
            _client = QdrantClient(url=f"https://{settings.qdrant_host}", api_key=settings.qdrant_api_key)
        else:
            _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return _client


def ensure_collection() -> None:
    """Create the Qdrant collection and required payload indexes."""
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=settings.embedding_dimensions, distance=Distance.COSINE),
        )
    try:  # payload index required for filtered search on Qdrant Cloud
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # local (in-memory) Qdrant warns that indexes are a no-op
            client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name="vendor_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
    except Exception:
        pass  # already exists


def reset_collection() -> None:
    """Delete and recreate the corpus collection (demo reset)."""
    client = get_qdrant()
    if settings.qdrant_collection in [c.name for c in client.get_collections().collections]:
        client.delete_collection(settings.qdrant_collection)
    ensure_collection()


def point_id_for(invoice_id: str) -> str:
    """Deterministic corpus point id for an invoice.

    Random ids made retrieval tie-breaking (and so every measured result) vary run to run. A stable id also
    makes write-back idempotent: confirming the same invoice twice replaces its line instead of duplicating it.
    """
    return str(uuid5(NAMESPACE_URL, f"ap-automation/invoice/{invoice_id}"))


def _build_filter(vendor_id: str) -> Filter:
    return Filter(must=[FieldCondition(key="vendor_id", match=MatchValue(value=vendor_id))])


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def _recency_weight(timestamp: datetime) -> float:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    days_old = max((datetime.now(timezone.utc) - timestamp).days, 0)
    return math.exp(-days_old / 730)


def _collection_size() -> int:
    try:
        info = get_qdrant().get_collection(settings.qdrant_collection)
        return info.points_count or 0
    except Exception:
        return 0


def _is_evidence(payload: dict | None) -> bool:
    """Auto-posted lines are never evidence (legacy corpora may still contain some)."""
    return bool(payload) and payload.get("source") != ConfirmationSource.AUTO_POST.value


def _scroll_vendor(client: QdrantClient, vendor_filter: Filter, limit: int) -> list:
    points: list = []
    offset = None
    while len(points) < limit:
        batch, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=vendor_filter,
            limit=min(256, limit - len(points)),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def _context_fields(payload: dict) -> dict:
    amt = payload.get("amount")
    try:
        amt = float(amt) if amt is not None else None
    except (TypeError, ValueError):
        amt = None
    return {"entity_id": str(payload.get("entity_id", "") or ""),
            "amount_band": str(payload.get("amount_band", "") or ""), "amount": amt}


def context_matches(payload: dict, entity_id: str, fund: str | None, amount_band: str) -> tuple[bool, bool, bool]:
    """(department match, fund match, amount-band match) of a stored line against the invoice being coded."""
    dept = bool(entity_id) and str(payload.get("entity_id", "") or "") == entity_id
    fnd = bool(fund) and str(payload.get("cost_centre", "") or "") == fund
    amt = bool(amount_band) and str(payload.get("amount_band", "") or "") == amount_band
    return dept, fnd, amt


def context_subset(payloads: list[dict], entity_id: str, fund: str | None, ladder: str,
                   min_lines: int) -> tuple[str, list[int]]:
    """
    Pick the evidence subset: walk the ladder (e.g. "dept+fund,fund,dept,vendor") and take the first level
    for which at least `min_lines` of the vendor's lines match. "vendor" always matches. Returns the level and
    the indices of the matching lines. A thin subset falls back to the next, broader level - never to nothing.
    """
    flags = [context_matches(p, entity_id, fund, "") for p in payloads]
    for level in (s.strip() for s in ladder.split(",") if s.strip()):
        if level == "dept+fund":
            idx = [i for i, (d, f, _) in enumerate(flags) if d and f]
        elif level == "fund":
            idx = [i for i, (_, f, _) in enumerate(flags) if f]
        elif level == "dept":
            idx = [i for i, (d, _, _) in enumerate(flags) if d]
        elif level == "vendor":
            idx = list(range(len(payloads)))
        else:
            raise ValueError(f"unknown context level {level!r}")
        if len(idx) >= min_lines or level == "vendor":
            return level, idx
    return "vendor", list(range(len(payloads)))


def _retrieve_with_context(description: str, vendor_id: str, entity_id: str, amount_band: AmountBand,
                           fund: str | None, top_k: int | None, info: dict | None) -> list[RetrievedEvidence]:
    """
    Context-aware retrieval. Same hybrid recipe (dense cosine + BM25, fused by RRF) but ranks are taken over the
    whole candidate set (so the top-k is a plain truncation of one ranking) and the candidate set / scores use the
    invoice's department, fund and amount band:

      subset - candidates are the vendor's lines at the most specific ladder level that has enough lines
      boost  - candidates are all of the vendor's lines; each matching signal adds `context_boost` to the RRF score

    Still vendor-scoped, still excludes auto-posted lines. Uses only department, fund and amount: never any of the
    columns derived from the account being predicted (public_data_import.LEAK_COLUMNS).
    """
    k = top_k or settings.top_k
    client = get_qdrant()
    vendor_filter = _build_filter(vendor_id)
    vendor_pts = sorted(
        (p for p in _scroll_vendor(client, vendor_filter, settings.bm25_max_candidates) if _is_evidence(p.payload)),
        key=lambda p: (str(p.payload.get("invoice_id", "")), str(p.id)),
    )
    if not vendor_pts:
        return []
    payloads = [p.payload for p in vendor_pts]
    flags = [context_matches(pl, entity_id, fund, amount_band.value) for pl in payloads]
    level, idx = ("vendor", list(range(len(vendor_pts)))) if settings.context_signals == "boost" else \
        context_subset(payloads, entity_id, fund, settings.context_ladder, settings.context_min_lines)
    candidates = [vendor_pts[i] for i in idx]
    cand_flags = [flags[i] for i in idx]

    if info is not None:
        info.update(
            context_level=level, vendor_lines=len(vendor_pts),
            vendor_distinct_codes=len({str(pl.get("gl_account", "")) for pl in payloads}),
            subset_lines=len(candidates),
            subset_distinct_codes=len({str(p.payload.get("gl_account", "")) for p in candidates}),
            n_dept_fund=sum(1 for d, f, _ in flags if d and f), n_fund=sum(1 for _, f, _ in flags if f),
            n_dept=sum(1 for d, _, _ in flags if d),
        )

    query_vector = embed(description)
    ids = [str(p.id) for p in candidates]
    scored = client.query_points(
        collection_name=settings.qdrant_collection, query=query_vector,
        query_filter=Filter(must=[HasIdCondition(has_id=ids)]), limit=len(ids),
        with_payload=False, with_vectors=False,
    )
    cosine_map = {str(r.id): r.score for r in scored.points}
    by_id = {str(p.id): p for p in candidates}
    dense_order = sorted(ids, key=lambda i: (-cosine_map.get(i, 0.0), str(by_id[i].payload.get("invoice_id", "")), i))
    dense_rank = {pid: r + 1 for r, pid in enumerate(dense_order)}

    bm25 = BM25Okapi([str(p.payload.get("description", "")).lower().split() for p in candidates])
    bm_scores = bm25.get_scores(description.lower().split())
    # candidates are already ordered by (invoice_id, id), so a stable sort breaks BM25 ties reproducibly
    bm_order = sorted(range(len(candidates)), key=lambda i: -bm_scores[i])
    sparse_rank = {str(candidates[i].id): r + 1 for r, i in enumerate(bm_order)}

    score: dict[str, float] = {}
    for pid, (d, f, a) in zip(ids, cand_flags, strict=True):
        s = _rrf_score(dense_rank[pid], settings.rrf_k) + _rrf_score(sparse_rank[pid], settings.rrf_k)
        if settings.context_signals == "boost":
            s += settings.context_boost * (int(d) + int(f) + (int(a) if settings.context_use_amount else 0))
        elif settings.context_use_amount and a:
            s += 1e-4  # subset mode: an amount-band match only breaks near-ties
        score[pid] = s
    top_ids = sorted(score, key=lambda i: (-score[i], i))[:k]

    evidence: list[RetrievedEvidence] = []
    for pid in top_ids:
        payload = by_id[pid].payload
        ts_raw = payload.get("confirmation_timestamp")
        try:
            ts = datetime.fromisoformat(str(ts_raw)) if ts_raw else datetime.now(timezone.utc)
        except ValueError:
            ts = datetime.now(timezone.utc)
        evidence.append(RetrievedEvidence(
            invoice_id=str(payload.get("invoice_id", "")), description=str(payload.get("description", "")),
            gl_account=str(payload.get("gl_account", "")), cost_centre=str(payload.get("cost_centre", "")),
            vendor_id=str(payload.get("vendor_id", "")), similarity_score=cosine_map.get(pid, 0.0),
            dense_rank=dense_rank[pid], sparse_rank=sparse_rank[pid], rrf_score=score[pid],
            confirmation_timestamp=ts,
            source=ConfirmationSource(payload.get("source", ConfirmationSource.HUMAN_CONFIRM.value)),
            **_context_fields(payload),
        ))
    return evidence


def retrieve(
    description: str,
    vendor_id: str,
    entity_id: str,
    amount_band: AmountBand,
    currency: str,
    top_k: int | None = None,
    fund: str | None = None,
    amount: float | None = None,
    info: dict | None = None,
) -> list[RetrievedEvidence]:
    """
    Retrieve vendor-specific evidence. Returns [] for a vendor with no usable history.

    `fund` / `amount` are only used when settings.context_signals is "subset" or "boost" (default "off").
    `info`, if given, is filled with diagnostics the gate variants use (vendor / subset line counts and distinct
    codes, the context level the evidence was drawn from).
    """
    if info is not None:
        info.clear()
    if _collection_size() == 0:
        return []
    if settings.context_signals in ("subset", "boost"):
        return _retrieve_with_context(description, vendor_id, entity_id, amount_band, fund, top_k, info)

    k = top_k or settings.top_k
    client = get_qdrant()
    query_vector = embed(description)
    vendor_filter = _build_filter(vendor_id)

    # Dense retrieval (vendor-filtered). Over-fetch so excluded auto-post lines don't starve k.
    dense_response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=vendor_filter,
        limit=k * 2,
        with_payload=True,
        with_vectors=False,
    )
    # Ties (identical descriptions embed identically) are broken by invoice id so the evidence set is reproducible.
    dense_results = sorted(
        (r for r in dense_response.points if _is_evidence(r.payload)),
        key=lambda r: (-r.score, str(r.payload.get("invoice_id", "")), str(r.id)),
    )[:k]
    if not dense_results:
        return []

    # Sparse retrieval (BM25) over the vendor's own history
    candidates = sorted(
        (p for p in _scroll_vendor(client, vendor_filter, settings.bm25_max_candidates) if _is_evidence(p.payload)),
        key=lambda p: (str(p.payload.get("invoice_id", "")), str(p.id)),
    )
    if not candidates:
        return []
    if info is not None:
        codes = len({str(p.payload.get("gl_account", "")) for p in candidates})
        info.update(context_level="vendor", vendor_lines=len(candidates), vendor_distinct_codes=codes,
                    subset_lines=len(candidates), subset_distinct_codes=codes)

    tokenised_corpus = [str(p.payload.get("description", "")).lower().split() for p in candidates]
    bm25 = BM25Okapi(tokenised_corpus)
    bm25_scores = bm25.get_scores(description.lower().split())
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
    bm25_rank_map = {str(candidates[idx].id): rank + 1 for rank, (idx, _) in enumerate(bm25_ranked)}
    dense_rank_map = {str(r.id): i + 1 for i, r in enumerate(dense_results)}

    rrf_scores: dict[str, float] = {}
    for pid in sorted(set(dense_rank_map) | set(bm25_rank_map)):
        rrf_scores[pid] = (
            _rrf_score(dense_rank_map.get(pid, k * 2), settings.rrf_k)
            + _rrf_score(bm25_rank_map.get(pid, k * 2), settings.rrf_k)
        )
    top_ids = sorted(rrf_scores, key=lambda x: (-rrf_scores[x], x))[:k]

    payload_map = {str(p.id): p.payload for p in candidates}
    payload_map.update({str(r.id): r.payload for r in dense_results})
    cosine_map = {str(r.id): r.score for r in dense_results}

    # BM25-only hits have no cosine yet. Score them with one id-restricted dense query
    # rather than leaving similarity at 0, which would silently drag confidence down.
    missing = [pid for pid in top_ids if pid not in cosine_map]
    if missing:
        scored = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            query_filter=Filter(must=[HasIdCondition(has_id=missing)]),
            limit=len(missing),
            with_payload=False,
            with_vectors=False,
        )
        cosine_map.update({str(r.id): r.score for r in scored.points})

    evidence: list[RetrievedEvidence] = []
    for pid in top_ids:
        payload = payload_map.get(pid)
        if not payload:
            continue
        ts_raw = payload.get("confirmation_timestamp")
        try:
            ts = datetime.fromisoformat(str(ts_raw)) if ts_raw else datetime.now(timezone.utc)
        except ValueError:
            ts = datetime.now(timezone.utc)
        evidence.append(RetrievedEvidence(
            invoice_id=str(payload.get("invoice_id", "")),
            description=str(payload.get("description", "")),
            gl_account=str(payload.get("gl_account", "")),
            cost_centre=str(payload.get("cost_centre", "")),
            vendor_id=str(payload.get("vendor_id", "")),
            similarity_score=cosine_map.get(pid, 0.0),
            dense_rank=dense_rank_map.get(pid, k * 2),
            sparse_rank=bm25_rank_map.get(pid, k * 2),
            rrf_score=rrf_scores[pid],
            confirmation_timestamp=ts,
            source=ConfirmationSource(payload.get("source", ConfirmationSource.HUMAN_CONFIRM.value)),
            **_context_fields(payload),
        ))

    gl_counts = Counter(e.gl_account for e in evidence)
    max_count = max(gl_counts.values(), default=1)

    def rerank_score(e: RetrievedEvidence) -> float:
        frequency = gl_counts[e.gl_account] / max_count
        return (e.similarity_score * 0.6) + (_recency_weight(e.confirmation_timestamp) * 0.25) + (frequency * 0.15)

    evidence.sort(key=rerank_score, reverse=True)
    return evidence[:k]


def _scroll_all(limit: int = 50000) -> list:
    client = get_qdrant()
    points: list = []
    offset = None
    while len(points) < limit:
        batch, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def corpus_vendors() -> list[dict]:
    """One summary row per vendor in the corpus: size, dominant GL, last confirmation, who taught it."""
    by_vendor: dict[str, list[dict]] = {}
    try:
        for p in _scroll_all():
            payload = p.payload or {}
            by_vendor.setdefault(str(payload.get("vendor_id", "")), []).append(payload)
    except Exception:
        return []
    rows = []
    for vendor_id, payloads in by_vendor.items():
        gl_counts = Counter(str(x.get("gl_account", "")) for x in payloads)
        top_gl, top_n = gl_counts.most_common(1)[0]
        names = Counter(str(x["vendor_name"]) for x in payloads if x.get("vendor_name"))
        rows.append({
            "vendor_id": vendor_id,
            # The vendor master has no canonical display names, so show the most common name seen for it.
            "vendor_name": max(names, key=lambda n: (names[n], len(n), n)) if names else None,
            "points": len(payloads),
            "top_gl": top_gl,
            "top_gl_share": round(top_n / len(payloads), 3),
            "last_confirmed": max((str(x.get("confirmation_timestamp", "")) for x in payloads), default=""),
            "sources": dict(Counter(str(x.get("source", "")) for x in payloads)),
        })
    return sorted(rows, key=lambda r: (-r["points"], r["vendor_id"]))


def corpus_points(vendor_id: str, limit: int = 500) -> list[dict]:
    """The evidence lines for one vendor, newest confirmation first."""
    points = _scroll_vendor(get_qdrant(), _build_filter(vendor_id), limit)
    rows = [{
        "point_id": str(p.id),
        "description": str((p.payload or {}).get("description", "")),
        "gl_account": str((p.payload or {}).get("gl_account", "")),
        "cost_centre": str((p.payload or {}).get("cost_centre", "")),
        "source": str((p.payload or {}).get("source", "")),
        "confirmed_by": str((p.payload or {}).get("confirming_user_id", "")),
        "confirmed_at": str((p.payload or {}).get("confirmation_timestamp", "")),
    } for p in points]
    return sorted(rows, key=lambda r: r["confirmed_at"], reverse=True)


def delete_corpus_point(point_id: str) -> bool:
    """Remove one evidence line (data-steward correction). Returns False if it does not exist."""
    client = get_qdrant()
    found = client.retrieve(settings.qdrant_collection, ids=[point_id], with_payload=False, with_vectors=False)
    if not found:
        return False
    client.delete(settings.qdrant_collection, points_selector=PointIdsList(points=[point_id]))
    return True


def index_outcome(outcome_payload: dict) -> None:
    vector = embed(outcome_payload["description"])
    get_qdrant().upsert(
        collection_name=settings.qdrant_collection,
        points=[PointStruct(id=outcome_payload["point_id"], vector=vector, payload=outcome_payload)],
    )
