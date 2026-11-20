"""
A small in-process stand-in for the slice of QdrantClient that core/retrieval.py uses, indexed by vendor_id.

Why: qdrant-client's local (":memory:") mode evaluates a payload filter by looping over EVERY point in Python,
about four times per prediction. At 30k points that is ~0.5 s per invoice (hours for a replay). This store keeps
one vector matrix per vendor, so a vendor-filtered query touches only that vendor's lines. It is used ONLY by
scripts/replay_public_data.py; retrieval, BM25, RRF fusion, confidence, gating and write-back are the real code.

Semantics replicated: cosine score (vectors L2-normalised), vendor filter (FieldCondition/MatchValue),
id filter (HasIdCondition), upsert-by-id replaces, scroll pagination, payload returned as stored.
tests/test_public_data.py checks it returns the same evidence as qdrant-client's local mode.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from qdrant_client import models

# Equal cosine scores (identical descriptions) are ordered newest-insert-first. qdrant-client's local mode leaves
# the order of ties to np.argsort; on a 1,200-line replay "newest" agreed with it on 95% of lines (oldest: 93%) and
# on aggregate automation / auto-post accuracy to within ~2 points.
_TIE = "newest"


class _VendorBlock:
    def __init__(self, dim: int):
        self.vecs = np.zeros((16, dim), dtype=np.float32)
        self.ids: list[str] = []
        self.payloads: list[dict] = []
        self.row: dict[str, int] = {}

    def upsert(self, pid: str, vec: np.ndarray, payload: dict) -> None:
        if pid in self.row:
            i = self.row[pid]
            self.vecs[i] = vec
            self.payloads[i] = payload
            return
        if len(self.ids) == len(self.vecs):
            self.vecs = np.concatenate([self.vecs, np.zeros_like(self.vecs)])
        self.vecs[len(self.ids)] = vec
        self.row[pid] = len(self.ids)
        self.ids.append(pid)
        self.payloads.append(payload)


class InProcessVectorStore:
    """Duck-types the QdrantClient methods retrieval.py calls. Single collection semantics per name."""

    def __init__(self) -> None:
        self._dim: dict[str, int] = {}
        self._blocks: dict[str, dict[str, _VendorBlock]] = {}
        self._where: dict[str, dict[str, str]] = {}  # collection -> point id -> vendor_id

    # -- collection management -------------------------------------------------
    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self._dim])

    def create_collection(self, collection_name: str, vectors_config, **_):
        self._dim[collection_name] = vectors_config.size
        self._blocks[collection_name] = {}
        self._where[collection_name] = {}

    def delete_collection(self, collection_name: str, **_):
        for d in (self._dim, self._blocks, self._where):
            d.pop(collection_name, None)

    def create_payload_index(self, *_, **__):
        return None

    def get_collection(self, collection_name: str):
        return SimpleNamespace(points_count=len(self._where[collection_name]))

    # -- writes ----------------------------------------------------------------
    def upsert(self, collection_name: str, points, **_):
        dim = self._dim[collection_name]
        for p in points:
            vec = np.asarray(p.vector, dtype=np.float32)
            if vec.shape != (dim,):
                raise ValueError(f"vector has {vec.shape}, collection expects ({dim},)")
            norm = float(np.linalg.norm(vec)) or 1.0
            vec = vec / norm
            pid, vendor = str(p.id), str((p.payload or {}).get("vendor_id", ""))
            old_vendor = self._where[collection_name].get(pid)
            if old_vendor is not None and old_vendor != vendor:
                raise ValueError("re-upserting a point under a different vendor is not supported")
            blocks = self._blocks[collection_name]
            blocks.setdefault(vendor, _VendorBlock(dim)).upsert(pid, vec, dict(p.payload or {}))
            self._where[collection_name][pid] = vendor

    def delete(self, collection_name: str, points_selector, **_):
        raise NotImplementedError("not needed by the replay")

    # -- reads -----------------------------------------------------------------
    @staticmethod
    def _parse_filter(flt) -> tuple[str | None, list[str] | None]:
        vendor, ids = None, None
        for cond in (flt.must if flt else None) or []:
            if isinstance(cond, models.FieldCondition) and cond.key == "vendor_id":
                vendor = str(cond.match.value)
            elif isinstance(cond, models.HasIdCondition):
                ids = [str(i) for i in cond.has_id]
            else:
                raise NotImplementedError(f"unsupported filter condition: {cond!r}")
        return vendor, ids

    def _select(self, collection_name: str, flt) -> list[tuple[_VendorBlock, int]]:
        vendor, ids = self._parse_filter(flt)
        blocks = self._blocks[collection_name]
        if ids is not None:
            out = []
            for pid in ids:
                v = self._where[collection_name].get(pid)
                if v is not None and (vendor is None or v == vendor):
                    out.append((blocks[v], blocks[v].row[pid]))
            return out
        if vendor is not None:
            b = blocks.get(vendor)
            return [(b, i) for i in range(len(b.ids))] if b else []
        return [(b, i) for b in blocks.values() for i in range(len(b.ids))]

    def query_points(self, collection_name: str, query, query_filter=None, limit: int = 10, **_):
        q = np.asarray(query, dtype=np.float32)
        q = q / (float(np.linalg.norm(q)) or 1.0)
        vendor, ids = self._parse_filter(query_filter)
        if vendor is not None and ids is None:  # fast path: one matrix product over this vendor's lines
            b = self._blocks[collection_name].get(vendor)
            if b is None or not b.ids:
                return SimpleNamespace(points=[])
            sel = [(b, i) for i in range(len(b.ids))]
            scores = (b.vecs[:len(b.ids)] @ q).astype(np.float64)
        else:
            sel = self._select(collection_name, query_filter)
            if not sel:
                return SimpleNamespace(points=[])
            scores = np.array([float(b.vecs[i] @ q) for b, i in sel], dtype=np.float64)
        order = np.lexsort((-np.arange(len(scores)) if _TIE == "newest" else np.arange(len(scores)), -scores))[:limit]
        pts = [models.ScoredPoint(id=sel[j][0].ids[sel[j][1]], version=0, score=float(scores[j]),
                                  payload=sel[j][0].payloads[sel[j][1]], vector=None) for j in order]
        return SimpleNamespace(points=pts)

    def scroll(self, collection_name: str, scroll_filter=None, limit: int = 10, offset=None, **_):
        sel = self._select(collection_name, scroll_filter)
        start = int(offset or 0)
        batch = sel[start:start + limit]
        recs = [models.Record(id=b.ids[i], payload=b.payloads[i], vector=None) for b, i in batch]
        nxt = start + limit if start + limit < len(sel) else None
        return recs, nxt

    def retrieve(self, collection_name: str, ids, **_):
        out = []
        for pid in ids:
            v = self._where[collection_name].get(str(pid))
            if v is not None:
                b = self._blocks[collection_name][v]
                out.append(models.Record(id=str(pid), payload=b.payloads[b.row[str(pid)]], vector=None))
        return out
