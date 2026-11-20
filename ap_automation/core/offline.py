"""
Offline backend: an in-memory Qdrant and a deterministic lexical embedder.

Lets the full pipeline - retrieval, confidence gate, agents, governance, write-back -
run with no network, no Docker and no API spend. Used by the test-suite and by
`scripts/evaluate_poc.py`.

IMPORTANT: the hashing embedder is purely lexical (hashed unigrams + bigrams). It
validates the MECHANICS of the loop (cold start -> review -> human confirmation ->
write-back -> auto-post) but says nothing about the semantic quality of
text-embedding-3-small. Run the evaluation with --live-embeddings for that.
"""

from __future__ import annotations

import math
import re
import zlib

from qdrant_client import QdrantClient

from ap_automation.core import embeddings, retrieval
from ap_automation.core.config import settings

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    def __init__(self, dim: int | None = None):
        self.dim = dim or settings.embedding_dimensions

    def __call__(self, text: str) -> list[float]:
        tokens = _TOKEN.findall(text.lower())
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vec = [0.0] * self.dim
        for feat in features:
            h = zlib.crc32(feat.encode())
            vec[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def use_offline_backend(hashing: bool = True) -> QdrantClient:
    """Switch retrieval to a fresh in-memory Qdrant (and, by default, hashing embeddings)."""
    client = QdrantClient(":memory:")
    retrieval.set_client(client)
    if hashing:
        embeddings.set_embedder(HashingEmbedder())
    retrieval.ensure_collection()
    return client


def reset_backend() -> None:
    """Restore the configured (remote) Qdrant and OpenAI embeddings."""
    retrieval.set_client(None)
    embeddings.set_embedder(None)
