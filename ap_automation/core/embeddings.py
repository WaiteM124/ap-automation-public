"""
Embedding generation via OpenAI text-embedding-3-small.

The embedder is pluggable (`set_embedder`) so tests and the offline evaluation can
run without network access. See core/offline.py.
"""

from __future__ import annotations

import time
from typing import Callable

from ap_automation.core.config import settings
from ap_automation.core.preprocessing import preprocess_description

_client = None
_embedder_override: Callable[[str], list[float]] | None = None


def set_embedder(fn: Callable[[str], list[float]] | None) -> None:
    """Replace the embedding function (None restores OpenAI)."""
    global _embedder_override
    _embedder_override = fn


def backend_name() -> str:
    """Which embedding backend is active - surfaced in the demo pre-flight so it is never hidden."""
    if _embedder_override is not None:
        return "offline-hashing (lexical, not a semantic model)"
    return f"openai/{settings.embedding_model}"


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set - cannot compute embeddings")
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed(text: str, langfuse_span=None) -> list[float]:
    """Embed a single text string. Preprocesses the description before embedding."""
    cleaned = preprocess_description(text)
    if _embedder_override is not None:
        return _embedder_override(cleaned)

    t0 = time.time()
    response = get_client().embeddings.create(model=settings.embedding_model, input=cleaned)
    latency_ms = int((time.time() - t0) * 1000)
    token_count = response.usage.total_tokens if response.usage else 0

    if langfuse_span is not None:
        try:
            langfuse_span.update(metadata={
                "embedding_model": settings.embedding_model,
                "embedding_tokens": token_count,
                "embedding_cost_usd": round(token_count * 0.00000002, 8),  # $0.02 per 1M tokens
                "embedding_latency_ms": latency_ms,
                "input_text": cleaned[:200],
            })
        except Exception:
            pass

    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. More efficient than calling embed() in a loop."""
    cleaned = [preprocess_description(t) for t in texts]
    if _embedder_override is not None:
        return [_embedder_override(c) for c in cleaned]
    response = get_client().embeddings.create(model=settings.embedding_model, input=cleaned)
    return [item.embedding for item in response.data]
