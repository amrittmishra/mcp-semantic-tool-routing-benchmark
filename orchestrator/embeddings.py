"""Gemini embeddings for taxonomy documents and incoming queries.

Authentication is *not* reimplemented here: :mod:`vertex_env` already owns the
service account, project, and location, and exposes a configured client plus an
``embed()`` helper. This module only decides model, dimensionality, task type,
and batching.

Task types matter for retrieval quality and are asymmetric on purpose:

    stored taxonomy documents -> RETRIEVAL_DOCUMENT
    incoming user queries     -> RETRIEVAL_QUERY

Vectors are returned un-normalized; :mod:`orchestrator.index` L2-normalizes on
ingest and at query time so that FAISS inner product == cosine similarity.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

import vertex_env
from orchestrator import config

BATCH_SIZE = 32
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


class EmbeddingError(RuntimeError):
    pass


def _embed_batch(texts: Sequence[str], task_type: str) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return vertex_env.embed(
                list(texts),
                dims=config.EMBED_DIMENSIONS,
                task_type=task_type,
                normalize=False,
            )
        except Exception as exc:  # noqa: BLE001 -- surfaced below with context
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise EmbeddingError(
        f"{config.EMBED_MODEL} failed after {MAX_RETRIES} attempts: {last_error}"
    ) from last_error


def embed_texts(texts: Sequence[str], task_type: str) -> np.ndarray:
    """Embed ``texts`` and return a validated ``(n, dims)`` float32 array."""
    if not texts:
        return np.zeros((0, config.EMBED_DIMENSIONS), dtype=np.float32)

    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        vectors.extend(_embed_batch(texts[start : start + BATCH_SIZE], task_type))

    for i, vector in enumerate(vectors):
        if len(vector) != config.EMBED_DIMENSIONS:
            raise EmbeddingError(
                f"Embedding {i} has {len(vector)} dimensions, "
                f"expected {config.EMBED_DIMENSIONS}"
            )

    array = np.asarray(vectors, dtype=np.float32)
    if array.dtype != np.float32:  # pragma: no cover -- asarray guarantees it
        raise EmbeddingError(f"Expected float32 vectors, got {array.dtype}")
    return array


def embed_documents(texts: Sequence[str]) -> np.ndarray:
    """Embed taxonomy documents destined for the FAISS index."""
    return embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> np.ndarray:
    """Embed one user query. Returns a ``(1, dims)`` float32 array."""
    return embed_texts([text], task_type="RETRIEVAL_QUERY")
