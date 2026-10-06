"""Semantic taxonomy router.

    query -> gemini-embedding-001 (RETRIEVAL_QUERY)
          -> L2 normalize
          -> persistent FAISS IndexFlatIP
          -> top-k semantic operations

Routing is operation-first. The downstream ``server_id`` is read off the winning
operation as execution metadata; we never search "which MCP server" and then
"which tool inside it".

The router abstains rather than forcing a call it is not confident about:
``no_match`` below the confidence threshold, ``ambiguous`` when the top two
candidates are within the ambiguity margin.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from orchestrator import config, embeddings, index as index_module
from orchestrator.registry import Registry, load_registry

STATUS_OK = "ok"
STATUS_NO_MATCH = "no_match"
STATUS_AMBIGUOUS = "ambiguous"


@dataclasses.dataclass
class RouteResult:
    status: str
    candidates: list[dict[str, Any]]
    embedding_ms: float
    faiss_ms: float
    reason: str | None = None

    @property
    def top(self) -> dict[str, Any] | None:
        return self.candidates[0] if self.candidates else None

    @property
    def confidence(self) -> float | None:
        return self.candidates[0]["score"] if self.candidates else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "candidates": self.candidates,
            "embedding_ms": self.embedding_ms,
            "faiss_ms": self.faiss_ms,
        }


class Router:
    """Holds the loaded FAISS index. Construct once per process."""

    def __init__(
        self,
        registry: Registry | None = None,
        loaded_index: index_module.LoadedIndex | None = None,
    ) -> None:
        self.registry = registry or load_registry()
        self.index = loaded_index or index_module.load(self.registry)

    def reload(self) -> None:
        """Re-read registry and index from disk (used by POST /index/reload)."""
        self.registry = load_registry()
        self.index = index_module.load(self.registry)

    @property
    def manifest(self) -> dict[str, Any]:
        return self.index.manifest

    def route(
        self,
        query: str,
        top_k: int | None = None,
        confidence_threshold: float | None = None,
        ambiguity_margin: float | None = None,
    ) -> RouteResult:
        top_k = top_k if top_k is not None else config.ROUTER_TOP_K
        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else config.ROUTER_CONFIDENCE_THRESHOLD
        )
        margin = (
            ambiguity_margin
            if ambiguity_margin is not None
            else config.ROUTER_AMBIGUITY_MARGIN
        )

        started = time.perf_counter()
        query_vector = embeddings.embed_query(query)
        embedding_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        candidates = self.index.search(query_vector, top_k)
        faiss_ms = (time.perf_counter() - started) * 1000

        if not candidates:
            return RouteResult(
                status=STATUS_NO_MATCH,
                candidates=[],
                embedding_ms=embedding_ms,
                faiss_ms=faiss_ms,
                reason="index returned no candidates",
            )

        top1 = candidates[0]["score"]
        if top1 < threshold:
            return RouteResult(
                status=STATUS_NO_MATCH,
                candidates=candidates,
                embedding_ms=embedding_ms,
                faiss_ms=faiss_ms,
                reason=(
                    f"top-1 similarity {top1:.4f} is below "
                    f"ROUTER_CONFIDENCE_THRESHOLD {threshold}"
                ),
            )

        if len(candidates) > 1:
            gap = top1 - candidates[1]["score"]
            if gap < margin:
                return RouteResult(
                    status=STATUS_AMBIGUOUS,
                    candidates=candidates,
                    embedding_ms=embedding_ms,
                    faiss_ms=faiss_ms,
                    reason=(
                        f"top-1 {candidates[0]['operation_id']} ({top1:.4f}) and top-2 "
                        f"{candidates[1]['operation_id']} ({candidates[1]['score']:.4f}) "
                        f"differ by {gap:.4f}, under ROUTER_AMBIGUITY_MARGIN {margin}"
                    ),
                )

        return RouteResult(
            status=STATUS_OK,
            candidates=candidates,
            embedding_ms=embedding_ms,
            faiss_ms=faiss_ms,
        )


def route(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Module-level convenience wrapper: returns the raw top-k candidates.

    Builds a Router per call, so it is for scripts and tests only -- long-lived
    processes should hold a :class:`Router`.
    """
    return Router().route(query, top_k=top_k).candidates
