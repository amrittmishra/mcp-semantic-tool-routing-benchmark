"""Persistent FAISS taxonomy index.

Three artifacts live side by side under ``data/cache/`` -- a *bind-mounted host
directory*, never the container writable layer:

    tools.faiss              the IndexFlatIP index + normalized 768-d vectors
    tools_metadata.json      FAISS row position -> operation / tool / server
    embedding_manifest.json  what built it, so staleness can be detected

Vectors are L2-normalized before insertion and queries are L2-normalized before
search, which makes ``IndexFlatIP`` inner product exactly cosine similarity.

The row ordering in ``tools_metadata.json`` MUST match the order vectors were
added to FAISS -- ``load()`` verifies the counts agree and refuses to serve a
mismatched pair rather than returning silently wrong routes.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib
from typing import Any

import faiss
import numpy as np

from orchestrator import config
from orchestrator.registry import Registry, ToolRecord

METRIC = "inner_product"


class IndexMissingError(RuntimeError):
    pass


class IndexStaleError(RuntimeError):
    pass


class IndexInvalidError(RuntimeError):
    pass


@dataclasses.dataclass
class LoadedIndex:
    index: faiss.Index
    metadata: list[dict[str, Any]]
    manifest: dict[str, Any]

    @property
    def size(self) -> int:
        return self.index.ntotal

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        """Cosine-similarity search. ``query_vector`` is normalized here."""
        vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        vector = l2_normalize(vector)
        k = max(1, min(top_k, self.size))
        scores, positions = self.index.search(vector, k)

        results = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:  # FAISS pads with -1 when fewer than k exist
                continue
            entry = self.metadata[int(position)]
            results.append(
                {
                    "operation_id": entry["operation_id"],
                    "tool_id": entry["tool_id"],
                    "server_id": entry["server_id"],
                    "score": float(score),
                }
            )
        return results


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Unit-length rows, as float32. Zero rows are left alone."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (array / norms).astype(np.float32)


def build_index(vectors: np.ndarray) -> faiss.Index:
    """Build an ``IndexFlatIP`` over already-normalized vectors."""
    if vectors.dtype != np.float32:
        raise IndexInvalidError(f"FAISS requires float32 vectors, got {vectors.dtype}")
    if vectors.shape[1] != config.EMBED_DIMENSIONS:
        raise IndexInvalidError(
            f"Expected {config.EMBED_DIMENSIONS}-dimensional vectors, "
            f"got {vectors.shape[1]}"
        )
    index = faiss.IndexFlatIP(config.EMBED_DIMENSIONS)
    index.add(vectors)
    return index


def build_metadata(tools: list[ToolRecord]) -> list[dict[str, Any]]:
    """Row-position metadata, in the same order the vectors were added."""
    return [
        {
            "operation_id": tool.operation_id,
            "tool_id": tool.tool_id,
            "server_id": tool.server_id,
            "name": tool.name,
            "taxonomy_class": tool.taxonomy_class,
            "taxonomy_type": tool.taxonomy_type,
        }
        for tool in tools
    ]


def build_manifest(
    registry: Registry,
    tools: list[ToolRecord],
    tool_limit: int | None,
) -> dict[str, Any]:
    return {
        "model": config.EMBED_MODEL,
        "dimensions": config.EMBED_DIMENSIONS,
        "normalized": True,
        "metric": METRIC,
        "tool_count": len(tools),
        "active_tool_count": len(tools),
        "registry_tool_count": len(registry.tools),
        "tool_limit": tool_limit,
        "branch": registry.branch,
        "qos_enabled": registry.qos_enabled,
        "registry_hash": registry.hash(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def save(
    index: faiss.Index,
    metadata: list[dict[str, Any]],
    manifest: dict[str, Any],
    cache_dir: pathlib.Path | None = None,
) -> dict[str, pathlib.Path]:
    """Persist the three artifacts to the (bind-mounted) cache directory."""
    cache_dir = cache_dir or config.CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    index_path = cache_dir / "tools.faiss"
    metadata_path = cache_dir / "tools_metadata.json"
    manifest_path = cache_dir / "embedding_manifest.json"

    faiss.write_index(index, str(index_path))
    # Keyed by stringified row position, matching FAISS insertion order.
    metadata_path.write_text(
        json.dumps({str(i): row for i, row in enumerate(metadata)}, indent=2),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "index": index_path,
        "metadata": metadata_path,
        "manifest": manifest_path,
    }


def _read_metadata(path: pathlib.Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    # Sort numerically -- JSON object key order is not guaranteed and "10"
    # sorts before "2" lexicographically.
    return [raw[key] for key in sorted(raw, key=int)]


def load(
    registry: Registry,
    cache_dir: pathlib.Path | None = None,
) -> LoadedIndex:
    """Load and validate the persisted index. Never re-embeds implicitly.

    Raises :class:`IndexMissingError` if artifacts are absent,
    :class:`IndexStaleError` if the registry changed since the build, and
    :class:`IndexInvalidError` if the artifacts disagree with each other.
    """
    cache_dir = cache_dir or config.CACHE_DIR
    index_path = cache_dir / "tools.faiss"
    metadata_path = cache_dir / "tools_metadata.json"
    manifest_path = cache_dir / "embedding_manifest.json"

    missing = [p.name for p in (index_path, metadata_path, manifest_path) if not p.exists()]
    if missing:
        raise IndexMissingError(
            f"FAISS index not found (missing: {', '.join(missing)} in {cache_dir}). "
            "Run python -m scripts.build_index"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = _read_metadata(metadata_path)
    index = faiss.read_index(str(index_path))

    if manifest.get("registry_hash") != registry.hash():
        raise IndexStaleError(
            "FAISS index is stale because tools.json changed.\n"
            f"  manifest registry_hash: {manifest.get('registry_hash')}\n"
            f"  current  registry_hash: {registry.hash()}\n"
            "Rebuild the index: python -m scripts.build_index"
        )

    if manifest.get("model") != config.EMBED_MODEL:
        raise IndexStaleError(
            f"Index was built with embedding model {manifest.get('model')!r} but the "
            f"orchestrator is configured for {config.EMBED_MODEL!r}. Rebuild the index."
        )

    if manifest.get("dimensions") != config.EMBED_DIMENSIONS:
        raise IndexStaleError(
            f"Index dimension {manifest.get('dimensions')} != configured "
            f"{config.EMBED_DIMENSIONS}. Rebuild the index."
        )

    if index.d != config.EMBED_DIMENSIONS:
        raise IndexInvalidError(
            f"FAISS index dimension {index.d} != configured {config.EMBED_DIMENSIONS}."
        )

    if index.ntotal != len(metadata):
        raise IndexInvalidError(
            f"FAISS holds {index.ntotal} vectors but tools_metadata.json describes "
            f"{len(metadata)} rows. The artifacts are out of sync; rebuild the index."
        )

    if manifest.get("tool_count") != len(metadata):
        raise IndexInvalidError(
            f"Manifest tool_count {manifest.get('tool_count')} != metadata rows "
            f"{len(metadata)}. Rebuild the index."
        )

    # The active set must be a subset of the registry (it is a subset, not an
    # equality, because --tool-limit builds a deterministic slice).
    known_operations = {tool.operation_id for tool in registry.tools}
    unknown = [
        row["operation_id"] for row in metadata if row["operation_id"] not in known_operations
    ]
    if unknown:
        raise IndexInvalidError(
            f"Index references operations absent from the registry: {unknown[:5]}. "
            "Rebuild the index."
        )

    return LoadedIndex(index=index, metadata=metadata, manifest=manifest)
