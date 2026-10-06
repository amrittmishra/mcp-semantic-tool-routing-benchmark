"""Rebuild the FAISS index from data/tools.json with gemini-embedding-001.

    python -m scripts.build_index

Requires Vertex AI credentials (see .env.example). The shipped data/cache/
already contains the index, so this is only needed to re-embed from scratch.
"""

from __future__ import annotations

from orchestrator import config, embeddings, index as index_module
from orchestrator.registry import embedding_document, load_registry


def main() -> None:
    registry = load_registry()
    tools = registry.tools
    print(f"Loaded {len(tools)} tools; embedding with {config.EMBED_MODEL} "
          f"({config.EMBED_DIMENSIONS} dims)")
    vectors = embeddings.embed_documents([embedding_document(t) for t in tools])
    if vectors.shape[0] != len(tools):
        raise RuntimeError("embedding count does not match tool count")
    faiss_index = index_module.build_index(index_module.l2_normalize(vectors))
    paths = index_module.save(
        faiss_index,
        index_module.build_metadata(tools),
        index_module.build_manifest(registry, tools, None),
    )
    for path in paths.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
