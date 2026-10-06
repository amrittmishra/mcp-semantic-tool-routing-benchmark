"""Does hierarchical (tree) routing beat flat routing over the same vectors?

    python -m benchmark.hierarchy_experiment

Flat routing searches all N operation vectors at once. Tree routing first picks
a taxonomy node, then searches only the operations under it. The tree is built
from the vectors already in the index -- a node's vector is the L2-normalized
centroid of its members -- so no capability is re-embedded and both arms are
compared over *identical* operation vectors. The only thing that varies is the
search procedure.

Query vectors are embedded once and cached to data/cache/, keyed by a hash of
the dataset, so every variant after the first run is free.

Variants:
  flat            argmax over all operations                      (the baseline)
  class-gate      best of 12 class centroids, then within it
  node-gate       best of 38 class>type centroids, then within it
  class-beam-k    union of the top-k classes, then within them
  node-beam-k     union of the top-k nodes, then within them
  soft-class      flat score + weight * parent class score        (no hard cut)

The metric that decides it is not just accuracy: a hard gate that picks the
wrong node makes the right answer *unreachable*, which flat search never does.
So we report gate accuracy and, crucially, how many answers each variant fixes
versus breaks relative to flat.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

import numpy as np

from orchestrator import config, embeddings, index as index_module
from orchestrator.registry import load_registry

BENCHMARK_DIR = pathlib.Path(__file__).resolve().parent
DATASET_PATH = BENCHMARK_DIR / "dataset.jsonl"


# --- query vector cache -----------------------------------------------------


def load_query_vectors(queries: list[str], cache_dir: pathlib.Path) -> np.ndarray:
    """Embed the benchmark queries once; reuse forever."""
    digest = hashlib.sha256("".join(queries).encode()).hexdigest()[:16]
    path = cache_dir / f"benchmark_query_vectors_{digest}.npy"

    if path.exists():
        vectors = np.load(path)
        if vectors.shape == (len(queries), config.EMBED_DIMENSIONS):
            print(f"Loaded cached query vectors: {path.name}")
            return vectors

    print(f"Embedding {len(queries)} queries as RETRIEVAL_QUERY (one time)...")
    vectors = embeddings.embed_texts(queries, task_type="RETRIEVAL_QUERY")
    vectors = index_module.l2_normalize(vectors)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    print(f"Cached to {path.name}")
    return vectors


# --- tree construction ------------------------------------------------------


def build_tree(operation_vectors: np.ndarray, operations: list[str], taxonomy: dict):
    """Group operations into class and class>type nodes; centroid each group.

    A centroid is the natural node vector here: it is the mean direction of the
    capabilities beneath it, so a query that is 'about' the node scores highly
    without any node ever being separately embedded or described.
    """
    groups: dict[str, dict[str, list[int]]] = {"class": {}, "node": {}}
    for i, operation in enumerate(operations):
        cls, typ = taxonomy[operation]
        groups["class"].setdefault(cls, []).append(i)
        groups["node"].setdefault(f"{cls}>{typ}", []).append(i)

    tree = {}
    for level, buckets in groups.items():
        names = sorted(buckets)
        centroids = np.vstack(
            [operation_vectors[buckets[name]].mean(axis=0) for name in names]
        )
        tree[level] = {
            "names": names,
            "centroids": index_module.l2_normalize(centroids),
            "members": [np.array(buckets[name]) for name in names],
            "of_operation": np.array(
                [names.index(_key(level, taxonomy[op])) for op in operations]
            ),
        }
    return tree


def _key(level: str, taxonomy_pair: tuple[str, str]) -> str:
    cls, typ = taxonomy_pair
    return cls if level == "class" else f"{cls}>{typ}"


# --- routing variants -------------------------------------------------------


def route_flat(query_vectors, operation_vectors, top_k=3):
    scores = query_vectors @ operation_vectors.T
    order = np.argsort(-scores, axis=1)[:, :top_k]
    return order, scores, None


def route_gated(query_vectors, operation_vectors, level_tree, beam=1, top_k=3):
    """Pick the top-`beam` nodes, then rank only the operations beneath them."""
    node_scores = query_vectors @ level_tree["centroids"].T
    chosen = np.argsort(-node_scores, axis=1)[:, :beam]

    operation_scores = query_vectors @ operation_vectors.T
    ranked = np.empty((len(query_vectors), top_k), dtype=int)

    for q in range(len(query_vectors)):
        candidates = np.concatenate([level_tree["members"][n] for n in chosen[q]])
        local = operation_scores[q, candidates]
        best = candidates[np.argsort(-local)][:top_k]
        # Pad when a node holds fewer than top_k operations.
        if len(best) < top_k:
            best = np.concatenate([best, np.full(top_k - len(best), -1)])
        ranked[q] = best

    return ranked, operation_scores, chosen


def route_soft(query_vectors, operation_vectors, level_tree, weight, top_k=3):
    """No hard cut: blend each operation's score with its parent node's score."""
    operation_scores = query_vectors @ operation_vectors.T
    node_scores = query_vectors @ level_tree["centroids"].T
    parent = node_scores[:, level_tree["of_operation"]]
    blended = (1 - weight) * operation_scores + weight * parent
    order = np.argsort(-blended, axis=1)[:, :top_k]
    return order, blended, None


# --- evaluation -------------------------------------------------------------


def evaluate(ranked, expected_idx, chosen_nodes, level_tree, flat_top1=None):
    top1 = ranked[:, 0]
    n = len(expected_idx)
    correct = top1 == expected_idx
    top3 = np.array([expected_idx[i] in ranked[i] for i in range(n)])

    result = {
        "top1": correct.mean(),
        "top3": top3.mean(),
        "gate": None,
        "fixed": None,
        "broken": None,
        "unreachable": None,
    }

    if chosen_nodes is not None and level_tree is not None:
        true_node = level_tree["of_operation"][expected_idx]
        in_beam = np.array(
            [true_node[i] in chosen_nodes[i] for i in range(n)]
        )
        result["gate"] = in_beam.mean()
        # The distinctive tree failure: the gate excluded the right answer, so
        # no amount of within-node ranking could recover it.
        result["unreachable"] = (~in_beam).sum()

    if flat_top1 is not None:
        flat_correct = flat_top1 == expected_idx
        result["fixed"] = int((correct & ~flat_correct).sum())
        result["broken"] = int((~correct & flat_correct).sum())

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Flat vs tree routing")
    parser.add_argument("--errors", action="store_true",
                        help="Show queries the tree broke that flat got right")
    args = parser.parse_args()

    registry = load_registry()
    loaded = index_module.load(registry)

    operations = [row["operation_id"] for row in loaded.metadata]
    operation_vectors = np.vstack(
        [loaded.index.reconstruct(i) for i in range(loaded.size)]
    )
    taxonomy = {
        tool.operation_id: (tool.taxonomy_class, tool.taxonomy_type)
        for tool in registry.tools
    }

    dataset = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset = [row for row in dataset if row["expected_operation"] in operations]
    queries = [row["query"] for row in dataset]
    expected_idx = np.array([operations.index(row["expected_operation"]) for row in dataset])

    query_vectors = load_query_vectors(queries, config.CACHE_DIR)
    tree = build_tree(operation_vectors, operations, taxonomy)

    print(f"\n{len(dataset)} queries | {len(operations)} operations | "
          f"{len(tree['class']['names'])} classes | {len(tree['node']['names'])} nodes")
    print("Tree nodes are centroids of the SAME operation vectors flat search uses,")
    print("so any difference is the search procedure, not the embeddings.\n")

    flat_ranked, _, _ = route_flat(query_vectors, operation_vectors)
    flat_top1 = flat_ranked[:, 0]
    baseline = evaluate(flat_ranked, expected_idx, None, None)

    rows = [("flat (baseline)", baseline, None)]

    for level in ("class", "node"):
        for beam in (1, 2, 3):
            ranked, _, chosen = route_gated(
                query_vectors, operation_vectors, tree[level], beam=beam
            )
            label = f"{level}-gate beam={beam}"
            rows.append((label, evaluate(ranked, expected_idx, chosen, tree[level], flat_top1), None))

    for level in ("class", "node"):
        for weight in (0.15, 0.30, 0.50):
            ranked, _, _ = route_soft(
                query_vectors, operation_vectors, tree[level], weight
            )
            label = f"soft-{level} w={weight:.2f}"
            rows.append((label, evaluate(ranked, expected_idx, None, None, flat_top1), None))

    header = f"{'variant':<22}{'top-1':>9}{'top-3':>9}{'gate':>9}{'fixed':>8}{'broke':>8}{'unreach':>9}"
    print(header)
    print("-" * len(header))
    for label, m, _ in rows:
        gate = f"{m['gate']*100:7.1f}%" if m["gate"] is not None else "       -"
        fixed = f"{m['fixed']:>7}" if m["fixed"] is not None else "      -"
        broke = f"{m['broken']:>7}" if m["broken"] is not None else "      -"
        unreach = f"{m['unreachable']:>8}" if m["unreachable"] is not None else "       -"
        print(f"{label:<22}{m['top1']*100:8.2f}%{m['top3']*100:8.2f}%{gate}{fixed} {broke} {unreach}")

    print("\ngate    = the right answer was inside the node(s) the gate admitted")
    print("fixed   = flat got it wrong, this variant got it right")
    print("broke   = flat got it right, this variant got it wrong")
    print("unreach = gate excluded the correct operation entirely (flat never does this)")

    if args.errors:
        ranked, _, chosen = route_gated(query_vectors, operation_vectors, tree["class"], beam=1)
        broke = [
            i for i in range(len(dataset))
            if ranked[i, 0] != expected_idx[i] and flat_top1[i] == expected_idx[i]
        ]
        print(f"\nBroken by class-gate beam=1 ({len(broke)}):")
        for i in broke[:15]:
            print(f"  {dataset[i]['expected_operation']:<24} -> {operations[ranked[i,0]]:<24}"
                  f"  {dataset[i]['query'][:44]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
