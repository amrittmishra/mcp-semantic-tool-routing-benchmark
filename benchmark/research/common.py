"""Shared loading for the research experiments.

Everything is driven off two cached arrays -- the 101 operation vectors already
inside tools.faiss, and the 505 benchmark query vectors cached by
hierarchy_experiment. No experiment below re-embeds anything, so sweeps are
free and byte-for-byte reproducible.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from orchestrator import config, index as index_module
from orchestrator.registry import Registry, load_registry

BENCHMARK_DIR = pathlib.Path(__file__).resolve().parents[1]
DATASET_PATH = BENCHMARK_DIR / "dataset.jsonl"


class Corpus:
    """Operation vectors, query vectors, and the labels tying them together."""

    def __init__(self) -> None:
        self.registry: Registry = load_registry()
        loaded = index_module.load(self.registry)

        self.operations: list[str] = [row["operation_id"] for row in loaded.metadata]
        self.op_index = {op: i for i, op in enumerate(self.operations)}
        self.V: np.ndarray = np.vstack(
            [loaded.index.reconstruct(i) for i in range(loaded.size)]
        )

        self.dataset = [
            json.loads(line)
            for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.dataset = [r for r in self.dataset if r["expected_operation"] in self.op_index]
        self.queries = [r["query"] for r in self.dataset]
        self.y = np.array([self.op_index[r["expected_operation"]] for r in self.dataset])

        self.Q = self._query_vectors()

        by_op = {t.operation_id: t for t in self.registry.tools}
        self.tools = [by_op[op] for op in self.operations]
        self.server_of = np.array([t.server_id for t in self.tools])

    def _query_vectors(self) -> np.ndarray:
        cached = sorted(config.CACHE_DIR.glob("benchmark_query_vectors_*.npy"))
        if not cached:
            raise SystemExit(
                "No cached query vectors. Run python -m benchmark.hierarchy_experiment once."
            )
        Q = np.load(cached[-1])
        if Q.shape[0] != len(self.dataset):
            raise SystemExit(
                f"Cached query vectors ({Q.shape[0]}) do not match the dataset "
                f"({len(self.dataset)}). Delete {cached[-1].name} and re-run."
            )
        return Q

    @property
    def n_ops(self) -> int:
        return len(self.operations)


def normalize(X: np.ndarray) -> np.ndarray:
    return index_module.l2_normalize(np.asarray(X, dtype=np.float32))


def top1_accuracy(scores: np.ndarray, y: np.ndarray) -> float:
    return float((scores.argmax(axis=1) == y).mean())


def topk_recall(scores: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    order = np.argpartition(-scores, min(k, scores.shape[1] - 1), axis=1)[:, :k]
    return float(np.mean([y[i] in order[i] for i in range(len(y))]))


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar. Returns (a-only wins, b-only wins, p)."""
    from math import comb

    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return a_only, b_only, p
