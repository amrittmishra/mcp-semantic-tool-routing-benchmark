"""BM25, RRF and score-interpolation baselines on the 505-query benchmark (PID 1135 revision).

    REGISTRY_PATH=data/tools.json CACHE_DIR=data/cache python -m benchmark.research.lexical_hybrid

Lexical side: Okapi BM25 (k1=1.5, b=0.75) over the exact documents the dense
index embeds (orchestrator.registry.embedding_document).
Dense side: either the saved FAISS top-5 lists in benchmark/results.jsonl
(always available) or, when --dense-matrix is given, a full 505x101 cosine
matrix saved as .npz (keys: scores) from the rebuilt vectors.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

from orchestrator.registry import embedding_document, load_registry

ROOT = Path(__file__).resolve().parent


def terms(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(terms(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avg = self.len.mean()
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.tf))
        qt = set(terms(query))
        for i, c in enumerate(self.tf):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            for t in qt:
                f = c.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out[i] = s
        return out


def mcnemar(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (centre - half), 100 * (centre + half)


def report(name: str, order: np.ndarray, y: np.ndarray, flat_correct: np.ndarray, out: dict) -> None:
    top1 = order[:, 0] == y
    top3 = np.array([y[i] in order[i, :3] for i in range(len(y))])
    fixed = int((top1 & ~flat_correct).sum())
    broke = int((~top1 & flat_correct).sum())
    p = mcnemar(fixed, broke)
    lo, hi = wilson(int(top1.sum()), len(y))
    print(f"{name:34s} top1 {top1.sum():3d}/505 = {100*top1.mean():6.2f}%  CI [{lo:.1f},{hi:.1f}]"
          f"  top3 {top3.sum():3d} = {100*top3.mean():6.2f}%  fixed/broke {fixed}/{broke}  p={p:.4g}")
    out[name] = {"top1": int(top1.sum()), "top3": int(top3.sum()), "fixed": fixed, "broke": broke,
                 "p_vs_flat": p, "ci95": [lo, hi]}


def minmax(v: np.ndarray) -> np.ndarray:
    span = v.max() - v.min()
    return (v - v.min()) / span if span > 1e-12 else np.zeros_like(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-matrix", type=Path, default=None,
                    help=".npz with key 'scores' (505x101 dense cosine), rows aligned to dataset order")
    ap.add_argument("--out", type=Path, default=ROOT / "lexical_hybrid_results.json")
    args = ap.parse_args()

    reg = load_registry()
    ops = [t.operation_id for t in reg.tools]
    op_i = {o: i for i, o in enumerate(ops)}
    docs = [embedding_document(t) for t in reg.tools]
    bm25 = BM25(docs)

    dataset = [json.loads(l) for l in open(ROOT.parents[0] / "dataset.jsonl") if l.strip()]
    results = {r["id"]: r for r in (json.loads(l) for l in open(ROOT.parents[0] / "results.jsonl") if l.strip())}
    y = np.array([op_i[r["expected_operation"]] for r in dataset])
    n, m = len(dataset), len(ops)

    # lexical score matrix
    L = np.vstack([bm25.scores(r["query"]) for r in dataset])
    # dense: saved top-5 lists (partial) and optional full matrix
    D_top5 = np.full((n, m), np.nan)
    dense_rank = np.full((n, m), np.inf)
    for qi, r in enumerate(dataset):
        for rank, c in enumerate(results[r["id"]]["candidates"]):
            D_top5[qi, op_i[c["operation_id"]]] = c["score"]
            dense_rank[qi, op_i[c["operation_id"]]] = rank + 1
    D_full = None
    if args.dense_matrix:
        D_full = np.load(args.dense_matrix)["scores"]
        assert D_full.shape == (n, m)

    flat_order_saved = np.argsort(-np.nan_to_num(D_top5, nan=-1), axis=1)
    flat_correct = flat_order_saved[:, 0] == y
    out: dict = {}
    print(f"documents: {m}; queries: {n}; saved FAISS flat top1 = {flat_correct.sum()}")
    report("flat dense (saved top-5 lists)", flat_order_saved, y, flat_correct, out)
    if D_full is not None:
        report("flat dense (rebuilt full matrix)", np.argsort(-D_full, axis=1), y, flat_correct, out)

    # BM25 (ties broken by document index, like official_services.lexical_router)
    L_order = np.lexsort((np.arange(m)[None, :].repeat(n, 0), -L), axis=1) if False else np.argsort(-L, axis=1, kind="stable")
    report("BM25 k1=1.5 b=0.75", L_order, y, flat_correct, out)

    # RRF k=60 over BM25 top-5 and dense top-5
    bm_rank = np.full((n, m), np.inf)
    for qi in range(n):
        for rank, j in enumerate(L_order[qi, :5]):
            bm_rank[qi, j] = rank + 1
    for k in (60, 20):
        R = np.where(np.isfinite(bm_rank), 1 / (k + bm_rank), 0) + np.where(np.isfinite(dense_rank), 1 / (k + dense_rank), 0)
        # tie-break by dense score so equal-RRF candidates follow the dense order
        R_order = np.lexsort((-np.nan_to_num(D_top5, nan=-1), -R), axis=1)
        report(f"RRF k={k}, BM25 top5 + dense top5", R_order, y, flat_correct, out)
    if D_full is not None:
        full_dense_rank = np.argsort(np.argsort(-D_full, axis=1), axis=1) + 1
        full_bm_rank = np.argsort(np.argsort(-L, axis=1, kind="stable"), axis=1) + 1
        R = 1 / (60 + full_dense_rank) + 1 / (60 + full_bm_rank)
        report("RRF k=60, full rankings", np.lexsort((-D_full, -R), axis=1), y, flat_correct, out)

    # Score interpolation: alpha * norm(dense) + (1-alpha) * norm(bm25)
    for label, D in (("top-5 dense (missing=0)", np.nan_to_num(D_top5, nan=np.nan)),) + ((("full dense", D_full),) if D_full is not None else ()):
        for a in (0.0, 0.3, 0.5, 0.7, 1.0):
            S = np.zeros((n, m))
            for qi in range(n):
                d = D[qi].copy()
                if np.isnan(d).any():
                    known = ~np.isnan(d)
                    dn = np.zeros(m)
                    dn[known] = minmax(d[known]) if known.sum() > 1 else 1.0
                else:
                    dn = minmax(d)
                S[qi] = a * dn + (1 - a) * minmax(L[qi])
            report(f"interp a={a} [{label}]", np.lexsort((-np.nan_to_num(D, nan=-1), -S), axis=1), y, flat_correct, out)

    args.out.write_text(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
