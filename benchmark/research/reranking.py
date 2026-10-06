"""Can top-1 routing be improved without changing the embedding model?

    python -m benchmark.research.reranking

All techniques here are post-hoc transforms of vectors that already exist, so
they cost nothing at build time and microseconds at query time. Each is a known
fix for a known pathology of dense embeddings:

  centering    embeddings share a large common component that carries no
               discriminative signal; subtracting the mean spreads them out.
  abtt         "all but the top" -- remove the top-k principal directions,
               which are dominated by that same shared structure.
  hubness      some vectors are hubs: close to everything, so they win queries
               they should not. Penalise a document by how close it sits to its
               own neighbours.
  csls         cross-domain local scaling: the standard hubness correction,
               using each document's document-side neighbourhood as its
               expected-similarity baseline.

Nothing here sees a test label. The hubness statistics are computed from the
DOCUMENT side only, so there is no leakage from the benchmark queries.
"""

from __future__ import annotations

import numpy as np

from benchmark.research.common import Corpus, mcnemar, normalize, top1_accuracy, topk_recall


def center(V: np.ndarray, Q: np.ndarray, mu: np.ndarray | None = None):
    mu = V.mean(axis=0) if mu is None else mu
    return normalize(V - mu), normalize(Q - mu)


def all_but_the_top(V: np.ndarray, Q: np.ndarray, d: int = 2):
    """Mu & Viswanath (2018): remove the dominant principal directions."""
    mu = V.mean(axis=0)
    Vc = V - mu
    # Right singular vectors of the centered doc matrix are the top directions.
    _, _, Wt = np.linalg.svd(Vc, full_matrices=False)
    W = Wt[:d]

    def strip(X):
        Xc = X - mu
        return normalize(Xc - (Xc @ W.T) @ W)

    return strip(V), strip(Q)


def hubness_penalty(V: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean similarity of each document to its k nearest *other* documents."""
    S = V @ V.T
    np.fill_diagonal(S, -np.inf)
    neighbours = np.sort(S, axis=1)[:, -k:]
    return neighbours.mean(axis=1)


def score_variants(corpus: Corpus) -> dict[str, np.ndarray]:
    V, Q = corpus.V, corpus.Q
    out: dict[str, np.ndarray] = {"baseline": Q @ V.T}

    Vc, Qc = center(V, Q)
    out["centering"] = Qc @ Vc.T

    for d in (1, 2, 4):
        Va, Qa = all_but_the_top(V, Q, d=d)
        out[f"abtt d={d}"] = Qa @ Va.T

    base = Q @ V.T
    for k in (3, 5, 10):
        r = hubness_penalty(V, k=k)
        for lam in (0.25, 0.5, 1.0):
            out[f"csls k={k} lam={lam}"] = base - lam * r[None, :]

    # The two most promising ideas, composed.
    Vc2, Qc2 = center(V, Q)
    r = hubness_penalty(Vc2, k=5)
    out["centering + csls k=5 lam=0.5"] = (Qc2 @ Vc2.T) - 0.5 * r[None, :]

    Va, Qa = all_but_the_top(V, Q, d=2)
    r = hubness_penalty(Va, k=5)
    out["abtt d=2 + csls k=5 lam=0.5"] = (Qa @ Va.T) - 0.5 * r[None, :]

    return out


def main() -> int:
    corpus = Corpus()
    variants = score_variants(corpus)

    baseline = variants["baseline"]
    base_correct = baseline.argmax(axis=1) == corpus.y

    print(f"{len(corpus.dataset)} queries | {corpus.n_ops} operations | "
          f"no re-embedding, no labels used\n")
    header = f"{'technique':<32}{'top-1':>9}{'top-3':>9}{'fixed':>8}{'broke':>8}{'p':>10}"
    print(header)
    print("-" * len(header))

    results = []
    for name, scores in variants.items():
        t1 = top1_accuracy(scores, corpus.y)
        t3 = topk_recall(scores, corpus.y, 3)
        correct = scores.argmax(axis=1) == corpus.y
        if name == "baseline":
            print(f"{name:<32}{t1*100:8.2f}%{t3*100:8.2f}%{'-':>8}{'-':>8}{'-':>10}")
            continue
        fixed, broke, p = mcnemar(correct, base_correct)
        flag = " *" if p < 0.05 else ""
        print(f"{name:<32}{t1*100:8.2f}%{t3*100:8.2f}%{fixed:>8}{broke:>8}{p:>9.4f}{flag}")
        results.append((name, t1, p, fixed, broke))

    print("\nfixed = this technique got it right where baseline was wrong")
    print("broke = baseline was right, this technique wrong")
    print("p     = exact McNemar, paired on the same 505 queries.  * = p<0.05")

    best = max(results, key=lambda r: r[1])
    print(f"\nbest: {best[0]} at {best[1]*100:.2f}% "
          f"(baseline {top1_accuracy(baseline, corpus.y)*100:.2f}%), p={best[2]:.4f}")
    if best[2] >= 0.05:
        print("NOT statistically distinguishable from baseline -- do not ship it.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
