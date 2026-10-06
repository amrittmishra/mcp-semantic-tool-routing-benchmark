"""One vector per operation, or several?

    python -m benchmark.research.multivector

The shipped index embeds one document per operation: name + taxonomy + purpose
+ positive examples + negative examples, all concatenated. Concatenation averages
the phrasings together, so a query that matches one example strongly gets diluted
by four that it does not match at all.

This tests the alternative: give each operation SEVERAL vectors -- the summary
document plus one vector per positive example -- and score an operation by its
best-matching vector. That is late interaction / max-pooling, the idea behind
multi-vector retrievers, applied at the capability level.

Also tested:
  * task type for the example vectors. Examples are phrased like queries, so
    embedding them as RETRIEVAL_QUERY (symmetric) may beat RETRIEVAL_DOCUMENT.
  * negative examples as repulsors: subtract similarity to what the capability
    explicitly says it is NOT for.

Benchmark queries were authored independently of positive_examples, so using
those examples as index-side data is legitimate rather than leakage.
"""

from __future__ import annotations

import pathlib

import numpy as np

from benchmark.research.common import Corpus, mcnemar, normalize, top1_accuracy, topk_recall
from orchestrator import config, embeddings


def embed_cached(texts: list[str], task_type: str, tag: str) -> np.ndarray:
    path = config.CACHE_DIR / f"research_{tag}.npy"
    if path.exists():
        cached = np.load(path)
        if cached.shape[0] == len(texts):
            return cached
    print(f"  embedding {len(texts)} texts as {task_type} -> {path.name}")
    vectors = normalize(embeddings.embed_texts(texts, task_type=task_type))
    np.save(path, vectors)
    return vectors


def build_example_matrix(corpus: Corpus, attr: str, task_type: str, tag: str):
    """Flat matrix of all examples plus the operation index each belongs to."""
    texts, owner = [], []
    for i, tool in enumerate(corpus.tools):
        for example in getattr(tool, attr):
            texts.append(example)
            owner.append(i)
    if not texts:
        return None, None
    return embed_cached(texts, task_type, tag), np.array(owner)


def pool(scores_flat: np.ndarray, owner: np.ndarray, n_ops: int, how: str) -> np.ndarray:
    """Collapse per-example scores down to one score per operation."""
    out = np.full((scores_flat.shape[0], n_ops), -np.inf, dtype=np.float32)
    for op in range(n_ops):
        cols = np.where(owner == op)[0]
        if len(cols) == 0:
            continue
        block = scores_flat[:, cols]
        out[:, op] = block.max(axis=1) if how == "max" else block.mean(axis=1)
    return out


def main() -> int:
    corpus = Corpus()
    Q, V = corpus.Q, corpus.V
    n = corpus.n_ops

    print(f"{len(corpus.dataset)} queries | {n} operations")
    n_pos = sum(len(t.positive_examples) for t in corpus.tools)
    n_neg = sum(len(t.negative_examples) for t in corpus.tools)
    print(f"{n_pos} positive examples, {n_neg} negative examples\n")

    baseline = Q @ V.T
    base_correct = baseline.argmax(axis=1) == corpus.y
    variants: dict[str, np.ndarray] = {"baseline (1 vector/op)": baseline}

    for task, tag in (("RETRIEVAL_DOCUMENT", "pos_doc"), ("RETRIEVAL_QUERY", "pos_query")):
        E, owner = build_example_matrix(corpus, "positive_examples", task, tag)
        flat = Q @ E.T
        ex_max = pool(flat, owner, n, "max")
        ex_mean = pool(flat, owner, n, "mean")
        short = "doc" if tag == "pos_doc" else "qry"

        variants[f"examples only, max  [{short}]"] = ex_max
        variants[f"examples only, mean [{short}]"] = ex_mean
        for a in (0.3, 0.5, 0.7):
            variants[f"blend {a:.1f}*summary + max [{short}]"] = a * baseline + (1 - a) * ex_max
        # Late interaction: the operation's score is its single best vector,
        # summary document included.
        variants[f"max(summary, examples) [{short}]"] = np.maximum(baseline, ex_max)

    N, nowner = build_example_matrix(
        corpus, "negative_examples", "RETRIEVAL_DOCUMENT", "neg_doc"
    )
    if N is not None:
        neg_max = pool(Q @ N.T, nowner, n, "max")
        for lam in (0.1, 0.2, 0.3):
            variants[f"baseline - {lam:.1f}*negatives"] = baseline - lam * neg_max

        Epos, powner = build_example_matrix(
            corpus, "positive_examples", "RETRIEVAL_QUERY", "pos_query"
        )
        best_pos = np.maximum(baseline, pool(Q @ Epos.T, powner, n, "max"))
        for lam in (0.1, 0.2):
            variants[f"max(summary,examples) - {lam:.1f}*negatives"] = best_pos - lam * neg_max

    header = f"{'variant':<42}{'top-1':>9}{'top-3':>9}{'fixed':>7}{'broke':>7}{'p':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for name, scores in variants.items():
        t1 = top1_accuracy(scores, corpus.y)
        t3 = topk_recall(scores, corpus.y, 3)
        correct = scores.argmax(axis=1) == corpus.y
        if name.startswith("baseline (1"):
            print(f"{name:<42}{t1*100:8.2f}%{t3*100:8.2f}%{'-':>7}{'-':>7}{'-':>10}")
            continue
        fixed, broke, p = mcnemar(correct, base_correct)
        flag = " *" if p < 0.05 else ""
        print(f"{name:<42}{t1*100:8.2f}%{t3*100:8.2f}%{fixed:>7}{broke:>7}{p:>9.4f}{flag}")
        rows.append((name, t1, t3, p, fixed, broke))

    best = max(rows, key=lambda r: r[1])
    print(f"\nbest: {best[0]}")
    print(f"  top-1 {best[1]*100:.2f}% vs baseline {top1_accuracy(baseline, corpus.y)*100:.2f}%"
          f"  (+{(best[1]-top1_accuracy(baseline, corpus.y))*100:.2f} points)")
    print(f"  top-3 {best[2]*100:.2f}%,  fixed {best[4]}, broke {best[5]},  p={best[3]:.4f}")
    print("  SIGNIFICANT" if best[3] < 0.05 else "  not significant -- treat as a wash")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
