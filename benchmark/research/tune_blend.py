"""Is the summary+examples blend a real gain, or test-set overfitting?

    python -m benchmark.research.tune_blend

multivector.py found `0.7*summary + 0.3*max(examples)` at +1.58 points -- but
the weight was chosen by looking at the same 505 queries it was scored on. That
is selection bias, and reporting it as-is would be dishonest.

This tunes the weight on one split and reports accuracy on the other, so every
number below is held out from the choice that produced it. Splits are stratified
by operation: each operation's 5 queries are divided across folds, so no fold is
missing a class.

Also fits the negative-example repulsion weight jointly, on the training fold
only.
"""

from __future__ import annotations

import numpy as np

from benchmark.research.common import Corpus, mcnemar, top1_accuracy, topk_recall
from benchmark.research.multivector import build_example_matrix, pool

BLEND_GRID = np.round(np.arange(0.30, 1.01, 0.05), 2)
NEG_GRID = np.round(np.arange(0.00, 0.31, 0.05), 2)


def stratified_folds(y: np.ndarray, n_folds: int, seed: int = 0) -> np.ndarray:
    """Assign each query a fold, balanced within every operation."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for op in np.unique(y):
        idx = np.where(y == op)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % n_folds
    return fold


def combined(baseline, ex_max, neg_max, alpha, lam):
    score = alpha * baseline + (1 - alpha) * ex_max
    return score if lam == 0 else score - lam * neg_max


def main() -> int:
    corpus = Corpus()
    Q, V, y = corpus.Q, corpus.V, corpus.y
    n = corpus.n_ops

    baseline = Q @ V.T
    E, owner = build_example_matrix(corpus, "positive_examples", "RETRIEVAL_DOCUMENT", "pos_doc")
    ex_max = pool(Q @ E.T, owner, n, "max")
    N, nowner = build_example_matrix(corpus, "negative_examples", "RETRIEVAL_DOCUMENT", "neg_doc")
    neg_max = pool(Q @ N.T, nowner, n, "max")

    print(f"{len(y)} queries | {n} operations")
    print(f"baseline top-1: {top1_accuracy(baseline, y)*100:.2f}%\n")

    for n_folds in (2, 5):
        folds = stratified_folds(y, n_folds)
        held_out_correct = np.zeros(len(y), dtype=bool)
        picks = []

        for f in range(n_folds):
            train = folds != f
            test = folds == f

            best, best_acc = (0.7, 0.0), -1.0
            for alpha in BLEND_GRID:
                for lam in NEG_GRID:
                    s = combined(baseline[train], ex_max[train], neg_max[train], alpha, lam)
                    acc = top1_accuracy(s, y[train])
                    if acc > best_acc:
                        best_acc, best = acc, (alpha, lam)

            alpha, lam = best
            picks.append((alpha, lam, best_acc))
            s = combined(baseline[test], ex_max[test], neg_max[test], alpha, lam)
            held_out_correct[test] = s.argmax(axis=1) == y[test]

        base_correct = baseline.argmax(axis=1) == y
        acc = held_out_correct.mean()
        fixed, broke, p = mcnemar(held_out_correct, base_correct)

        chosen = ", ".join(f"a={a:.2f}/l={l:.2f}" for a, l, _ in picks)
        print(f"{n_folds}-fold cross-validated")
        print(f"  weights chosen per fold: {chosen}")
        print(f"  HELD-OUT top-1: {acc*100:.2f}%   "
              f"(baseline {base_correct.mean()*100:.2f}%, "
              f"{(acc - base_correct.mean())*100:+.2f} points)")
        print(f"  fixed {fixed}, broke {broke}, p={p:.4f}"
              f"  {'SIGNIFICANT' if p < 0.05 else 'not significant'}\n")

    # For reference: the full-data optimum, which is the number NOT to quote.
    best_alpha, best_lam, best_acc = 0.0, 0.0, -1.0
    for alpha in BLEND_GRID:
        for lam in NEG_GRID:
            acc = top1_accuracy(combined(baseline, ex_max, neg_max, alpha, lam), y)
            if acc > best_acc:
                best_acc, best_alpha, best_lam = acc, alpha, lam
    print(f"in-sample optimum (OVERFIT, do not quote): alpha={best_alpha:.2f} "
          f"lam={best_lam:.2f} -> {best_acc*100:.2f}%")

    # Sensitivity: how flat is the surface around the optimum?
    print("\nblend sensitivity at lam=0 (in-sample, for shape only):")
    for alpha in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
        s = combined(baseline, ex_max, neg_max, alpha, 0.0)
        print(f"  alpha={alpha:.2f}  top-1 {top1_accuracy(s, y)*100:.2f}%  "
              f"top-3 {topk_recall(s, y, 3)*100:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
