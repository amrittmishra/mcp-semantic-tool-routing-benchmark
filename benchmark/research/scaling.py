"""How does routing accuracy move as the catalogue grows?

    python -m benchmark.research.scaling

The central claim of the router architecture is that the agent's cost is O(1)
in catalogue size. That is trivially true of the *schema* it sees. This asks the
harder question: does routing QUALITY hold up, or does it decay as N grows and
the vector space fills with near neighbours?

Runs entirely on cached vectors, so each point is a genuine subset evaluation
rather than an extrapolation: for a subset S we score only the queries whose
expected operation is in S, against only the operations in S. Random subsets at
each N give error bars; the deterministic `select_tools` subset is the one the
shipped `--tool-limit` flag actually produces.
"""

from __future__ import annotations

import numpy as np

from benchmark.research.common import Corpus
from orchestrator.registry import select_tools


def evaluate_subset(corpus: Corpus, subset: np.ndarray) -> tuple[float, float, int]:
    """Top-1 and top-3 restricted to `subset` (indices into corpus.operations)."""
    mask = np.isin(corpus.y, subset)
    if not mask.any():
        return float("nan"), float("nan"), 0

    Q = corpus.Q[mask]
    y_global = corpus.y[mask]
    scores = Q @ corpus.V[subset].T

    local_of_global = {g: i for i, g in enumerate(subset)}
    y_local = np.array([local_of_global[g] for g in y_global])

    top1 = float((scores.argmax(axis=1) == y_local).mean())
    k = min(3, len(subset))
    order = np.argsort(-scores, axis=1)[:, :k]
    top3 = float(np.mean([y_local[i] in order[i] for i in range(len(y_local))]))
    return top1, top3, int(mask.sum())


def main() -> int:
    corpus = Corpus()
    sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 101]
    trials = 40
    rng = np.random.default_rng(0)

    print(f"{len(corpus.dataset)} queries | {corpus.n_ops} operations | "
          f"vectors cached, no API calls\n")
    print("Random subsets (mean +/- 1 s.d. over 40 draws) vs the deterministic")
    print("--tool-limit subset the build script actually produces.\n")

    header = (f"{'N':>5}{'queries':>9}{'top-1 random':>18}{'top-3 random':>16}"
              f"{'top-1 determ.':>15}{'top-3 determ.':>15}")
    print(header)
    print("-" * len(header))

    curve = []
    for n in sizes:
        if n >= corpus.n_ops:
            subset = np.arange(corpus.n_ops)
            t1, t3, cnt = evaluate_subset(corpus, subset)
            print(f"{n:>5}{cnt:>9}{t1*100:>13.2f}%{'':>4}{t3*100:>15.2f}%"
                  f"{t1*100:>14.2f}%{t3*100:>14.2f}%")
            curve.append((n, t1, 0.0, t1))
            continue

        t1s, t3s = [], []
        for _ in range(trials):
            subset = np.sort(rng.choice(corpus.n_ops, n, replace=False))
            t1, t3, _ = evaluate_subset(corpus, subset)
            t1s.append(t1)
            t3s.append(t3)

        det_ops = [t.operation_id for t in select_tools(corpus.registry.tools, n)]
        det_subset = np.sort(np.array([corpus.op_index[o] for o in det_ops]))
        dt1, dt3, cnt = evaluate_subset(corpus, det_subset)

        print(f"{n:>5}{cnt:>9}{np.mean(t1s)*100:>11.2f}% +/-{np.std(t1s)*100:4.2f}"
              f"{np.mean(t3s)*100:>15.2f}%{dt1*100:>14.2f}%{dt3*100:>14.2f}%")
        curve.append((n, float(np.mean(t1s)), float(np.std(t1s)), dt1))

    # Is the trend real? Fit accuracy against log N over the random draws.
    ns = np.array([c[0] for c in curve], dtype=float)
    accs = np.array([c[1] for c in curve])
    slope, intercept = np.polyfit(np.log(ns), accs, 1)
    print(f"\nlinear fit of top-1 against log(N):  slope = {slope*100:+.2f} points per e-fold")
    print(f"  extrapolated to N=1000:  {(intercept + slope*np.log(1000))*100:.1f}%")
    print(f"  extrapolated to N=10000: {(intercept + slope*np.log(10000))*100:.1f}%")
    print("\n(Extrapolation assumes new capabilities are as separable as these 101.")
    print(" A denser catalogue with more near-duplicates would decay faster --")
    print(" that is what the collision detector is for.)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
