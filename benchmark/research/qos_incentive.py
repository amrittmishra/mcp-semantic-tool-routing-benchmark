"""Does routing error break the QoS incentive?

    python -m benchmark.research.qos_incentive

The Branch 2 thesis is a marketplace: many providers implement the same
operation, the router picks the operation, a QoS stage picks the provider, and
providers are thereby incentivised to raise quality. That only works if the
traffic a provider receives is genuinely a function of its quality.

Routing error attacks exactly that, in two very different ways:

  RANDOM misroutes spray traffic over unrelated operations. Annoying, but it is
  noise -- it washes out in expectation and merely slows learning.

  SYSTEMATIC misroutes are the real danger. The measured errors are not random:
  PAPER_METADATA lost to PAPER_CITATIONS on 3 of 3 queries; MESSAGE_SEARCH to
  CHANNEL_HISTORY on 2 of 2. A provider on the winning side of a systematic
  confusion collects a permanent subsidy it never earned, and one on the losing
  side is punished no matter how good it becomes. That is a broken market, not
  a noisy one.

This simulation is driven by the REAL top-5 routing scores recorded in
benchmark/results.jsonl -- 505 actual queries with actual cosine similarities --
so the routing behaviour is measured, and only the providers are synthetic.

Metrics
  incentive fidelity    Spearman rank correlation between a provider's true
                        quality and the traffic it receives, computed within each
                        operation and averaged. 1.0 = the market rewards quality
                        exactly. 0.0 = traffic is unrelated to quality, and no
                        rational provider would invest in QoS.
  improvement gradient  extra traffic share won by raising one's own quality
                        +0.10. This is the number a provider actually feels.
  routing accuracy      fraction of queries that reached the correct operation
                        after the QoS stage had its say.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

RESULTS_PATH = pathlib.Path(__file__).resolve().parents[1] / "results.jsonl"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, no scipy dependency."""
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


class Market:
    """Real routing candidates + synthetic competing providers."""

    def __init__(self, providers_per_op: int = 4, seed: int = 0):
        rows = [json.loads(l) for l in RESULTS_PATH.read_text().splitlines() if l.strip()]
        self.ops = sorted({r["expected_operation"] for r in rows}
                          | {c["operation_id"] for r in rows for c in r["candidates"]})
        self.op_index = {o: i for i, o in enumerate(self.ops)}

        self.truth = np.array([self.op_index[r["expected_operation"]] for r in rows])
        # Real candidate lists: (operation index, cosine score) per query.
        self.cand_ops = [np.array([self.op_index[c["operation_id"]] for c in r["candidates"]])
                         for r in rows]
        self.cand_scores = [np.array([c["score"] for c in r["candidates"]]) for r in rows]

        rng = np.random.default_rng(seed)
        self.M = providers_per_op
        self.n_ops = len(self.ops)
        # True quality of each provider, uniform so rank correlation is meaningful.
        self.quality = rng.uniform(0.05, 0.95, size=(self.n_ops, self.M))

    def run(self, top_k: int, w_quality: float, quality=None) -> dict:
        """Route each query, let QoS choose, and tally who got paid.

        w_quality blends the QoS score into the selection:
            score = (1 - w) * normalized_routing_score + w * quality
        w=0 is pure routing (QoS only breaks ties inside the winning operation);
        w=1 ignores routing entirely and always picks the highest-quality
        provider among the candidates, wherever it lives.
        """
        quality = self.quality if quality is None else quality
        traffic = np.zeros((self.n_ops, self.M))
        correct = 0

        for q in range(len(self.truth)):
            ops = self.cand_ops[q][:top_k]
            scores = self.cand_scores[q][:top_k]
            # Normalize routing scores within the candidate set so the blend is
            # scale-free; cosine values here sit in a narrow ~0.5-0.8 band.
            span = scores.max() - scores.min()
            routing = (scores - scores.min()) / span if span > 1e-9 else np.ones_like(scores)

            best_value, best = -np.inf, (ops[0], 0)
            for slot, op in enumerate(ops):
                for m in range(self.M):
                    value = (1 - w_quality) * routing[slot] + w_quality * quality[op, m]
                    if value > best_value:
                        best_value, best = value, (op, m)

            op, m = best
            traffic[op, m] += 1
            correct += int(op == self.truth[q])

        return {
            "traffic": traffic,
            "routing_accuracy": correct / len(self.truth),
            "fidelity": self._fidelity(traffic, quality),
        }

    def _fidelity(self, traffic: np.ndarray, quality: np.ndarray) -> float:
        """Within-operation quality/traffic rank correlation, averaged.

        Measured within an operation because that is where a provider actually
        competes; across operations, traffic differences reflect demand, not merit.
        """
        per_op = []
        for op in range(self.n_ops):
            if traffic[op].sum() == 0:
                continue
            r = spearman(quality[op], traffic[op])
            if not np.isnan(r):
                per_op.append(r)
        return float(np.mean(per_op)) if per_op else float("nan")

    def gradient(self, top_k: int, w_quality: float, bump: float = 0.10,
                 samples: int = 40, seed: int = 1) -> float:
        """Traffic share a provider gains by improving its own quality by +bump."""
        rng = np.random.default_rng(seed)
        base = self.run(top_k, w_quality)["traffic"]
        total = base.sum()
        gains = []
        for _ in range(samples):
            op = int(rng.integers(self.n_ops))
            m = int(rng.integers(self.M))
            q2 = self.quality.copy()
            q2[op, m] = min(1.0, q2[op, m] + bump)
            after = self.run(top_k, w_quality, quality=q2)["traffic"]
            gains.append((after[op, m] - base[op, m]) / total)
        return float(np.mean(gains))


def main() -> int:
    parser = argparse.ArgumentParser(description="QoS incentive under routing noise")
    parser.add_argument("--providers", type=int, default=4)
    parser.add_argument("--gradient", action="store_true",
                        help="Also estimate the improvement gradient (slow)")
    args = parser.parse_args()

    if not RESULTS_PATH.exists():
        raise SystemExit(f"Need {RESULTS_PATH}. Run python -m benchmark.run first.")

    market = Market(providers_per_op=args.providers)
    print(f"{len(market.truth)} real queries | {market.n_ops} operations | "
          f"{args.providers} synthetic providers each "
          f"({market.n_ops * args.providers} providers competing)\n")

    print("Routing accuracy is what fraction of queries reached the CORRECT operation")
    print("after QoS had its say. Fidelity is whether the market rewards quality.\n")

    header = f"{'top-k':>6}{'w_quality':>11}{'routing acc':>13}{'fidelity':>11}"
    print(header)
    print("-" * len(header))

    best_combo = None
    for top_k in (1, 2, 3, 5):
        for w in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0):
            if top_k == 1 and w not in (0.0, 0.5, 1.0):
                continue  # inside one operation, w only rescales the same ranking
            out = market.run(top_k, w)
            print(f"{top_k:>6}{w:>11.2f}{out['routing_accuracy']*100:>12.2f}%"
                  f"{out['fidelity']:>11.3f}")
            score = out["routing_accuracy"] + 0.5 * out["fidelity"]
            if best_combo is None or score > best_combo[0]:
                best_combo = (score, top_k, w, out)
        print()

    _, k, w, out = best_combo
    print(f"best joint operating point: top-k={k}, w_quality={w:.2f} -> "
          f"routing {out['routing_accuracy']*100:.2f}%, fidelity {out['fidelity']:.3f}")

    if args.gradient:
        print("\nimprovement gradient (traffic share gained per +0.10 quality):")
        for top_k, w in ((1, 0.5), (3, 0.2), (3, 0.5)):
            g = market.gradient(top_k, w)
            print(f"  top-k={top_k} w={w:.2f}:  {g*100:+.4f}% of all traffic")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
