"""One-factor sensitivity sweeps for the QoS marketplace (paper Table 10).

    python -m benchmark.research.qos_sensitivity
    python -m benchmark.research.qos_sensitivity --requests 10000 --seeds 20 --json out.json

``qos_dynamic`` varies the allocation policy, the temperature, and the prior on
one fixed market. This varies the market instead, one provider attribute at a
time, with everything else held at the ``qos_dynamic`` values:

  quality   beta's success rate and output correctness, from clearly worse
            than alpha to clearly better (latency and cost unchanged)
  cost      alpha's cost per call, from cheapest to five times the base
  latency   alpha's latency, from fastest to eight times the base

Each sweep is scored under the ``default`` task-class weights and, for cost and
latency, also under the class that weights that attribute most (``budget``,
``interactive``), because whether a cost or latency change should move traffic
depends on gamma_{t,j}. For every cell we report the true best provider, the
provider the learner ends up sending most traffic to, how often it is the true
best across seeds, the swept provider's final traffic share, and regret per
request. Two exploration settings are run per cell: softmax at tau = 0.10 with
an uninformative prior, and argmax with an optimistic prior (the two settings
that found the best provider in every seed of the static experiment).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from collections import Counter
from typing import Any

import numpy as np

from benchmark.research.qos_dynamic import MARKET, Market
from orchestrator.qos import POLICY_ARGMAX, POLICY_SOFTMAX
from orchestrator.qos_sim import simulate

SETTINGS = (
    ("softmax t=0.10, uninformative", POLICY_SOFTMAX, 0.10, "neutral"),
    ("argmax, optimistic", POLICY_ARGMAX, 0.0, "optimistic"),
)

BASE = {t.provider_id: t for t in MARKET}


def _with(provider_id: str, **changes: Any) -> tuple:
    return tuple(dataclasses.replace(t, **changes) if t.provider_id == provider_id else t
                 for t in MARKET)


SWEEPS: dict[str, dict[str, Any]] = {
    "quality": {
        "provider": "provider_beta",
        "label": "beta success / correctness",
        "task_classes": ("default",),
        "values": [
            ("0.85 / 0.76", _with("provider_beta", p_success=0.85, correctness=0.76)),
            ("0.93 / 0.86 (base)", MARKET),
            ("0.97 / 0.93", _with("provider_beta", p_success=0.97, correctness=0.93)),
            ("0.99 / 0.97", _with("provider_beta", p_success=0.99, correctness=0.97)),
            ("1.00 / 0.99", _with("provider_beta", p_success=1.00, correctness=0.99)),
        ],
    },
    "cost": {
        "provider": "provider_alpha",
        "label": "alpha cost per call",
        "task_classes": ("default", "budget"),
        "values": [
            ("0.6", _with("provider_alpha", cost_units=0.6)),
            ("1.2 (base)", MARKET),
            ("2.4", _with("provider_alpha", cost_units=2.4)),
            ("3.6", _with("provider_alpha", cost_units=3.6)),
            ("6.0", _with("provider_alpha", cost_units=6.0)),
        ],
    },
    "latency": {
        "provider": "provider_alpha",
        "label": "alpha latency (ms)",
        "task_classes": ("default", "interactive"),
        "values": [
            ("60", _with("provider_alpha", latency_ms=60.0)),
            ("120 (base)", MARKET),
            ("240", _with("provider_alpha", latency_ms=240.0)),
            ("480", _with("provider_alpha", latency_ms=480.0)),
            ("960", _with("provider_alpha", latency_ms=960.0)),
        ],
    },
}


def _short(p: str) -> str:
    return p.replace("provider_", "")


def run_cell(truths: tuple, task_class: str, policy: str, tau: float, prior: str,
             requests: int, seeds: list[int], swept: str) -> dict[str, Any]:
    oracle = Market(truths=truths, seed=seeds[0], task_class=task_class, prior=prior).oracle()
    true_best = max(oracle, key=oracle.get)
    tail = min(requests, 1000)
    leaders, swept_share, regret, best_share = [], [], [], []
    for seed in seeds:
        market = Market(truths=truths, seed=seed, task_class=task_class, prior=prior)
        run = simulate(market, requests, policy, tau)
        share = run.share(start=requests - tail)
        leaders.append(max(share.items(), key=lambda kv: kv[1])[0])
        swept_share.append(share.get(swept, 0.0))
        best_share.append(share.get(true_best, 0.0))
        regret.append(float(np.mean(run.regret)))
    modal = Counter(leaders).most_common(1)[0][0]
    return {
        "oracle": {p: round(q, 4) for p, q in sorted(oracle.items(), key=lambda kv: -kv[1])},
        "true_best": true_best,
        "learned_leader": modal,
        "found_best_rate": float(np.mean([l == true_best for l in leaders])),
        "swept_final_share": float(np.mean(swept_share)),
        "best_final_share": float(np.mean(best_share)),
        "regret_per_request": float(np.mean(regret)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="QoS one-factor sensitivity sweeps")
    parser.add_argument("--requests", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sweep", choices=(*SWEEPS, "all"), default="all")
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()
    seeds = [args.seed + i for i in range(args.seeds)]
    wanted = SWEEPS if args.sweep == "all" else {args.sweep: SWEEPS[args.sweep]}

    out: dict[str, Any] = {"requests": args.requests, "seeds": seeds, "sweeps": {}}
    for name, sweep in wanted.items():
        swept = sweep["provider"]
        for task_class in sweep["task_classes"]:
            print(f"\n=== {name} sweep: {sweep['label']} | task class '{task_class}' "
                  f"| {args.requests} requests x {len(seeds)} seeds")
            print(f"{'value':>20}  {'true best':>9}  {'setting':>30}  {'leader':>7}  "
                  f"{'found':>6}  {_short(swept)+' share':>12}  {'regret':>7}")
            for value_label, truths in sweep["values"]:
                for setting, policy, tau, prior in SETTINGS:
                    cell = run_cell(truths, task_class, policy, tau, prior,
                                    args.requests, seeds, swept)
                    out["sweeps"].setdefault(name, {}).setdefault(task_class, {}) \
                        .setdefault(value_label, {})[setting] = cell
                    print(f"{value_label:>20}  {_short(cell['true_best']):>9}  {setting:>30}  "
                          f"{_short(cell['learned_leader']):>7}  "
                          f"{cell['found_best_rate']*100:>5.0f}%  "
                          f"{cell['swept_final_share']*100:>11.1f}%  "
                          f"{cell['regret_per_request']:>7.4f}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
