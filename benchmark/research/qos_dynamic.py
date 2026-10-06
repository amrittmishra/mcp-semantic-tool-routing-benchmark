"""Does provider quality become provider traffic when nobody is told the answer?

    python -m benchmark.research.qos_dynamic --experiment all

``qos_incentive`` measures the *static* market: quality is known, allocation is
computed once, and fidelity asks whether the allocation rule respects a ranking
it was handed. That establishes the rule is order-preserving. It does not
establish that the system can find the ranking.

This experiment removes the oracle. Every provider starts from an identical
uninformative profile, so at request zero the orchestrator cannot distinguish
the best provider from the worst. Quality exists only inside the simulated
execution, and reaches the registry one observation at a time through the same
``QoSRegistry.observe`` path the live orchestrator calls after a downstream
tool returns (``orchestrator/server.py``). Nothing here reimplements scoring or
allocation: selection is ``QoSRegistry.select`` with the shipped task-class
weights from ``data/providers.json``.

The loop is the one specified in the manuscript, run to convergence:

    select provider -> execute -> observe outcome -> update reputation -> repeat

Generative model. A provider is defined by hidden parameters (success
probability, mean grade of a successful execution, schema validity, latency,
cost). The registry never sees them. It sees sampled outcomes, and its EWMA
converges to the *observable expectation* of those parameters -- for example
observed correctness is ``p_success * correctness``, because a failed execution
produces nothing to grade and is recorded as a zero, which is what the live
orchestrator does on ``DownstreamError``. ``Truth.observable`` states that fixed
point explicitly, and the oracle ranking is computed from it, so "the learner
converged" and "the learner was right" are the same claim rather than two.

Availability and capability fidelity are held constant across providers: neither
is observable from a single execution (the first needs liveness probing, the
second needs declared-versus-observed behaviour comparison), so a per-execution
loop cannot learn them and pretending otherwise would flatter the result.

Experiments
  static        can traffic share recover the true ranking from nothing?
  degradation   the best provider silently gets worse. How fast does traffic
                leave, and how much is misallocated before it does?
  cold-start    a superior provider joins after the market has converged. Does
                it ever get tried?
  task-class    same providers, same measurements, different task weights.
                Does the market's answer change with the question?
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter
from typing import Any

import numpy as np

from orchestrator.qos import POLICY_ARGMAX, POLICY_PROPORTIONAL, POLICY_SOFTMAX
from orchestrator.qos_sim import POLICY_RANDOM, Market as _Market, Truth, simulate

OPERATION = "CONTESTED_OP"

WINDOW = 200        # trailing requests used for "where is traffic going right now"
ADAPT_WINDOW = 50   # finer window for detecting a migration; also its resolution floor


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation with *average* ranks for ties.

    ``qos_incentive.spearman`` breaks ties by array position, which is harmless
    when every provider receives traffic. Here it is not: a starving policy
    leaves several providers on exactly zero, and position-broken ties would
    score that as a perfect ranking whenever the input happens to be sorted --
    crediting a policy for an order it never expressed. Averaged ranks make the
    starved providers indistinguishable, which is the truth about them.
    """
    if len(a) < 2:
        return float("nan")

    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(len(x), dtype=float)
        sorted_x = x[order]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
            i = j + 1
        return r

    ra, rb = ranks(np.asarray(a, dtype=float)), ranks(np.asarray(b, dtype=float))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom else float("nan")


def _fidelity(run, oracle: dict[str, float]) -> float:
    """Rank correlation between true quality and the traffic a provider received.

    The market rewards quality exactly at 1.0; at 0.0 traffic is unrelated to
    quality and no rational provider would invest in it.
    """
    providers = sorted(oracle)
    traffic = run.traffic
    return spearman(
        np.array([oracle[p] for p in providers]),
        np.array([float(traffic[p]) for p in providers]),
    )


class Market(_Market):
    """The experiment's market: one synthetic operation, four competing providers."""

    def __init__(self, truths: tuple[Truth, ...] = None, **kwargs: Any) -> None:
        super().__init__(MARKET if truths is None else truths,
                         operation_id=OPERATION, **kwargs)


# Four providers with genuinely different trade-offs: alpha is the most reliable
# and most expensive, beta is the fastest, gamma is cheap and mediocre, delta is
# bad at everything except price. Under the `default` task class the true
# ranking is alpha > beta > gamma > delta; under `interactive` and `budget` it
# is not, which is what the task-class experiment exercises.
MARKET: tuple[Truth, ...] = (
    Truth("provider_alpha", p_success=0.98, correctness=0.96, p_schema=0.99,
          latency_ms=120.0, cost_units=1.20),
    Truth("provider_beta", p_success=0.93, correctness=0.86, p_schema=0.99,
          latency_ms=90.0, cost_units=1.00),
    Truth("provider_gamma", p_success=0.88, correctness=0.80, p_schema=0.97,
          latency_ms=200.0, cost_units=0.80),
    Truth("provider_delta", p_success=0.70, correctness=0.62, p_schema=0.93,
          latency_ms=260.0, cost_units=0.60),
)

# Post-degradation alpha: same latency and cost, collapsed reliability. The
# signature of a provider that quietly broke rather than one that went offline,
# which is the case availability monitoring does not catch.
DEGRADED_ALPHA = MARKET[0].degraded(success=0.55, correctness=0.45, schema=0.90)

# The newcomer: strictly better than the converged incumbent on every learnable
# component except cost.
NEWCOMER = Truth("provider_epsilon", p_success=0.99, correctness=0.97, p_schema=0.995,
                 latency_ms=80.0, cost_units=0.90)


# --- reporting helpers -----------------------------------------------------

def _short(provider_id: str) -> str:
    return provider_id.replace("provider_", "")


def _fmt_share(share: dict[str, float], order: list[str]) -> str:
    return "  ".join(f"{_short(p):>6} {share.get(p, 0.0)*100:5.1f}%" for p in order)


def _bar(fraction: float, width: int = 26) -> str:
    return "#" * int(round(fraction * width))


def _ranked(oracle: dict[str, float]) -> list[str]:
    return [p for p, _ in sorted(oracle.items(), key=lambda kv: -kv[1])]


def _window_leader(choices: list[str], end: int, window: int = WINDOW) -> str | None:
    segment = choices[max(0, end - window):end]
    if len(segment) < window:
        return None
    return Counter(segment).most_common(1)[0][0]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _mean_share(shares: list[dict[str, float]], providers: list[str]) -> dict[str, float]:
    return {p: _mean([s.get(p, 0.0) for s in shares]) for p in providers}


# Policies compared throughout. Random is a floor, not a proposal; argmax and
# softmax are the shipped ones. Proportional is included because it is the
# obvious middle option and turns out to be the wrong one.
STATIC_POLICIES = (
    (POLICY_RANDOM, 0.0), (POLICY_ARGMAX, 0.0), (POLICY_PROPORTIONAL, 0.0),
    (POLICY_SOFTMAX, 0.25), (POLICY_SOFTMAX, 0.10), (POLICY_SOFTMAX, 0.05),
)


def _label(policy: str, tau: float) -> str:
    return f"{policy} t={tau:g}" if policy == POLICY_SOFTMAX else policy


# --- experiment 1: can the ranking be found at all? ------------------------

def experiment_static(requests: int, seeds: list[int], task_class: str,
                      prior: str = "neutral") -> dict[str, Any]:
    print(f"\n{'='*94}\nSTATIC MARKET -- four providers, hidden quality, {requests} requests, "
          f"{len(seeds)} seeds, task class '{task_class}', {prior} prior\n{'='*94}")

    oracle = Market(seed=seeds[0], task_class=task_class, prior=prior).oracle()
    order = _ranked(oracle)
    print("\ntrue quality, unknown to the router at request 0 "
          "(every provider starts from an identical profile):")
    for p in order:
        print(f"  {_short(p):>6}  Q = {oracle[p]:.4f}")

    header = (f"{'policy':>14}{'fidelity':>10}{'starved':>9}{'found best':>12}{'success':>9}"
              f"{'latency':>9}{'regret/req':>12}   mean traffic share (true rank order)")
    print(f"\n{header}\n{'-'*len(header)}")

    results: dict[str, Any] = {}
    for policy, tau in STATIC_POLICIES:
        runs = []
        for seed in seeds:
            market = Market(seed=seed, task_class=task_class, prior=prior)
            runs.append(simulate(market, requests, policy, tau))

        label = _label(policy, tau)
        # "found best" is the honest convergence question: at the end of the
        # run, is the provider actually receiving the traffic the best one?
        found = [
            max(r.share(start=requests - min(requests, 1000)).items(), key=lambda kv: kv[1])[0]
            == order[0]
            for r in runs
        ]
        results[label] = {
            "fidelity": _mean([_fidelity(r, oracle) for r in runs]),
            "starved": _mean([r.starved(oracle) for r in runs]),
            "found_best_rate": float(np.mean(found)),
            "success_rate": _mean([float(np.mean(r.successes)) for r in runs]),
            "mean_latency_ms": _mean([float(np.mean(r.latencies)) for r in runs]),
            "regret_per_request": _mean([float(np.mean(r.regret)) for r in runs]),
            "share": _mean_share([r.share() for r in runs], order),
            "final_share": _mean_share(
                [r.share(start=requests - min(requests, 1000)) for r in runs], order
            ),
        }
        r = results[label]
        print(f"{label:>14}{r['fidelity']:>10.3f}{r['starved']:>9.2f}"
              f"{r['found_best_rate']*100:>11.0f}%{r['success_rate']*100:>8.1f}%"
              f"{r['mean_latency_ms']:>8.0f}m{r['regret_per_request']:>12.4f}   "
              f"{_fmt_share(r['share'], order)}")

    print("\nfidelity   rank correlation between true quality and traffic received, "
          "averaged over seeds")
    print("starved    providers that received zero traffic, and so could never be evaluated")
    print("found best fraction of seeds whose final-1000-request leader is the true best provider")

    best_tau = max(
        (l for l in results if l.startswith("softmax")),
        key=lambda l: (results[l]["found_best_rate"], -results[l]["regret_per_request"]),
    )
    print(f"\nconverged allocation under {best_tau}, mean over the final 1000 requests:")
    for p in order:
        frac = results[best_tau]["final_share"].get(p, 0.0)
        print(f"  {_short(p):>6}  {_bar(frac):<26} {frac*100:5.1f}%   (true Q {oracle[p]:.3f})")

    market = Market(seed=seeds[0], task_class=task_class, prior=prior)
    simulate(market, requests, POLICY_ARGMAX, 0.0)
    learned = market.learned()
    print(f"\nwhy argmax fails, seed {seeds[0]}: learned quality and observation count "
          "after the run")
    print(f"  {'provider':>8}{'learned':>10}{'true':>10}{'observations':>14}")
    for p in order:
        provider = market.registry.get(OPERATION, p)
        print(f"  {_short(p):>8}{learned[p]:>10.4f}{oracle[p]:>10.4f}"
              f"{provider.observations:>14}")
    print("  A provider argmax never selects keeps its prior forever, so its estimate is")
    print("  not wrong in a way further evidence can correct -- there is no further evidence.")

    return {"oracle": oracle, "policies": results}


# --- experiment 2: does the market notice when quality drops? --------------

def experiment_degradation(requests: int, seeds: list[int], task_class: str) -> dict[str, Any]:
    change_at = requests // 2
    print(f"\n{'='*94}\nDEGRADATION -- alpha's success collapses 0.98 -> 0.55 at request "
          f"{change_at}, {len(seeds)} seeds\n{'='*94}")

    before = Market(seed=seeds[0], task_class=task_class).oracle()
    degraded_market = Market(seed=seeds[0], task_class=task_class)
    degraded_market.degrade(DEGRADED_ALPHA)
    after = degraded_market.oracle()
    new_best = _ranked(after)[0]
    print(f"\ntrue best before: {_short(_ranked(before)[0])} (Q {max(before.values()):.3f})"
          f"   after: {_short(new_best)} (Q {max(after.values()):.3f}); "
          f"alpha falls to Q {after['provider_alpha']:.3f}")
    print("Latency and cost are unchanged: this is a provider that quietly broke, not one")
    print("that went offline, so availability monitoring would not catch it.")

    header = (f"{'policy':>14}{'testable':>10}{'adapted':>9}{'adaptation':>12}{'wasted':>9}"
              f"{'phase-2 success':>17}{'phase-2 regret':>16}")
    print(f"\n{header}\n{'-'*len(header)}")

    results: dict[str, Any] = {}
    for policy, tau in ((POLICY_ARGMAX, 0.0), (POLICY_SOFTMAX, 0.25),
                        (POLICY_SOFTMAX, 0.10), (POLICY_SOFTMAX, 0.05)):
        adaptations, wasted, successes, regrets, shares = [], [], [], [], []
        testable = 0
        for seed in seeds:
            market = Market(seed=seed, task_class=task_class)
            # Two phases against one registry: the oracle switches with the
            # truth, so phase-2 regret is scored against the post-change best,
            # while the learner keeps the reputation it built before the change.
            run = simulate(market, change_at, policy, tau)
            market.degrade(DEGRADED_ALPHA)
            tail = simulate(market, requests - change_at, policy, tau)
            run.choices += tail.choices
            run.successes += tail.successes
            run.regret += tail.regret

            successes.append(float(np.mean(run.successes[change_at:])))
            regrets.append(float(np.mean(run.regret[change_at:])))
            shares.append(run.share(start=change_at))

            # A seed only tests adaptation if the policy was actually sending
            # traffic to alpha when alpha broke. Where it had converged
            # elsewhere there is nothing to migrate, and counting those as
            # instant adaptations would credit a policy for its own earlier
            # failure to find the best provider.
            if _window_leader(run.choices, change_at) != "provider_alpha":
                continue
            testable += 1
            adaptations.append(next(
                (t - change_at for t in range(change_at + ADAPT_WINDOW, requests + 1)
                 if _window_leader(run.choices, t, ADAPT_WINDOW) == new_best),
                None,
            ))
            wasted.append(
                run.choices[change_at:change_at + (adaptations[-1] or (requests - change_at))]
                .count("provider_alpha")
            )

        label = _label(policy, tau)
        hit = [a for a in adaptations if a is not None]
        results[label] = {
            "testable_seeds": testable,
            "adapted_rate": len(hit) / testable if testable else float("nan"),
            "adaptation_requests": _mean(hit),
            "wasted_on_degraded": _mean(wasted),
            "phase2_success": _mean(successes),
            "phase2_regret": _mean(regrets),
            "phase2_share": _mean_share(shares, _ranked(after)),
        }
        r = results[label]
        shown = "never" if not hit else f"{r['adaptation_requests']:.0f} req"
        print(f"{label:>14}{testable:>7}/{len(seeds):<2}{r['adapted_rate']*100:>8.0f}%{shown:>12}"
              f"{r['wasted_on_degraded']:>9.0f}{r['phase2_success']*100:>16.1f}%"
              f"{r['phase2_regret']:>16.4f}")

    print("\ntestable   seeds where alpha actually held the traffic when it broke; the rest")
    print("           cannot test migration and are excluded from the three columns after it")
    print(f"adapted    fraction of those seeds where traffic migrated to {_short(new_best)}")
    print(f"adaptation requests until the trailing-{ADAPT_WINDOW} leader is the new best "
          f"({ADAPT_WINDOW} is the resolution floor)")
    print("wasted     requests still sent to the degraded provider during that interval")
    return {"before": before, "after": after, "new_best": new_best, "policies": results}


# --- experiment 3: does a newcomer ever get tried? -------------------------

def experiment_cold_start(requests: int, seeds: list[int], task_class: str) -> dict[str, Any]:
    join_at = int(requests * 0.75)
    print(f"\n{'='*94}\nCOLD START -- a better provider joins at request {join_at}, after the "
          f"market has converged, {len(seeds)} seeds\n{'='*94}")

    reference = Market(seed=seeds[0], task_class=task_class)
    reference.join(NEWCOMER)
    oracle = reference.oracle()
    print(f"\nnewcomer true quality Q = {oracle['provider_epsilon']:.4f}; best incumbent "
          f"{_short(_ranked(oracle)[1])} Q = {sorted(oracle.values())[-2]:.4f}. "
          "It enters with a prior, not a measurement.")

    header = (f"{'prior':>11}{'policy':>16}{'discovered':>12}{'discovery':>11}"
              f"{'tried in 500':>14}{'final share':>13}{'post-join regret':>18}")
    print(f"\n{header}\n{'-'*len(header)}")

    results: dict[str, Any] = {}
    for prior in ("neutral", "optimistic"):
        for policy, tau in ((POLICY_ARGMAX, 0.0), (POLICY_SOFTMAX, 0.05),
                            (POLICY_SOFTMAX, 0.10), (POLICY_SOFTMAX, 0.25)):
            discoveries, tried, finals, regrets = [], [], [], []
            for seed in seeds:
                market = Market(seed=seed, prior=prior, task_class=task_class)
                run = simulate(market, join_at, policy, tau)
                market.join(NEWCOMER)
                tail = simulate(market, requests - join_at, policy, tau)
                run.choices += tail.choices

                discoveries.append(next(
                    (t - join_at for t in range(join_at + ADAPT_WINDOW, requests + 1)
                     if _window_leader(run.choices, t, ADAPT_WINDOW) == "provider_epsilon"),
                    None,
                ))
                after = run.choices[join_at:]
                tried.append(after[:500].count("provider_epsilon"))
                finals.append(run.share(start=requests - min(len(after), 1000))
                              .get("provider_epsilon", 0.0))
                regrets.append(float(np.mean(tail.regret)))

            label = _label(policy, tau)
            hit = [d for d in discoveries if d is not None]
            results[f"{prior}/{label}"] = {
                "discovered_rate": len(hit) / len(seeds),
                "discovery_requests": _mean(hit),
                "tried_in_first_500": _mean(tried),
                "final_share": _mean(finals),
                "post_join_regret": _mean(regrets),
            }
            r = results[f"{prior}/{label}"]
            shown = "never" if not hit else f"{r['discovery_requests']:.0f} req"
            print(f"{prior:>11}{label:>16}{r['discovered_rate']*100:>11.0f}%{shown:>11}"
                  f"{r['tried_in_first_500']:>14.1f}{r['final_share']*100:>12.1f}%"
                  f"{r['post_join_regret']:>18.4f}")

    print(f"\ndiscovered  fraction of seeds where the newcomer ever leads the "
          f"trailing-{ADAPT_WINDOW} window")
    print("tried       executions the newcomer received in its first 500 requests of market time")
    print("regret      measured after entry, so it prices the exploration that found the newcomer")
    return {"oracle": oracle, "policies": results}


# --- experiment 4: does the answer depend on the question? -----------------

def experiment_task_class(requests: int, seeds: list[int]) -> dict[str, Any]:
    print(f"\n{'='*94}\nTASK-CONDITIONED ALLOCATION -- one market, four weightings, "
          f"{len(seeds)} seeds\n{'='*94}")

    header = f"{'task class':>13}{'true best':>12}{'learned best':>14}{'share':>8}   mean allocation"
    print(f"\n{header}\n{'-'*len(header)}")

    results: dict[str, Any] = {}
    for task_class in ("default", "analysis", "interactive", "budget"):
        oracle = Market(seed=seeds[0], task_class=task_class).oracle()
        order = _ranked(oracle)
        runs = [
            simulate(Market(seed=seed, task_class=task_class), requests, POLICY_SOFTMAX, 0.10)
            for seed in seeds
        ]
        share = _mean_share([r.share(start=requests - min(requests, 1000)) for r in runs], order)
        winner = max(share, key=share.get)
        results[task_class] = {
            "true_best": order[0],
            "learned_best": winner,
            "winner_share": share[winner],
            "fidelity": _mean([_fidelity(r, oracle) for r in runs]),
            "share": share,
        }
        print(f"{task_class:>13}{_short(order[0]):>12}{_short(winner):>14}"
              f"{share[winner]*100:>7.1f}%   {_fmt_share(share, order)}")

    distinct = {r["learned_best"] for r in results.values()}
    agree = sum(r["true_best"] == r["learned_best"] for r in results.values())
    print(f"\n{agree} of {len(results)} task classes converged on the true best provider; "
          f"{len(distinct)} distinct winners overall "
          f"({', '.join(sorted(_short(p) for p in distinct))}).")
    print("Same providers, same measurements, different gamma_{t,j}. The weighting decides,")
    print("which is why the weights are per task class and not global.")
    return results


EXPERIMENTS = ("static", "degradation", "cold-start", "task-class")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dynamic QoS marketplace: reputation learned from execution"
    )
    parser.add_argument("--experiment", default="all", choices=(*EXPERIMENTS, "all"))
    parser.add_argument("--requests", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=int, default=20,
                        help="number of consecutive seeds to average over")
    parser.add_argument("--task-class", default="default")
    parser.add_argument("--prior", default="neutral", choices=("neutral", "optimistic"),
                        help="initial profile for an unmeasured provider")
    parser.add_argument("--json", type=pathlib.Path, help="write the measured numbers here")
    args = parser.parse_args()

    seeds = [args.seed + i for i in range(max(1, args.seeds))]
    wanted = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    out: dict[str, Any] = {"requests": args.requests, "seeds": seeds,
                           "task_class": args.task_class,
                           "prior": args.prior}

    if "static" in wanted:
        out["static"] = experiment_static(args.requests, seeds, args.task_class, args.prior)
    if "degradation" in wanted:
        out["degradation"] = experiment_degradation(args.requests, seeds, args.task_class)
    if "cold-start" in wanted:
        out["cold_start"] = experiment_cold_start(args.requests, seeds, args.task_class)
    if "task-class" in wanted:
        out["task_class"] = experiment_task_class(args.requests, seeds)

    if args.json:
        args.json.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
