"""Simulated execution of a provider marketplace, for evaluation and demonstration.

The QoS stage in :mod:`orchestrator.qos` selects a provider from measured
profiles. This module supplies the other half of the loop -- executions that
produce measurements -- without requiring 10,000 real downstream calls to see
what the loop does over 10,000 requests.

A provider is defined by hidden generative parameters that the router never
reads: success probability, mean grade of a successful execution, schema
validity, latency, and cost. The learner starts from a prior, selects with
``QoSRegistry.select``, and folds each sampled outcome back through
``QoSRegistry.observe`` -- the same call the orchestrator makes after a
downstream tool returns. Nothing here reimplements scoring or allocation.

A failed execution is recorded as a schema violation and zero correctness,
matching the orchestrator's ``DownstreamError`` path, so the exponentially
weighted profile converges to the *observable expectation* of the hidden
parameters rather than to the parameters themselves. :meth:`Truth.observable`
states that fixed point, which is what makes "the learner converged" and "the
learner was right" the same claim.

Used by ``benchmark.research.qos_dynamic`` for the reported experiments and by
the orchestrator's ``/qos/simulate`` endpoint for the live UI panel.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import random
import tempfile
from collections import Counter
from typing import Any

from orchestrator import config
from orchestrator.qos import POLICY_SOFTMAX, Provider, QoSRegistry

# Neither is observable from a single execution -- the first needs liveness
# probing, the second needs declared-versus-observed behaviour comparison -- so
# a per-execution loop holds them fixed rather than pretending to learn them.
AVAILABILITY = 0.99
CAPABILITY_FIDELITY = 0.95

# Multiplicative noise on latency and cost, additive noise on the grade of a
# successful execution. Small enough that the ranking is learnable, large enough
# that a single observation is not decisive.
LATENCY_SIGMA = 0.25
COST_SIGMA = 0.15
CORRECTNESS_SIGMA = 0.03

POLICY_RANDOM = "random"

PRIORS: dict[str, dict[str, float]] = {
    # Uninformative: every provider looks identical, so the first selection
    # carries no information and min-max normalization is degenerate by
    # construction (all components map to 1.0).
    "neutral": {
        "success_rate": 0.50, "output_correctness": 0.50, "latency_ms": 200.0,
        "cost_units": 1.00, "schema_compliance": 0.50,
        "availability": AVAILABILITY, "capability_fidelity": CAPABILITY_FIDELITY,
    },
    # Optimistic initialization, the standard bandit remedy for starvation: an
    # unmeasured provider is assumed excellent, so it is tried and then demoted
    # on evidence rather than never tried at all.
    "optimistic": {
        "success_rate": 1.00, "output_correctness": 1.00, "latency_ms": 50.0,
        "cost_units": 0.50, "schema_compliance": 1.00,
        "availability": AVAILABILITY, "capability_fidelity": CAPABILITY_FIDELITY,
    },
}


@dataclasses.dataclass(frozen=True)
class Truth:
    """Hidden generative parameters of one provider. Never read by the router."""

    provider_id: str
    p_success: float
    correctness: float   # mean grade of a *successful* execution
    p_schema: float      # schema validity of a *successful* execution
    latency_ms: float
    cost_units: float
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_profile(cls, provider_id: str, profile: dict[str, float],
                     **meta: Any) -> "Truth":
        """Treat a measured profile as ground truth for a simulated market.

        The stored profile is an observable expectation, so the per-execution
        parameters are recovered by dividing out the success rate: a provider
        whose successes are always correct still shows an observed correctness
        of only ``p_success`` once failures are graded as zero.
        """
        p_success = max(1e-6, float(profile["success_rate"]))
        return cls(
            provider_id=provider_id,
            p_success=p_success,
            correctness=min(1.0, float(profile["output_correctness"]) / p_success),
            p_schema=min(1.0, float(profile["schema_compliance"]) / p_success),
            latency_ms=float(profile["latency_ms"]),
            cost_units=float(profile["cost_units"]),
            meta=meta,
        )

    def observable(self) -> dict[str, float]:
        """The profile an infinitely long EWMA would converge to."""
        return {
            "success_rate": self.p_success,
            "output_correctness": self.p_success * self.correctness,
            "latency_ms": self.latency_ms,
            "cost_units": self.cost_units,
            "schema_compliance": self.p_success * self.p_schema,
            "availability": AVAILABILITY,
            "capability_fidelity": CAPABILITY_FIDELITY,
        }

    def degraded(self, success: float, correctness: float,
                 schema: float | None = None) -> "Truth":
        """The same provider, quietly worse. Latency and cost are unchanged."""
        return dataclasses.replace(
            self, p_success=success, correctness=correctness,
            p_schema=self.p_schema if schema is None else schema,
        )

    def execute(self, rng: random.Random) -> dict[str, Any]:
        """Sample one execution outcome."""
        # Log-normal centred so the arithmetic mean is the declared value; a
        # median-centred draw would bias every learned latency upward.
        latency = rng.lognormvariate(
            math.log(self.latency_ms) - LATENCY_SIGMA**2 / 2, LATENCY_SIGMA
        )
        cost = rng.lognormvariate(
            math.log(self.cost_units) - COST_SIGMA**2 / 2, COST_SIGMA
        )
        if rng.random() >= self.p_success:
            return {
                "success": False, "latency_ms": latency, "cost_units": cost,
                "schema_ok": False, "correctness": 0.0,
            }
        return {
            "success": True,
            "latency_ms": latency,
            "cost_units": cost,
            "schema_ok": rng.random() < self.p_schema,
            "correctness": min(1.0, max(0.0, rng.gauss(self.correctness, CORRECTNESS_SIGMA))),
        }


def _provider_record(truth: Truth, profile: dict[str, float],
                     operation_id: str) -> dict[str, Any]:
    meta = truth.meta
    return {
        "provider_id": truth.provider_id,
        "display_name": meta.get("display_name", truth.provider_id.replace("provider_", "")),
        "tier": meta.get("tier", "simulated"),
        "incumbent": bool(meta.get("incumbent", False)),
        "server_id": meta.get("server_id", f"{truth.provider_id}_mcp"),
        "tool_id": meta.get("tool_id", f"{operation_id.lower()}__{truth.provider_id}"),
        "profile": dict(profile),
        "observations": 0,
    }


class Market:
    """A QoSRegistry driven by simulated executions of hidden-quality providers."""

    def __init__(
        self,
        truths: tuple[Truth, ...],
        operation_id: str = "CONTESTED_OP",
        prior: str = "neutral",
        task_class: str = "default",
        seed: int = 7,
        learning_rate: float = 0.05,
    ) -> None:
        self.operation_id = operation_id
        self.truth: dict[str, Truth] = {t.provider_id: t for t in truths}
        self.prior = PRIORS[prior]
        self.task_class = task_class
        self.learning_rate = learning_rate
        self.rng = random.Random(seed + 1)
        self._dir = tempfile.TemporaryDirectory(prefix="qos_sim_")
        self._task_classes = _task_classes()

        path = self._write(
            pathlib.Path(self._dir.name) / "market.json",
            {pid: self.prior for pid in self.truth},
        )
        self.registry = QoSRegistry(path, seed=seed)

    @classmethod
    def from_registry(
        cls,
        qos: QoSRegistry,
        operation_id: str,
        **kwargs: Any,
    ) -> "Market":
        """Build a sandbox market from the live provider set of one operation.

        The shipped profiles become the hidden truth and the learner is reset to
        a prior, so the demonstration asks whether those profiles could have been
        discovered rather than assuming them. The live registry is not mutated:
        this market owns its own QoSRegistry over a temporary file.
        """
        providers = qos.providers.get(operation_id)
        if not providers:
            raise ValueError(f"No providers registered for operation {operation_id!r}")
        truths = tuple(
            Truth.from_profile(
                p.provider_id, p.profile, display_name=p.display_name, tier=p.tier,
                incumbent=p.incumbent, server_id=p.server_id, tool_id=p.tool_id,
            )
            for p in providers
        )
        return cls(truths, operation_id=operation_id, **kwargs)

    def _write(self, path: pathlib.Path, profiles: dict[str, dict[str, float]]) -> pathlib.Path:
        path.write_text(
            json.dumps({
                "version": 1,
                "branch": "qos-simulation",
                "note": "generated by orchestrator.qos_sim; not a real registry",
                "task_classes": self._task_classes,
                "providers": {
                    self.operation_id: [
                        _provider_record(self.truth[pid], profile, self.operation_id)
                        for pid, profile in profiles.items()
                    ]
                },
            }),
            encoding="utf-8",
        )
        return path

    # --- ground truth -----------------------------------------------------

    def oracle(self) -> dict[str, float]:
        """True Q_{p|t}, scored by the shipped scorer over observable profiles.

        Built as a second QoSRegistry rather than by reimplementing the sum, so
        the oracle and the learner cannot disagree about normalization.
        """
        path = self._write(
            pathlib.Path(self._dir.name) / "oracle.json",
            {pid: t.observable() for pid, t in self.truth.items()},
        )
        oracle = QoSRegistry(path)
        return {p.provider_id: q for p, q in oracle.quality(self.operation_id, self.task_class)}

    def learned(self) -> dict[str, float]:
        return {
            p.provider_id: q
            for p, q in self.registry.quality(self.operation_id, self.task_class)
        }

    # --- mutation ---------------------------------------------------------

    def degrade(self, truth: Truth) -> None:
        """Silently change a provider's hidden behaviour. The registry is not told."""
        self.truth[truth.provider_id] = truth

    def join(self, truth: Truth) -> None:
        """A new provider enters with a prior, not a measurement."""
        self.truth[truth.provider_id] = truth
        self.registry.add_provider(
            self.operation_id,
            Provider(**_provider_record(truth, self.prior, self.operation_id)),
        )

    # --- one request ------------------------------------------------------

    def step(self, policy: str, temperature: float) -> tuple[str, dict[str, Any]]:
        if policy == POLICY_RANDOM:
            provider_id = self.rng.choice(list(self.truth))
        else:
            provider_id = self.registry.select(
                self.operation_id,
                task_class=self.task_class,
                policy=policy,
                temperature=temperature,
                deterministic=False,
            ).provider.provider_id

        outcome = self.truth[provider_id].execute(self.rng)
        self.registry.observe(
            self.operation_id,
            provider_id,
            success=outcome["success"],
            latency_ms=outcome["latency_ms"],
            schema_ok=outcome["schema_ok"],
            correctness=outcome["correctness"],
            cost_units=outcome["cost_units"],
            alpha=self.learning_rate,
        )
        return provider_id, outcome


def _task_classes() -> dict[str, dict[str, float]]:
    """The shipped gamma weights, so a simulation scores what production scores."""
    raw = json.loads(config.PROVIDERS_PATH.read_text(encoding="utf-8"))
    return raw["task_classes"]


@dataclasses.dataclass
class Run:
    policy: str
    temperature: float
    choices: list[str]
    successes: list[bool]
    latencies: list[float]
    regret: list[float]
    oracle_at_end: dict[str, float]
    checkpoints: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    @property
    def traffic(self) -> Counter:
        return Counter(self.choices)

    def share(self, start: int = 0, stop: int | None = None) -> dict[str, float]:
        window = self.choices[start:stop]
        counts = Counter(window)
        return {p: counts[p] / len(window) for p in sorted(set(self.choices))} if window else {}

    def starved(self, oracle: dict[str, float]) -> int:
        """Providers that received no traffic at all, and so no way to be judged.

        Rank correlation cannot see this: a policy that sends everything to one
        provider expresses no opinion about the rest, yet still scores a number
        for them.
        """
        traffic = self.traffic
        return sum(1 for p in oracle if traffic[p] == 0)


def simulate(
    market: Market,
    requests: int,
    policy: str = POLICY_SOFTMAX,
    temperature: float = 0.05,
    checkpoint_every: int | None = None,
    checkpoint_window: int = 200,
    offset: int = 0,
) -> Run:
    """Run the loop for one market phase.

    A phase has fixed ground truth, so the oracle is resolved once. The
    degradation and cold-start scenarios run two phases against the same live
    registry, which is what lets the learner carry its now-stale reputation
    across the change instead of being reset by it. ``offset`` only labels
    checkpoints, for a caller stitching phases into one timeline.
    """
    oracle = market.oracle()
    best = max(oracle.values())
    run = Run(policy, temperature, [], [], [], [], oracle)

    for t in range(requests):
        provider_id, outcome = market.step(policy, temperature)
        run.choices.append(provider_id)
        run.successes.append(outcome["success"])
        run.latencies.append(outcome["latency_ms"])
        # Quality regret: what this request gave up against the best available
        # provider, scored on true quality. Accumulates while the market is
        # wrong and flattens once it is right.
        run.regret.append(best - oracle[provider_id])

        if checkpoint_every and (t + 1) % checkpoint_every == 0:
            window = run.choices[max(0, t + 1 - checkpoint_window):t + 1]
            counts = Counter(window)
            learned = market.learned()
            run.checkpoints.append({
                "request": offset + t + 1,
                "share": {p: counts[p] / len(window) for p in market.truth},
                "learned": {p: round(learned.get(p, 0.0), 4) for p in market.truth},
                "observations": {
                    p: market.registry.get(market.operation_id, p).observations
                    for p in market.truth
                },
            })

    run.oracle_at_end = oracle
    return run


# --- scenarios -------------------------------------------------------------

# How badly a degrading provider breaks, and how good an entrant is. Both are
# deliberate: the degradation leaves latency and cost untouched so availability
# monitoring would not catch it, and the entrant is better than every incumbent
# so failing to find it is unambiguously a loss.
DEGRADE_SUCCESS = 0.55
DEGRADE_CORRECTNESS = 0.45
ENTRANT_ID = "provider_entrant"


def _entrant(operation_id: str) -> Truth:
    return Truth(
        ENTRANT_ID, p_success=0.99, correctness=0.97, p_schema=0.995,
        latency_ms=60.0, cost_units=0.70,
        meta={"display_name": "new entrant", "tier": "entrant",
              "server_id": f"{operation_id.lower()}_entrant_mcp"},
    )


def run_scenario(
    qos: QoSRegistry,
    operation_id: str,
    scenario: str = "static",
    requests: int = 2000,
    policy: str = POLICY_SOFTMAX,
    temperature: float = 0.05,
    task_class: str = "default",
    prior: str = "neutral",
    seed: int = 11,
    checkpoints: int = 40,
) -> dict[str, Any]:
    """Run one marketplace scenario and return a timeline the UI can plot.

    ``static`` learns a fixed market. ``degradation`` breaks the best provider
    halfway through. ``cold-start`` admits a superior provider at three quarters,
    which is the case a converged winner-take-all market cannot see.
    """
    market = Market.from_registry(
        qos, operation_id, prior=prior, task_class=task_class, seed=seed,
    )
    every = max(1, requests // max(1, checkpoints))
    events: list[dict[str, Any]] = []

    if scenario == "static":
        run = simulate(market, requests, policy, temperature, checkpoint_every=every)
        phases = [run]
    else:
        cut = requests // 2 if scenario == "degradation" else (requests * 3) // 4
        first = simulate(market, cut, policy, temperature, checkpoint_every=every)
        before_best = max(first.oracle_at_end, key=first.oracle_at_end.get)

        if scenario == "degradation":
            market.degrade(
                market.truth[before_best].degraded(DEGRADE_SUCCESS, DEGRADE_CORRECTNESS)
            )
            events.append({
                "request": cut, "type": "degradation", "provider_id": before_best,
                "label": f"{market.truth[before_best].meta.get('display_name', before_best)} "
                         f"success falls to {DEGRADE_SUCCESS:.2f}",
            })
        else:
            market.join(_entrant(operation_id))
            events.append({
                "request": cut, "type": "entry", "provider_id": ENTRANT_ID,
                "label": "a better provider joins, with a prior and no history",
            })

        second = simulate(market, requests - cut, policy, temperature,
                          checkpoint_every=every, offset=cut)
        phases = [first, second]

    oracle = market.oracle()
    choices = [c for run in phases for c in run.choices]
    successes = [s for run in phases for s in run.successes]
    latencies = [l for run in phases for l in run.latencies]
    regret = [r for run in phases for r in run.regret]
    timeline = [cp for run in phases for cp in run.checkpoints]

    # Scored over the tail of the final phase, not the whole of it: in the
    # changing scenarios the earlier phase measures a market that no longer
    # exists, and even within the final phase the migration itself is a
    # transient. Averaging across it would report a market as undecided long
    # after it had decided.
    tail = phases[-1]
    window = tail.choices[-max(200, len(tail.choices) // 4):]
    tail_counts = Counter(window)
    final_share = {p: tail_counts[p] / len(window) for p in market.truth}
    leader = max(final_share, key=final_share.get)
    true_best = max(oracle, key=oracle.get)

    return {
        "operation_id": operation_id,
        "scenario": scenario,
        "policy": policy,
        "temperature": temperature,
        "task_class": task_class,
        "prior": prior,
        "requests": len(choices),
        "providers": [
            {
                "provider_id": t.provider_id,
                "display_name": t.meta.get("display_name", t.provider_id),
                "server_id": t.meta.get("server_id"),
                "tier": t.meta.get("tier", "simulated"),
                "true_quality": round(oracle.get(t.provider_id, 0.0), 4),
                "learned_quality": round(market.learned().get(t.provider_id, 0.0), 4),
                "observations": market.registry.get(operation_id, t.provider_id).observations,
                "final_share": round(final_share.get(t.provider_id, 0.0), 4),
                "true_best": t.provider_id == true_best,
                "entered_late": t.provider_id == ENTRANT_ID,
                "hidden_profile": {k: round(v, 4) for k, v in t.observable().items()},
            }
            for t in market.truth.values()
        ],
        "timeline": timeline,
        "events": events,
        "summary": {
            "found_best": leader == true_best,
            "leader": leader,
            "true_best": true_best,
            "starved": sum(1 for p in market.truth if choices.count(p) == 0),
            "success_rate": round(sum(successes) / len(successes), 4),
            "mean_latency_ms": round(sum(latencies) / len(latencies), 1),
            "regret_per_request": round(sum(regret) / len(regret), 4),
        },
    }
