"""Quality-of-service provider selection (Branch 2).

Implements the stage specified in the manuscript. Retrieval selects an operation
``o_hat``; the registry resolves it to a provider set ``P(o_hat)``; this module
selects one provider from that set before execution.

Two properties are structural rather than incidental:

1. **Selection operates only inside P(o_hat).** Quality is never allowed to vote
   across operations. Measured: in a joint semantic-quality blend, weights above
   w = 0.5 let a high-quality provider of the WRONG operation win, and at
   top-k = 5 with w = 1.0 routing accuracy fell to 19.6%. ``joint_score`` exists
   for that experiment and refuses w > 0.5.

2. **Allocation is probabilistic, not winner-take-all.** Under argmax, providers
   below rank one receive no traffic and therefore no gradient to improve on;
   measured incentive fidelity caps at 0.584. Softmax reaches 1.000 at identical
   routing accuracy.

   The shipped default of tau = 0.05 is calibrated for the shipped registry,
   whose profiles are already measured. It does NOT transfer to a market being
   learned from scratch: with uninformative priors it concentrates before the
   estimates mean anything, finding the best of four providers in 35% of seeds
   against 100% at tau = 0.10 or 0.25 (``benchmark.research.qos_dynamic``).
   Initialize an unmeasured provider optimistically rather than lowering tau --
   that restores every policy to 100% and cuts regret by an order of magnitude,
   because a provider that must be sampled before it can be dismissed cannot be
   excluded unsampled. Tau then buys the incentive property rather than
   convergence: it is what keeps a measured, second-best provider receiving
   enough traffic to be re-measured.

Profiles live in ``data/providers.json``, deliberately not in ``data/tools.json``:
the orchestrator validates a SHA-256 of the tool registry at startup, so writing
a reputation update into the registry would invalidate the FAISS index on every
observation.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import random
import threading
from typing import Any

from orchestrator import config

# Profile components, and whether a larger raw value is better. Latency, cost,
# and error-like quantities are inverted during normalization.
COMPONENTS: dict[str, bool] = {
    "success_rate": True,
    "output_correctness": True,
    "latency": False,          # stored as latency_ms
    "cost": False,             # stored as cost_units
    "schema_compliance": True,
    "availability": True,
    "capability_fidelity": True,
}

RAW_FIELD = {"latency": "latency_ms", "cost": "cost_units"}

POLICY_SOFTMAX = "softmax"
POLICY_ARGMAX = "argmax"
POLICY_PROPORTIONAL = "proportional"
POLICIES = (POLICY_SOFTMAX, POLICY_ARGMAX, POLICY_PROPORTIONAL)

MAX_JOINT_WEIGHT = 0.5


class QoSError(RuntimeError):
    pass


@dataclasses.dataclass
class Provider:
    provider_id: str
    display_name: str
    tier: str
    incumbent: bool
    server_id: str
    tool_id: str
    profile: dict[str, float]
    observations: int = 0

    def raw(self, component: str) -> float:
        return float(self.profile[RAW_FIELD.get(component, component)])

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "display_name": self.display_name,
            "tier": self.tier,
            "incumbent": self.incumbent,
            "server_id": self.server_id,
            "tool_id": self.tool_id,
            "profile": dict(self.profile),
            "observations": self.observations,
        }


@dataclasses.dataclass
class Selection:
    provider: Provider
    score: float
    probability: float
    policy: str
    task_class: str
    ranked: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider.provider_id,
            "display_name": self.provider.display_name,
            "tier": self.provider.tier,
            "server_id": self.provider.server_id,
            "tool_id": self.provider.tool_id,
            "quality": round(self.score, 6),
            "probability": round(self.probability, 6),
            "policy": self.policy,
            "task_class": self.task_class,
            "candidates": self.ranked,
        }


def normalize(values: list[float], higher_is_better: bool) -> list[float]:
    """Min-max normalize within the provider set, inverting where required.

    Normalizing within P(o) rather than globally keeps the comparison to the
    providers actually competing for this operation. A degenerate set where every
    provider is identical yields 1.0 for all, so no provider is arbitrarily
    favoured.
    """
    low, high = min(values), max(values)
    if math.isclose(high, low):
        return [1.0] * len(values)
    span = high - low
    if higher_is_better:
        return [(v - low) / span for v in values]
    return [(high - v) / span for v in values]


class QoSRegistry:
    """Provider sets, task-conditioned scoring, and allocation."""

    def __init__(self, path: pathlib.Path | None = None, seed: int | None = None) -> None:
        self.path = pathlib.Path(path) if path else config.PROVIDERS_PATH
        self._lock = threading.Lock()
        self._rng = random.Random(config.QOS_SEED if seed is None else seed)
        self.load()

    # --- loading ----------------------------------------------------------

    def load(self) -> None:
        if not self.path.exists():
            raise QoSError(
                f"Provider registry not found at {self.path}. "
                "Run python -m scripts.build_providers"
            )
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.task_classes: dict[str, dict[str, float]] = raw["task_classes"]
        self.providers: dict[str, list[Provider]] = {
            operation: [Provider(**entry) for entry in entries]
            for operation, entries in raw["providers"].items()
        }
        self._validate_task_classes()

    def _validate_task_classes(self) -> None:
        for name, weights in self.task_classes.items():
            missing = set(COMPONENTS) - set(weights)
            if missing:
                raise QoSError(f"Task class {name!r} is missing weights for {sorted(missing)}")
            total = sum(weights.values())
            if not math.isclose(total, 1.0, abs_tol=1e-6):
                raise QoSError(
                    f"Task class {name!r} weights sum to {total:.4f}, expected 1.0"
                )

    # --- scoring ----------------------------------------------------------

    def quality(self, operation_id: str, task_class: str = "default") -> list[tuple[Provider, float]]:
        """Q_{p|t} for every provider of an operation, highest first."""
        providers = self.providers.get(operation_id)
        if not providers:
            raise QoSError(f"No providers registered for operation {operation_id!r}")
        weights = self.task_classes.get(task_class)
        if weights is None:
            raise QoSError(
                f"Unknown task class {task_class!r}. "
                f"Known: {', '.join(sorted(self.task_classes))}"
            )

        normalized: dict[str, list[float]] = {}
        for component, higher_is_better in COMPONENTS.items():
            normalized[component] = normalize(
                [p.raw(component) for p in providers], higher_is_better
            )

        scored = []
        for i, provider in enumerate(providers):
            score = sum(weights[c] * normalized[c][i] for c in COMPONENTS)
            scored.append((provider, score))
        return sorted(scored, key=lambda pair: -pair[1])

    # --- allocation -------------------------------------------------------

    def allocation(
        self,
        operation_id: str,
        task_class: str = "default",
        policy: str | None = None,
        temperature: float | None = None,
    ) -> list[tuple[Provider, float, float]]:
        """Return (provider, quality, probability), highest quality first."""
        policy = policy or config.QOS_POLICY
        temperature = temperature if temperature is not None else config.QOS_TEMPERATURE
        if policy not in POLICIES:
            raise QoSError(f"Unknown policy {policy!r}. Known: {', '.join(POLICIES)}")

        scored = self.quality(operation_id, task_class)
        scores = [s for _, s in scored]

        if policy == POLICY_ARGMAX:
            probs = [0.0] * len(scores)
            probs[0] = 1.0
        elif policy == POLICY_PROPORTIONAL:
            total = sum(scores)
            probs = [s / total for s in scores] if total > 0 else [1 / len(scores)] * len(scores)
        else:
            tau = max(temperature, 1e-6)
            # Shift by the maximum before exponentiating; at tau = 0.05 the raw
            # exponentials overflow float64 otherwise.
            top = max(scores)
            weights = [math.exp((s - top) / tau) for s in scores]
            total = sum(weights)
            probs = [w / total for w in weights]

        return [(p, s, prob) for (p, s), prob in zip(scored, probs)]

    def select(
        self,
        operation_id: str,
        task_class: str = "default",
        policy: str | None = None,
        temperature: float | None = None,
        deterministic: bool | None = None,
    ) -> Selection:
        """Choose one provider from P(operation_id)."""
        policy = policy or config.QOS_POLICY
        allocation = self.allocation(operation_id, task_class, policy, temperature)
        deterministic = (
            config.QOS_DETERMINISTIC if deterministic is None else deterministic
        )

        if policy == POLICY_ARGMAX or deterministic:
            chosen_index = 0
        else:
            with self._lock:
                roll = self._rng.random()
            cumulative, chosen_index = 0.0, len(allocation) - 1
            for i, (_, _, prob) in enumerate(allocation):
                cumulative += prob
                if roll <= cumulative:
                    chosen_index = i
                    break

        provider, score, probability = allocation[chosen_index]
        ranked = [
            {
                "provider_id": p.provider_id,
                "display_name": p.display_name,
                "tier": p.tier,
                "incumbent": p.incumbent,
                "quality": round(s, 6),
                "probability": round(prob, 6),
                "selected": i == chosen_index,
                "profile": dict(p.profile),
            }
            for i, (p, s, prob) in enumerate(allocation)
        ]
        return Selection(
            provider=provider,
            score=score,
            probability=probability,
            policy=policy,
            task_class=task_class,
            ranked=ranked,
        )

    def joint_score(
        self,
        operation_id: str,
        routing_score: float,
        task_class: str = "default",
        w: float = 0.3,
    ) -> list[tuple[Provider, float]]:
        """Experimental joint semantic-quality score, capped at w = 0.5.

        Retained only so the bound can be reproduced. Above w = 0.5 a
        high-quality provider of the wrong operation begins to win; at top-k = 5
        with w = 1.0 routing accuracy fell to 19.6 percent. Production selection
        must use ``select()``, which never lets quality vote across operations.
        """
        if w > MAX_JOINT_WEIGHT:
            raise QoSError(
                f"joint quality weight w={w} exceeds the measured safe bound "
                f"{MAX_JOINT_WEIGHT}; above it a high-quality provider of the "
                "wrong operation wins and routing accuracy collapses"
            )
        return [
            (p, (1 - w) * routing_score + w * q)
            for p, q in self.quality(operation_id, task_class)
        ]

    # --- reputation -------------------------------------------------------

    def observe(
        self,
        operation_id: str,
        provider_id: str,
        success: bool,
        latency_ms: float | None = None,
        schema_ok: bool = True,
        correctness: float | None = None,
        cost_units: float | None = None,
        alpha: float | None = None,
    ) -> dict[str, Any]:
        """Fold one execution outcome back into the provider's profile.

        This is the "Evaluate -> Update reputation" edge of the feedback loop.
        An exponentially weighted update with a small alpha keeps a single bad
        execution from evicting an otherwise reliable provider, while a run of
        failures moves it steadily down the allocation.

        ``correctness`` and ``cost_units`` are optional because the orchestrator
        cannot always measure them: correctness needs a grader or a reference
        answer, cost needs the provider to report one. The dynamic marketplace
        experiment supplies both, so the same update path covers every scored
        component except availability, which is observed by liveness probing
        rather than by execution.
        """
        alpha = config.QOS_LEARNING_RATE if alpha is None else alpha
        provider = self.get(operation_id, provider_id)
        if provider is None:
            raise QoSError(f"No provider {provider_id!r} for operation {operation_id!r}")

        with self._lock:
            profile = provider.profile
            profile["success_rate"] = round(
                (1 - alpha) * profile["success_rate"] + alpha * (1.0 if success else 0.0), 6
            )
            profile["schema_compliance"] = round(
                (1 - alpha) * profile["schema_compliance"] + alpha * (1.0 if schema_ok else 0.0), 6
            )
            if latency_ms is not None:
                profile["latency_ms"] = round(
                    (1 - alpha) * profile["latency_ms"] + alpha * float(latency_ms), 3
                )
            if correctness is not None:
                profile["output_correctness"] = round(
                    (1 - alpha) * profile["output_correctness"] + alpha * float(correctness), 6
                )
            if cost_units is not None:
                profile["cost_units"] = round(
                    (1 - alpha) * profile["cost_units"] + alpha * float(cost_units), 6
                )
            provider.observations += 1

        return provider.as_dict()

    def add_provider(self, operation_id: str, provider: Provider) -> dict[str, Any]:
        """Register a provider for an operation at runtime.

        Entry is the cold-start case: a newcomer has no execution history, so its
        profile is a prior rather than a measurement, and it can only be
        evaluated on traffic it has not yet earned. Whether that traffic ever
        arrives is a property of the allocation policy, which is what
        ``benchmark.research.qos_dynamic --experiment cold-start`` measures.
        """
        if self.get(operation_id, provider.provider_id) is not None:
            raise QoSError(
                f"Provider {provider.provider_id!r} already serves operation {operation_id!r}"
            )
        with self._lock:
            self.providers.setdefault(operation_id, []).append(provider)
        return provider.as_dict()

    # --- mutation, for the simulation controls -----------------------------

    def set_profile(self, operation_id: str, provider_id: str, updates: dict[str, float]) -> dict:
        """Overwrite profile components directly. Used by the UI simulator."""
        provider = self.get(operation_id, provider_id)
        if provider is None:
            raise QoSError(f"No provider {provider_id!r} for operation {operation_id!r}")
        allowed = set(provider.profile)
        unknown = set(updates) - allowed
        if unknown:
            raise QoSError(f"Unknown profile fields: {sorted(unknown)}")
        with self._lock:
            provider.profile.update({k: float(v) for k, v in updates.items()})
        return provider.as_dict()

    def get(self, operation_id: str, provider_id: str) -> Provider | None:
        return next(
            (p for p in self.providers.get(operation_id, []) if p.provider_id == provider_id),
            None,
        )

    def reset(self) -> None:
        """Reload from disk, discarding runtime mutations."""
        self.load()

    # --- introspection ----------------------------------------------------

    @property
    def operation_count(self) -> int:
        return len(self.providers)

    @property
    def provider_count(self) -> int:
        return sum(len(v) for v in self.providers.values())

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": config.QOS_ENABLED,
            "operations": self.operation_count,
            "providers": self.provider_count,
            "providers_per_operation": (
                self.provider_count / self.operation_count if self.operation_count else 0
            ),
            "policy": config.QOS_POLICY,
            "temperature": config.QOS_TEMPERATURE,
            "task_classes": sorted(self.task_classes),
            "default_task_class": config.QOS_TASK_CLASS,
            "deterministic": config.QOS_DETERMINISTIC,
            "registry_path": str(self.path),
        }
