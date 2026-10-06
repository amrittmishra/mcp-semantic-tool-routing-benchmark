"""Paper figures for the dynamic QoS marketplace simulation (PR #1).

    python -m benchmark.research.plot_qos_dynamic

Reads the committed summaries in ``benchmark/qos_dynamic_*.json`` for the
aggregate bars, and re-runs a handful of seeded trajectories for the traffic-
share timelines (same generative model and ``select``/``observe`` loop as
``qos_dynamic`` / ``/qos/simulate``).
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from orchestrator.qos import POLICY_ARGMAX, POLICY_SOFTMAX
from orchestrator.qos_sim import simulate

from benchmark.research.qos_dynamic import (
    DEGRADED_ALPHA,
    MARKET,
    NEWCOMER,
    Market,
)

BENCHMARK_DIR = pathlib.Path(__file__).resolve().parents[1]
RESULTS = BENCHMARK_DIR / "qos_dynamic_results.json"
OPTIMISTIC = BENCHMARK_DIR / "qos_dynamic_optimistic.json"

# Match the agent-benchmark palette: muted, print-friendly, no purple glow.
INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ALPHA_C = "#0f766e"   # teal — true best / entrant when discovered
BETA_C = "#2563eb"    # blue
GAMMA_C = "#d97706"   # amber
DELTA_C = "#9ca3af"   # gray — weak provider
FAIL_C = "#b91c1c"    # red — broken / starved path
OK_C = "#15803d"      # green — recovered path

PROVIDER_COLORS = {
    "provider_alpha": ALPHA_C,
    "provider_beta": BETA_C,
    "provider_gamma": GAMMA_C,
    "provider_delta": DELTA_C,
    "provider_epsilon": "#7c3aed",
}

SHORT = {
    "provider_alpha": "alpha",
    "provider_beta": "beta",
    "provider_gamma": "gamma",
    "provider_delta": "delta",
    "provider_epsilon": "epsilon",
}


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _rolling_share(choices: list[str], providers: list[str],
                   window: int = 200, every: int = 50) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Trailing traffic share, sampled every ``every`` requests."""
    xs, series = [], {p: [] for p in providers}
    for t in range(window, len(choices) + 1, every):
        seg = choices[t - window:t]
        counts = Counter(seg)
        xs.append(t)
        for p in providers:
            series[p].append(counts[p] / len(seg))
    return np.asarray(xs), {p: np.asarray(v) for p, v in series.items()}


def _stacked(ax, xs: np.ndarray, series: dict[str, np.ndarray],
             providers: list[str], event: int | None = None, event_label: str = "") -> None:
    base = np.zeros(len(xs))
    for p in providers:
        top = base + series[p]
        ax.fill_between(xs, base, top, color=PROVIDER_COLORS[p], alpha=0.85,
                        linewidth=0, label=SHORT[p])
        base = top
    if event is not None:
        ax.axvline(event, color=INK, linestyle="--", linewidth=1.0, alpha=0.7)
        if event_label:
            # Keep the marker label inside the axes, with a halo so it does not
            # disappear into the stacked fill.
            ax.text(
                event + (xs[-1] - xs[0]) * 0.015, 0.92, event_label,
                fontsize=8, color=INK, va="top", ha="left", weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.92),
                zorder=6, clip_on=False,
            )
    ax.set_ylim(0, 1.02)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylabel("Trailing traffic share")
    ax.set_xlabel("Request")
    _style(ax)


def plot_regret_comparison(out: pathlib.Path) -> pathlib.Path:
    """Table XII as a grouped bar chart: regret under both priors."""
    neutral = json.loads(RESULTS.read_text())["static"]["policies"]
    optimistic = json.loads(OPTIMISTIC.read_text())["static"]["policies"]

    # Policies that carry the paper's claim; omit random (floor) for clarity.
    order = ["argmax", "softmax t=0.25", "softmax t=0.1", "softmax t=0.05"]
    labels = ["argmax", r"softmax $\tau{=}0.25$", r"softmax $\tau{=}0.10$",
              r"softmax $\tau{=}0.05$"]
    n_vals = [neutral[k]["regret_per_request"] for k in order]
    o_vals = [optimistic[k]["regret_per_request"] for k in order]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(order))
    w = 0.36
    ymax = max(n_vals) * 1.45
    ax.set_ylim(0, ymax)
    b1 = ax.bar(x - w / 2, n_vals, w, color=FAIL_C, label="Uninformative prior")
    b2 = ax.bar(x + w / 2, o_vals, w, color=OK_C, label="Optimistic prior")
    # Label above every bar; tiny optimistic bars get a minimum lift off the axis.
    # Stagger labels when paired bars are equal height so 0.086/0.086 never collide.
    for i, (r1, r2) in enumerate(zip(b1, b2)):
        h1, h2 = r1.get_height(), r2.get_height()
        lift1 = max(h1 + ymax * 0.03, ymax * 0.06)
        lift2 = max(h2 + ymax * 0.03, ymax * 0.06)
        if abs(h1 - h2) < ymax * 0.02:
            lift1 += ymax * 0.06
            lift2 += ymax * 0.02
        ax.text(r1.get_x() + r1.get_width() / 2, lift1, f"{h1:.3f}",
                ha="center", va="bottom", fontsize=8, color=FAIL_C, weight="bold",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.92),
                zorder=5, clip_on=False)
        ax.text(r2.get_x() + r2.get_width() / 2, lift2, f"{h2:.3f}",
                ha="center", va="bottom", fontsize=8, color=OK_C, weight="bold",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.92),
                zorder=5, clip_on=False)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean quality regret per request")
    ax.set_title("Learning provider quality — regret by prior and allocation policy")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.tick_params(axis="x", pad=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_fidelity_found(out: pathlib.Path) -> pathlib.Path:
    """Companion bars: incentive fidelity and fraction of seeds that found best."""
    neutral = json.loads(RESULTS.read_text())["static"]["policies"]
    optimistic = json.loads(OPTIMISTIC.read_text())["static"]["policies"]
    order = ["argmax", "softmax t=0.25", "softmax t=0.1", "softmax t=0.05"]
    labels = ["argmax", r"$\tau{=}0.25$", r"$\tau{=}0.10$", r"$\tau{=}0.05$"]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=False)

    for ax, metric, title, ylim in (
        (axes[0], "fidelity", "Incentive fidelity (quality↔traffic)", (-0.05, 1.08)),
        (axes[1], "found_best_rate", "Seeds that found the true best", (0, 1.12)),
    ):
        n_vals = [neutral[k][metric] for k in order]
        o_vals = [optimistic[k][metric] for k in order]
        x = np.arange(len(order))
        w = 0.36
        b1 = ax.bar(x - w / 2, n_vals, w, color=FAIL_C, label="Uninformative")
        b2 = ax.bar(x + w / 2, o_vals, w, color=OK_C, label="Optimistic")
        ax.set_xticks(x, labels)
        ax.set_ylim(*ylim)
        y0, y1 = ylim
        for rect, color, vals in (
            (b1, FAIL_C, n_vals), (b2, OK_C, o_vals),
        ):
            for r, v in zip(rect, vals):
                # Format fidelity as 0.xx, found-best as percent.
                label = f"{v:.2f}" if metric == "fidelity" else f"{v * 100:.0f}%"
                lift = max(v + (y1 - y0) * 0.04, (y1 - y0) * 0.08 + y0)
                # Keep labels inside the axes for negative/near-zero fidelity.
                lift = min(lift, y1 - (y1 - y0) * 0.06)
                ax.text(r.get_x() + r.get_width() / 2, lift, label,
                        ha="center", va="bottom", fontsize=7.5, color=color,
                        weight="bold",
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                                  edgecolor="none", alpha=0.92),
                        zorder=5, clip_on=False)
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=7)
        ax.tick_params(axis="x", pad=6)
        _style(ax)

    fig.suptitle("Table XII metrics — 10,000 requests × 20 seeds", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cold_start_timelines(out: pathlib.Path, requests: int = 4000,
                              seed: int = 11) -> pathlib.Path:
    """Stacked share: uninformative argmax never tries the entrant; optimistic does."""
    join_at = (requests * 3) // 4
    providers_base = [t.provider_id for t in MARKET]
    providers_all = providers_base + [NEWCOMER.provider_id]

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.6), sharex=True)

    configs = [
        ("neutral", POLICY_ARGMAX, 0.0,
         "Uninformative prior · argmax — entrant never sampled"),
        ("optimistic", POLICY_ARGMAX, 0.0,
         "Optimistic prior · argmax — entrant discovered within detection floor"),
    ]

    for ax, (prior, policy, tau, title) in zip(axes, configs):
        market = Market(seed=seed, prior=prior)
        first = simulate(market, join_at, policy, tau)
        market.join(NEWCOMER)
        second = simulate(market, requests - join_at, policy, tau)
        choices = first.choices + second.choices
        xs, series = _rolling_share(choices, providers_all, window=200, every=40)
        # Providers that never appear stay at zero; keep a stable stack order.
        order = providers_base + (
            [NEWCOMER.provider_id] if NEWCOMER.provider_id in series else []
        )
        _stacked(ax, xs, series, order, event=join_at,
                 event_label="ε joins")
        # Annotate final entrant share in the top-right, clear of the stack.
        final = series[NEWCOMER.provider_id][-1] if len(series[NEWCOMER.provider_id]) else 0.0
        ax.set_title(f"{title}\n(ε final share {final * 100:.0f}%)", fontsize=10)

    handles = [Patch(facecolor=PROVIDER_COLORS[p], label=SHORT[p]) for p in providers_all]
    axes[0].legend(handles=handles, frameon=False, fontsize=7, ncol=5,
                   loc="upper left", bbox_to_anchor=(0, 1.02))
    axes[1].set_xlabel("Request")
    fig.suptitle("Cold start at request 75% — traffic share over market time",
                 fontsize=11, color=INK, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_degradation_timeline(out: pathlib.Path, requests: int = 4000,
                              seed: int = 7) -> pathlib.Path:
    """Softmax τ=0.10 migrates off a silently broken leader."""
    change_at = requests // 2
    providers = [t.provider_id for t in MARKET]

    market = Market(seed=seed, prior="neutral")
    first = simulate(market, change_at, POLICY_SOFTMAX, 0.10)
    market.degrade(DEGRADED_ALPHA)
    second = simulate(market, requests - change_at, POLICY_SOFTMAX, 0.10)
    choices = first.choices + second.choices
    xs, series = _rolling_share(choices, providers, window=200, every=40)

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    _stacked(ax, xs, series, providers, event=change_at,
             event_label="α degrades")
    ax.set_title(r"Silent degradation — softmax $\tau{=}0.10$ migrates to β",
                 fontsize=11)
    handles = [Patch(facecolor=PROVIDER_COLORS[p], label=SHORT[p]) for p in providers]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_combined(out: pathlib.Path) -> pathlib.Path:
    """One composition figure for the paper / README."""
    regret = BENCHMARK_DIR / "qos_regret.png"
    cold = BENCHMARK_DIR / "qos_cold_start.png"
    deg = BENCHMARK_DIR / "qos_degradation.png"
    # Ensure siblings exist.
    if not regret.exists():
        plot_regret_comparison(regret)
    if not cold.exists():
        plot_cold_start_timelines(cold)
    if not deg.exists():
        plot_degradation_timeline(deg)

    # Combined is already produced as separate panels; copy a note file path.
    # For convenience, also emit a 2×2 overview built live.
    neutral = json.loads(RESULTS.read_text())["static"]["policies"]
    optimistic = json.loads(OPTIMISTIC.read_text())["static"]["policies"]
    cold_json = json.loads(RESULTS.read_text())["cold_start"]["policies"]

    fig = plt.figure(figsize=(9.0, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

    # (0,0) regret
    ax = fig.add_subplot(gs[0, 0])
    order = ["argmax", "softmax t=0.05"]
    labels = ["argmax", r"softmax $\tau{=}0.05$"]
    x = np.arange(len(order))
    w = 0.36
    n_vals = [neutral[k]["regret_per_request"] for k in order]
    o_vals = [optimistic[k]["regret_per_request"] for k in order]
    ymax = max(n_vals) * 1.4
    ax.set_ylim(0, ymax)
    b1 = ax.bar(x - w / 2, n_vals, w, color=FAIL_C, label="Uninformative")
    b2 = ax.bar(x + w / 2, o_vals, w, color=OK_C, label="Optimistic")
    for rect, color in [(r, FAIL_C) for r in b1] + [(r, OK_C) for r in b2]:
        h = rect.get_height()
        lift = max(h + ymax * 0.04, ymax * 0.08)
        ax.text(rect.get_x() + rect.get_width() / 2, lift, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8, color=color, weight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                          edgecolor="none", alpha=0.92),
                zorder=5, clip_on=False)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Regret / request")
    ax.set_title("A. Prior fixes premature concentration")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.tick_params(axis="x", pad=6)
    _style(ax)

    # (0,1) cold-start final share of entrant
    ax = fig.add_subplot(gs[0, 1])
    keys = [
        ("neutral/argmax", "uninf.\nargmax"),
        ("neutral/softmax t=0.05", "uninf.\nτ=0.05"),
        ("neutral/softmax t=0.25", "uninf.\nτ=0.25"),
        ("optimistic/argmax", "opt.\nargmax"),
        ("optimistic/softmax t=0.05", "opt.\nτ=0.05"),
    ]
    shares = [cold_json[k]["final_share"] * 100 for k, _ in keys]
    colors = [FAIL_C, FAIL_C, MUTED, OK_C, OK_C]
    ax.set_ylim(0, 125)
    bars = ax.bar([lab for _, lab in keys], shares, color=colors, width=0.7)
    for rect, share in zip(bars, shares):
        lift = max(share + 4, 8)
        ax.text(rect.get_x() + rect.get_width() / 2, lift, f"{share:.0f}%",
                ha="center", va="bottom", fontsize=8, weight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                          edgecolor="none", alpha=0.92),
                zorder=5, clip_on=False)
    ax.set_ylabel("Entrant final traffic share (%)")
    ax.set_title("B. Cold start — barrier vs discovery")
    ax.tick_params(axis="x", pad=6)
    _style(ax)

    # (1,:) cold-start timelines side by side, short runs
    for col, (prior, title) in enumerate((
        ("neutral", "C. Uninformative · argmax timeline"),
        ("optimistic", "D. Optimistic · argmax timeline"),
    )):
        ax = fig.add_subplot(gs[1, col])
        requests = 3000
        join_at = (requests * 3) // 4
        providers_all = [t.provider_id for t in MARKET] + [NEWCOMER.provider_id]
        market = Market(seed=11, prior=prior)
        first = simulate(market, join_at, POLICY_ARGMAX, 0.0)
        market.join(NEWCOMER)
        second = simulate(market, requests - join_at, POLICY_ARGMAX, 0.0)
        choices = first.choices + second.choices
        xs, series = _rolling_share(choices, providers_all, window=150, every=30)
        _stacked(ax, xs, series, providers_all, event=join_at, event_label="ε joins")
        ax.set_title(title, fontsize=10)
        if col == 0:
            ax.legend(
                handles=[Patch(facecolor=PROVIDER_COLORS[p], label=SHORT[p])
                         for p in providers_all],
                frameon=False, fontsize=6, ncol=5, loc="upper left",
            )

    fig.suptitle("QoS marketplace simulation — dynamic reputation loop",
                 fontsize=12, color=INK, y=0.98)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    paths = [
        plot_regret_comparison(BENCHMARK_DIR / "qos_regret.png"),
        plot_fidelity_found(BENCHMARK_DIR / "qos_fidelity.png"),
        plot_cold_start_timelines(BENCHMARK_DIR / "qos_cold_start.png"),
        plot_degradation_timeline(BENCHMARK_DIR / "qos_degradation.png"),
        plot_combined(BENCHMARK_DIR / "qos_simulation.png"),
    ]
    for p in paths:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
